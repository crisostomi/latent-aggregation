import dataclasses
import random
from functools import partial
from pathlib import Path
from pydoc import locate
from typing import Dict, List, Union

import hydra
import omegaconf
import pytorch_lightning as pl
from datasets import Dataset, DatasetDict
from hydra import compose, initialize
from hydra.utils import instantiate
from pytorch_lightning import Callback, Trainer, seed_everything
from torch.utils.data import DataLoader
from tqdm import tqdm

from nn_core.callbacks import NNTemplateCore
from nn_core.common import PROJECT_ROOT
from nn_core.model_logging import NNLogger
from nn_core.serialization import NNCheckpointIO, load_model

from la.data.my_dataset_dict import MyDatasetDict
from la.pl_modules.classifier import Classifier
from la.utils.io_utils import add_ids_to_dataset, load_data, preprocess_dataset
from la.utils.utils import build_callbacks, get_checkpoint_callback


@dataclasses.dataclass
class Task:
    class_idxs: list
    classes: list
    global_to_local: Dict
    id: str
    dataset: Union[DatasetDict, MyDatasetDict] = None
    embedded_dataset: MyDatasetDict = None
    model: pl.LightningModule = None

    def metadata(self):
        return {
            "id": self.id,
            "class_idxs": self.class_idxs,
            "classes": self.classes,
            "global_to_local": self.global_to_local,
            "model": self.model,
        }


def run(cfg: omegaconf.DictConfig):
    seed_everything(cfg.seed)

    original_dataset = dataset = load_data(cfg)  # .shard(num_shards=10, index=0)
    dataset = preprocess_dataset(dataset, cfg)
    dataset = add_ids_to_dataset(dataset)

    img_size = dataset["train"][0]["x"].shape[1]

    class_names = original_dataset["train"].features["fine_label"].names
    class_idxs = [original_dataset["train"].features["fine_label"].str2int(class_name) for class_name in class_names]

    subtask_dataset_path = Path(cfg.subtask_dataset_path)
    subtask_dataset_path.mkdir(exist_ok=True)

    # dataset = load_existing_tasks(subtask_dataset_path)

    transform_func = instantiate(cfg.nn.model.transform_func)

    dataset = dataset.map(
        desc=f"Preprocessing samples",
        function=lambda x: {"x": transform_func(x["x"])},
    )

    # create new tasks, task_0 is the dummy task with all the dataset, remaining tasks will be composed
    # of K random classes each
    all_classes_task = Task(
        class_idxs=class_idxs,
        classes=class_names,
        global_to_local={i: i for i in range(len(class_names))},
        id="all_classes",
        dataset=MyDatasetDict(train=dataset["train"], test=dataset["test"]),
        embedded_dataset=MyDatasetDict(train=DatasetDict(), test=DatasetDict()),
    )

    tasks = [all_classes_task]

    for _ in range(cfg.num_tasks):
        task = create_task(cfg.num_task_classes, dataset, class_names, class_idxs)
        tasks.append(task)

    loader_func = partial(DataLoader, batch_size=100, pin_memory=False, num_workers=8)

    for task in tasks:
        print(f"Training model for task {task.id}")

        num_classes = len(task.classes)
        model: pl.LightningModule = hydra.utils.instantiate(
            cfg.nn.model,
            _recursive_=False,
            num_classes=num_classes,
            model=cfg.nn.model.model,
            input_dim=img_size,
        )

        train_loader = loader_func(
            task.dataset["train"],
            shuffle=True,
        )

        val_loader = loader_func(
            task.dataset["test"],
            shuffle=False,
        )

        template_core: NNTemplateCore = NNTemplateCore(
            restore_cfg=cfg.train.get("restore", None),
        )
        callbacks: List[Callback] = build_callbacks(cfg.train.callbacks, template_core)

        storage_dir: str = cfg.core.storage_dir
        logger: NNLogger = NNLogger(logging_cfg=cfg.train.logging, cfg=cfg, resume_id=template_core.resume_id)

        # Use this in case we need to restore models, search for it in the wandb UI
        logger.experiment.config["task_classes"] = task.id

        trainer = pl.Trainer(
            default_root_dir=storage_dir,
            plugins=[NNCheckpointIO(jailing_dir=logger.run_dir)],
            logger=logger,
            callbacks=callbacks,
            **cfg.train.trainer,
        )
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        best_model_path = get_checkpoint_callback(callbacks).best_model_path

        task.model = {
            "path": best_model_path,
            "class": str(model.__class__.__module__ + "." + model.__class__.__qualname__),
        }

        logger.experiment.finish()

        task_dataset = MyDatasetDict(task.dataset)
        task_dataset["metadata"] = task.metadata()

        Path(f"{cfg.subtask_dataset_path}/{task.id}").mkdir(exist_ok=True, parents=True)
        task_dataset.save_to_disk(f"{cfg.subtask_dataset_path}/{task.id}")

        task_dataset = MyDatasetDict.load_from_disk(f"{cfg.subtask_dataset_path}/{task.id}")
        task.dataset = task_dataset

    # embed the tasks with the corresponding pretrained model
    for task in tasks:
        embed_and_save_samples(task, cfg)


def create_task(num_task_classes, dataset, class_names, class_idxs) -> Task:
    task_class_indices = sorted(random.sample(class_idxs, k=num_task_classes))

    global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(task_class_indices)}

    task_classes = [class_names[i] for i in task_class_indices]
    task_str = "_".join([str(i) for i in task_class_indices])

    task_dataset = dataset.filter(lambda row: row["y"] in task_class_indices)
    task_dataset = task_dataset.map(lambda row: {"y": global_to_local[row["y"]]})

    task_dataset.set_format(type="torch", columns=["x", "y", "id"])

    embeds = MyDatasetDict(train=DatasetDict(), test=DatasetDict())

    task = Task(
        class_idxs=task_class_indices,
        classes=task_classes,
        global_to_local=global_to_local,
        id=task_str,
        embedded_dataset=embeds,
        dataset=task_dataset,
    )

    return task


def load_existing_tasks(subtask_dataset_path):
    tasks = []
    for dataset_path in subtask_dataset_path.glob("*"):
        dataset = MyDatasetDict.load_from_disk(dataset_path)

        task = Task(
            id=dataset_path.name,
            class_idxs=dataset["metadata"]["class_idxs"],
            classes=dataset["metadata"]["classes"],
            global_to_local=dataset["metadata"]["global_to_local"],
            dataset=dataset,
        )

        tasks.append(task)
    return dataset


def embed_and_save_samples(task, cfg, batch_size=100) -> Dict:
    modes = ["train", "test"]

    model_path = task.model["path"]
    model_class = locate(task.model["class"])

    model = load_model(model_class, checkpoint_path=Path(model_path + ".zip"))
    model.eval().cuda()

    for mode in modes:
        mode_embeddings = []
        mode_ids = []
        mode_labels = []

        mode_loader = DataLoader(
            task.dataset[mode],
            batch_size=batch_size,
            pin_memory=True,
            shuffle=False,
            num_workers=4,
        )

        for batch in tqdm(mode_loader, desc=f"Embedding {mode} samples for task {task.id}"):
            x = batch["x"].to("cuda")
            embeds = model(x)["embeds"].detach()

            mode_embeddings.extend(embeds)
            mode_ids.extend(batch["id"])
            mode_labels.extend(batch["y"])

        task.embedded_dataset[mode] = Dataset.from_dict(
            {
                "embeds": mode_embeddings,
                "id": mode_ids,
                "y": mode_labels,
            }
        )

    model.cpu()
    task.embedded_dataset["metadata"] = task.metadata()

    subtask_embedding_path = Path(cfg.subtask_embedding_path)
    (subtask_embedding_path / task.id).mkdir(exist_ok=True, parents=True)
    task.embedded_dataset.save_to_disk(subtask_embedding_path / task.id)


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="class_subsets")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
