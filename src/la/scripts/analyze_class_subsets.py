from functools import partial
from pathlib import Path
import hydra
from hydra import initialize, compose
from typing import Dict, List

import omegaconf
import pytorch_lightning
from la.data.my_dataset_dict import MyDatasetDict
from nn_core.common import PROJECT_ROOT
from la.utils.io_utils import add_ids_to_dataset, load_data
from la.utils.io_utils import preprocess_dataset
import dataclasses
from typing import List
from nn_core.callbacks import NNTemplateCore
from nn_core.model_logging import NNLogger
from nn_core.serialization import NNCheckpointIO
from datasets import Dataset, DatasetDict
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from torch.utils.data import DataLoader
from pytorch_lightning import Callback
from la.utils.utils import build_callbacks
from torch.utils.data import DataLoader
from la.data.my_dataset_dict import MyDatasetDict
from tqdm import tqdm
import dataclasses
from typing import Union
import random
from hydra.utils import instantiate
from la.utils.utils import get_checkpoint_callback
from pydoc import locate
from nn_core.serialization import load_model
from typing import List
from nn_core.callbacks import NNTemplateCore
from nn_core.model_logging import NNLogger
from nn_core.serialization import NNCheckpointIO
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from pytorch_lightning import Callback
from la.pl_modules.classifier import Classifier
import logging
from la.utils.utils import build_callbacks
import json

pylogger = logging.getLogger(__name__)


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


@dataclasses.dataclass
class Result:
    task_id: str
    num_train_classes: int
    metric_name: str
    score: float


def run(cfg: omegaconf.DictConfig):
    seed_everything(cfg.seed)

    subtasks, global_task = load_existing_tasks(Path(cfg.subtask_embedding_path), cfg.num_task_classes)

    all_performances = {task.id: {} for task in subtasks}

    global_dataset = global_task.embedded_dataset
    global_dataset.set_format(type="torch", columns=["embeds", "y"])

    for task in subtasks:
        task_dataset = task.embedded_dataset
        task_dataset.set_format(type="torch", columns=["embeds", "y"])

        # performance of a classifier trained on the task-specific class set
        task_specific_performances = run_classification_on_space(task, task_dataset, len(task.classes), cfg)

        task_subspace_in_global_space = global_dataset.filter(lambda row: row["y"].item() in task.class_idxs)

        global_to_local = {int(k): v for k, v in task.global_to_local.items()}
        task_subspace_in_global_space = task_subspace_in_global_space.map(
            lambda row: {"y": global_to_local[row["y"].item()], "embeds": row["embeds"]},
        )

        # performance of a classifier trained on the subregion of the global space corresponding to the task-specific class set
        restricted_performances = run_classification_on_space(
            task, task_subspace_in_global_space, len(task.classes), cfg
        )

        all_performances[task.id] = {
            "task_specific_test_acc": task_specific_performances.score,
            "restricted_test_acc": restricted_performances.score,
        }

    Path(cfg.results_path).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.results_path, "w+") as f:
        json.dump(all_performances, f)


def load_existing_tasks(subtask_dataset_path: str, num_task_classes: int):
    """
    Load existing tasks from a directory of subtask datasets.
    Returns both the global task over all the classes and the subtasks on the subsets of classes.
    """
    subtasks = []
    global_task = None
    for dataset_path in subtask_dataset_path.glob("*"):
        dataset = MyDatasetDict.load_from_disk(dataset_path)

        task = Task(
            id=dataset_path.name,
            class_idxs=dataset["metadata"]["class_idxs"],
            classes=dataset["metadata"]["classes"],
            global_to_local=dataset["metadata"]["global_to_local"],
            embedded_dataset=dataset,
        )

        if task.id == "all_classes":
            global_task = task
            continue

        if len(task.classes) != num_task_classes:
            pylogger.info(
                f"Number of task classes {len(task.classes)} is different from expected number {num_task_classes}"
            )
            continue

        subtasks.append(task)

    return subtasks, global_task


def run_classification_on_space(task, dataset, num_classes, cfg):
    eval_train_loader = DataLoader(
        dataset["train"],
        batch_size=64,
        pin_memory=True,
        shuffle=True,
        num_workers=8,
    )

    eval_test_loader = DataLoader(
        dataset["test"],
        batch_size=64,
        pin_memory=True,
        shuffle=False,
        num_workers=0,
    )

    model = Classifier(
        input_dim=dataset["train"]["embeds"].size(1),
        num_classes=num_classes,
        lr=1e-4,
        deep=True,
        x_feature="embeds",
        y_feature="y",
    )

    storage_dir: str = cfg.core.storage_dir

    trainer = pl.Trainer(
        default_root_dir=storage_dir,
        logger=None,
        fast_dev_run=False,
        gpus=1,
        precision=32,
        max_epochs=250,
        accumulate_grad_batches=1,
        num_sanity_val_steps=2,
        gradient_clip_val=10.0,
        val_check_interval=1.0,
        callbacks=[pytorch_lightning.callbacks.EarlyStopping(monitor="val_loss", patience=2)],
    )

    trainer.fit(model, train_dataloaders=eval_train_loader, val_dataloaders=eval_test_loader)

    classifier_model = trainer.model.eval().cpu().requires_grad_(False)
    run_results = trainer.test(model=classifier_model, dataloaders=eval_test_loader)[0]

    return Result(
        task_id=task.id,
        num_train_classes=len(task.classes),
        metric_name="test_accuracy",
        score=run_results["accuracy"],
    )


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="class_subsets")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
