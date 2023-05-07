import logging
import os
from pathlib import Path
from typing import List, Optional

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from datasets import disable_caching
from nn_core.callbacks import NNTemplateCore
from nn_core.common import PROJECT_ROOT
from nn_core.common.utils import enforce_tags, seed_index_everything
from nn_core.model_logging import NNLogger
from nn_core.serialization import NNCheckpointIO, load_model
from omegaconf import DictConfig
from pytorch_lightning import Callback
from timm.data import resolve_data_config, create_transform, ToTensor
from torchvision.transforms import Compose
from tqdm import tqdm
from torch.utils.data import DataLoader

import la  # noqa
from la.data.datamodule import MetaData
from la.pl_modules.efficient_net import MyEfficientNet
from la.pl_modules.pl_module import DataAugmentation
from la.utils.utils import ToFloatRange, get_checkpoint_callback, build_callbacks

disable_caching()
pylogger = logging.getLogger(__name__)


def run(cfg: DictConfig) -> str:
    """Generic train loop.

    Args:
        cfg: run configuration, defined by Hydra in /conf

    Returns:
        the run directory inside the storage_dir used by the current experiment
    """
    seed_index_everything(cfg.train)

    pylogger.info(f"Running experiment on {cfg.nn.dataset_name}")

    cfg.core.tags = enforce_tags(cfg.core.get("tags", None))

    pylogger.info(f"Instantiating <{cfg.nn.data['_target_']}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.nn.data, _recursive_=False)

    num_tasks = datamodule.data["metadata"]["num_tasks"]
    num_classes = datamodule.data["metadata"]["num_classes"]

    for task_ind in range(num_tasks + 1):
        seed_index_everything(cfg.train)

        pylogger.info(f"Instantiating <{cfg.nn.model['_target_']}>")
        model: pl.LightningModule = hydra.utils.instantiate(
            cfg.nn.model,
            _recursive_=False,
            num_classes=num_classes,
            model=cfg.nn.model.model,
            input_dim=datamodule.img_size,
        )

        datamodule.task_ind = task_ind
        datamodule.transform_func = model.transform_func
        datamodule.setup()

        template_core: NNTemplateCore = NNTemplateCore(
            restore_cfg=cfg.train.get("restore", None),
        )
        callbacks: List[Callback] = build_callbacks(cfg.train.callbacks, template_core)

        logger: NNLogger = NNLogger(logging_cfg=cfg.train.logging, cfg=cfg, resume_id=template_core.resume_id)

        pylogger.info("Instantiating the <Trainer>")
        trainer = pl.Trainer(
            default_root_dir=cfg.core.storage_dir,
            plugins=[NNCheckpointIO(jailing_dir=logger.run_dir)],
            logger=logger,
            callbacks=callbacks,
            **cfg.train.trainer,
        )

        pylogger.info("Starting training!")
        trainer.fit(
            model=model,
            datamodule=datamodule,
            ckpt_path=template_core.trainer_ckpt_path,
        )

        if trainer.checkpoint_callback.best_model_path is not None:
            pylogger.info("Starting testing!")
            trainer.test(datamodule=datamodule)

        if logger is not None:
            logger.experiment.finish()

        best_model_path = get_checkpoint_callback(callbacks).best_model_path
        # TODO: check that the best_model_path is different for different tasks

        best_model = load_model(model.__class__, checkpoint_path=Path(best_model_path + ".zip"))

        best_model.eval().cuda()

        training_samples, test_samples, anchors = embed_all_samples(datamodule, best_model, task_ind)

        datamodule.data[f"task_{task_ind}_train"] = training_samples
        datamodule.data[f"task_{task_ind}_test"] = test_samples
        datamodule.data[f"task_{task_ind}_anchors"] = anchors

    if not os.path.exists(cfg.nn.output_path):
        os.makedirs(cfg.nn.output_path)

    datamodule.data.save_to_disk(cfg.nn.output_path)

    return logger.run_dir


def embed_all_samples(datamodule, model, task_ind):
    training_samples = datamodule.data[f"task_{task_ind}_train"]
    test_samples = datamodule.data[f"task_{task_ind}_test"]

    datamodule.shuffle_train = False

    train_embeddings = []
    for batch in tqdm(datamodule.train_dataloader(), desc="Embedding training samples"):
        x = batch["x"].to("cuda")
        train_embeddings.extend(model(x)["embeds"].detach())
    train_embeddings = torch.stack(train_embeddings)

    test_embeddings = []
    for batch in tqdm(datamodule.test_dataloader(), desc="Embedding test samples"):
        x = batch["x"].to("cuda")
        test_embeddings.extend(model(x)["embeds"].detach())
    test_embeddings = torch.stack(test_embeddings)

    map_params = {
        "with_indices": True,
        "batched": True,
        "batch_size": 128,
        "num_proc": 1,
        "writer_batch_size": 10,
    }

    training_samples = training_samples.map(
        function=lambda x, ind: {
            "embedding": train_embeddings[ind],
        },
        desc="Storing embedded training samples",
        remove_columns=["x"],
        **map_params,
    )

    test_samples = test_samples.map(
        function=lambda x, ind: {
            "embedding": test_embeddings[ind],
        },
        desc="Storing embedded test samples",
        remove_columns=["x"],
        **map_params,
    )

    anchors = datamodule.data[f"task_{task_ind}_anchors"]
    anchors_dataloader = DataLoader(anchors, batch_size=128, num_workers=8)

    anchor_embeddings = []
    for batch in tqdm(anchors_dataloader, desc="Embedding anchors"):
        x = batch["x"].to("cuda")
        anchor_embeddings.extend(model(x)["embeds"].detach())
    anchor_embeddings = torch.stack(anchor_embeddings)

    anchors = anchors.map(
        function=lambda x, ind: {
            "embedding": anchor_embeddings[ind],
        },
        desc="Storing embedded anchors",
        remove_columns=["x"],
        **map_params,
    )

    return training_samples, test_samples, anchors


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="run_same_classes_disj_samples")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
