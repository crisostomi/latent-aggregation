import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

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
from la.utils.utils import ToFloatRange, embed_task_samples, get_checkpoint_callback, build_callbacks

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

        embedded_samples = embed_task_samples(
            datamodule, best_model, task_ind, modes=["train", "val", "test", "anchors"]
        )

        datamodule.data[f"task_{task_ind}_train"] = embedded_samples["train"]
        datamodule.data[f"task_{task_ind}_val"] = embedded_samples["val"]
        datamodule.data[f"task_{task_ind}_test"] = embedded_samples["test"]
        datamodule.data[f"task_{task_ind}_anchors"] = embedded_samples["anchors"]

    if not os.path.exists(cfg.nn.output_path):
        os.makedirs(cfg.nn.output_path)

    datamodule.data.save_to_disk(cfg.nn.output_path)

    return logger.run_dir


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="run_same_classes_disj_samples")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
