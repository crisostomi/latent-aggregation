import la  # noqa
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

from la.data.datamodule import MetaData
from la.pl_modules.efficient_net import MyEfficientNet
from la.utils.io_utils import save_dataset_to_disk
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

    pylogger.info(
        f"Running experiment on {cfg.nn.dataset_name} with {cfg.nn.num_shared_classes} shared classes and {cfg.nn.num_novel_classes_per_task} novel classes for task."
    )

    cfg.core.tags = enforce_tags(cfg.core.get("tags", None))

    pylogger.info(f"Instantiating <{cfg.nn.data['_target_']}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.nn.data, _recursive_=False)

    metadata: Optional[MetaData] = getattr(datamodule, "metadata", None)
    if metadata is None:
        pylogger.warning(f"No 'metadata' attribute found in datamodule <{datamodule.__class__.__name__}>")

    num_tasks = datamodule.data["metadata"]["num_tasks"]
    for task_ind in range(num_tasks + 1):
        seed_index_everything(cfg.train)

        pylogger.info(f"Instantiating <{cfg.nn.model['_target_']}>")

        task_class_vocab = datamodule.data["metadata"]["global_to_local_class_mappings"][f"task_{task_ind}"]

        model: pl.LightningModule = hydra.utils.instantiate(
            cfg.nn.model,
            _recursive_=False,
            num_classes=len(task_class_vocab),
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

        storage_dir: str = cfg.core.storage_dir

        logger: NNLogger = NNLogger(logging_cfg=cfg.train.logging, cfg=cfg, resume_id=template_core.resume_id)

        pylogger.info("Instantiating the <Trainer>")
        trainer = pl.Trainer(
            default_root_dir=storage_dir,
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

        # TODO: check that the best_model_path is different for different tasks
        best_model_path = get_checkpoint_callback(callbacks).best_model_path

        best_model = load_model(model.__class__, checkpoint_path=Path(best_model_path + ".zip"))
        best_model.eval().cuda()

        embedded_samples = embed_task_samples(datamodule, best_model, task_ind)

        datamodule.data[f"task_{task_ind}_train"] = embedded_samples["train"]
        datamodule.data[f"task_{task_ind}_val"] = embedded_samples["val"]
        datamodule.data[f"task_{task_ind}_test"] = embedded_samples["test"]

    save_dataset_to_disk(datamodule.data, cfg.nn.output_path)

    return logger.run_dir


def embed_task_samples(datamodule, model, task_ind) -> Dict:
    datamodule.shuffle_train = False

    modes = ["train", "val", "test"]

    embeddings = {mode: None for mode in modes}

    for mode in modes:
        mode_embeddings = []

        for batch in tqdm(datamodule.dataloader(mode), desc=f"Embedding {mode} samples"):
            x = batch["x"].to("cuda")
            mode_embeddings.extend(model(x)["embeds"].detach())

        embeddings[mode] = torch.stack(mode_embeddings)

    embedded_samples = {mode: None for mode in modes}

    map_params = {
        "with_indices": True,
        "batched": True,
        "batch_size": 128,
        "num_proc": 1,
        "writer_batch_size": 10,
    }

    for mode in modes:
        embedded_samples[mode] = datamodule.data[f"task_{task_ind}_{mode}"].map(
            function=lambda x, ind: {
                "embedding": embeddings[mode][ind],
            },
            desc=f"Storing embedded {mode} samples",
            **map_params,
            remove_columns=["x"],
        )

    return embedded_samples


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="run_part_shared_part_novel")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
