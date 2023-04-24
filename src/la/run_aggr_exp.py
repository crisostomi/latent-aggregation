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
from omegaconf import DictConfig, ListConfig
from pytorch_lightning import Callback
from timm.data import resolve_data_config, create_transform, ToTensor
from torchvision.transforms import Compose
from tqdm import tqdm

# Force the execution of __init__.py if this file is executed directly.
import la  # noqa
from la.data.prelim_exp_datamodule import MetaData
from la.pl_modules.efficient_net import MyEfficientNet
from la.utils.utils import ToFloatRange, get_checkpoint_callback

disable_caching()
pylogger = logging.getLogger(__name__)


def build_callbacks(cfg: ListConfig, *args: Callback) -> List[Callback]:
    """Instantiate the callbacks given their configuration.

    Args:
        cfg: a list of callbacks instantiable configuration
        *args: a list of extra callbacks already instantiated

    Returns:
        the complete list of callbacks to use
    """
    callbacks: List[Callback] = list(args)

    for callback in cfg:
        pylogger.info(f"Adding callback <{callback['_target_'].split('.')[-1]}>")
        callbacks.append(hydra.utils.instantiate(callback, _recursive_=False))

    return callbacks


def run(cfg: DictConfig) -> str:
    """Generic train loop.

    Args:
        cfg: run configuration, defined by Hydra in /conf

    Returns:
        the run directory inside the storage_dir used by the current experiment
    """
    seed_index_everything(cfg.train)

    pylogger.info(
        f"Running experiment on {cfg.nn.dataset_name} with {cfg.nn.num_shared_classes} shared classes and {cfg.nn.num_novel_classes} novel classes for task."
    )

    fast_dev_run: bool = cfg.train.trainer.fast_dev_run
    if fast_dev_run:
        pylogger.info(f"Debug mode <{cfg.train.trainer.fast_dev_run=}>. Forcing debugger friendly configuration!")
        # Debuggers don't like GPUs nor multiprocessing
        cfg.train.trainer.gpus = 0
        cfg.nn.data.num_workers.train = 0
        cfg.nn.data.num_workers.val = 0
        cfg.nn.data.num_workers.test = 0

    cfg.core.tags = enforce_tags(cfg.core.get("tags", None))

    # Instantiate datamodule
    pylogger.info(f"Instantiating <{cfg.nn.data['_target_']}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.nn.data, _recursive_=False)

    metadata: Optional[MetaData] = getattr(datamodule, "metadata", None)
    if metadata is None:
        pylogger.warning(f"No 'metadata' attribute found in datamodule <{datamodule.__class__.__name__}>")

    num_tasks = datamodule.data["metadata"]["num_tasks"]
    for task_ind in range(num_tasks + 1):
        seed_index_everything(cfg.train)

        # Instantiate model
        pylogger.info(f"Instantiating <{cfg.nn.model['_target_']}>")

        task_class_vocab = datamodule.data["metadata"]["global_to_local_class_mappings"][f"task_{task_ind}"]

        model: pl.LightningModule = hydra.utils.instantiate(
            cfg.nn.model,
            _recursive_=False,
            class_vocab=task_class_vocab,
            model=cfg.nn.model.model,
            input_dim=datamodule.img_size,
        )

        transform_func = Compose(
            [
                ToTensor(),
                ToFloatRange(),
            ]
        )

        if isinstance(model, MyEfficientNet):
            config = resolve_data_config({}, model=model.embedder)
            transform_func = create_transform(**config)

        datamodule.task_ind = task_ind
        datamodule.transform_func = transform_func
        datamodule.setup()

        # Instantiate the callbacks
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

        if "test" in cfg.nn.data.datasets and trainer.checkpoint_callback.best_model_path is not None:
            pylogger.info("Starting testing!")
            trainer.test(datamodule=datamodule)

        if logger is not None:
            logger.experiment.finish()

        best_model_path = get_checkpoint_callback(callbacks).best_model_path

        best_model = load_model(model.__class__, checkpoint_path=Path(best_model_path + ".zip"))

        best_model.eval().cuda()

        training_samples = datamodule.data[f"task_{task_ind}_train"]
        test_samples = datamodule.data[f"task_{task_ind}_test"]

        datamodule.shuffle_train = False
        train_embeddings = []
        for batch in tqdm(datamodule.train_dataloader(), desc="Embedding training samples"):
            x = batch["x"].to("cuda")
            train_embeddings.extend(best_model(x)["embeds"].detach())
        train_embeddings = torch.stack(train_embeddings)

        test_embeddings = []
        for batch in tqdm(datamodule.val_dataloader(), desc="Embedding test samples"):
            x = batch["x"].to("cuda")
            test_embeddings.extend(best_model(x)["embeds"].detach())
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
            **map_params,
            remove_columns=["x"],
        )

        test_samples = test_samples.map(
            function=lambda x, ind: {
                "embedding": test_embeddings[ind],
            },
            desc="Storing embedded test samples",
            remove_columns=["x"],
            **map_params,
        )

        datamodule.data[f"task_{task_ind}_train"] = training_samples
        datamodule.data[f"task_{task_ind}_test"] = test_samples

    if not os.path.exists(cfg.nn.output_path):
        os.makedirs(cfg.nn.output_path)

    datamodule.data.save_to_disk(cfg.nn.output_path)

    return logger.run_dir


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="aggr_exp")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
