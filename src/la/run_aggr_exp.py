import logging
from typing import List, Optional

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from datasets import Dataset
from omegaconf import DictConfig, ListConfig
from pytorch_lightning import Callback

from nn_core.callbacks import NNTemplateCore
from nn_core.common import PROJECT_ROOT
from nn_core.common.utils import enforce_tags, seed_index_everything
from nn_core.model_logging import NNLogger
from nn_core.serialization import NNCheckpointIO

# Force the execution of __init__.py if this file is executed directly.
import la  # noqa
from la.data.prelim_exp_datamodule import MetaData

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
        pylogger.info(f"Instantiating <{cfg.nn.module['_target_']}>")

        task_class_vocab = datamodule.data["metadata"]["global_to_local_class_mappings"][f"task_{task_ind}"]

        model: pl.LightningModule = hydra.utils.instantiate(
            cfg.nn.module, _recursive_=False, class_vocab=task_class_vocab, model=cfg.nn.module.model
        )

        datamodule.task_ind = task_ind
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
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=template_core.trainer_ckpt_path)

        if fast_dev_run:
            pylogger.info("Skipping testing in 'fast_dev_run' mode!")
        else:
            if "test" in cfg.nn.data.datasets and trainer.checkpoint_callback.best_model_path is not None:
                pylogger.info("Starting testing!")
                trainer.test(datamodule=datamodule)

        # Logger closing to release resources/avoid multi-run conflicts
        if logger is not None:
            logger.experiment.finish()

        # embed all the samples with the trained model
        model.eval()

        training_samples: Dataset = datamodule.data[f"task_{task_ind}_train"]
        test_samples: Dataset = datamodule.data[f"task_{task_ind}_test"]
        training_samples.set_format(type="torch")
        test_samples.set_format(type="torch")

        def preprocess(x):
            return torch.permute(x, (0, 3, 1, 2)).float() / 255.0

        # training_samples = training_samples.map(lambda x: {'embedding': model.model.forward_pre_head(preprocess(x['img'])).detach().numpy()},
        #                                         batched=True)

        # test_samples = test_samples.map(lambda x: {'embedding': model.model.forward_pre_head(preprocess(x['img'])).detach().numpy()},
        #                                         batched=True)

        training_samples = training_samples.map(
            lambda x: {"embedding": model(preprocess(x["img"]))["embeds"].detach().numpy()}, batched=True
        )
        test_samples = test_samples.map(
            lambda x: {"embedding": model(preprocess(x["img"]))["embeds"].detach().numpy()}, batched=True
        )

        datamodule.data[f"task_{task_ind}_train"] = training_samples
        datamodule.data[f"task_{task_ind}_test"] = test_samples

    datamodule.data.save_to_disk(cfg.nn.output_path)

    return logger.run_dir


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="aggr_exp")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
