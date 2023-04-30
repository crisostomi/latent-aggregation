import logging
from typing import Any, Mapping

import pytorch_lightning as pl
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from nn_core.model_logging import NNLogger
from kornia.augmentation import (
    ColorJitter,
    RandomChannelShuffle,
    RandomHorizontalFlip,
    RandomThinPlateSpline,
    RandomRotation,
    RandomCrop,
    Normalize,
)

pylogger = logging.getLogger(__name__)


class MyLightningModule(pl.LightningModule):
    logger: NNLogger

    def __init__(self, num_classes, *args, **kwargs) -> None:
        super().__init__()

        self.num_classes = num_classes

        metric = torchmetrics.Accuracy()
        self.train_accuracy = metric.clone()
        self.val_accuracy = metric.clone()
        self.test_accuracy = metric.clone()
        self.data_augm = DataAugmentation()
        self.preprocess = None

    def step(self, x, y) -> Mapping[str, Any]:
        logits = self(x)["logits"]
        loss = F.cross_entropy(logits, y)
        return {"logits": logits.detach(), "loss": loss}

    def on_after_batch_transfer(self, batch, dataloader_idx):
        x = batch["x"]
        x = self.preprocess(x)

        if self.trainer.training:
            x = self.data_augm(x)

        batch["x"] = x

        return batch

    def training_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        x, y = batch["x"], batch["y"]
        step_out = self.step(x, y)

        self.log_dict(
            {"loss/train": step_out["loss"].cpu().detach()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.train_accuracy(torch.softmax(step_out["logits"], dim=-1), y)
        self.log_dict(
            {
                "acc/train": self.train_accuracy,
            },
            on_epoch=True,
        )

        return step_out

    def validation_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        x, y = batch["x"], batch["y"]
        step_out = self.step(x, y)

        self.log_dict(
            {"loss/val": step_out["loss"].cpu().detach()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.val_accuracy(torch.softmax(step_out["logits"], dim=-1), y)
        self.log_dict(
            {
                "acc/val": self.val_accuracy,
            },
            on_epoch=True,
        )

        return step_out

    def test_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        x, y = batch["x"], batch["y"]
        step_out = self.step(x, y)

        self.log_dict(
            {"loss/test": step_out["loss"].cpu().detach()},
        )

        self.test_accuracy(torch.softmax(step_out["logits"], dim=-1), y)
        self.log_dict(
            {
                "acc/test": self.test_accuracy,
            },
            on_epoch=True,
        )

        return step_out


class DataAugmentation(nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(self) -> None:
        super().__init__()

        self.transforms = nn.Sequential(
            RandomHorizontalFlip(p=0.5),
            RandomRotation(degrees=30),
            RandomCrop((32, 32)),
        )

    def forward(self, x: Tensor) -> Tensor:
        x_out = self.transforms(x)  # BxCxHxW
        return x_out


class PreProcess(nn.Module):
    """Module to perform preprocessing on torch tensors."""

    def __init__(self, std, mean) -> None:
        super().__init__()

        self.transforms = nn.Sequential(
            Normalize(mean=mean, std=std),
        )

    def forward(self, x: Tensor) -> Tensor:
        x_out = self.transforms(x)  # BxCxHxW
        return x_out
