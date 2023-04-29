import logging
from typing import Any, Mapping

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from nn_core.model_logging import NNLogger

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

    def step(self, x, y) -> Mapping[str, Any]:
        logits = self(x)["logits"]
        loss = F.cross_entropy(logits, y)
        return {"logits": logits.detach(), "loss": loss}

    def training_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        x, y = batch["x"], batch["y"]
        step_out = self.step(x, y)

        self.log_dict(
            {"loss/train": step_out["loss"].cpu().detach()},
            on_step=True,
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
