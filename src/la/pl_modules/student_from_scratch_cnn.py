import logging
from typing import Any, Mapping, Sequence, Tuple, Union

import hydra
import torch
from hydra.utils import instantiate
from nn_core.model_logging import NNLogger
from omegaconf import DictConfig
from torch.optim import Optimizer
from la.pl_modules.from_scratch_cnn import FromScratchCNN
import torch.nn.functional as F
import torch.nn as nn
from la.pl_modules.pl_module import MyLightningModule


class StudentFromScratchCNN(FromScratchCNN):
    def __init__(self, num_classes, model: DictConfig, input_dim, anchors, *args, **kwargs) -> None:
        super().__init__(num_classes=num_classes, model=model, input_dim=input_dim, *args, **kwargs)

        self.register_buffer("anchors", anchors["x"])
        self.distillation_loss = nn.MSELoss()

        self.save_hyperparameters(logger=False, ignore=("metadata",))

    def step(self, x, y, teacher_embeds) -> Mapping[str, Any]:
        step_out = self.model(x, self.anchors)
        logits, relative_embeds = step_out["logits"], step_out["relative_embeds"]

        loss = F.cross_entropy(logits, y)

        distillation_loss = self.distillation_loss(relative_embeds, teacher_embeds)

        alpha = 0.1
        total_loss = loss + alpha * distillation_loss
        return {"logits": logits.detach(), "loss": total_loss}

    def training_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        x, y, teacher_embeds = batch["x"], batch["y"], batch["teacher_embeds"]
        step_out = self.step(x, y, teacher_embeds)

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
        x, y, teacher_embeds = batch["x"], batch["y"], batch["teacher_embeds"]
        step_out = self.step(x, y, teacher_embeds)

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
            prog_bar=True,
        )

        return step_out

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Mapping[str, Any]:
        x, y, teacher_embeds = batch["x"], batch["y"], batch["teacher_embeds"]
        step_out = self.step(x, y, teacher_embeds)

        self.log_dict(
            {"loss/test": step_out["loss"].cpu().detach()},
        )

        self.test_accuracy(torch.softmax(step_out["logits"], dim=-1), y)
        self.log_dict(
            {
                f"acc/test": self.test_accuracy,
            },
            on_epoch=True,
        )

        return step_out
