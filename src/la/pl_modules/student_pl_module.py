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


class StudentPLModule(FromScratchCNN):
    def __init__(self, num_classes, model: DictConfig, input_dim, anchors, *args, **kwargs) -> None:
        super().__init__(num_classes=num_classes, model=model, input_dim=input_dim, *args, **kwargs)

        self.register_buffer("anchors", anchors["x"])
        self.register_buffer("teacher_embedded_anchors", anchors["embedding"])
        self.distillation_loss = nn.MSELoss()

        self.save_hyperparameters(logger=False, ignore=("metadata",))

    def step(self, x, y, teacher_embeds) -> Mapping[str, Any]:
        step_out = self.model(x)
        logits = step_out["logits"]

        loss = F.cross_entropy(logits, y)

        with torch.no_grad():
            anchor_embeds = self.model(self.anchors)["embeds"]

        relative_embeds = F.normalize(step_out["embeds"]) @ F.normalize(anchor_embeds).T
        teacher_relative_embeds = F.normalize(teacher_embeds) @ F.normalize(self.teacher_embedded_anchors).T

        distillation_loss = self.distillation_loss(relative_embeds, teacher_relative_embeds)

        alpha = 1
        distillation_weight = self.calculate_adaptive_weight(loss, distillation_loss, self.model.proj.weight)
        total_loss = loss + alpha * distillation_weight *  distillation_loss

        return {"logits": logits.detach(), "loss": total_loss}

    def training_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        x, y, teacher_embeds = batch["x"], batch["y"], batch["embedding"]
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
        x, y, teacher_embeds = batch["x"], batch["y"], batch["embedding"]
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
        x, y, teacher_embeds = batch["x"], batch["y"], batch["embedding"]
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

    def calculate_adaptive_weight(self, loss_a: torch.Tensor, loss_b: torch.Tensor, last_layer) -> float:
        """Compute the re-scaling factor to apply to loss_b to be comparable with loss_a

        Args:
            loss_a: first loss
            loss_b: second loss
            last_layer: consider the gradients in this layer

        Returns:
            the scaling factor to apply to loss_b
        """
        try:
            loss_a_grads = torch.autograd.grad(loss_a, last_layer, retain_graph=True)[0]
            loss_b_grads = torch.autograd.grad(loss_b, last_layer, retain_graph=True)[0]

            loss_b_weight = torch.norm(loss_a_grads) / (torch.norm(loss_b_grads) + 1e-4)
            loss_b_weight = torch.clamp(loss_b_weight, 0.0, 1e6).detach()
        except RuntimeError:
            print("ERROR")
            loss_b_weight = torch.tensor(0.0)
        return loss_b_weight