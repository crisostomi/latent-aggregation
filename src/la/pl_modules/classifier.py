from typing import Optional

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import nn
from torchmetrics import Accuracy, FBetaScore, MetricCollection


class LambdaModule(nn.Module):
    def __init__(self, lambda_func) -> None:
        super().__init__()

        self.lambda_func = lambda_func

    def forward(self, x: torch.Tensor):
        return self.lambda_func(x)


class Classifier(LightningModule):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        lr: float,
        deep: bool,
        bias: bool = True,
        x_feature: str = "encoding",
        y_feature: str = "target",
        first_activation: nn.Module = nn.Tanh(),
        second_activation: nn.Module = nn.ReLU(),
        first_projection_dim: Optional[int] = None,
    ):
        super().__init__()
        assert deep or (first_activation is None and second_activation is None and first_projection_dim is None)

        if callable(first_activation) and getattr(first_activation, "__name__", None) == "<lambda>":
            first_activation = LambdaModule(first_activation)

        if callable(second_activation) and getattr(second_activation, "__name__", None) == "<lambda>":
            second_activation = LambdaModule(second_activation)

        first_projection_dim = input_dim if first_projection_dim is None else first_projection_dim

        self.class_proj = (
            nn.Sequential(
                #
                nn.Linear(input_dim, first_projection_dim, bias=bias),
                first_activation,
                #
                nn.Linear(first_projection_dim, first_projection_dim // 2, bias=bias),
                second_activation,
                #
                nn.Linear(first_projection_dim // 2, num_classes, bias=bias),
            )
            if deep
            else nn.Sequential(nn.Linear(input_dim, num_classes))
        )

        self.train_metrics = MetricCollection(
            {
                "accuracy": Accuracy(num_classes=num_classes),
                "f1": FBetaScore(num_classes=num_classes),
            }
        )
        self.val_metrics = self.train_metrics.clone()
        self.test_metrics = self.train_metrics.clone()
        self.lr: float = lr

        self.x_feature: str = x_feature
        self.y_feature: str = y_feature

    def forward(self, x):
        x = self.class_proj(x)
        return F.log_softmax(x, dim=1)

    def _step(self, batch, split: str):
        logits = self(batch[self.x_feature])
        loss = F.cross_entropy(logits, batch[self.y_feature])
        preds = torch.argmax(logits, dim=1)
        metrics = getattr(self, f"{split}_metrics")
        metrics.update(preds, batch[self.y_feature])

        self.log(f"{split}_loss", loss, prog_bar=True)
        self.log_dict(metrics, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch=batch, split="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch=batch, split="val")

    def test_step(self, batch, batch_idx):
        return self._step(batch=batch, split="test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
