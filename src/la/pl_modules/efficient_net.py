import logging
from typing import Any, Sequence, Tuple, Union, Dict

import hydra
import torch
from hydra.utils import instantiate
from nn_core.model_logging import NNLogger
from omegaconf import DictConfig
from torch.optim import Optimizer

from la.pl_modules.pl_module import DataAugmentation, MyLightningModule

pylogger = logging.getLogger(__name__)


class MyEfficientNet(MyLightningModule):
    logger: NNLogger

    def __init__(self, num_classes, input_dim, model: DictConfig, *args, **kwargs) -> None:
        super().__init__(num_classes=num_classes, input_dim=input_dim, *args, **kwargs)

        self.save_hyperparameters(logger=False, ignore=("metadata",))

        self.embedder = instantiate(model.embedder)
        self.projector = instantiate(
            model.projector,
        )

        self.classifier = instantiate(model.classifier, out_features=num_classes)
        # differently from other models, EfficientNet works on 224x224 images
        self.data_augm = DataAugmentation(input_dim=224)

    def forward(self, x: torch.Tensor) -> Dict:
        """Method for the forward pass.

        'training_step', 'validation_step' and 'test_step' should call
        this method in order to compute the output predictions and the loss.

        Returns:
            output_dict: forward output containing the predictions (output logits ecc...) and the loss if any.
        """
        embeds = self.embedder(x)
        projected_embeds = self.projector(embeds)
        logits = self.classifier(projected_embeds)

        return {"embeds": projected_embeds, "logits": logits}

    def on_train_epoch_start(self) -> None:
        self.embedder.eval()
        self.embedder.requires_grad_(False)

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Tuple[Sequence[Optimizer], Sequence[Any]]]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.

        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Return:
            Any of these 6 options.
            - Single optimizer.
            - List or Tuple - List of optimizers.
            - Two lists - The first list has multiple optimizers, the second a list of LR schedulers (or lr_dict).
            - Dictionary, with an 'optimizer' key, and (optionally) a 'lr_scheduler'
              key whose value is a single LR scheduler or lr_dict.
            - Tuple of dictionaries as described, with an optional 'frequency' key.
            - None - Fit will run without any optimizer.
        """
        opt = hydra.utils.instantiate(self.hparams.optimizer, params=self.parameters(), _convert_="partial")
        if "lr_scheduler" not in self.hparams:
            return [opt]
        scheduler = hydra.utils.instantiate(self.hparams.lr_scheduler, optimizer=opt)
        return [opt], [scheduler]
