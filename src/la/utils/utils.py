import json
import logging
from os import PathLike
from pathlib import Path
from typing import Dict, List, Optional, Union

import hydra
from datasets import Dataset, DatasetDict, Features
from omegaconf import ListConfig
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from traitlets import Callable

pylogger = logging.getLogger(__name__)


class IdentityTransform:
    def __call__(self, x):
        return x


class ToFloatRange:
    def __call__(self, x):
        """
        Convert [0, 255] to [0, 1]
        :param x:
        :return:
        """
        return x.float() / 255


def encode_field(batch, src_field: str, tgt_field: str, transformation):
    """
    Create a new field with name `tgt_field` by applying `transformation` to `src_field`.
    """
    src_data = batch[src_field]
    transformed = transformation(src_data)

    return {tgt_field: transformed}


def preprocess_img(x):
    """
    (H, W, C) --> (C, H, W)
    :param x: (H, W, C)
    :return:
    """
    if x.ndim == 2:
        x = x.unsqueeze(-1)
        x = x.repeat_interleave(3, axis=2)

    if x.ndim == 4:
        x = x.permute(0, 3, 1, 2)
    else:
        x = x.permute(2, 0, 1)

    return x.float() / 255.0


class ConvertToRGB:
    def __call__(self, image):
        convert_to_rgb(image)


def convert_to_rgb(image):
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def get_checkpoint_callback(callbacks):
    for callback in callbacks:
        if isinstance(callback, ModelCheckpoint):
            return callback


def add_tensor_column(dataset, column, tensor):
    dataset_dict = dataset.to_dict()
    dataset_dict[column] = tensor.tolist()
    dataset = Dataset.from_dict(dataset_dict)

    return dataset


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


def save_dict_to_file(content, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w+") as f:
        json.dump(content, f)
