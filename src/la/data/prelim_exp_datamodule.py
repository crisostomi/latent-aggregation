import logging
from functools import cached_property, partial
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Union

import hydra
import omegaconf
import pytorch_lightning as pl
from nn_core.common import PROJECT_ROOT
from nn_core.nn_types import Split
from omegaconf import DictConfig
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
from torchvision.datasets import MNIST

from la.data.prelim_exp_dataset import MyDataset

pylogger = logging.getLogger(__name__)


class MetaData:
    def __init__(self, class_vocab: Mapping[str, int]):
        """The data information the Lightning Module will be provided with.

        This is a "bridge" between the Lightning DataModule and the Lightning Module.
        There is no constraint on the class name nor in the stored information, as long as it exposes the
        `save` and `load` methods.

        The Lightning Module will receive an instance of MetaData when instantiated,
        both in the train loop or when restored from a checkpoint.

        This decoupling allows the architecture to be parametric (e.g. in the number of classes) and
        DataModule/Trainer independent (useful in prediction scenarios).
        MetaData should contain all the information needed at test time, derived from its train dataset.

        Examples are the class names in a classification task or the vocabulary in NLP tasks.
        MetaData exposes `save` and `load`. Those are two user-defined methods that specify
        how to serialize and de-serialize the information contained in its attributes.
        This is needed for the checkpointing restore to work properly.

        Args:
            class_vocab: association between class names and their indices
        """
        self.class_vocab: Mapping[str, int] = class_vocab

    def save(self, dst_path: Path) -> None:
        """
        Serialize the MetaData attributes into the zipped checkpoint in dst_path.

        Args:
            dst_path: the root folder of the metadata inside the zipped checkpoint
        """
        pylogger.debug(f"Saving MetaData to '{dst_path}'")

        (dst_path / "class_vocab.tsv").write_text(
            "\n".join(f"{key}\t{value}" for key, value in self.class_vocab.items())
        )

    @staticmethod
    def load(src_path: Path) -> "MetaData":
        """Deserialize the MetaData from the information contained inside the zipped checkpoint in src_path.

        Args:
            src_path: the root folder of the metadata inside the zipped checkpoint

        Returns:
            an instance of MetaData containing the information in the checkpoint
        """
        pylogger.debug(f"Loading MetaData from '{src_path}'")

        lines = (src_path / "class_vocab.tsv").read_text(encoding="utf-8").splitlines()

        class_vocab = {}
        for line in lines:
            key, value = line.strip().split("\t")
            class_vocab[key] = value

        return MetaData(
            class_vocab=class_vocab,
        )


def collate_fn(samples: List, split: Split, metadata: MetaData):
    """Custom collate function for dataloaders with access to split and metadata.

    Args:
        samples: A list of samples coming from the Dataset to be merged into a batch
        split: The data split (e.g. train/val/test)
        metadata: The MetaData instance coming from the DataModule or the restored checkpoint

    Returns:
        A batch generated from the given samples
    """
    return default_collate(samples)


class MyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datasets: DictConfig,
        num_workers: DictConfig,
        batch_size: DictConfig,
        gpus: Optional[Union[List[int], str, int]],
        val_percentage: float,
        classes_to_discard: set,
        mnist_version: str = "vanilla",
    ):
        super().__init__()
        self.datasets = datasets
        self.num_workers = num_workers
        self.batch_size = batch_size

        self.mnist_version = mnist_version

        self.pin_memory: bool = gpus is not None and str(gpus) != "0"
        self.pin_memory = False

        self.train_dataset: Optional[MyDataset] = None
        self.val_datasets: Optional[Sequence[MyDataset]] = None
        self.test_datasets: Optional[Sequence[MyDataset]] = None

        self.val_percentage: float = val_percentage

        transform = transforms.Compose([transforms.ToTensor()])  # , grayscale_to_rgb
        data_path = PROJECT_ROOT / "data"

        self.train_val_mnist = MNIST(
            data_path,
            train=True,
            download=True,
            transform=transform,
        )

        train_length = int(len(self.train_val_mnist) * (1 - self.val_percentage))
        val_length = len(self.train_val_mnist) - train_length

        self.mnist = {"vanilla": {}, "colored": {}, "three_dim": {}}

        self.mnist["vanilla"]["train"], self.mnist["vanilla"]["val"] = random_split(
            self.train_val_mnist, [train_length, val_length]
        )

        self.mnist["vanilla"]["test"] = MNIST(
            data_path,
            train=False,
            download=True,
            transform=transform,
        )

        self.filter_out_classes(set(classes_to_discard))

        pylogger.info("Preprocessing done.")

    @cached_property
    def metadata(self) -> MetaData:
        """Data information to be fed to the Lightning Module as parameter.

        Examples are vocabularies, number of classes...

        Returns:
            metadata: everything the model should know about the data, wrapped in a MetaData object.
        """
        if self.train_dataset is None:
            self.setup(stage="fit")

        class_vocab = self.train_dataset.class_vocab

        return MetaData(class_vocab=class_vocab)

    def filter_out_classes(self, classes_to_discard: set):
        for split in ["train", "val", "test"]:

            if split in {"train", "val"}:
                data = self.mnist[self.mnist_version][split].dataset.data
                split_indices = self.mnist[self.mnist_version][split].indices
                targets = self.mnist[self.mnist_version][split].dataset.targets
                split_data = data[split_indices].float().unsqueeze(1)
                split_targets = targets[split_indices]
            else:
                split_data = self.mnist[self.mnist_version][split].data.float().unsqueeze(1)
                split_targets = self.mnist[self.mnist_version][split].targets

            self.mnist[self.mnist_version][split] = [
                (split_data[i], split_targets[i])
                for i in range(len(split_data))
                if split_targets[i] not in classes_to_discard
            ]

            new_targets = {self.mnist[self.mnist_version][split][i][1] for i in range(len(split_data))}
            assert classes_to_discard.intersection(set(new_targets)) == set()

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None):
        if (stage is None or stage == "fit") and (self.train_dataset is None and self.val_datasets is None):
            self.train_dataset = hydra.utils.instantiate(
                self.datasets.train,
                split="train",
                samples=self.mnist[self.mnist_version]["train"],
                class_vocab=self.train_val_mnist.class_to_idx,
            )

            self.val_datasets = [
                hydra.utils.instantiate(
                    self.datasets.val[0],
                    split="val",
                    samples=self.mnist[self.mnist_version]["val"],
                    class_vocab=self.train_val_mnist.class_to_idx,
                )
            ]

        if stage is None or stage == "test":
            self.test_datasets = [
                hydra.utils.instantiate(
                    self.datasets.test[0],
                    split="test",
                    samples=self.mnist[self.mnist_version]["test"],
                    class_vocab=self.train_val_mnist.class_to_idx,
                )
            ]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle=False,
            batch_size=self.batch_size.train,
            num_workers=self.num_workers.train,
            pin_memory=self.pin_memory,
            collate_fn=partial(collate_fn, split="train", metadata=self.metadata),
        )

    def val_dataloader(self) -> Sequence[DataLoader]:
        return [
            DataLoader(
                dataset,
                shuffle=False,
                batch_size=self.batch_size.val,
                num_workers=self.num_workers.val,
                pin_memory=self.pin_memory,
                collate_fn=partial(collate_fn, split="val", metadata=self.metadata),
            )
            for dataset in self.val_datasets
        ]

    def test_dataloader(self) -> Sequence[DataLoader]:

        return [
            DataLoader(
                dataset,
                shuffle=False,
                batch_size=self.batch_size.test,
                num_workers=self.num_workers.test,
                pin_memory=self.pin_memory,
                collate_fn=partial(collate_fn, split="test", metadata=self.metadata),
            )
            for dataset in self.test_datasets
        ]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(" f"{self.datasets=}, " f"{self.num_workers=}, " f"{self.batch_size=})"


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig) -> None:
    """Debug main to quickly develop the DataModule.

    Args:
        cfg: the hydra configuration
    """
    _: pl.LightningDataModule = hydra.utils.instantiate(cfg.data.datamodule, _recursive_=False)


if __name__ == "__main__":
    main()
