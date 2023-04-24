import logging
from functools import cached_property, partial
from pathlib import Path
from typing import List, Mapping, Optional, Union

import pytorch_lightning as pl
from nn_core.nn_types import Split
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from la.data.prelim_exp_dataset import MyDataset
from la.utils.utils import MyDatasetDict

pylogger = logging.getLogger(__name__)


class MetaData:
    def __init__(self, tasks_info: Mapping[str, int]):
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
        self.tasks_info: Mapping[str, int] = tasks_info

    def save(self, dst_path: Path) -> None:
        """
        Serialize the MetaData attributes into the zipped checkpoint in dst_path.

        Args:
            dst_path: the root folder of the metadata inside the zipped checkpoint
        """
        pylogger.debug(f"Saving MetaData to '{dst_path}'")

        (dst_path / "tasks_info.tsv").write_text("\n".join(f"{key}\t{value}" for key, value in self.tasks_info.items()))

    @staticmethod
    def load(src_path: Path) -> "MetaData":
        """Deserialize the MetaData from the information contained inside the zipped checkpoint in src_path.

        Args:
            src_path: the root folder of the metadata inside the zipped checkpoint

        Returns:
            an instance of MetaData containing the information in the checkpoint
        """
        pylogger.debug(f"Loading MetaData from '{src_path}'")

        lines = (src_path / "tasks_info.tsv").read_text(encoding="utf-8").splitlines()

        tasks_info = {}
        for line in lines:
            key, value = line.strip().split("\t")
            tasks_info[key] = value

        return MetaData(
            tasks_info=tasks_info,
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
        data_path: Path,
        only_use_sample_num: int = -1,
    ):
        super().__init__()
        self.datasets = datasets
        self.num_workers = num_workers
        self.batch_size = batch_size

        self.pin_memory: bool = gpus is not None and str(gpus) != "0"
        self.pin_memory = False

        self.train_dataset: Optional[MyDataset] = None
        self.val_dataset: Optional[MyDataset] = None

        self.val_percentage: float = val_percentage

        self.data: MyDatasetDict = MyDatasetDict.load_from_disk(dataset_dict_path=str(data_path))

        self.img_size = self.data["task_0_train"][0]["x"].size[1]

        self.tasks = {key for key in self.data.keys() if key != "metadata"}

        self.task_ind = None  # will be set in setup
        self.transform_func = None  # will be set in setup
        self.shuffle_train = True

        self.only_use_sample_num = only_use_sample_num
        if only_use_sample_num >= 0:
            for task in self.tasks:
                self.data[task] = self.data[task].select(range(only_use_sample_num))

        self.seen_tasks = set()

        pylogger.info("Preprocessing done.")

    @cached_property
    def metadata(self) -> MetaData:
        """Data information to be fed to the Lightning Module as parameter.

        Examples are vocabularies, number of classes...

        Returns:
            metadata: everything the model should know about the data, wrapped in a MetaData object.
        """

        return MetaData(tasks_info=self.data["metadata"])

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        # to avoid reprocessing the data
        if self.task_ind in self.seen_tasks:
            return

        self.shuffle_train = True

        train_samples = self.data[f"task_{self.task_ind}_train"]
        test_samples = self.data[f"task_{self.task_ind}_test"]

        map_params = {
            "function": lambda x: {"x": self.transform_func(x["x"])},
            "writer_batch_size": 10,
            "num_proc": 1,
        }

        train_samples = train_samples.map(
            desc=f"Transforming task {self.task_ind} train samples",
            **map_params,
        )

        test_samples = test_samples.map(desc=f"Transforming task {self.task_ind} test samples", **map_params)

        train_samples.set_format(type="torch", columns=["x", "y"])
        test_samples.set_format(type="torch", columns=["x", "y"])

        self.data[f"task_{self.task_ind}_train"] = train_samples
        self.data[f"task_{self.task_ind}_test"] = test_samples

        self.train_dataset = train_samples
        # TODO: use train subset for validation
        self.val_dataset = test_samples

        self.seen_tasks.add(self.task_ind)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle=self.shuffle_train,
            batch_size=self.batch_size.train,
            num_workers=self.num_workers.train,
            pin_memory=self.pin_memory,
            collate_fn=partial(collate_fn, split="train", metadata=self.metadata),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.batch_size.val,
            num_workers=self.num_workers.val,
            pin_memory=self.pin_memory,
            collate_fn=partial(collate_fn, split="val", metadata=self.metadata),
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(" f"{self.datasets=}, " f"{self.num_workers=}, " f"{self.batch_size=})"
