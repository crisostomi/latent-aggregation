from os import PathLike
from pathlib import Path
from typing import Optional, Union, Dict

from datasets import DatasetDict

import json


class MyDatasetDict(DatasetDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def save_to_disk(
        self,
        dataset_dict_path: PathLike,
        fs="deprecated",
        max_shard_size: Optional[Union[str, int]] = None,
        num_shards: Optional[Dict[str, int]] = None,
        num_proc: Optional[int] = None,
        storage_options: Optional[dict] = None,
    ):
        self.save_metadata(Path(dataset_dict_path) / "metadata.json")
        tmp = self["metadata"]
        del self["metadata"]
        super().save_to_disk(
            dataset_dict_path,
            fs=fs,
            max_shard_size=max_shard_size,
            num_shards=num_shards,
            num_proc=num_proc,
            storage_options=storage_options,
        )
        self["metadata"] = tmp

    def save_metadata(self, metadata_path: PathLike):
        with open(metadata_path, "w") as f:
            json.dump(self["metadata"], f)

    @staticmethod
    def load_from_disk(
        dataset_dict_path: PathLike,
        fs="deprecated",
        keep_in_memory: Optional[bool] = None,
        storage_options: Optional[dict] = None,
    ) -> "MyDatasetDict":
        dataset = MyDatasetDict(
            DatasetDict.load_from_disk(
                dataset_dict_path, fs=fs, keep_in_memory=keep_in_memory, storage_options=storage_options
            )
        )
        dataset["metadata"] = json.load(open(Path(dataset_dict_path) / "metadata.json"))
        return dataset
