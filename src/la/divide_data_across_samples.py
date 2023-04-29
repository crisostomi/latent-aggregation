from typing import List, Dict

import nn_core  # noqa
import logging
from collections import namedtuple
from pathlib import Path

# Force the execution of __init__.py if this file is executed directly.
import la  # noqa

import hydra
from datasets import (
    load_dataset,
    DatasetDict,
    load_from_disk,
    Dataset,
    concatenate_datasets,
    Value,
)
from nn_core.common import PROJECT_ROOT
from omegaconf import DictConfig, omegaconf
from pytorch_lightning import seed_everything

from la.utils.utils import MyDatasetDict, convert_to_rgb

pylogger = logging.getLogger(__name__)


def run(cfg: DictConfig):
    seed_everything(cfg.seed)

    pylogger.info(f"Subdividing dataset {cfg.dataset.name}")

    dataset = load_data(cfg)

    # standardize label key and image key
    dataset = dataset.map(
        lambda x: {cfg.label_key: x[cfg.dataset.label_key]},
        remove_columns=[cfg.dataset.label_key],
        desc="Standardizing label key",
    )
    dataset = dataset.map(
        lambda x: {cfg.image_key: x[cfg.dataset.image_key]},
        batched=True,
        remove_columns=[cfg.dataset.image_key],
        desc="Standardizing image key",
    )

    # in case some images are not RGB, convert them to RGB
    dataset = dataset.map(lambda x: {cfg.image_key: convert_to_rgb(x["x"])}, desc="Converting to RGB")

    # add ids
    N = len(dataset["train"])
    M = len(dataset["test"])
    indices = {"train": list(range(N)), "test": list(range(N, N + M))}

    for mode in ["train", "test"]:
        dataset[mode] = dataset[mode].map(lambda row, ind: {"id": indices[mode][ind]}, with_indices=True)

    if isinstance(dataset["train"].features[cfg.label_key], Value):
        all_classes = [str(class_id) for class_id in range(cfg.dataset.num_classes)]
    else:
        all_classes = dataset["train"].features[cfg.label_key].names

    num_classes = len(all_classes)

    all_classes_ids = [id for id, _ in enumerate(all_classes)]

    num_tasks = cfg.num_tasks

    classes_partitions = cfg.classes_partitions

    classes_sets = [set(range(*partition)) for partition in classes_partitions]

    pylogger.info(classes_sets)
    subset_percentages = cfg.subset_percentages

    new_dataset = MyDatasetDict()

    # task 0 is a dummy task that consists of the samples for all the classes
    val_train_split = dataset["train"].train_test_split(test_size=cfg.val_percentage)

    new_dataset["task_0_train"] = val_train_split["train"]
    new_dataset["task_0_val"] = val_train_split["test"]
    new_dataset["task_0_test"] = dataset["test"]

    anchors = dataset["train"].shuffle(seed=cfg.seed).select(range(cfg.num_anchors))
    anchor_ids = set(anchors["id"])

    new_dataset["anchors"] = anchors

    num_partitions = len(classes_partitions)

    # for each class set C_1, .., C_k contains the train and test samples for that class set
    partitions_by_class_set: Dict[str, List] = {"train": [], "test": []}

    for part_ind in range(num_partitions):
        for mode in ["train", "test"]:
            # e.g. C = {0, ... , 49}
            partition_classes = classes_sets[part_ind]

            # partition is a dataset containing only samples belonging to the corresponding class partition
            partition = dataset[mode].filter(
                lambda x: x[cfg.label_key] in partition_classes and x["id"] not in anchor_ids,
                desc=f"Creating partition {part_ind}",
            )
            partitions_by_class_set[mode].append(partition)

    all_tasks = {"train": [], "test": []}
    remaining_percentages = {"train": [1.0, 1.0], "test": [1.0, 1.0]}

    # for each task we will have a different model

    for task in range(1, num_tasks + 1):
        # e.g. [0.8, 0.2]
        task_subset_percentages = subset_percentages[task - 1]

        # each task will have a train and test set
        for mode in ["train", "test"]:
            task_samples = []

            for part_ind in range(num_partitions):
                # percentage of the task samples that will be sampled from this partition
                # e.g. 0.8 means that 80% of the samples of the task will have class in this class partition
                part_percentage = task_subset_percentages[part_ind]

                # percentage of the samples having class in the partition that remain for the following tasks
                remaining_percentages[mode][part_ind] -= part_percentage

                if is_zero(remaining_percentages[mode][part_ind]):
                    task_partition_samples = partitions_by_class_set[mode][part_ind]
                    partitions_by_class_set[mode][part_ind] = None
                else:
                    task_partition_samples, remaining_samples = split(
                        partition_samples=partitions_by_class_set[mode][part_ind],
                        split_percentage=remaining_percentages[mode][part_ind],
                    )

                    partitions_by_class_set[mode][part_ind] = remaining_samples

                task_samples.append(task_partition_samples)

            task_samples = concatenate_datasets(task_samples)

            all_tasks[mode].append(task_samples)

            if mode == "train":
                task_samples_split = task_samples.train_test_split(test_size=cfg.perc_test_per_task)

                new_dataset[f"task_{task}_train"] = task_samples_split["train"]
                new_dataset[f"task_{task}_val"] = task_samples_split["test"]
            else:
                new_dataset[f"task_{task}_test"] = task_samples

    # safety check

    for mode in ["train", "test"]:
        for task in range(num_tasks):
            for part in range(num_partitions):
                task_samples = all_tasks[mode][task].filter(
                    lambda x: x[cfg.label_key] in classes_sets[part],
                )
                pylogger.info(f"Task {task} has {len(task_samples)} {mode} samples for partition {part}")

    metadata = {
        "num_train_samples_per_class": cfg.dataset.num_train_samples_per_class,
        "num_test_samples_per_class": cfg.dataset.num_test_samples_per_class,
        "num_tasks": num_tasks,
        "all_classes": all_classes,
        "all_classes_ids": all_classes_ids,
        "num_classes": num_classes,
    }

    new_dataset["metadata"] = metadata

    save_to_disk(cfg, new_dataset)


def split(partition_samples, split_percentage):
    split_samples = partition_samples.train_test_split(test_size=split_percentage)

    partition_samples = split_samples["train"]
    remaining_samples = split_samples["test"]

    return partition_samples, remaining_samples


def is_zero(remaining_percentage):
    # to avoid numerical issues
    return abs(remaining_percentage - 0.0) < 1e-6


def load_data(cfg):
    DatasetParams = namedtuple("DatasetParams", ["name", "fine_grained", "train_split", "test_split", "hf_key"])
    dataset_params: DatasetParams = DatasetParams(
        cfg.dataset.ref,
        None,
        cfg.dataset.train_split,
        cfg.dataset.test_split,
        (cfg.dataset.ref,),
    )
    DATASET_KEY = "_".join(
        map(
            str,
            [v for k, v in dataset_params._asdict().items() if k != "hf_key" and v is not None],
        )
    )
    DATASET_DIR: Path = PROJECT_ROOT / "data" / "encoded_data" / DATASET_KEY
    if not DATASET_DIR.exists() or not cfg.use_cached:
        train_dataset = load_dataset(
            dataset_params.name,
            split=dataset_params.train_split,
            use_auth_token=True,
        )
        test_dataset = load_dataset(dataset_params.name, split=dataset_params.test_split)
        dataset: DatasetDict = DatasetDict(train=train_dataset, test=test_dataset)
    else:
        dataset: Dataset = load_from_disk(dataset_path=str(DATASET_DIR))

    return dataset


def save_to_disk(cfg, new_dataset):
    """

    :param cfg:
    :param new_dataset:
    :return:
    """

    dataset_folder = PROJECT_ROOT / "data" / f"{cfg.dataset.name}"

    if not dataset_folder.exists():
        dataset_folder.mkdir()

    output_folder = dataset_folder / "partitioned"

    if not (output_folder).exists():
        (output_folder).mkdir()

    new_dataset.save_to_disk(output_folder)


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="divide_in_nondisjoint_tasks")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
