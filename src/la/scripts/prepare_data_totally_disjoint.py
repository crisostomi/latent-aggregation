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
from la.utils.io_utils import add_ids_to_dataset, load_data, preprocess_dataset, save_dataset_to_disk

from la.utils.utils import MyDatasetDict, convert_to_rgb

pylogger = logging.getLogger(__name__)


def run(cfg: DictConfig):
    seed_everything(cfg.seed)

    pylogger.info(
        f"Subdividing dataset {cfg.dataset.name} into totally disjoint subsets, class-level and sample-level."
    )

    dataset = load_data(cfg)

    dataset = preprocess_dataset(dataset, cfg)

    dataset = add_ids_to_dataset(dataset)

    if isinstance(dataset["train"].features[cfg.label_key], Value):
        all_classes = [str(class_id) for class_id in range(cfg.dataset.num_classes)]
    else:
        all_classes = dataset["train"].features[cfg.label_key].names

    num_classes = len(all_classes)

    all_classes_ids = [id for id, _ in enumerate(all_classes)]
    class_str_to_id = {c: i for i, c in enumerate(all_classes)}

    classes_partitions = cfg.classes_partitions

    num_tasks = len(classes_partitions)
    num_partitions = len(classes_partitions)

    classes_sets = [set(range(*partition)) for partition in classes_partitions]

    pylogger.info(classes_sets)

    new_dataset = MyDatasetDict()

    # task 0 is a dummy task that consists of the samples for all the classes
    val_train_split = dataset["train"].train_test_split(test_size=cfg.val_perc_per_task)

    new_dataset["task_0_train"] = val_train_split["train"]
    new_dataset["task_0_val"] = val_train_split["test"]
    new_dataset["task_0_test"] = dataset["test"]

    anchors = dataset["train"].shuffle(seed=cfg.seed).select(range(cfg.num_anchors))
    anchor_ids = set(anchors["id"])

    new_dataset["anchors"] = anchors
    global_to_local_class_mappings = {}
    global_to_local_class_mappings["task_0"] = {class_str_to_id[c]: i for i, c in enumerate(all_classes)}

    for part_ind in range(num_partitions):
        task_ind = part_ind + 1

        # e.g. C = {0, ... , 49}
        partition_classes = classes_sets[part_ind]

        global_to_local_class_map = {c: i for i, c in enumerate(partition_classes)}
        global_to_local_class_mappings[f"task_{task_ind}"] = global_to_local_class_map

        for mode in ["train", "test"]:
            # partition is a dataset containing only samples belonging to the corresponding class partition
            task_samples = dataset[mode].filter(
                lambda x: x[cfg.label_key] in partition_classes and x["id"] not in anchor_ids,
                desc=f"Creating partition {part_ind}",
            )

            task_samples = task_samples.map(lambda row: {cfg.label_key: global_to_local_class_map[row[cfg.label_key]]})

            if mode == "train":
                task_samples_split = task_samples.train_test_split(test_size=cfg.val_perc_per_task)

                new_dataset[f"task_{task_ind}_train"] = task_samples_split["train"]
                new_dataset[f"task_{task_ind}_val"] = task_samples_split["test"]

                pylogger.info(
                    f'Task {task_ind} has {len(task_samples_split["train"])} train samples and {len(task_samples_split["test"])} val samples'
                )
            else:
                new_dataset[f"task_{task_ind}_test"] = task_samples
                pylogger.info(f"Task {task_ind} has {len(task_samples)} test samples")

    metadata = {
        "num_train_samples_per_class": cfg.dataset.num_train_samples_per_class,
        "num_test_samples_per_class": cfg.dataset.num_test_samples_per_class,
        "num_tasks": num_tasks,
        "all_classes": all_classes,
        "all_classes_ids": all_classes_ids,
        "num_classes": num_classes,
        "global_to_local_class_mappings": global_to_local_class_mappings,
    }

    new_dataset["metadata"] = metadata

    save_dataset_to_disk(new_dataset, cfg.output_path)


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="prepare_data_totally_disjoint")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
