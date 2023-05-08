import os
import nn_core  # noqa
import logging
import random
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
        f"Subdividing dataset {cfg.dataset.name} with {cfg.num_shared_classes} shared classes and {cfg.num_novel_classes_per_task} novel classes for task."
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

    # Sample shared classes
    shared_classes = set(random.sample(all_classes_ids, k=cfg.num_shared_classes))

    non_shared_classes = set([c for c in all_classes_ids if c not in shared_classes])

    assert len(non_shared_classes) == num_classes - cfg.num_shared_classes

    # Subdivide data into tasks defined by different classes subsets
    num_tasks = (num_classes - cfg.num_shared_classes) // cfg.num_novel_classes_per_task
    new_dataset = MyDatasetDict()
    global_to_local_class_mappings = {}

    # task 0 is a dummy task that consists of the samples for all the classes
    train_val_split = dataset["train"].train_test_split(test_size=cfg.per_task_val_percentage)
    new_dataset["task_0_train"], new_dataset["task_0_val"] = train_val_split["train"], train_val_split["test"]
    new_dataset["task_0_test"] = dataset["test"]

    global_to_local_class_mappings["task_0"] = {class_str_to_id[c]: i for i, c in enumerate(all_classes)}

    shared_train_samples = dataset["train"].filter(lambda x: x[cfg.label_key] in shared_classes)
    shared_test_samples = dataset["test"].filter(lambda x: x[cfg.label_key] in shared_classes)

    for i in range(1, num_tasks + 1):
        task_output = prepare_task(
            cfg,
            dataset,
            non_shared_classes,
            shared_classes,
            shared_test_samples,
            shared_train_samples,
        )

        non_shared_classes = task_output["non_shared_classes"]
        pylogger.info(f"Task {i}, remaining classes: {non_shared_classes}")

        global_to_local_class_mappings[f"task_{i}"] = task_output["global_to_local_class_map"]
        new_dataset[f"task_{i}_train"] = task_output["task_train_samples"]
        new_dataset[f"task_{i}_val"] = task_output["task_val_samples"]
        new_dataset[f"task_{i}_test"] = task_output["task_test_samples"]

    metadata = {
        "num_train_samples_per_class": cfg.dataset.num_train_samples_per_class,
        "num_test_samples_per_class": cfg.dataset.num_test_samples_per_class,
        "num_shared_classes": cfg.num_shared_classes,
        "num_novel_classes_per_task": cfg.num_novel_classes_per_task,
        "num_tasks": num_tasks,
        "shared_classes": list(shared_classes),
        "non_shared_classes": list(non_shared_classes),
        "all_classes": all_classes,
        "all_classes_ids": all_classes_ids,
        "num_classes": num_classes,
        "global_to_local_class_mappings": global_to_local_class_mappings,
    }

    new_dataset["metadata"] = metadata
    pylogger.info(metadata["global_to_local_class_mappings"])

    run_safety_checks(new_dataset, shared_classes, num_tasks)

    save_dataset_to_disk(new_dataset, cfg.output_path)


def prepare_task(
    cfg,
    dataset,
    non_shared_classes,
    shared_classes,
    shared_test_samples,
    shared_train_samples,
):
    """

    :param dataset:
    :param non_shared_classes:
    :param num_novel_classes_per_task:
    :param shared_classes:
    :param shared_test_samples:
    :param shared_train_samples:
    :return:
    """
    label_key = cfg.label_key
    task_novel_classes = set(random.sample(list(non_shared_classes), k=cfg.num_novel_classes_per_task))

    # remove the classes sampled for this task so that all tasks have disjoint novel classes
    non_shared_classes = non_shared_classes.difference(task_novel_classes)
    task_classes = shared_classes.union(task_novel_classes)
    global_to_local_class_map = {c: i for i, c in enumerate(list(task_classes))}
    novel_train_samples = dataset["train"].filter(lambda x: x[label_key] in task_novel_classes)

    task_train_samples = concatenate_datasets([shared_train_samples, novel_train_samples])

    task_train_samples = task_train_samples.map(lambda row: {cfg.label_key: global_to_local_class_map[row[label_key]]})

    assert len(task_train_samples) == cfg.dataset.num_train_samples_per_class * len(task_classes)

    train_val_split = task_train_samples.train_test_split(test_size=cfg.per_task_val_percentage)
    task_train_samples, task_val_samples = train_val_split["train"], train_val_split["test"]

    novel_test_samples = dataset["test"].filter(lambda x: x[label_key] in task_novel_classes)
    task_test_samples = concatenate_datasets([shared_test_samples, novel_test_samples])
    task_test_samples = task_test_samples.map(lambda row: {cfg.label_key: global_to_local_class_map[row[label_key]]})

    assert len(task_test_samples) == cfg.dataset.num_test_samples_per_class * len(task_classes)

    return {
        "non_shared_classes": non_shared_classes,
        "task_train_samples": task_train_samples,
        "task_val_samples": task_val_samples,
        "task_test_samples": task_test_samples,
        "global_to_local_class_map": global_to_local_class_map,
    }


def run_safety_checks(dataset, shared_classes, num_tasks):
    """
    Check that the samples belonging to novel task-specific classes don't overlap
    """
    for mode in ["train", "test"]:
        for task_ind in range(1, num_tasks):
            global_to_local_class_map = dataset["metadata"]["global_to_local_class_mappings"][f"task_{task_ind}"]
            local_to_global_class_map = {v: k for k, v in global_to_local_class_map.items()}

            task_i_samples = dataset[f"task_{task_ind}_{mode}"].map(
                lambda row: {"y": local_to_global_class_map[row["y"]]},
                desc="Mapping to global classes",
            )

            task_i_samples = task_i_samples.map(
                lambda row: {"shared": row["y"] in shared_classes},
                desc="Adding shared column to samples",
            )

            task_novel_samples = task_i_samples.filter(lambda x: x["shared"] == False)
            task_novel_ids = task_novel_samples["id"]

            for task_j in range(task_ind + 1, num_tasks + 1):
                global_to_local_class_map = dataset["metadata"]["global_to_local_class_mappings"][f"task_{task_ind}"]
                local_to_global_class_map = {v: k for k, v in global_to_local_class_map.items()}

                task_j_samples = dataset[f"task_{task_j}_{mode}"].map(
                    lambda row: {"y": local_to_global_class_map[row["y"]]},
                    desc="Mapping to global classes",
                )

                task_j_samples = task_j_samples.map(
                    lambda row: {"shared": row["y"] in shared_classes},
                    desc="Adding shared column to samples",
                )

                other_task_novel_samples = task_j_samples.filter(lambda x: x["shared"] == False)
                other_task_novel_ids = other_task_novel_samples["id"]

                common_novel_samples = set(task_novel_ids).intersection(set(other_task_novel_ids))
                assert len(common_novel_samples) == 0


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="prepare_data_part_shared_part_novel")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
