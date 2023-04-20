import logging
import random
from collections import namedtuple
from pathlib import Path

import hydra
from datasets import (
    load_dataset,
    DatasetDict,
    load_from_disk,
    Dataset,
    concatenate_datasets,
)
from nn_core.common import PROJECT_ROOT
from omegaconf import DictConfig, omegaconf
from pytorch_lightning import seed_everything

from la.utils.utils import MyDatasetDict

pylogger = logging.getLogger(__name__)


def run(cfg: DictConfig):
    seed_everything(cfg.seed)

    dataset = load_data(cfg)

    # add ids
    dataset["train"] = dataset["train"].map(lambda row, ind: {"id": ind}, batched=True, with_indices=True)
    dataset["test"] = dataset["test"].map(lambda row, ind: {"id": ind}, batched=True, with_indices=True)

    all_classes = dataset["train"].features[cfg.dataset.label_key].names
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
    new_dataset["task_0_train"] = dataset["train"]
    new_dataset["task_0_test"] = dataset["test"]

    global_to_local_class_mappings["task_0"] = {class_str_to_id[c]: i for i, c in enumerate(all_classes)}

    shared_train_samples = dataset["train"].filter(lambda x: x[cfg.dataset.label_key] in shared_classes)
    shared_test_samples = dataset["test"].filter(lambda x: x[cfg.dataset.label_key] in shared_classes)

    for i in range(1, num_tasks + 1):
        (non_shared_classes, task_test_samples, task_train_samples, global_to_local_class_map,) = prepare_task(
            cfg,
            dataset,
            non_shared_classes,
            shared_classes,
            shared_test_samples,
            shared_train_samples,
        )

        global_to_local_class_mappings[f"task_{i}"] = global_to_local_class_map
        new_dataset[f"task_{i}_train"] = task_train_samples
        new_dataset[f"task_{i}_test"] = task_test_samples

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

    save_to_disk(cfg, new_dataset, cfg.num_novel_classes_per_task, cfg.num_shared_classes)


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
    label_key = cfg.dataset.label
    task_novel_classes = set(random.sample(list(non_shared_classes), k=cfg.num_novel_classes_per_task))

    # remove the classes sampled for this task so that all tasks have disjoint novel classes
    non_shared_classes = non_shared_classes.difference(task_novel_classes)
    task_classes = shared_classes.union(task_novel_classes)
    global_to_local_class_map = {c: i for i, c in enumerate(list(task_classes))}
    novel_train_samples = dataset["train"].filter(lambda x: x[label_key] in task_novel_classes)

    task_train_samples = concatenate_datasets([shared_train_samples, novel_train_samples])

    task_train_samples = task_train_samples.map(lambda row: {"fine_label": global_to_local_class_map[row[label_key]]})

    novel_test_samples = dataset["test"].filter(lambda x: x[label_key] in task_novel_classes)
    task_test_samples = concatenate_datasets([shared_test_samples, novel_test_samples])
    task_test_samples = task_test_samples.map(lambda row: {"fine_label": global_to_local_class_map[row[label_key]]})

    assert len(task_train_samples) == cfg.dataset.num_train_samples_per_class * len(task_classes)
    assert len(task_test_samples) == cfg.dataset.num_test_samples_per_class * len(task_classes)

    return (
        non_shared_classes,
        task_test_samples,
        task_train_samples,
        global_to_local_class_map,
    )


def save_to_disk(cfg, new_dataset, num_novel_classes_per_task, num_shared_classes):
    """

    :param cfg:
    :param new_dataset:
    :param num_novel_classes_per_task:
    :param num_shared_classes:
    :return:
    """

    dataset_folder = PROJECT_ROOT / "data" / f"{cfg.dataset.name}"

    if not dataset_folder.exists():
        dataset_folder.mkdir()

    output_folder = dataset_folder / f"S{num_shared_classes}_N{num_novel_classes_per_task}"

    if not (output_folder).exists():
        (output_folder).mkdir()

    new_dataset.save_to_disk(output_folder)


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="divide_in_tasks")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
