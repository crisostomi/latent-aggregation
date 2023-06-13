import json
import logging
import random
from functools import partial
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import omegaconf
import pytorch_lightning
import torch
import torchmetrics
from datasets import Dataset, concatenate_datasets
from nn_core.common import PROJECT_ROOT
from nn_core.common.utils import seed_index_everything
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything
from functools import partial

from torch import nn
from torch.nn import functional as F
import torch.nn as nn

import la  # noqa
from la.utils.cka import CKA
from la.utils.class_analysis import Classifier, TaskEmbeddingModel
from la.utils.relative_analysis import compare_merged_original_qualitative
from la.utils.utils import add_tensor_column, save_dict_to_file
from pytorch_lightning import Trainer
from la.data.my_dataset_dict import MyDatasetDict
from la.utils.class_analysis import Classifier, KNNClassifier, Model

pylogger = logging.getLogger(__name__)


def run(cfg: DictConfig) -> str:
    """
    Main entry point for the experiment.
    """
    seed_index_everything(cfg)

    all_results = {}

    analyses = ["cka", "classification", "clustering", "knn"]

    for analysis in analyses:
        all_results[analysis] = {
            dataset_name: {
                model_name: {partition: {} for partition in cfg.partitions} for model_name in cfg.model_names
            }
            for dataset_name in cfg.dataset_names
        }

    # check_runs_exist(cfg.configurations)

    for single_cfg in cfg.configurations:
        single_cfg_results = single_configuration_experiment(cfg, single_cfg)

        for analysis in analyses:
            if cfg.run_analysis[analysis]:
                all_results[analysis][single_cfg.dataset_name][single_cfg.model_name][
                    single_cfg.partition_id
                ] = single_cfg_results[analysis]

    for analysis in analyses:

        if cfg.run_analysis[analysis]:
            save_dict_to_file(path=cfg.results_path[analysis], content=all_results[analysis])


def single_configuration_experiment(global_cfg: DictConfig, single_cfg: DictConfig):
    """
    Run a single experiment with the given configurations.

    :param global_cfg: shared configurations for the suite of experiments
    :param single_cfg: configurations for the current experiment
    """
    dataset_name, model_name, partition_id = (single_cfg.dataset_name, single_cfg.model_name, single_cfg.partition_id)

    has_coarse_label = global_cfg.has_coarse_label[dataset_name]

    pylogger.info(f"Running experiment on {dataset_name} embedded with {model_name}.")

    dataset_path = (
        f"{PROJECT_ROOT}/data/{dataset_name}/totally_disjoint/partition-{partition_id}_{model_name}_embedding"
    )

    data: MyDatasetDict = MyDatasetDict.load_from_disk(dataset_dict_path=dataset_path)

    num_tasks = data["metadata"]["num_tasks"]
    num_total_classes = global_cfg.num_total_classes[dataset_name]

    map_labels_to_global(data, num_tasks)

    tensor_columns = ["embedding", "y", "id"]
    if has_coarse_label:
        tensor_columns.append("coarse_label")
    set_torch_format(data, num_tasks, modes=["train", "test", "anchors"], tensor_columns=tensor_columns)

    num_anchors = len(data["task_0_anchors"])

    SUBSAMPLE_ANCHORS = global_cfg.num_anchors < num_anchors

    if SUBSAMPLE_ANCHORS:
        pylogger.info(f"Selecting only {global_cfg.num_anchors} anchors out of {num_anchors}")
        num_anchors = global_cfg.num_anchors
        for task in range(num_tasks + 1):
            anchors_subsample = data[f"task_{task}_anchors"].select(range(num_anchors))
            data[f"task_{task}_anchors"] = anchors_subsample

    check_same_anchor_ids(data, num_tasks)

    centering = False
    if centering:
        for task_ind in range(num_tasks + 1):
            for mode in ["train", "test", "anchors"]:
                embedding_mean = data[f"task_{task_ind}_{mode}"]["embedding"].mean(dim=0)
                data[f"task_{task_ind}_{mode}"] = data[f"task_{task_ind}_train"].map(
                    lambda row: {"embedding": row["embedding"] - embedding_mean}
                )

    # map to relative
    for task_ind in range(0, num_tasks + 1):
        task_anchors = data[f"task_{task_ind}_anchors"]["embedding"]
        norm_anchors = F.normalize(task_anchors, p=2, dim=-1)

        for mode in ["train", "test"]:
            task_embeddings = data[f"task_{task_ind}_{mode}"]["embedding"]

            abs_space = F.normalize(task_embeddings, p=2, dim=-1)

            rel_space = abs_space @ norm_anchors.T

            data[f"task_{task_ind}_{mode}"] = add_tensor_column(
                data[f"task_{task_ind}_{mode}"], "relative_embeddings", rel_space
            )

    tensor_columns = tensor_columns + ["relative_embeddings"]
    set_torch_format(data, num_tasks, modes=["train", "test"], tensor_columns=tensor_columns)

    merged_dataset_train = concatenate_datasets([data[f"task_{i}_train"] for i in range(1, num_tasks + 1)])
    merged_dataset_test = concatenate_datasets([data[f"task_{i}_test"] for i in range(1, num_tasks + 1)])

    # sort the datasets by ID to have a consistent order
    original_dataset_train = data[f"task_0_train"].sort("id")
    original_dataset_test = data[f"task_0_test"].sort("id")

    merged_dataset_train = merged_dataset_train.sort("id")
    merged_dataset_test = merged_dataset_test.sort("id")

    # this fails because original_dataset_train has more samples than merged_dataset_train because of the anchors
    # assert torch.all(torch.eq(merged_dataset_train["id"], original_dataset_train["id"]))

    assert torch.all(torch.eq(merged_dataset_test["id"], original_dataset_test["id"]))

    if global_cfg.run_analysis["qualitative"]:
        # qualitative comparison absolute -- merged
        plots_path = Path(global_cfg.plots_path) / dataset_name / model_name / f"partition-{partition_id}"
        plots_path.mkdir(parents=True, exist_ok=True)

        compare_merged_original_qualitative(
            original_dataset_test, merged_dataset_test, has_coarse_label, plots_path, prefix="", suffix="all_classes"
        )

    results = {}

    if global_cfg.run_analysis["cka"]:

        cka = CKA(mode="linear", device="cuda")

        cka_rel_abs = cka(merged_dataset_test["relative_embeddings"], merged_dataset_test["embedding"])

        cka_tot = cka(merged_dataset_test["relative_embeddings"], original_dataset_test["relative_embeddings"])

        results["cka"] = {
            "cka_rel_abs": cka_rel_abs.detach().item(),
            "cka_tot": cka_tot.detach().item(),
        }

    if global_cfg.run_analysis["knn"]:

        knn_results_original_abs = run_knn_class_experiment(
            num_total_classes,
            train_dataset=original_dataset_train,
            test_dataset=original_dataset_test,
            use_relatives=False,
        )

        knn_results_original_rel = run_knn_class_experiment(
            num_total_classes,
            train_dataset=original_dataset_train,
            test_dataset=original_dataset_test,
            use_relatives=True,
        )

        knn_results_merged = run_knn_class_experiment(
            num_total_classes, train_dataset=merged_dataset_train, test_dataset=merged_dataset_test, use_relatives=True
        )

        results["knn"] = {
            "original_abs": knn_results_original_abs,
            "original_rel": knn_results_original_rel,
            "merged": knn_results_merged,
        }

    if global_cfg.run_analysis["classification"]:

        label_to_task = {
            int(label): i
            for i in range(1, num_tasks + 1)
            for label in data["metadata"]["global_to_local_class_mappings"][f"task_{i}"].keys()
        }

        original_dataset_train = add_task_id(original_dataset_train, label_to_task)
        original_dataset_test = add_task_id(original_dataset_test, label_to_task)

        class_exp = partial(
            run_classification_experiment,
            num_total_classes=num_total_classes,
            classifier_embed_dim=global_cfg.classifier_embed_dim,
            num_tasks=num_tasks,
        )

        # add num_tasks dimensions to the absolute space which are the one-hot encoding of the task
        task_onehot_dataset_train = add_task_one_hot(original_dataset_train, num_tasks)
        task_onehot_dataset_test = add_task_one_hot(original_dataset_test, num_tasks)

        class_results_task_aware_abs = class_exp(
            train_dataset=task_onehot_dataset_train,
            test_dataset=task_onehot_dataset_test,
            use_relatives=False,
            input_dim=task_onehot_dataset_train["embedding"].shape[1],
        )

        jumble_train = concatenate_datasets([data[f"task_{i}_train"] for i in range(1, num_tasks + 1)])
        jumble_test = concatenate_datasets([data[f"task_{i}_test"] for i in range(1, num_tasks + 1)])
        class_results_jumble = class_exp(
            train_dataset=jumble_train,
            test_dataset=jumble_test,
            use_relatives=False,
            input_dim=original_dataset_train["embedding"].shape[1],
        )

        class_results_original_task_embedding = class_exp(
            train_dataset=original_dataset_train,
            test_dataset=original_dataset_test,
            use_relatives=False,
            input_dim=original_dataset_train["embedding"].shape[1],
            embed_tasks=True,
        )

        class_results_original_abs = class_exp(
            train_dataset=original_dataset_train,
            test_dataset=original_dataset_test,
            use_relatives=False,
            input_dim=original_dataset_train["embedding"].shape[1],
        )

        class_results_original_rel = class_exp(
            train_dataset=original_dataset_train,
            test_dataset=original_dataset_test,
            use_relatives=True,
            input_dim=num_anchors,
        )

        class_results_merged = class_exp(
            train_dataset=merged_dataset_train,
            test_dataset=merged_dataset_test,
            use_relatives=True,
            input_dim=num_anchors,
        )

        results["classification"] = {
            "original_abs": class_results_original_abs,
            "original_rel": class_results_original_rel,
            "jumble": class_results_jumble,
            "merged": class_results_merged,
            "task_onehot_abs": class_results_task_aware_abs,
            "task_embed_abs": class_results_original_task_embedding,
        }

    return results


def add_task_id(dataset, label_to_task):
    dataset = dataset.map(
        lambda row: {"task": label_to_task[row["y"].item()]},
        desc="Mapping labels to tasks.",
    )
    dataset.set_format("torch", columns=["embedding", "task", "y", "relative_embeddings"])

    return dataset


def add_task_one_hot(dataset, num_tasks):

    task_one_hot = torch.nn.functional.one_hot(dataset["task"] - 1, num_tasks).float()

    task_aware_dataset = dataset.map(
        lambda row: {"embedding": torch.cat([row["embedding"], task_one_hot[row["task"] - 1]])},
    )

    return task_aware_dataset


def map_labels_to_global(data, num_tasks):
    for task_ind in range(1, num_tasks + 1):
        global_to_local_map = data["metadata"]["global_to_local_class_mappings"][f"task_{task_ind}"]
        local_to_global_map = {v: int(k) for k, v in global_to_local_map.items()}

        for mode in ["train", "val", "test"]:
            data[f"task_{task_ind}_{mode}"] = data[f"task_{task_ind}_{mode}"].map(
                lambda row: {"y": local_to_global_map[row["y"].item()]},
                desc="Mapping labels back to global.",
            )


def set_torch_format(data, num_tasks, modes, tensor_columns):
    for task_ind in range(0, num_tasks + 1):
        for mode in modes:
            key = f"task_{task_ind}_{mode}"
            if key in data:
                data[key].set_format(type="torch", columns=tensor_columns)


def check_same_anchor_ids(data, num_tasks):
    for task_i in range(num_tasks + 1):
        for task_j in range(task_i, num_tasks + 1):
            assert torch.all(data[f"task_{task_i}_anchors"]["id"] == data[f"task_{task_j}_anchors"]["id"])


def run_classification_experiment(
    num_total_classes: int,
    input_dim: int,
    train_dataset,
    test_dataset,
    use_relatives: bool,
    classifier_embed_dim: int,
    num_tasks: int,
    embed_tasks: bool = False,
):
    """ """
    seed_everything(42)

    dataloader_func = partial(
        torch.utils.data.DataLoader,
        batch_size=128,
        num_workers=8,
    )

    trainer_func = partial(Trainer, gpus=1, max_epochs=100, logger=False, enable_progress_bar=True)

    if embed_tasks:
        classifier = Classifier(
            input_dim=input_dim + input_dim // 2,
            classifier_embed_dim=classifier_embed_dim,
            num_classes=num_total_classes,
        )

        task_embedding_dim = input_dim // 2
        model = TaskEmbeddingModel(
            classifier=classifier,
            use_relatives=use_relatives,
            num_tasks=num_tasks,
            task_embedding_dim=task_embedding_dim,
        )

    else:
        classifier = Classifier(
            input_dim=input_dim,
            classifier_embed_dim=classifier_embed_dim,
            num_classes=num_total_classes,
        )
        model = Model(
            classifier=classifier,
            use_relatives=use_relatives,
        )

    trainer = trainer_func(callbacks=[pytorch_lightning.callbacks.EarlyStopping(monitor="val_loss", patience=10)])

    split_dataset = train_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["test"]

    train_dataloader = dataloader_func(train_dataset, shuffle=True)
    val_dataloader = dataloader_func(val_dataset, shuffle=False)
    test_dataloader = dataloader_func(test_dataset, shuffle=False)

    trainer.fit(model, train_dataloader, val_dataloader)

    results = trainer.test(model, test_dataloader)[0]

    results = {
        "total_acc": results["test_acc"],
    }

    return results


def run_knn_class_experiment(
    num_total_classes: int,
    train_dataset,
    test_dataset,
    use_relatives: bool,
):
    seed_everything(42)
    torch.backends.cudnn.deterministic = True

    dataloader_func = partial(
        torch.utils.data.DataLoader,
        batch_size=128,
        num_workers=8,
    )

    trainer_func = partial(Trainer, gpus=1, max_epochs=1, logger=False, enable_progress_bar=True)

    model = KNNClassifier(train_dataset, num_total_classes, use_relatives=use_relatives)
    trainer = trainer_func()

    train_dataloader = dataloader_func(train_dataset, shuffle=True)
    test_dataloader = dataloader_func(test_dataset, shuffle=False)

    trainer.fit(model, train_dataloader)

    results = trainer.test(model, test_dataloader)[0]

    results = {
        "total_acc": results["test_acc"],
    }

    return results


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="analyze_totally_disjoint")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
