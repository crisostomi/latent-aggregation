import json
import logging
import os
import random
from functools import partial
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Union

import hydra
import matplotlib.pyplot as plt
import omegaconf
import pytorch_lightning
import torch
import torchmetrics
from datasets import Dataset, concatenate_datasets
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything
from sklearn import metrics
from sklearn.cluster import KMeans

# classification analysis stuff
from torch import nn
from torch.nn import functional as F

from nn_core.common import PROJECT_ROOT
from nn_core.common.utils import seed_index_everything

# Force the execution of __init__.py if this file is executed directly.
import la  # noqa
from la.data.my_dataset_dict import MyDatasetDict
from la.utils.cka import CKA
from la.utils.class_analysis import Classifier, Model, PartSharedPartNovelModel
from la.utils.relative_analysis import (
    Reduction,
    compare_merged_original_qualitative,
    plot_pairwise_dist,
    plot_self_dist,
    plot_space_grid,
    reduce,
    self_sim_comparison,
)
from la.utils.task_utils import get_shared_samples_ids, map_labels_to_global
from la.utils.utils import add_tensor_column, save_dict_to_file, standard_normalization

# plt.style.use("dark_background")


pylogger = logging.getLogger(__name__)


def run(cfg: DictConfig) -> str:
    """ """
    seed_index_everything(cfg)

    all_results = {}

    analyses = ["cka", "classification", "clustering"]

    for analysis in analyses:
        all_results[analysis] = {
            dataset_name: {model_name: {} for model_name in cfg.model_names} for dataset_name in cfg.dataset_names
        }

    check_runs_exist(cfg.configurations)

    for single_cfg in cfg.configurations:
        single_cfg_results = single_configuration_experiment(cfg, single_cfg)

        for analysis in analyses:
            if cfg.run_analysis[analysis]:
                all_results[analysis][single_cfg.dataset_name]["_".join(single_cfg.model_name)][
                    f"S{single_cfg.num_shared_classes}_N{single_cfg.num_novel_classes}"
                ] = single_cfg_results[analysis]

    for analysis in analyses:

        if cfg.run_analysis[analysis]:
            save_dict_to_file(path=cfg.results_path[analysis], content=all_results[analysis])


def check_runs_exist(configurations):
    for single_cfg in configurations:
        dataset_name, num_shared_classes, num_novel_classes, model_name = (
            single_cfg.dataset_name,
            single_cfg.num_shared_classes,
            single_cfg.num_novel_classes,
            # single_cfg.model_name,
            "_".join(single_cfg.model_name),
        )

        dataset_path = f"{PROJECT_ROOT}/data/{dataset_name}/part_shared_part_novel/S{num_shared_classes}_N{num_novel_classes}_{model_name}"

        assert os.path.exists(dataset_path), f"Path {dataset_path} does not exist."


def single_configuration_experiment(global_cfg, single_cfg):
    dataset_name, num_shared_classes, num_novel_classes, model_name = (
        single_cfg.dataset_name,
        single_cfg.num_shared_classes,
        single_cfg.num_novel_classes,
        # single_cfg.model_name,
        "_".join(single_cfg.model_name),
    )

    has_coarse_label = global_cfg.has_coarse_label[dataset_name]

    pylogger.info(
        f"Running experiment on {dataset_name} embedded with {model_name} with {num_shared_classes} shared classes and {num_novel_classes} novel classes for task."
    )

    dataset_path = f"{PROJECT_ROOT}/data/{dataset_name}/part_shared_part_novel/S{num_shared_classes}_N{num_novel_classes}_{model_name}"

    data: MyDatasetDict = MyDatasetDict.load_from_disk(dataset_dict_path=dataset_path)

    num_total_classes = global_cfg.num_total_classes[dataset_name]
    non_shared_classes = set(range(num_total_classes)).difference(data["metadata"]["shared_classes"])
    num_tasks = data["metadata"]["num_tasks"]
    shared_classes = set(data["metadata"]["shared_classes"])

    # TODO: discuss this, is there anything better we can do? this is correct but it worsens the results
    # re-unite train and val sets for each task
    for task_ind in range(num_tasks + 1):
        data[f"task_{task_ind}_train"] = concatenate_datasets(
            [data[f"task_{task_ind}_train"], data[f"task_{task_ind}_val"]]
        ).sort("id")

    tensor_columns = ["embedding", "y", "id"]
    set_torch_format(data, num_tasks, tensor_columns)

    map_labels_to_global(data, num_tasks)

    num_anchors = data["task_0_train"]["embedding"].shape[1]
    pylogger.info(f"Using {num_anchors} anchors")

    shared_ids = get_shared_samples_ids(data, num_tasks, shared_classes)

    anchor_ids = random.sample(shared_ids, num_anchors)

    add_anchor_column(data, num_tasks, anchor_ids)

    map_to_relative_spaces(data, num_tasks)

    tensor_columns.extend(["relative_embeddings", "shared"])
    set_torch_format(data, num_tasks, tensor_columns)

    shared_samples, disjoint_samples = subdivide_shared_and_non_shared(data, num_tasks)

    novel_samples = {
        "train": concatenate_datasets(disjoint_samples["train"]),
        "test": concatenate_datasets(disjoint_samples["test"]),
    }

    # Merge the subspaces to obtain a single unified space
    merged_datasets = {"train": [], "test": []}

    for mode in ["train", "test"]:
        merged_datasets[mode] = merge_subspaces(shared_samples[mode], novel_samples[mode])

    # Analysis
    mode = "test"  # train or test
    merged_dataset = merged_datasets[mode].sort("id")
    original_dataset = data[f"task_0_{mode}"].sort("id")

    columns = ["relative_embeddings", "y", "embedding"]
    if has_coarse_label:
        columns.append("coarse_label")

    merged_dataset.set_format(type="torch", columns=columns)
    original_dataset.set_format(type="torch", columns=columns)

    merged_dataset_nonshared = merged_dataset.filter(lambda row: row["y"].item() in non_shared_classes).sort("id")
    original_dataset_nonshared = original_dataset.filter(lambda row: row["y"].item() in non_shared_classes).sort("id")

    merged_dataset_shared = merged_dataset.filter(lambda row: row["y"].item() in shared_classes)
    original_dataset_shared = original_dataset.filter(lambda row: row["y"].item() in shared_classes)

    if global_cfg.run_qualitative_analysis:
        plots_path = (
            Path(global_cfg.plots_path) / dataset_name / model_name / f"S{num_shared_classes}_N{num_novel_classes}"
        )
        plots_path.mkdir(parents=True, exist_ok=True)

        compare_merged_original_qualitative(
            original_dataset, merged_dataset, has_coarse_label, plots_path, suffix="_all_classes"
        )

        compare_merged_original_qualitative(
            original_dataset_nonshared,
            merged_dataset_nonshared,
            has_coarse_label,
            plots_path,
            suffix="_nonshared_classes",
        )

        compare_merged_original_qualitative(
            original_dataset_shared,
            merged_dataset_shared,
            has_coarse_label,
            plots_path,
            suffix="_shared_classes",
        )

    results = {}

    if global_cfg.run_clustering_analysis:
        clustering_results_original = compute_clustering_metrics(
            original_dataset["embedding"], space_y=original_dataset["y"], num_classes=num_total_classes
        )

        clustering_results_rel = compute_clustering_metrics(
            original_dataset["relative_embeddings"], space_y=original_dataset["y"], num_classes=num_total_classes
        )

        clustering_results_merged = compute_clustering_metrics(
            merged_dataset["relative_embeddings"], space_y=merged_dataset["y"], num_classes=num_total_classes
        )

        results["clustering"] = {
            "original_abs": clustering_results_original,
            "original_rel": clustering_results_rel,
            "merged": clustering_results_merged,
        }

    if global_cfg.run_cka_analysis:
        cka = CKA(mode="linear", device="cuda")

        cka_rel_abs = None
        try:
            cka_rel_abs = cka(original_dataset["relative_embeddings"], torch.stack(original_dataset["embedding"]))
            cka_rel_abs.detach().item()
        except:
            pylogger.info(
                "CKA between relative and absolute embeddings not possible when using different architectures"
            )

        cka_tot = cka(merged_dataset["relative_embeddings"], original_dataset["relative_embeddings"])

        cka_nonshared = cka(
            merged_dataset_nonshared["relative_embeddings"],
            original_dataset_nonshared["relative_embeddings"],
        )

        cka_shared = cka(
            merged_dataset_shared["relative_embeddings"],
            original_dataset_shared["relative_embeddings"],
        )

        results["cka"] = {
            "cka_rel_abs": cka_rel_abs,
            "cka_tot": cka_tot.detach().item(),
            "cka_shared": cka_shared.detach().item(),
            "cka_non_shared": cka_nonshared.detach().item(),
        }

    if global_cfg.run_classification_analysis:
        # construct a dataset that is just the concatenation of the absolute embeddings of the different tasks
        jumble_dataset = concatenate_datasets([data[f"task_{i}_{mode}"] for i in range(1, num_tasks + 1)])

        class_exp = partial(
            run_classification_experiment,
            shared_classes,
            non_shared_classes,
            num_total_classes,
            global_cfg,
            num_anchors,
        )

        class_results_original_abs = class_exp(original_dataset, use_relatives=False)

        class_results_original_rel = class_exp(original_dataset, use_relatives=True)

        class_results_jumble = class_exp(jumble_dataset, use_relatives=False)

        class_results_merged = class_exp(merged_dataset, use_relatives=True)

        results["class"] = {
            "original_abs": class_results_original_abs,
            "original_rel": class_results_original_rel,
            "jumble_abs": class_results_jumble,
            "merged": class_results_merged,
        }

    return results


def add_anchor_column(data, num_tasks, anchor_ids):
    # only training samples can be anchors
    for task_ind in range(num_tasks + 1):
        data[f"task_{task_ind}_train"] = data[f"task_{task_ind}_train"].map(
            lambda row: {"anchor": row["id"].item() in anchor_ids},
            desc="Adding anchor column to train samples",
        )


def map_to_relative_spaces(data, num_tasks, normalize=False):
    """

    :param data:
    :param num_tasks:
    """
    for task_ind in range(0, num_tasks + 1):
        task_anchors = data[f"task_{task_ind}_train"]["embedding"][data[f"task_{task_ind}_train"]["anchor"]]

        if normalize:
            task_anchors = standard_normalization(task_anchors)

        norm_anchors = F.normalize(task_anchors, p=2, dim=-1)

        for mode in ["train", "test"]:
            task_embeddings = data[f"task_{task_ind}_{mode}"]["embedding"]

            if normalize:
                task_embeddings = standard_normalization(task_embeddings)

            abs_space = F.normalize(task_embeddings, p=2, dim=-1)

            rel_space = abs_space @ norm_anchors.T

            data[f"task_{task_ind}_{mode}"] = data[f"task_{task_ind}_{mode}"].map(
                desc=f"Mapping {mode} task {task_ind} to relative space",
                function=lambda row, ind: {"relative_embeddings": rel_space[ind]},
                with_indices=True,
            )


def set_torch_format(data, num_tasks, tensor_columns):
    for task_ind in range(0, num_tasks + 1):
        for mode in ["train", "test"]:
            data[f"task_{task_ind}_{mode}"].set_format(type="torch", columns=tensor_columns)


def subdivide_shared_and_non_shared(
    data: MyDatasetDict, num_tasks: int
) -> Tuple[Dict[str, Dataset], Dict[str, Dataset]]:
    pylogger.info("Subdividing shared and non-shared samples")

    shared_samples = {"train": {}, "test": {}}
    disjoint_samples = {"train": {}, "test": {}}

    for task_ind in range(1, num_tasks + 1):
        for mode in ["train", "test"]:
            task_shared_samples = (
                data[f"task_{task_ind}_{mode}"]
                .filter(
                    lambda row: row["shared"],
                    desc=f"Selecting shared samples for {mode} task {task_ind}",
                )
                .sort("id")
            )

            task_novel_samples = data[f"task_{task_ind}_{mode}"].filter(
                lambda row: ~row["shared"],
                desc=f"Selecting novel samples for {mode} task {task_ind}",
            )

            shared_samples[mode][f"task_{task_ind}"] = task_shared_samples
            disjoint_samples[mode][f"task_{task_ind}"] = task_novel_samples

    # check that novel samples have disjoint ids
    for task_ind in range(0, num_tasks):
        for mode in ["train", "test"]:
            task_novel_samples = disjoint_samples[mode][f"task_{task_ind + 1}"]

            task_novel_ids = task_novel_samples["id"]

            for task_j in range(task_ind + 1, num_tasks):
                other_task_novel_samples = disjoint_samples[mode][f"task_{task_j + 1}"]

                other_task_novel_ids = other_task_novel_samples["id"]

                common_novel_samples = set(task_novel_ids.tolist()).intersection(set(other_task_novel_ids.tolist()))
                assert len(common_novel_samples) == 0

    pylogger.info("Checked that all tasks have disjoint novel samples")

    return shared_samples, disjoint_samples


def merge_subspaces(shared_samples, novel_samples, method="mean"):
    """
    Average the shared samples and then concat the task-specific samples and the shared samples to go to the merged space

    # compute the mean of the shared_samples and put them back in the dataset
    # Extract the 'embedding' columns from each dataset
    """
    shared_rel_embeddings = [dataset["relative_embeddings"] for dataset in shared_samples.values()]

    first_dataset = list(shared_samples.values())[0]
    # Create a new dataset with the same features as the original datasets
    new_features = first_dataset.features.copy()

    # Replace the 'embedding' column in the new dataset with the mean embeddings
    new_data = {column: first_dataset[column] for column in new_features}

    if method == "mean":
        # Calculate the mean of the embeddings for each sample
        merged_embeddings = torch.mean(torch.stack(shared_rel_embeddings), dim=0)

    elif method == "single":
        # Just take the first task's embeddings
        merged_embeddings = shared_rel_embeddings[0]

    new_data["relative_embeddings"] = merged_embeddings.tolist()

    # Create the new Hugging Face dataset
    shared_dataset = Dataset.from_dict(new_data, features=new_features)

    merged_dataset = concatenate_datasets([shared_dataset, novel_samples])

    merged_dataset = merged_dataset

    return merged_dataset


def compute_clustering_metrics(space, space_y, num_classes):
    """
    Compute the clustering metrics for the dataset
    """

    kmeans = KMeans(n_clusters=num_classes, random_state=0).fit(space)

    assignment = kmeans.labels_

    clustering_metrics = {
        "silhouette_score": metrics.silhouette_score(space, assignment, metric="euclidean").item(),
        "completeness_score": metrics.completeness_score(labels_true=space_y, labels_pred=assignment).item(),
        "homogeneity_score": metrics.homogeneity_score(labels_true=space_y, labels_pred=assignment).item(),
        "mutual_info_score": metrics.mutual_info_score(labels_true=space_y, labels_pred=assignment).item(),
    }

    return clustering_metrics


def run_classification_experiment(
    shared_classes,
    non_shared_classes,
    num_total_classes,
    global_cfg,
    num_anchors,
    dataset,
    use_relatives,
):
    seed_everything(42)

    dataloader_func = partial(
        torch.utils.data.DataLoader,
        batch_size=128,
        num_workers=8,
    )

    trainer_func = partial(Trainer, gpus=1, max_epochs=100, logger=False, enable_progress_bar=True)

    classifier = Classifier(
        input_dim=num_anchors,
        classifier_embed_dim=global_cfg.classifier_embed_dim,
        num_classes=num_total_classes,
    )
    model = PartSharedPartNovelModel(
        classifier=classifier,
        shared_classes=shared_classes,
        non_shared_classes=non_shared_classes,
        use_relatives=use_relatives,
    )
    trainer = trainer_func(callbacks=[pytorch_lightning.callbacks.EarlyStopping(monitor="val_loss", patience=10)])

    # split dataset in train, val and test
    split_dataset = dataset.train_test_split(test_size=0.3, seed=42)
    train_dataset = split_dataset["train"]
    val_test_dataset = split_dataset["test"]

    split_val_test = val_test_dataset.train_test_split(test_size=0.5, seed=42)
    val_dataset = split_val_test["train"]
    test_dataset = split_val_test["test"]

    train_dataloader = dataloader_func(train_dataset, shuffle=True)
    val_dataloader = dataloader_func(val_dataset, shuffle=False)
    test_dataloader = dataloader_func(test_dataset, shuffle=False)

    trainer.fit(model, train_dataloader, val_dataloader)

    results = trainer.test(model, test_dataloader)[0]

    results = {
        "total_acc": results["test_acc_epoch"],
        "shared_class_acc": results["test_acc_shared_classes_epoch"],
        "non_shared_class_acc": results["test_acc_non_shared_classes_epoch"],
    }

    return results


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="analyze_part_shared_part_novel")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
