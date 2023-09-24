import json
import logging
import os
import random
from functools import partial
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Union
from hydra.utils import instantiate

import pytorch_lightning as pl
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
    plot_prototypes,
    plot_self_dist,
    plot_space_grid,
    reduce,
    self_sim_comparison,
)
from la.utils.separability_analysis import compute_separabilities
from la.utils.task_utils import get_shared_samples_ids, map_labels_to_global
from la.utils.utils import add_tensor_column, save_dict_to_file, standard_normalization
from torch.utils.data import DataLoader

# plt.style.use("dark_background")


pylogger = logging.getLogger(__name__)


def run(cfg: DictConfig) -> str:
    """ """
    seed_index_everything(cfg)

    all_results = {}

    analyses = ["cka", "classification", "clustering", "separability"]

    for analysis in analyses:
        all_results[analysis] = {
            dataset_name: {
                model_name: {
                    f"S{i}": {f"N{j}": {} for j in cfg.num_novel_classes[dataset_name]}
                    for i in cfg.num_shared_classes[dataset_name]
                }
                for model_name in cfg.model_names
            }
            for dataset_name in cfg.dataset_names
        }

    check_runs_exist(cfg.configurations)

    analysis_state = {}
    for single_cfg in cfg.configurations:
        single_cfg_results, analysis_state = single_configuration_experiment(cfg, single_cfg, analysis_state)

        for analysis in analyses:
            model_name = (
                "_".join(single_cfg.model_name) if isinstance(single_cfg.model_name, list) else single_cfg.model_name
            )

            if cfg.run_analysis[analysis]:
                all_results[analysis][single_cfg.dataset_name][model_name][f"S{single_cfg.num_shared_classes}"][
                    f"N{single_cfg.num_novel_classes}"
                ] = single_cfg_results[analysis]

    for analysis in analyses:
        if cfg.run_analysis[analysis]:
            save_dict_to_file(path=cfg.results_path[analysis], content=all_results[analysis])


def check_runs_exist(configurations):
    non_existing_runs = []

    for single_cfg in configurations:
        model_name = (
            "_".join(single_cfg.model_name) if isinstance(single_cfg.model_name, list) else single_cfg.model_name
        )

        dataset_name, num_shared_classes, num_novel_classes, model_name = (
            single_cfg.dataset_name,
            single_cfg.num_shared_classes,
            single_cfg.num_novel_classes,
            model_name,
        )

        dataset_path = f"{PROJECT_ROOT}/data/{dataset_name}/part_shared_part_novel/S{num_shared_classes}_N{num_novel_classes}_{model_name}"

        if not os.path.exists(dataset_path):
            run_identifier = (
                f"S{single_cfg.num_shared_classes}_N{single_cfg.num_novel_classes}_{model_name}_{dataset_name}"
            )
            non_existing_runs.append(run_identifier)

    assert len(non_existing_runs) == 0, f"The following runs do not exist: {non_existing_runs}"


def single_configuration_experiment(global_cfg, single_cfg, analysis_state):
    model_name = "_".join(single_cfg.model_name) if isinstance(single_cfg.model_name, list) else single_cfg.model_name
    dataset_name, num_shared_classes, num_novel_classes, model_name = (
        single_cfg.dataset_name,
        single_cfg.num_shared_classes,
        single_cfg.num_novel_classes,
        model_name,
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
        "train": concatenate_datasets(list(disjoint_samples["train"].values())),
        "test": concatenate_datasets(list(disjoint_samples["test"].values())),
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

    # construct a dataset that is just the concatenation of the absolute embeddings of the different tasks
    jumble_dataset = concatenate_datasets([data[f"task_{i}_{mode}"] for i in range(1, num_tasks + 1)])

    if global_cfg.run_analysis.qualitative:
        plots_path = (
            Path(global_cfg.results_path.plots)
            / dataset_name
            / model_name
            / f"S{num_shared_classes}_N{num_novel_classes}"
        )
        plots_path.mkdir(parents=True, exist_ok=True)

        compare_func = partial(
            compare_merged_original_qualitative,
            has_coarse_label=has_coarse_label,
            plots_path=plots_path,
            num_classes=num_total_classes,
            cfg=global_cfg,
        )

        compare_func(
            original_dataset_nonshared,
            merged_dataset_nonshared,
            suffix="_nonshared_classes",
        )

        compare_func(
            original_dataset_shared,
            merged_dataset_shared,
            suffix="_shared_classes",
        )

        if has_coarse_label:
            plot_prototypes(
                original_dataset,
                merged_dataset,
                orig_dataset_embed_key="embedding",
                merged_dataset_embed_key="relative_embeddings",
                reduction=Reduction.INDEPENDENT_PCA,
                plots_path=plots_path,
                suffix="_ours",
                prefix="",
                cfg=global_cfg,
            )

            plot_prototypes(
                original_dataset,
                jumble_dataset,
                orig_dataset_embed_key="embedding",
                merged_dataset_embed_key="embedding",
                reduction=Reduction.INDEPENDENT_PCA,
                plots_path=plots_path,
                suffix="_jumble",
                prefix="",
                cfg=global_cfg,
            )

    results = {}

    if global_cfg.run_analysis.clustering:
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

    if global_cfg.run_analysis.cka:
        cka = CKA(mode="linear", device="cuda")

        cka_orig_orig_rel_abs = cka(original_dataset["relative_embeddings"], original_dataset["embedding"])

        cka_orig_aggr_abs_rel = cka(original_dataset["embedding"], merged_dataset["relative_embeddings"])

        cka_orig_aggr_rel_rel = cka(merged_dataset["relative_embeddings"], original_dataset["relative_embeddings"])

        cka_nonshared = cka(
            merged_dataset_nonshared["relative_embeddings"],
            original_dataset_nonshared["relative_embeddings"],
        )

        cka_shared = cka(
            merged_dataset_shared["relative_embeddings"],
            original_dataset_shared["relative_embeddings"],
        )

        results["cka"] = {
            "cka_orig_orig_rel_abs": cka_orig_orig_rel_abs.detach().item(),
            "cka_orig_aggr_abs_rel": cka_orig_aggr_abs_rel.detach().item(),
            "cka_orig_aggr_rel_rel": cka_orig_aggr_rel_rel.detach().item(),
            "cka_shared": cka_shared.detach().item(),
            "cka_non_shared": cka_nonshared.detach().item(),
        }

    if global_cfg.run_analysis.separability:
        separabilities_original = compute_separabilities(
            original_dataset["embedding"], original_dataset["y"], non_shared_classes
        )
        mean_separabilities_original = torch.mean(torch.tensor([s[2] for s in separabilities_original]))

        separabilities_ours = compute_separabilities(
            merged_dataset["relative_embeddings"], merged_dataset["y"], non_shared_classes
        )
        mean_separability_ours = torch.mean(torch.tensor([s[2] for s in separabilities_ours]))
        separabilities_naive = compute_separabilities(
            jumble_dataset["embedding"], jumble_dataset["y"], non_shared_classes
        )

        mean_separability_naive = torch.mean(torch.tensor([s[2] for s in separabilities_naive]))

        results["separability"] = {
            "mean_separability_ours": mean_separability_ours.item(),
            "mean_separability_naive": mean_separability_naive.item(),
            "mean_separability_original": mean_separabilities_original.item(),
        }

    if global_cfg.run_analysis.classification:
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

        results["classification"] = {
            "original_abs": class_results_original_abs,
            "original_rel": class_results_original_rel,
            "jumble_abs": class_results_jumble,
            "merged": class_results_merged,
        }

    # distill a new model on the merged space
    if global_cfg.run_analysis.distillation:
        transform_func = instantiate(global_cfg.model.transform_func)

        merged_dataset.set_format("numpy", columns=["img"])

        map_params = {
            "function": lambda x: {"x": transform_func(x["img"])},
            "writer_batch_size": 100,
            "num_proc": 1,
        }

        merged_dataset = merged_dataset.map(desc=f"Transforming merged samples", **map_params)

        model: pl.LightningModule = hydra.utils.instantiate(
            global_cfg.model,
            _recursive_=False,
            num_classes=num_total_classes,
            input_dim=merged_dataset["x"][0].shape[-1],
        )

        model = model.to("cuda")

        merged_dataset = merged_dataset.rename_column("relative_embeddings", "teacher_embeds")

        merged_dataset.set_format(type="torch", columns=["teacher_embeds", "y", "x"])

        loader_func = partial(
            torch.utils.data.DataLoader,
            batch_size=512,
            num_workers=8,
        )

        trainer_func = partial(Trainer, gpus=1, max_epochs=100, logger=False, enable_progress_bar=True)

        # split dataset in train, val and test
        split_dataset = merged_dataset.train_test_split(test_size=0.3, seed=42)
        train_dataset = split_dataset["train"]
        val_test_dataset = split_dataset["test"]

        split_val_test = val_test_dataset.train_test_split(test_size=0.5, seed=42)
        val_dataset = split_val_test["train"]
        test_dataset = split_val_test["test"]

        train_loader = loader_func(train_dataset, shuffle=True)
        val_loader = loader_func(val_dataset, shuffle=False)
        test_loader = loader_func(test_dataset, shuffle=False)

        trainer = trainer_func()
        trainer.fit(model, train_loader, val_loader)
        trainer.test(model, test_loader)

    return results, analysis_state


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
