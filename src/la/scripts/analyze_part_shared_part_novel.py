import json
import logging
import os
from pathlib import Path
import random
from functools import partial

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

# classification analysis stuff
from torch import nn
from torch.nn import functional as F

# Force the execution of __init__.py if this file is executed directly.
import la  # noqa
from la.utils.cka import CKA
from la.utils.class_analysis import Classifier
from la.utils.utils import MyDatasetDict, add_tensor_column


from la.utils.relative_analysis import (
    compare_merged_original_qualitative,
    plot_space_grid,
    plot_pairwise_dist,
    plot_self_dist,
    self_sim_comparison,
    Reduction,
    reduce,
)

plt.style.use("dark_background")


pylogger = logging.getLogger(__name__)


def map_labels_to_global(data, num_tasks):
    for task_ind in range(1, num_tasks + 1):
        global_to_local_map = data["metadata"]["global_to_local_class_mappings"][f"task_{task_ind}"]
        local_to_global_map = {v: int(k) for k, v in global_to_local_map.items()}

        for mode in ["train", "test"]:
            data[f"task_{task_ind}_{mode}"] = data[f"task_{task_ind}_{mode}"].map(
                lambda row: {"y": local_to_global_map[row["y"].item()]},
                desc="Mapping labels back to global.",
            )


def run(cfg: DictConfig) -> str:
    """ """
    seed_index_everything(cfg)

    all_cka_results = {
        dataset_name: {model_name: {} for model_name in cfg.model_names} for dataset_name in cfg.dataset_names
    }
    all_class_results = {
        dataset_name: {model_name: {} for model_name in cfg.model_names} for dataset_name in cfg.dataset_names
    }

    check_runs_exist(cfg.configurations)

    for single_cfg in cfg.configurations:
        cka_results, class_results = single_configuration_experiment(cfg, single_cfg)

        all_cka_results[single_cfg.dataset_name][single_cfg.model_name][
            f"S{single_cfg.num_shared_classes}_N{single_cfg.num_novel_classes}"
        ] = cka_results

        all_class_results[single_cfg.dataset_name][single_cfg.model_name][
            f"S{single_cfg.num_shared_classes}_N{single_cfg.num_novel_classes}"
        ] = class_results

    with open(cfg.cka_results_path, "w+") as f:
        json.dump(all_cka_results, f)

    with open(cfg.class_results_path, "w+") as f:
        json.dump(all_class_results, f)


def check_runs_exist(configurations):
    for single_cfg in configurations:
        dataset_name, num_shared_classes, num_novel_classes, model_name = (
            single_cfg.dataset_name,
            single_cfg.num_shared_classes,
            single_cfg.num_novel_classes,
            single_cfg.model_name,
        )

        dataset_path = f"{PROJECT_ROOT}/data/{dataset_name}/part_shared_part_novel/S{num_shared_classes}_N{num_novel_classes}_{model_name}"

        assert os.path.exists(dataset_path), f"Path {dataset_path} does not exist."


def single_configuration_experiment(global_cfg, single_cfg):
    dataset_name, num_shared_classes, num_novel_classes, model_name = (
        single_cfg.dataset_name,
        single_cfg.num_shared_classes,
        single_cfg.num_novel_classes,
        single_cfg.model_name,
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

    prefix = f"S{num_shared_classes}_N{num_novel_classes}"
    compare_merged_original_qualitative(
        original_dataset, merged_dataset, has_coarse_label, global_cfg.plots_path, prefix, suffix="all_classes"
    )

    compare_merged_original_qualitative(
        original_dataset_nonshared,
        merged_dataset_nonshared,
        has_coarse_label,
        global_cfg.plots_path,
        prefix,
        suffix="nonshared_classes",
    )

    compare_merged_original_qualitative(
        original_dataset_shared,
        merged_dataset_shared,
        has_coarse_label,
        global_cfg.plots_path,
        prefix,
        suffix="shared_classes",
    )

    # # CKA analysis
    # cka = CKA(mode="linear", device="cuda")

    # cka_rel_abs = cka(merged_dataset["relative_embeddings"], merged_dataset["embedding"])

    # cka_tot = cka(merged_dataset["relative_embeddings"], original_dataset["relative_embeddings"])

    # cka_nonshared = cka(
    #     merged_dataset_nonshared["relative_embeddings"],
    #     original_dataset_nonshared["relative_embeddings"],
    # )

    # cka_shared = cka(
    #     merged_dataset_shared["relative_embeddings"],
    #     original_dataset_shared["relative_embeddings"],
    # )

    # cka_results = {
    #     "cka_rel_abs": cka_rel_abs.detach().item(),
    #     "cka_tot": cka_tot.detach().item(),
    #     "cka_shared": cka_shared.detach().item(),
    #     "cka_non_shared": cka_nonshared.detach().item(),
    # }

    # # classification analysis

    # class_exp = partial(
    #     run_classification_experiment,
    #     shared_classes,
    #     non_shared_classes,
    #     num_total_classes,
    #     global_cfg,
    #     num_anchors,
    # )

    # class_results_original_abs = class_exp(original_dataset, use_relatives=False)

    # class_results_original_rel = class_exp(original_dataset, use_relatives=True)

    # class_results_merged = class_exp(merged_dataset, use_relatives=True)

    # class_results = {
    #     "original_abs": class_results_original_abs,
    #     "original_rel": class_results_original_rel,
    #     "merged": class_results_merged,
    # }

    # pylogger.info(class_results_original_abs)
    # pylogger.info(class_results_original_rel)
    # pylogger.info(class_results_merged)

    cka_results, class_results = None, None

    return cka_results, class_results


def get_shared_samples_ids(data, num_tasks, shared_classes):
    """
    Get shared samples indices
    """

    # Add *shared* column, True for samples belonging to shared classes and False otherwise
    for task_ind in range(num_tasks + 1):
        for mode in ["train", "test"]:
            data[f"task_{task_ind}_{mode}"] = data[f"task_{task_ind}_{mode}"].map(
                lambda row: {"shared": row["y"].item() in shared_classes},
                desc="Adding shared column to samples",
            )

    shared_ids = []

    for task_ind in range(num_tasks + 1):
        all_ids = data[f"task_{task_ind}_train"]["id"]

        # get the indices of samples having shared to True
        task_shared_ids = all_ids[data[f"task_{task_ind}_train"]["shared"]].tolist()

        shared_ids.append(sorted(task_shared_ids))

    check_same_shared_ids(num_tasks, shared_ids)

    shared_ids = shared_ids[0]

    return shared_ids


def check_same_shared_ids(num_tasks, shared_ids):
    """
    Verify that each task has the same shared IDs
    """
    for task_i in range(num_tasks + 1):
        for task_j in range(task_i, num_tasks + 1):
            assert shared_ids[task_i] == shared_ids[task_j]


def add_anchor_column(data, num_tasks, anchor_ids):
    # only training samples can be anchors
    for task_ind in range(num_tasks + 1):
        data[f"task_{task_ind}_train"] = data[f"task_{task_ind}_train"].map(
            lambda row: {"anchor": row["id"].item() in anchor_ids},
            desc="Adding anchor column to train samples",
        )


def map_to_relative_spaces(data, num_tasks):
    for task_ind in range(0, num_tasks + 1):
        task_anchors = data[f"task_{task_ind}_train"]["embedding"][data[f"task_{task_ind}_train"]["anchor"]]
        norm_anchors = F.normalize(task_anchors, p=2, dim=-1)

        for mode in ["train", "test"]:
            task_embeddings = data[f"task_{task_ind}_{mode}"]["embedding"]

            abs_space = F.normalize(task_embeddings, p=2, dim=-1)

            rel_space = abs_space @ norm_anchors.T

            data[f"task_{task_ind}_{mode}"] = add_tensor_column(
                data[f"task_{task_ind}_{mode}"], "relative_embeddings", rel_space
            )


def set_torch_format(data, num_tasks, tensor_columns):
    for task_ind in range(0, num_tasks + 1):
        for mode in ["train", "test"]:
            data[f"task_{task_ind}_{mode}"].set_format(type="torch", columns=tensor_columns)


def subdivide_shared_and_non_shared(data, num_tasks):
    shared_samples = {"train": [], "test": []}
    disjoint_samples = {"train": [], "test": []}

    for task_ind in range(1, num_tasks + 1):
        for mode in ["train", "test"]:
            task_shared_samples = data[f"task_{task_ind}_{mode}"].filter(lambda row: row["shared"]).sort("id")

            task_novel_samples = data[f"task_{task_ind}_{mode}"].filter(lambda row: ~row["shared"])

            shared_samples[mode].append(task_shared_samples)
            disjoint_samples[mode].append(task_novel_samples)

    # check that novel samples have disjoint ids
    for task_ind in range(0, num_tasks):
        for mode in ["train", "test"]:
            task_novel_samples = disjoint_samples[mode][task_ind]

            task_novel_ids = task_novel_samples["id"]

            for task_j in range(task_ind + 1, num_tasks):
                other_task_novel_samples = disjoint_samples[mode][task_j]

                other_task_novel_ids = other_task_novel_samples["id"]

                common_novel_samples = set(task_novel_ids.tolist()).intersection(set(other_task_novel_ids.tolist()))
                assert len(common_novel_samples) == 0

    pylogger.info("Checked that all tasks have disjoint novel samples")

    return shared_samples, disjoint_samples


def merge_subspaces(shared_samples, novel_samples):
    """
    Average the shared samples and then concat the task-specific samples and the shared samples to go to the merged space

    # compute the mean of the shared_samples and put them back in the dataset
    # Extract the 'embedding' columns from each dataset
    """
    shared_rel_embeddings = [dataset["relative_embeddings"] for dataset in shared_samples]

    # Calculate the mean of the embeddings for each sample
    mean_embeddings = torch.mean(torch.stack(shared_rel_embeddings), dim=0)

    # Create a new dataset with the same features as the original datasets
    new_features = shared_samples[0].features.copy()

    # Replace the 'embedding' column in the new dataset with the mean embeddings
    new_data = {column: shared_samples[0][column] for column in new_features}
    new_data["relative_embeddings"] = mean_embeddings.tolist()

    # Create the new Hugging Face dataset
    shared_dataset = Dataset.from_dict(new_data, features=new_features)

    merged_dataset = concatenate_datasets([shared_dataset, novel_samples])

    merged_dataset = merged_dataset

    return merged_dataset


class Model(pytorch_lightning.LightningModule):
    def __init__(
        self,
        classifier: nn.Module,
        shared_classes: set,
        non_shared_classes: set,
        use_relatives: bool,
    ):
        super().__init__()
        self.classifier = classifier

        shared_classes = torch.Tensor(list(shared_classes)).long()
        non_shared_classes = torch.Tensor(list(non_shared_classes)).long()

        self.register_buffer("shared_classes", shared_classes)
        self.register_buffer("non_shared_classes", non_shared_classes)

        self.accuracy = torchmetrics.Accuracy()

        self.use_relatives = use_relatives
        self.embedding_key = "relative_embeddings" if self.use_relatives else "embedding"

    def forward(self, x):
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        x, y = batch[self.embedding_key], batch["y"]
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[self.embedding_key], batch["y"]
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss, on_step=True, prog_bar=True)

        val_acc = self.accuracy(y_hat, y)
        self.log("val_acc", val_acc, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch[self.embedding_key], batch["y"]
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("test_loss", loss, on_step=True)

        test_acc = self.accuracy(y_hat, y)
        self.log("test_acc", test_acc, on_step=True, on_epoch=True, prog_bar=True)

        # compute accuracy for shared classes
        shared_classes_mask = torch.isin(y, self.shared_classes)
        shared_classes_y = y[shared_classes_mask]

        y_hat = torch.argmax(y_hat, dim=1)
        shared_classes_y_hat = y_hat[shared_classes_mask]

        shared_classes_acc = torch.sum(shared_classes_y == shared_classes_y_hat) / len(shared_classes_y)
        self.log(
            "test_acc_shared_classes",
            shared_classes_acc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        # compute accuracy for non-shared classes
        non_shared_classes_mask = torch.isin(y, self.non_shared_classes)
        non_shared_classes_y = y[non_shared_classes_mask]
        non_shared_classes_y_hat = y_hat[non_shared_classes_mask]

        non_shared_classes_acc = torch.sum(non_shared_classes_y == non_shared_classes_y_hat) / len(non_shared_classes_y)
        self.log(
            "test_acc_non_shared_classes",
            non_shared_classes_acc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


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
    model = Model(
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
