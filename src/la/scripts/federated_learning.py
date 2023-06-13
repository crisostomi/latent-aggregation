import json
import logging
import os
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
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything
from sklearn import metrics
from sklearn.cluster import KMeans
import pytorch_lightning as pl

# classification analysis stuff
from torch import nn
from torch.nn import functional as F

from nn_core.common import PROJECT_ROOT
from nn_core.common.utils import seed_index_everything

# Force the execution of __init__.py if this file is executed directly.
import la  # noqa
from la.data.my_dataset_dict import MyDatasetDict
from la.scripts.analyze_part_shared_part_novel import (
    add_anchor_column,
    map_to_relative_spaces,
    merge_subspaces,
    set_torch_format,
    subdivide_shared_and_non_shared,
)
from la.scripts.embed_totally_disjoint import get_task_model
from la.utils.cka import CKA
from la.utils.class_analysis import Classifier, Model, PartSharedPartNovelModel
from la.utils.rel_utils import project_to_relative
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
from la.utils.utils import add_tensor_column, save_dict_to_file
import datasets
from datasets import set_caching_enabled

set_caching_enabled(True)

# plt.style.use("dark_background")

pylogger = logging.getLogger(__name__)
pylogger.info(f"Using cache: {datasets.is_caching_enabled()}")


def run(cfg):
    seed_everything(cfg.seed_index)

    dataset_name, num_shared_classes, num_novel_classes, model_name = (
        cfg.dataset_name,
        cfg.num_shared_classes,
        cfg.num_novel_classes,
        "_".join(cfg.model_name),
    )

    has_coarse_label = cfg.has_coarse_label[dataset_name]

    pylogger.info(
        f"Running experiment on {dataset_name} embedded with {model_name} with {num_shared_classes} shared classes and {num_novel_classes} novel classes for task."
    )

    dataset_path = f"{PROJECT_ROOT}/data/{dataset_name}/part_shared_part_novel/S{num_shared_classes}_N{num_novel_classes}_{model_name}"

    data: MyDatasetDict = MyDatasetDict.load_from_disk(dataset_dict_path=dataset_path)

    num_total_classes = cfg.num_total_classes[dataset_name]
    non_shared_classes = set(range(num_total_classes)).difference(data["metadata"]["shared_classes"])
    num_tasks = data["metadata"]["num_tasks"]
    shared_classes = set(data["metadata"]["shared_classes"])

    for task_ind in range(num_tasks + 1):
        data[f"task_{task_ind}_train"] = concatenate_datasets(
            [data[f"task_{task_ind}_train"], data[f"task_{task_ind}_val"]]
        ).sort("id")

    tensor_columns = ["embedding", "y", "id"]
    data.set_format(type="torch", columns=tensor_columns)

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

    # calculate the shift for each task
    shared_rel_embeddings = [dataset["relative_embeddings"] for dataset in shared_samples["train"].values()]
    mean_embeddings = torch.mean(torch.stack(shared_rel_embeddings), dim=0)
    shifts = {
        task: torch.mean(mean_embeddings - dataset["relative_embeddings"], dim=0)
        for task, dataset in shared_samples["train"].items()
    }

    for mode in ["train", "test"]:
        for task_ind in range(1, num_tasks + 1):
            disjoint_samples[mode][f"task_{task_ind}"] = disjoint_samples[mode][f"task_{task_ind}"].map(
                lambda row: {"relative_embeddings": row["relative_embeddings"] + shifts[f"task_{task_ind}"]},
            )

    novel_samples = {
        "train": concatenate_datasets(list(disjoint_samples["train"].values())),
        "test": concatenate_datasets(list(disjoint_samples["test"].values())),
    }

    # Merge the subspaces to obtain a single unified space
    merged_datasets = {"train": [], "test": []}

    for mode in ["train", "test"]:
        merged_datasets[mode] = merge_subspaces(shared_samples[mode], novel_samples[mode], method="mean")
        merged_datasets[mode].set_format(type="torch", columns=["relative_embeddings", "y", "id", "shared"])

    classifier = Classifier(
        input_dim=num_anchors,
        classifier_embed_dim=cfg.classifier_embed_dim,
        num_classes=num_total_classes,
    )

    model = Model(
        classifier=classifier,
        use_relatives=True,
    )

    dataloader_func = partial(
        torch.utils.data.DataLoader,
        batch_size=512,
        num_workers=8,
    )

    trainer_func = partial(Trainer, gpus=1, max_epochs=1, logger=False, enable_progress_bar=True)
    trainer = trainer_func(callbacks=[pytorch_lightning.callbacks.EarlyStopping(monitor="val_loss", patience=3)])

    split_dataset = merged_datasets["train"].train_test_split(test_size=0.3, seed=cfg.seed_index)
    train_dataset = split_dataset["train"]
    val_test_dataset = split_dataset["test"]

    split_val_test = val_test_dataset.train_test_split(test_size=0.5, seed=cfg.seed_index)
    val_dataset = split_val_test["train"]
    test_dataset = split_val_test["test"]

    train_dataloader = dataloader_func(train_dataset, shuffle=True)
    val_dataloader = dataloader_func(val_dataset, shuffle=False)
    test_dataloader = dataloader_func(test_dataset, shuffle=False)

    trainer.fit(model, train_dataloader, val_dataloader)

    results = trainer.test(model, test_dataloader)[0]

    # task-specific models
    whole_test_dataset = data["task_0_test"]
    whole_test_dataset.set_format(type="torch", columns=["x", "y"])
    test_dataloader = dataloader_func(whole_test_dataset, shuffle=False)

    for task_ind in range(1, num_tasks + 1):
        data.set_format(type="torch", columns=["x", "y"])
        anchors = data[f"task_{task_ind}_train"]["x"][data[f"task_{task_ind}_train"]["anchor"]]
        task_model = get_task_model(data, task_ind)

        body = task_model.model
        head = model.classifier

        all_task_classes = set(
            int(key) for key in data["metadata"]["global_to_local_class_mappings"][f"task_{task_ind}"].keys()
        )
        task_unseen_classes = set(data["metadata"]["all_classes_ids"]).difference(all_task_classes)

        head_replaced_model = HeadReplacedModel(
            body,
            head,
            anchors,
            shared_classes,
            non_shared_classes,
            shifts[f"task_{task_ind}"],
            torch.tensor(list(task_unseen_classes)),
        )

        task_trainer = trainer_func()
        task_trainer.test(head_replaced_model, test_dataloader)

        pylogger.info(f"Task {task_ind} results: {task_trainer.callback_metrics}")


class HeadReplacedModel(pl.LightningModule):
    def __init__(
        self, body, head, anchors, shared_classes, non_shared_classes, shift, task_unseen_classes, *args, **kwargs
    ) -> None:
        super().__init__()

        self.body = body
        self.head = head
        self.accuracy = torchmetrics.Accuracy()

        shared_classes = torch.Tensor(list(shared_classes)).long()
        non_shared_classes = torch.Tensor(list(non_shared_classes)).long()

        self.register_buffer("shared_classes", shared_classes)
        self.register_buffer("non_shared_classes", non_shared_classes)
        self.register_buffer("shift", shift)

        self.register_buffer("anchors", anchors)
        self.register_buffer("task_unseen_classes", task_unseen_classes)

    def forward(self, x):

        sample_embeds = self.body.backbone(x)
        sample_embeds = sample_embeds.reshape(sample_embeds.size(0), -1)

        sample_embeds = self.body.proj(sample_embeds)

        anchor_embeds = self.body.backbone(self.anchors)
        anchor_embeds = anchor_embeds.reshape(anchor_embeds.size(0), -1)
        anchor_embeds = self.body.proj(anchor_embeds)

        sample_rel_embeds = project_to_relative(sample_embeds, anchor_embeds, normalize=False)

        sample_rel_embeds = sample_rel_embeds + self.shift
        logits = self.head(sample_rel_embeds)

        return logits

    def test_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]

        y_hat = self(x)

        loss = F.cross_entropy(y_hat, y)
        self.log("test_loss", loss, on_epoch=True)

        test_acc = self.accuracy(y_hat, y)
        self.log("test_acc", test_acc, on_step=False, on_epoch=True, prog_bar=True)

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

        # compute accuracy for task-specific classes
        task_unseen_classes_mask = torch.isin(y, self.task_unseen_classes)
        task_unseen_classes_y = y[task_unseen_classes_mask]
        task_unseen_classes_y_hat = y_hat[task_unseen_classes_mask]

        task_unseen_classes_acc = torch.sum(task_unseen_classes_y == task_unseen_classes_y_hat) / len(
            task_unseen_classes_y_hat
        )
        self.log(
            "test_acc_task_unseen_classes",
            task_unseen_classes_acc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        return loss


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="federated_learning")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
