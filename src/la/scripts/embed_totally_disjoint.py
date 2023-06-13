import logging
import os
from pathlib import Path
from typing import List, Optional
import torch.nn.functional as F
from pydoc import locate

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from datasets import disable_caching, Dataset
from nn_core.callbacks import NNTemplateCore
from nn_core.common import PROJECT_ROOT
from nn_core.common.utils import enforce_tags, seed_index_everything
from nn_core.model_logging import NNLogger
from nn_core.serialization import NNCheckpointIO, load_model
from omegaconf import DictConfig
from pytorch_lightning import Callback
from timm.data import resolve_data_config, create_transform, ToTensor
from torchmetrics import Accuracy
from torchvision.transforms import Compose
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.functional import cosine_similarity
import la  # noqa
from la.data.datamodule import MetaData
from la.pl_modules.efficient_net import MyEfficientNet
from la.pl_modules.pl_module import DataAugmentation
from la.utils.utils import (
    ToFloatRange,
    embed_task_samples,
    get_checkpoint_callback,
    build_callbacks,
    standard_normalization,
)
from la.utils.utils import scatter_mean

disable_caching()
pylogger = logging.getLogger(__name__)


def run(cfg: DictConfig) -> str:
    """Generic train loop.

    Args:
        cfg: run configuration, defined by Hydra in /conf

    Returns:
        the run directory inside the storage_dir used by the current experiment
    """
    seed_index_everything(cfg.train)

    pylogger.info(f"Running experiment on {cfg.nn.dataset_name}")

    cfg.core.tags = enforce_tags(cfg.core.get("tags", None))

    pylogger.info(f"Instantiating <{cfg.nn.data['_target_']}>")

    cfg.nn.data.data_path = cfg.nn.output_path
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.nn.data, _recursive_=False)

    num_tasks = datamodule.data["metadata"]["num_tasks"]

    for task_ind in range(num_tasks + 1):
        seed_index_everything(cfg.train)

        pylogger.info(f"Instantiating <{cfg.nn.model['_target_']}>")

        model = get_task_model(datamodule.data, task_ind)

        datamodule.task_ind = task_ind
        datamodule.transform_func = model.transform_func
        datamodule.setup()

        modes = ["train", "val", "test", "anchors"] if task_ind == 0 else ["train", "val", "anchors"]
        embedded_samples = embed_task_samples(datamodule, model, task_ind, modes=modes)

        for mode in modes:
            datamodule.data[f"task_{task_ind}_{mode}"] = embedded_samples[mode]

    label_to_task = {
        int(label): i
        for i in range(1, num_tasks + 1)
        for label in datamodule.data["metadata"]["global_to_local_class_mappings"][f"task_{i}"].keys()
    }

    num_closest_anchors = 10

    # assignments = compute_assignments_cross_space_anchor_heuristic(
    #     datamodule,
    #     model_ind=1,
    #     num_tasks=num_tasks,
    #     label_to_task=label_to_task,
    #     num_closest_anchors=num_closest_anchors,
    # )

    # assignments = compute_assignments_anchor_heuristic(datamodule, model_ind=1, num_tasks=num_tasks, num_closest_anchors=num_closest_anchors, label_to_task=label_to_task)

    # assignments = compute_assignments_cosine_trainset(datamodule, num_tasks, use_relatives=True)

    # assignments = compute_assignments_cosine_trainset(datamodule, num_tasks, use_relatives=False)

    assignments = compute_perfect_assignment(datamodule, num_tasks)

    task_embeds = compute_embeddings(datamodule, assignments, num_tasks)

    store_task_specific_test_embeds(datamodule, task_embeds, num_tasks)

    datamodule.data.save_to_disk(cfg.nn.embedding_path)


def get_task_model(data, task_ind):
    model_path = data["metadata"]["task_embedders"][f"task_{task_ind}"]["path"]
    model_class = locate(data["metadata"]["task_embedders"][f"task_{task_ind}"]["class"])

    model = load_model(model_class, checkpoint_path=Path(model_path + ".zip"))
    model.eval().cuda()
    return model


def compute_assignments_anchor_heuristic(datamodule, model_ind, num_tasks, num_closest_anchors, label_to_task):

    anchors = datamodule.data[f"task_{model_ind}_anchors"]
    norm_anchors = standard_normalization(anchors["embedding"])
    norm_anchors = F.normalize(norm_anchors, p=2, dim=-1).cuda()

    model = get_task_model(datamodule.data, model_ind)

    accuracy = Accuracy().to("cuda")

    batch_size = 10

    labels = anchors["y"].cuda()

    assignments = {f"task_{i}": None for i in range(1, num_tasks + 1)}

    for task_ind in range(1, num_tasks + 1):
        datamodule.task_ind = task_ind
        datamodule.batch_size.test = batch_size

        test_dataloader = datamodule.dataloader("test", only_task_specific_test=True)

        task_assignments = []
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Assigning test samples to tasks"):
                # (batch_size, C, H, W)
                x = batch["x"].to("cuda")

                embeds = model(x)["embeds"]

                norm_embeds = standard_normalization(embeds)
                norm_embeds = F.normalize(norm_embeds, p=2, dim=-1)

                relative = norm_embeds @ norm_anchors.T
                closest_anchors = torch.topk(relative, dim=-1, k=num_closest_anchors)

                closest_labels = labels[closest_anchors.indices]

                # map labels to task
                all_closest_tasks = []
                for sample_ind in range(batch_size):
                    closest_tasks = torch.tensor(
                        [label_to_task[label.item()] for label in closest_labels[sample_ind]]
                    ).cuda()
                    all_closest_tasks.append(closest_tasks)

                all_closest_tasks = torch.stack(all_closest_tasks)
                # assign to each sample a single task, the one with the majority of the closest anchors

                # for each task, sum the similarity scores of the closest anchors
                nearest_models = torch.zeros(batch_size, num_tasks).cuda()
                for task_j in range(1, num_tasks + 1):
                    task_inds = (all_closest_tasks == task_j).type_as(relative)
                    model_value = closest_anchors.values[task_inds.bool()].sum(dim=-1)
                    nearest_models[:, task_j - 1] = model_value

                nearest_models = torch.argmax(nearest_models, dim=-1)

                # (batch_size, 1)
                ground_truth_task_ind = torch.full_like(nearest_models, task_ind - 1).type_as(nearest_models)

                # (batch_size, embed_dim)
                accuracy(nearest_models, ground_truth_task_ind)

                task_assignments.append(nearest_models)

        # num_samples_task
        assignments[f"task_{task_ind}"] = (
            torch.cat(task_assignments, dim=0) + 1
        )  # +1 because the subtask indices start from 1

    pylogger.info(f"Accuracy of the task assignment using the similarity wrt the anchors: {accuracy.compute().item()}")

    return assignments


def compute_assignments_cross_space_anchor_heuristic(
    datamodule, model_ind, num_tasks, label_to_task, num_closest_anchors
):
    accuracy = Accuracy().to("cuda")

    batch_size = 10

    assignments = {f"task_{i}": None for i in range(1, num_tasks + 1)}

    for task_ind in range(1, num_tasks + 1):

        per_task_acc = Accuracy().to("cuda")

        datamodule.task_ind = task_ind
        datamodule.batch_size.test = batch_size

        test_dataloader = datamodule.dataloader("test", only_task_specific_test=True)

        task_assignments = []

        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Assigning test samples to tasks"):
                # (batch_size, C, H, W)
                x = batch["x"].to("cuda")

                similarity_wrt_tasks_anchors = []
                for model_ind in range(1, num_tasks + 1):
                    model = get_task_model(datamodule.data, model_ind)

                    embeds = model(x)["embeds"]

                    anchors = datamodule.data[f"task_{model_ind}_anchors"]
                    norm_anchors = standard_normalization(anchors["embedding"].cuda())

                    norm_anchors = F.normalize(norm_anchors, p=2, dim=-1)

                    anchor_labels = anchors["y"].cuda()

                    embeds = standard_normalization(embeds)
                    embeds = F.normalize(embeds, p=2, dim=-1)

                    # (batch_size, num_anchors)
                    rel_embeds = embeds @ norm_anchors.T

                    # (batch_size, num_classes)

                    logits = scatter_mean(index=anchor_labels, src=rel_embeds, dim=-1)

                    probs = F.softmax(logits, dim=-1)

                    # num_classes = 100
                    # class_ids = torch.arange(num_classes)
                    # model_task_ids = torch.tensor([label_to_task[label.item()] for label in class_ids])
                    # model_probs = torch.sum(probs[:, model_task_ids == model_ind], dim=-1)
                    # other_model_logits = torch.sum(logits[:, model_task_ids != model_ind], dim=-1)

                    # model_logits = model_logits - other_model_logits
                    # (num_anchors)
                    anchor_task_labels = torch.tensor([label_to_task[label.item()] for label in anchor_labels]).cuda()

                    # # only consider the anchors that have class in task model_ind
                    # # (batch_size, anchors_with_class_in_model_task)
                    similarities_wrt_task_anchors = rel_embeds[:, anchor_task_labels == model_ind]

                    # # only take the average of the most similar anchors
                    # # (batch_size, num_closest_anchors)
                    similarities_wrt_task_anchors = torch.topk(
                        similarities_wrt_task_anchors, dim=-1, k=num_closest_anchors
                    ).values

                    # # (batch_size)
                    sum_similarities_wrt_task_anchors = similarities_wrt_task_anchors.mean(dim=-1)

                    # compute the sum of the relative representation for the
                    # anchors that have class in task task_ind
                    similarity_wrt_tasks_anchors.append(sum_similarities_wrt_task_anchors)

                # (batch_sizue, num_tasks)
                similarity_wrt_tasks_anchors = torch.stack(similarity_wrt_tasks_anchors)

                # (batch_size)
                batch_assignments = similarity_wrt_tasks_anchors.argmax(dim=0)

                # (batch_size, 1)
                ground_truth_task_ind = torch.full_like(batch_assignments, task_ind - 1).type_as(batch_assignments)

                # (batch_size, embed_dim)
                accuracy(batch_assignments, ground_truth_task_ind)

                per_task_acc(batch_assignments, ground_truth_task_ind)

                task_assignments.append(batch_assignments)

        # num_samples_task
        task_assignments = torch.cat(task_assignments, dim=0)
        assignments[f"task_{task_ind}"] = task_assignments
        pylogger.info(
            f"Accuracy for task {task_ind} of the task assignment using the cross-space similarity wrt the anchors: {per_task_acc.compute().item()}"
        )

    pylogger.info(
        f"Accuracy of the task assignment using the cross-space similarity wrt the anchors: {accuracy.compute().item()}"
    )

    return assignments


def compute_assignments_cosine_trainset(datamodule, num_tasks, use_relatives, top_k_similarities):
    """ """
    accuracy = Accuracy().to("cuda")
    batch_size = 10

    assignments = {f"task_{i}": None for i in range(1, num_tasks + 1)}

    for task_ind in range(1, num_tasks + 1):

        datamodule.task_ind = task_ind
        datamodule.batch_size.test = batch_size

        test_dataloader = datamodule.dataloader("test", only_task_specific_test=True)
        task_assignments = []
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Assigning test samples to tasks"):

                model_similarities = []

                # (batch_size, C, H, W)
                x = batch["x"].to("cuda")

                for model_ind in range(1, num_tasks + 1):

                    train_embeddings = datamodule.data[f"task_{model_ind}_train"]["embedding"].cuda()

                    model = get_task_model(datamodule.data, model_ind)

                    embeds = model(x)["embeds"]

                    if use_relatives:
                        anchors = datamodule.data[f"task_{model_ind}_anchors"]["embedding"].cuda()

                        norm_anchors = standard_normalization(anchors)
                        norm_anchors = F.normalize(norm_anchors, p=2, dim=-1)

                        norm_train_embeddings = standard_normalization(train_embeddings)
                        norm_train_embeddings = F.normalize(norm_train_embeddings, p=2, dim=-1)

                        rel_train_embeddings = norm_train_embeddings @ norm_anchors.T

                        norm_embeds = standard_normalization(embeds)
                        norm_embeds = F.normalize(norm_embeds, p=2, dim=-1)

                        rel_embeds = norm_embeds @ norm_anchors.T

                        similarities = rel_embeds @ rel_train_embeddings

                    else:
                        # shape (batch_size, num_train_embeddings)
                        similarities = cosine_similarity(embeds.unsqueeze(1), train_embeddings.unsqueeze(0), dim=2)

                    # only consider top K most similar embeddings
                    similarities = torch.topk(similarities, dim=1, k=top_k_similarities).values
                    similarities = similarities.mean(dim=1)

                    # shape (batch_size, 1)
                    model_similarities.append(similarities)

                # (batch_size, num_models)
                model_similarities_tens = torch.stack(model_similarities, dim=1)

                # for each sample in the batch, contains the index of the model that is most similar
                # (batch_size, 1)
                batch_assignments = torch.argmax(model_similarities_tens, dim=1)

                # (batch_size, 1)
                ground_truth_task_ind = torch.full_like(batch_assignments, task_ind - 1).type_as(batch_assignments)

                # only keep the embeddings of the samples assigned to the current model

                # (batch_size, embed_dim)

                accuracy(batch_assignments, ground_truth_task_ind)

                task_assignments.append(batch_assignments)

        assignments[f"task_{task_ind}"] = torch.cat(task_assignments, dim=0)

    pylogger.info(
        f"Accuracy of the task assignment using the cosine wrt the training samples with relatives to {use_relatives}: {accuracy.compute().item()}"
    )

    return assignments


def compute_perfect_assignment(datamodule, num_tasks):
    assignments = {f"task_{i}": None for i in range(1, num_tasks + 1)}

    for task_ind in range(1, num_tasks + 1):

        num_task_samples = len(datamodule.data[f"task_{task_ind}_test"])
        ground_truth_task_ind = torch.full(size=(num_task_samples,), fill_value=task_ind)

        assignments[f"task_{task_ind}"] = ground_truth_task_ind

    return assignments


def compute_embeddings(datamodule, assignments, num_tasks):

    batch_size = 1
    task_embeds = {f"task_{i}": {"embedding": [], "y": [], "id": []} for i in range(1, num_tasks + 1)}

    for task_ind in range(1, num_tasks + 1):
        datamodule.task_ind = task_ind
        datamodule.batch_size.test = batch_size

        test_dataloader = datamodule.dataloader("test", only_task_specific_test=True)

        with torch.no_grad():
            for sample_ind, sample in enumerate(tqdm(test_dataloader, desc="Embedding test samples")):

                x = sample["x"].to("cuda")
                y = sample["y"][0].to("cuda")
                ids = sample["id"][0]
                # coarse_labels = batch["coarse_label"]

                model_ind = assignments[f"task_{task_ind}"][sample_ind]
                model = get_task_model(datamodule.data, model_ind)

                embeds = model(x)["embeds"][0]

                task_embeds[f"task_{model_ind}"]["embedding"].append(embeds)
                task_embeds[f"task_{model_ind}"]["y"].append(y)
                task_embeds[f"task_{model_ind}"]["id"].append(ids)
                # task_embeds[f'task_{task_ind}']['coarse_label'].append(coarse_labels)

    return task_embeds


def store_task_specific_test_embeds(datamodule, task_embeds, num_tasks):
    for task_ind in range(1, num_tasks + 1):
        task_test_embeds = task_embeds[f"task_{task_ind}"]

        y_list = task_test_embeds["y"]
        embeds = task_test_embeds["embedding"]
        ids = task_test_embeds["id"]

        data_dict = {"embedding": embeds, "y": y_list, "id": ids}

        dataset = Dataset.from_dict(data_dict)
        dataset.set_format(type="torch", columns=["embedding", "y"])

        datamodule.data[f"task_{task_ind}_test"] = dataset


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="run_totally_disjoint")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
