import itertools
from enum import auto
import logging
from pathlib import Path
import random
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from backports.strenum import StrEnum
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch import cosine_similarity
from torch.nn.functional import mse_loss, pairwise_distance
from torchmetrics.functional import pearson_corrcoef, spearman_corrcoef

CMAP = "jet"

pylogger = logging.getLogger(__name__)


def compare_merged_original_qualitative(
    original_dataset, merged_dataset, has_coarse_label, plots_path: Path, prefix="", suffix=""
):
    pylogger.info("Running the qualitative analysis")

    plots_path.mkdir(parents=True, exist_ok=True)

    merged_space = merged_dataset["relative_embeddings"]
    original_space = original_dataset["relative_embeddings"]

    original_space_y = original_dataset["y"]

    subsample_dim: int = 1000
    subsample_indices = random.sample(range(0, original_space.shape[0]), subsample_dim)

    subsample_original = original_space[subsample_indices]
    subsample_merged = merged_space[subsample_indices]
    subsample_labels = original_space_y[subsample_indices]

    sort_indices: torch.Tensor = subsample_labels.sort().indices

    subsample_original_sorted: torch.Tensor = subsample_original[sort_indices]
    subsample_merged_sorted: torch.Tensor = subsample_merged[sort_indices]
    subsample_labels_sorted: torch.Tensor = subsample_labels[sort_indices]

    fig = plot_pairwise_dist(space1=subsample_original_sorted, space2=subsample_merged_sorted, prefix="Relative")
    fig.savefig(plots_path / f"{prefix}pairwise_dist{suffix}.png")

    self_sim_comp = self_sim_comparison(
        space1=subsample_original_sorted, space2=subsample_merged_sorted, normalize=True
    )
    pylogger.info(self_sim_comp)

    fig = plot_self_dist(space1=subsample_original_sorted, space2=subsample_merged_sorted, prefix="Relative")
    fig.savefig(plots_path / f"{prefix}self_dist{suffix}.png")

    y_header = ["Original Space", "Aggregated Space"]
    x_header = [reduction.upper() for reduction in Reduction]

    spaces = [
        [
            *reduce(space1=subsample_original_sorted, space2=subsample_merged_sorted, reduction=reduction),
        ]
        for reduction in Reduction
    ]

    fig = plot_space_grid(x_header=x_header, y_header=y_header, spaces=spaces, c=subsample_labels_sorted)
    fig.savefig(plots_path / f"{prefix}space_grid{suffix}.png")

    if has_coarse_label:
        original_space_coarse_labels = original_dataset["coarse_label"]
        subsample_coarse_labels = original_space_coarse_labels[subsample_indices]
        subsample_coarse_labels_sorted: torch.Tensor = subsample_coarse_labels[sort_indices]
        fig = plot_space_grid(x_header=x_header, y_header=y_header, spaces=spaces, c=subsample_coarse_labels_sorted)
        fig.savefig(plots_path / f"{prefix}space_grid_coarse{suffix}.png")

    reductions = [Reduction.INDEPENDENT_PCA, Reduction.TSNE]
    for reduction in reductions:
        y_header = ["Original Space", "Aggregated Space"]
        x_header = reduction.upper()

        spaces = [
            *reduce(space1=subsample_original_sorted, space2=subsample_merged_sorted, reduction=reduction),
        ]

        fig = plot_spaces(x_header=x_header, y_header=y_header, spaces=spaces, c=subsample_labels_sorted)
        fig.savefig(plots_path / f"{prefix}{reduction.upper()}{suffix}.png")


class DistMethod(StrEnum):
    COSINE = auto()
    INNER = auto()
    MSE = auto()


class Reduction(StrEnum):
    INDEPENDENT_PCA = auto()
    SHARED_PCA = auto()
    TSNE = auto()
    # UMAP = auto()
    FIRST_DIMS = auto()


def self_sim_comparison(
    space1: torch.Tensor,
    space2: torch.Tensor,
    normalize: bool = False,
):
    """
    For each sample, compute the similarities wrt to all the other points in the same space,
    obtaining a vector for each sample and hence a matrix for each space.
    Compute the correlation of the matrices from the two spaces.
    Intuitively, if the self similarities are correlated, the two datasets are semantically similar.

    :param space1:
    :param space2:
    :param normalize:
    :return:
    """
    if normalize:
        space1 = F.normalize(space1, p=2, dim=-1)
        space2 = F.normalize(space2, p=2, dim=-1)

    self_sim1 = space1 @ space1.T
    self_sim2 = space2 @ space2.T

    spearman = spearman_corrcoef(self_sim1.T, self_sim2.T)
    pearson = pearson_corrcoef(self_sim1.T, self_sim2.T)
    cosine = cosine_similarity(self_sim1, self_sim2)

    return dict(
        spearman_mean=spearman.mean().item(),
        spearman_std=spearman.std().item(),
        pearson_mean=pearson.mean().item(),
        pearson_std=pearson.std().item(),
        cosine_mean=cosine.mean().item(),
        cosine_std=cosine.std().item(),
    )


def pairwise_dist(space1: torch.Tensor, space2: torch.Tensor, method: DistMethod):
    if method == DistMethod.COSINE:
        dists = cosine_similarity(space1, space2)
    elif method == DistMethod.INNER:
        dists = pairwise_distance(space1, space2, p=2)
    elif method == DistMethod.MSE:
        dists = mse_loss(space1, space2, reduction="none").mean(dim=1)
    else:
        raise NotImplementedError

    return dists


def all_dist(space1: torch.Tensor, space2: torch.Tensor, method: DistMethod):
    if method == DistMethod.COSINE:
        space1 = F.normalize(space1, p=2, dim=-1)
        space2 = F.normalize(space2, p=2, dim=-1)
        dists = space1 @ space2.T
    elif method == DistMethod.INNER:
        dists = space1 @ space2.T
    elif method == DistMethod.MSE:
        dists = ((space1[:, None, :] - space2[None, :, :]) ** 2).mean(dim=-1)
    else:
        raise NotImplementedError

    return dists


def plot_pairwise_dist(space1: torch.Tensor, space2: torch.Tensor, prefix: str):
    """

    :param space1:
    :param space2:
    :param prefix:
    :return:
    """
    fig, axs = plt.subplots(nrows=1, ncols=len(DistMethod), figsize=(20, 6), sharey=False)

    for dist_method, ax in zip(DistMethod, axs):
        dists = pairwise_dist(space1=space1, space2=space2, method=dist_method)
        ax.hist(dists, bins=42)
        ax.set_title(f"{prefix} pairwise similarities ({dist_method})")
        ax.axvline(dists.mean(), color="k", linestyle="dashed", linewidth=1)

        min_ylim, max_ylim = ax.get_ylim()
        min_xlim, max_xlim = ax.get_xlim()
        ax.text(dists.mean() + (max_xlim - min_xlim) / 20, max_ylim * 0.95, "Mean: {:.2f}".format(dists.mean()))

    plt.close()

    return fig


def plot_self_dist_varying_distances(space1: torch.Tensor, space2: torch.Tensor, prefix: str):
    # S[i, j] = distance between sample_i from space 1 and sample_j from space 2
    # s[j, i] = distance between sample_j from space 1 and sample_i from space 2
    fig, axs = plt.subplots(nrows=2, ncols=len(DistMethod), figsize=(20, 11), sharey=False)

    # unnormalized
    for dist_method, ax in zip(DistMethod, axs[0]):
        dists = all_dist(space1=space1, space2=space2, method=dist_method)
        ax.set_title(f"{prefix} self-similarities ({dist_method})")
        img = ax.imshow(dists, cmap=CMAP)
        plt.colorbar(img, ax=ax)

    # normalized
    for dist_method, ax in zip(DistMethod, axs[1]):
        dists = all_dist(
            space1=F.normalize(space1, p=2, dim=-1), space2=F.normalize(space2, p=2, dim=-1), method=dist_method
        )
        ax.set_title(f"L2({prefix}) self-similarities ({dist_method})")
        img = ax.imshow(dists, cmap=CMAP)
        plt.colorbar(img, ax=ax)

    plt.close()

    return fig


def plot_self_dist(space1: torch.Tensor, space2: torch.Tensor, prefix: str):
    # S[i, j] = distance between sample_i from space 1 and sample_j from space 2
    # s[j, i] = distance between sample_j from space 1 and sample_i from space 2

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 12), sharey=False)

    dists = all_dist(space1=space1, space2=space2, method=DistMethod.COSINE)
    ax.set_title(f"self-similarities")
    img = ax.imshow(dists, cmap=CMAP)
    plt.colorbar(img, ax=ax)

    plt.close()

    return fig


def plot_space_grid(
    x_header: Sequence[str], y_header: Sequence[str], spaces: Sequence[Sequence[np.ndarray]], c=None, cmap=CMAP
):
    """Plots a grid of scatter plots using matplotlib.

    Args:
        x_header: A sequence of strings for the x-axis labels.
        y_header: A sequence of strings for the y-axis labels.
        spaces: A sequence of sequences of tensors containing the data to be plotted.
        c: Optional. The colors of the plotted points.
        cmap: The colormap to use for the plotted points.
    Returns:
        The figure object representing the complete plot.
    """
    n_rows = len(spaces)
    n_cols = len(spaces[0])

    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols * 5, n_rows * 5))

    for x, row in zip(x_header, axs):
        row[0].set_ylabel(x, rotation=90, size="xx-large")

    for y, col in zip(y_header, axs[0]):
        col.set_title(y, size="xx-large")

    for i, j in itertools.product(range(n_rows), range(n_cols)):
        space = spaces[i][j]
        assert space.shape[1] == 2
        axs[i, j].scatter(x=space[:, 0], y=space[:, 1], c=c, cmap=cmap)

    plt.close()
    return fig


def plot_spaces(x_header: str, y_header: Sequence[str], spaces: Sequence[Sequence[np.ndarray]], c=None, cmap=CMAP):
    """

    Args:
        x_header: A sequence of strings for the x-axis labels.
        y_header: A sequence of strings for the y-axis labels.
        spaces: A sequence of sequences of tensors containing the data to be plotted.
        c: Optional. The colors of the plotted points.
        cmap: The colormap to use for the plotted points.
    Returns:
        The figure object representing the complete plot.
    """
    n_rows = 1
    n_cols = len(spaces)

    fig, axs = plt.subplots(nrows=1, ncols=n_cols, figsize=(n_cols * 5, n_rows * 5))

    axs[0].set_ylabel(x_header, rotation=90, size="xx-large")

    for y, col in zip(y_header, axs):
        col.set_title(y, size="xx-large")

    for j in range(n_cols):
        space = spaces[j]
        assert space.shape[1] == 2
        x = space[:, 0]
        y = space[:, 1]
        axs[j].scatter(x=x, y=y, c=c, cmap=cmap)

    plt.close()
    return fig


def reduce(space1: torch.Tensor, space2: torch.Tensor, reduction: Reduction, seed: int = 42, perplexity=30):
    if reduction == Reduction.INDEPENDENT_PCA:
        space1 = PCA(2, random_state=seed).fit_transform(space1)
        space2 = PCA(2, random_state=seed).fit_transform(space2)
    elif reduction == Reduction.SHARED_PCA:
        pca = PCA(2, random_state=seed)
        space1 = pca.fit_transform(space1)
        space2 = pca.transform(space2)
    elif reduction == Reduction.TSNE:
        space1 = TSNE(2, random_state=seed, learning_rate="auto", init="pca", perplexity=perplexity).fit_transform(
            space1
        )
        space2 = TSNE(2, random_state=seed, learning_rate="auto", init="pca", perplexity=perplexity).fit_transform(
            space2
        )
    elif reduction == Reduction.FIRST_DIMS:
        space1 = space1[:, [0, 1]]
        space2 = space2[:, [0, 1]]
    else:
        raise NotImplementedError

    return space1, space2
