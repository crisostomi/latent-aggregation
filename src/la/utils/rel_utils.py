import torch
import torch.nn.functional as F

from la.utils.utils import standard_normalization


def project_to_relative(sample_embeds, anchor_embeds, normalize=False):

    if normalize:
        anchor_embeds = standard_normalization(anchor_embeds)
        sample_embeds = standard_normalization(sample_embeds)

    norm_anchors = F.normalize(anchor_embeds, p=2, dim=-1)

    sample_embeds = F.normalize(sample_embeds, p=2, dim=-1)

    rel_sample_embeds = sample_embeds @ norm_anchors.T

    return rel_sample_embeds
