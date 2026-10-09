"""DLBM v005: scene-wise differentiable discrete suffix search.

There is deliberately no hand-crafted feature encoder.  A scene owns one categorical distribution
over legal contiguous suffix boundaries.  A hard Gumbel sample selects a real suffix in the forward
pass, while the straight-through gradient lets a held-out objective update the boundary logits.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.boundary_supernet import suffix_code_from_all_layer_ocbe
from model.loss import per_layer_topk_dist


class ExactBoundarySearchSupernet(nn.Module):
    """Evaluate a small list of exact suffixes through shared all-layer Probe modules."""

    def __init__(self, ocbe, student, boundaries=(0, 4, 8, 11), topk=5):
        super().__init__()
        boundaries = tuple(int(value) for value in boundaries)
        if not boundaries or tuple(sorted(set(boundaries))) != boundaries:
            raise ValueError('boundaries must be unique and sorted')
        if boundaries[0] < 0 or boundaries[-1] >= ocbe.n_taps:
            raise ValueError(f'boundary outside [0,{ocbe.n_taps - 1}]')
        if ocbe.n_taps != student.n_taps or student.tap_align != 'absolute':
            raise ValueError('search supernet requires matching all-layer OCBE/absolute Student')
        self.ocbe = ocbe
        self.student = student
        self.boundaries = boundaries
        self.topk = int(topk)

    def forward(self, taps):
        if len(taps) != self.ocbe.n_taps:
            raise ValueError(f'expected {self.ocbe.n_taps} taps, got {len(taps)}')
        batch = taps[0].shape[0]
        codes = torch.cat([
            suffix_code_from_all_layer_ocbe(self.ocbe, taps, boundary)
            for boundary in self.boundaries
        ], dim=0)
        reconstructions = self.student(codes)
        scores = []
        for candidate, boundary in enumerate(self.boundaries):
            begin, end = candidate * batch, (candidate + 1) * batch
            layer_scores = per_layer_topk_dist(
                taps[boundary:],
                [value[begin:end] for value in reconstructions[boundary:]],
                k=self.topk,
            )
            scores.append(layer_scores.mean(dim=1))
        result = torch.stack(scores, dim=1)
        if not torch.isfinite(result).all():
            raise FloatingPointError('non-finite exact-boundary scores')
        return result


class SceneBoundaryDistribution(nn.Module):
    """The complete per-scene DLBM: categorical logits over ordered legal boundaries."""

    def __init__(self, boundaries=(0, 4, 8, 11)):
        super().__init__()
        if len(boundaries) < 2:
            raise ValueError('at least two candidate boundaries are required')
        self.boundaries = tuple(int(value) for value in boundaries)
        self.logits = nn.Parameter(torch.zeros(len(self.boundaries)))

    def sample(self, temperature=1.0):
        if temperature <= 0:
            raise ValueError('temperature must be positive')
        return F.gumbel_softmax(self.logits, tau=temperature, hard=True, dim=0)

    def probabilities(self):
        return self.logits.softmax(dim=0)

    def boundary(self):
        return self.boundaries[int(self.logits.argmax().item())]


def area_rank_loss(first_scores, second_scores, first_areas, second_areas):
    """Rank two pseudo views only by their observed CutPaste area; no Normal-Pseudo delta."""
    if first_scores.shape != second_scores.shape or first_scores.shape != first_areas.shape:
        raise ValueError('score/area shape mismatch')
    if first_areas.shape != second_areas.shape:
        raise ValueError('area shape mismatch')
    direction = torch.where(second_areas >= first_areas, 1.0, -1.0)
    return F.softplus(-direction * (second_scores - first_scores)).mean()


def boundary_policy(logits, boundaries):
    probabilities = logits.softmax(dim=0)
    index = int(probabilities.argmax().item())
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    entropy = entropy / torch.log(probabilities.new_tensor(float(len(probabilities))))
    return {
        'boundary': int(boundaries[index]),
        'selected_taps': list(range(int(boundaries[index]), 12)),
        'probabilities': {str(boundary): float(probabilities[i].detach().cpu())
                          for i, boundary in enumerate(boundaries)},
        'logits': {str(boundary): float(logits[i].detach().cpu())
                   for i, boundary in enumerate(boundaries)},
        'confidence': float(probabilities[index].detach().cpu()),
        'normalized_entropy': float(entropy.detach().cpu()),
    }

