"""DLBM v001 primitives: top-k sequence evaluator and counterfactual gain math.

This module deliberately stays independent of the old map-based DLBM.  Its observable is the
training-matched per-frame/per-layer top-k mean matrix ``(B, L)``.  A frozen evaluator supplies a
held-out normal/pseudo risk, while opening gain is measured through the change in the normal-only
distillation update direction.  No normal-minus-pseudo feature is constructed.
"""
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn as nn


class TopKSequenceEvaluator(nn.Module):
    """Map one frame's ordered layer-score sequence to a normal/pseudo logit."""

    def __init__(self, num_layers=12, model_dim=32, depth=1, heads=4, dropout=0.0):
        super().__init__()
        if num_layers < 2:
            raise ValueError('num_layers must be at least two')
        if model_dim % heads:
            raise ValueError('model_dim must be divisible by heads')
        self.num_layers = int(num_layers)
        self.model_dim = int(model_dim)
        self.scalar_projection = nn.Sequential(
            nn.Linear(1, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        self.layer_embedding = nn.Parameter(torch.empty(num_layers, model_dim))
        self.cls_token = nn.Parameter(torch.empty(1, 1, model_dim))
        block = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=heads, dim_feedforward=4 * model_dim,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=depth)
        self.output_norm = nn.LayerNorm(model_dim)
        self.classifier = nn.Linear(model_dim, 1)
        nn.init.normal_(self.layer_embedding, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)

    def encode(self, scores):
        if scores.ndim != 2 or scores.shape[1] != self.num_layers:
            raise ValueError(
                f'expected scores (B,{self.num_layers}), got {tuple(scores.shape)}')
        if not torch.isfinite(scores).all():
            raise FloatingPointError('non-finite top-k scores')
        tokens = self.scalar_projection(scores.unsqueeze(-1))
        tokens = tokens + self.layer_embedding.unsqueeze(0)
        cls = self.cls_token.expand(scores.shape[0], -1, -1)
        encoded = self.encoder(torch.cat((cls, tokens), dim=1))
        return self.output_norm(encoded[:, 0])

    def forward(self, scores):
        return self.classifier(self.encode(scores)).squeeze(-1)


def suffix_means(per_layer_losses):
    """Return losses for all contiguous suffixes ``[k, ..., L-1]``."""
    if per_layer_losses.ndim != 1 or len(per_layer_losses) < 2:
        raise ValueError(f'expected one loss per layer, got {tuple(per_layer_losses.shape)}')
    return torch.stack([per_layer_losses[k:].mean() for k in range(len(per_layer_losses))])


def adjacent_gains_from_risks(risks):
    """Gain of opening layer k: risk([k+1..]) - risk([k..]), for k=0..L-2."""
    if risks.ndim != 1 or len(risks) < 2:
        raise ValueError(f'expected one risk per boundary, got {tuple(risks.shape)}')
    return risks[1:] - risks[:-1]


@dataclass(frozen=True)
class BoundaryDecision:
    boundary: int
    cumulative_benefits: torch.Tensor


def boundary_from_adjacent_gains(gains):
    """Choose the suffix with maximum cumulative benefit relative to the L-1-only base.

    ``gains[k] = R[k+1] - R[k]``.  This is intentionally not a greedy stop rule: a later shallow
    bridge may compensate for locally negative opening steps.
    """
    if gains.ndim != 1 or len(gains) < 1:
        raise ValueError(f'expected adjacent gains vector, got {tuple(gains.shape)}')
    if not torch.isfinite(gains).all():
        raise FloatingPointError('non-finite adjacent gains')
    benefits = torch.zeros(len(gains) + 1, dtype=gains.dtype, device=gains.device)
    benefits[:-1] = torch.flip(torch.cumsum(torch.flip(gains, dims=(0,)), dim=0), dims=(0,))
    return BoundaryDecision(int(benefits.argmax().item()), benefits)


def detached_gradients(loss, parameters, retain_graph=False):
    """Differentiate a scalar and return detached gradients, preserving unused entries as None."""
    parameters = tuple(parameters)
    if loss.ndim != 0:
        raise ValueError(f'loss must be scalar, got {tuple(loss.shape)}')
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=retain_graph, allow_unused=True, create_graph=False,
    )
    return tuple(None if grad is None else grad.detach() for grad in gradients)


def gradient_dot(left: Sequence[torch.Tensor | None], right: Sequence[torch.Tensor | None]):
    if len(left) != len(right):
        raise ValueError(f'gradient length mismatch: {len(left)} vs {len(right)}')
    terms = [(a * b).sum() for a, b in zip(left, right) if a is not None and b is not None]
    if not terms:
        raise RuntimeError('no shared gradient-bearing parameters')
    return torch.stack(terms).sum()


def suffix_gradient_alignments(per_layer_losses, parameters, reference_gradients):
    """Compute ``<reference_grad, grad suffix_loss[k]>`` for every boundary k."""
    parameters = tuple(parameters)
    losses = suffix_means(per_layer_losses)
    alignments = []
    for k, loss in enumerate(losses):
        gradients = torch.autograd.grad(
            loss, parameters, retain_graph=k + 1 < len(losses), allow_unused=True,
            create_graph=False,
        )
        alignments.append(gradient_dot(reference_gradients, gradients).detach())
    return torch.stack(alignments)


def first_order_adjacent_gains(alignments, inner_lr):
    """First-order ``R_closed - R_open`` from suffix gradient alignments."""
    if alignments.ndim != 1 or len(alignments) < 2:
        raise ValueError(f'expected suffix alignments, got {tuple(alignments.shape)}')
    if inner_lr <= 0:
        raise ValueError('inner_lr must be positive')
    return float(inner_lr) * (alignments[:-1] - alignments[1:])


def video_balanced_mean(values, video_ids: Iterable[str]):
    """Equal-weight videos, then equal-weight frames within each video."""
    video_ids = list(video_ids)
    if values.ndim < 1 or len(values) != len(video_ids):
        raise ValueError('values/video_ids length mismatch')
    unique = sorted(set(video_ids))
    if not unique:
        raise ValueError('empty video_ids')
    by_video = []
    for video in unique:
        indices = [i for i, item in enumerate(video_ids) if item == video]
        by_video.append(values[indices].mean(dim=0))
    return torch.stack(by_video).mean(dim=0)

