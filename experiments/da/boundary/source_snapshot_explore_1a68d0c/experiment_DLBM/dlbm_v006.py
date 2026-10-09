"""DLBM v006 paired incremental relation network.

The module receives frozen-Probe patch discrepancy maps.  It never receives the
manual Gate statistics (rank flips, tail gaps, z scores, D/M/S groups, or their
thresholds).  For every adjacent opening ``b <- b+1`` it compares the candidate
layer's top-k patch tokens with the already selected deeper suffix through a
shared relation block, then aggregates the paired response over frames.
"""
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class BoundaryPolicy:
    boundary: int
    adjacent_logits: torch.Tensor
    cumulative_utilities: torch.Tensor


def policy_from_adjacent_logits(logits):
    """Choose the contiguous suffix with maximum cumulative open utility."""
    if logits.ndim != 1 or logits.numel() < 1:
        raise ValueError(f'expected adjacent logits vector, got {tuple(logits.shape)}')
    if not torch.isfinite(logits).all():
        raise FloatingPointError('non-finite adjacent logits')
    utilities = logits.new_zeros(logits.numel() + 1)
    utilities[:-1] = torch.flip(
        torch.cumsum(torch.flip(logits, dims=(0,)), dim=0), dims=(0,),
    )
    return BoundaryPolicy(
        boundary=int(utilities.argmax().item()),
        adjacent_logits=logits,
        cumulative_utilities=utilities,
    )


class PairedIncrementalRelationNetwork(nn.Module):
    """Learn scene-level adjacent opening values from raw top-k patch relations.

    Normal and pseudo branches share every encoder parameter.  Pseudo is compared
    internally as ``candidate versus base``; no hand-written Normal-Pseudo delta is
    formed.  A single value head sees the two learned branch summaries.
    """

    def __init__(self, num_layers=12, topk=5, model_dim=32, heads=4,
                 frame_depth=1, dropout=0.0):
        super().__init__()
        if num_layers < 2:
            raise ValueError('num_layers must be at least two')
        if topk < 1:
            raise ValueError('topk must be positive')
        if model_dim % heads:
            raise ValueError('model_dim must be divisible by heads')
        self.num_layers = int(num_layers)
        self.topk = int(topk)
        self.model_dim = int(model_dim)

        # Token features are distance, x/y patch coordinate, and within-top-k rank.
        self.token_encoder = nn.Sequential(
            nn.Linear(4, model_dim), nn.GELU(), nn.Linear(model_dim, model_dim),
        )
        self.layer_embedding = nn.Parameter(torch.empty(num_layers, model_dim))
        self.role_embedding = nn.Parameter(torch.empty(2, model_dim))
        self.cross_relation = nn.MultiheadAttention(
            model_dim, heads, dropout=dropout, batch_first=True,
        )
        self.relation_norm = nn.LayerNorm(model_dim)
        self.relation_ffn = nn.Sequential(
            nn.LayerNorm(model_dim), nn.Linear(model_dim, 2 * model_dim),
            nn.GELU(), nn.Linear(2 * model_dim, model_dim),
        )
        frame_block = nn.TransformerEncoderLayer(
            model_dim, heads, 4 * model_dim, dropout=dropout, activation='gelu',
            batch_first=True, norm_first=True,
        )
        self.frame_encoder = nn.TransformerEncoder(frame_block, num_layers=frame_depth)
        self.frame_query = nn.Parameter(torch.empty(1, 1, model_dim))
        self.frame_pool = nn.MultiheadAttention(
            model_dim, heads, dropout=dropout, batch_first=True,
        )
        self.value_head = nn.Sequential(
            nn.LayerNorm(2 * model_dim), nn.Linear(2 * model_dim, model_dim),
            nn.GELU(), nn.Linear(model_dim, 1),
        )
        nn.init.normal_(self.layer_embedding, std=0.02)
        nn.init.normal_(self.role_embedding, std=0.02)
        nn.init.normal_(self.frame_query, std=0.02)

    def _validate_maps(self, maps):
        if maps.ndim != 3 or maps.shape[1] != self.num_layers:
            raise ValueError(
                f'expected maps (frames,{self.num_layers},patches), got {tuple(maps.shape)}')
        if maps.shape[0] < 1 or maps.shape[2] < self.topk:
            raise ValueError('empty frame set or fewer patches than top-k')
        if not torch.isfinite(maps).all():
            raise FloatingPointError('non-finite discrepancy maps')
        side = int(round(maps.shape[2] ** 0.5))
        if side * side != maps.shape[2]:
            raise ValueError(f'patch count must form a square grid, got {maps.shape[2]}')
        return side

    def _topk_tokens(self, maps):
        side = self._validate_maps(maps)
        values, indices = maps.topk(self.topk, dim=-1)
        denom = float(max(1, side - 1))
        x = (indices % side).to(values.dtype) / denom
        y = torch.div(indices, side, rounding_mode='floor').to(values.dtype) / denom
        rank = torch.arange(self.topk, device=maps.device, dtype=values.dtype)
        rank = rank / float(max(1, self.topk - 1))
        rank = rank.view(1, 1, self.topk).expand_as(values)
        raw = torch.stack((values, x, y, rank), dim=-1)
        return self.token_encoder(raw)

    def _branch_relation(self, tokens, boundary, candidate_permutation=None):
        # tokens: frames x layers x topk x dim
        frames = tokens.shape[0]
        candidate = tokens[:, boundary]
        if candidate_permutation is not None:
            candidate = candidate.index_select(0, candidate_permutation)
        candidate = (
            candidate + self.layer_embedding[boundary].view(1, 1, -1)
            + self.role_embedding[0].view(1, 1, -1)
        )
        base = tokens[:, boundary + 1:].clone()
        layer = self.layer_embedding[boundary + 1:].view(
            1, self.num_layers - boundary - 1, 1, self.model_dim,
        )
        base = base + layer + self.role_embedding[1].view(1, 1, 1, -1)
        base = base.reshape(frames, -1, self.model_dim)
        related, _ = self.cross_relation(candidate, base, base, need_weights=False)
        related = self.relation_norm(candidate + related)
        related = related + self.relation_ffn(related)
        frame_tokens = related.mean(dim=1)
        frame_tokens = self.frame_encoder(frame_tokens.unsqueeze(0))
        query = self.frame_query.expand(1, -1, -1)
        pooled, _ = self.frame_pool(query, frame_tokens, frame_tokens, need_weights=False)
        return pooled[0, 0]

    def forward(self, normal_maps, pseudo_maps, candidate_layers=None,
                candidate_permutation=None):
        if normal_maps.shape != pseudo_maps.shape:
            raise ValueError('normal/pseudo discrepancy-map shape mismatch')
        normal_tokens = self._topk_tokens(normal_maps)
        pseudo_tokens = self._topk_tokens(pseudo_maps)
        if candidate_layers is None:
            candidate_layers = range(self.num_layers - 1)
        candidate_layers = tuple(int(value) for value in candidate_layers)
        if not candidate_layers or any(value < 0 or value >= self.num_layers - 1
                                       for value in candidate_layers):
            raise ValueError(f'invalid candidate layers: {candidate_layers}')
        if candidate_permutation is not None:
            candidate_permutation = torch.as_tensor(
                candidate_permutation, device=normal_maps.device, dtype=torch.long,
            )
            if candidate_permutation.shape != (normal_maps.shape[0],):
                raise ValueError('candidate permutation must contain one index per frame')

        logits = []
        for boundary in candidate_layers:
            normal_relation = self._branch_relation(
                normal_tokens, boundary, candidate_permutation,
            )
            pseudo_relation = self._branch_relation(
                pseudo_tokens, boundary, candidate_permutation,
            )
            # Concatenation leaves the interaction learnable; no fixed branch delta.
            logits.append(self.value_head(
                torch.cat((normal_relation, pseudo_relation), dim=-1),
            ).squeeze(-1))
        result = torch.stack(logits)
        if not torch.isfinite(result).all():
            raise FloatingPointError('non-finite DLBM logits')
        return result

