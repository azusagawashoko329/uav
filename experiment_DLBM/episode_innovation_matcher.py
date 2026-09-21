"""Same-frame cross-layer retrieval on frozen predictive innovations."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiment_DLBM.episode_innovation_boundary import (
    SCALE_BOUNDARIES, SCALE_TRANSITIONS, SeedAttentionPool,
)


class EpisodeInnovationMatcher(nn.Module):
    """Learn whether an opening contains identifiable information about its base frame.

    All scales share every parameter.  The only target is the diagonal identity of
    a base/opening similarity matrix; no boundary, Gate statistic, or score delta is
    supplied to the network.
    """

    def __init__(self, predictor_channels=16, topk=5, model_dim=32, heads=4,
                 num_layers=12):
        super().__init__()
        channels = int(predictor_channels)
        self.topk = int(topk)
        self.model_dim = int(model_dim)
        self.innovation_encoder = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1), nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.GELU(),
        )
        self.relation = nn.Sequential(
            nn.Conv2d(4 * channels, 2 * channels, 3, padding=1), nn.GELU(),
            nn.Conv2d(2 * channels, channels, 3, padding=1), nn.GELU(),
        )
        self.response = nn.Conv2d(channels, 1, 1)
        self.patch_projection = nn.Sequential(
            nn.LayerNorm(topk * (channels + 1)),
            nn.Linear(topk * (channels + 1), model_dim), nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        self.state_response = nn.Conv2d(channels, 1, 1)
        self.state_projection = nn.Sequential(
            nn.LayerNorm(topk * (channels + 1)),
            nn.Linear(topk * (channels + 1), model_dim), nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        self.layer_embedding = nn.Parameter(torch.empty(num_layers, model_dim))
        self.layer_pool = SeedAttentionPool(model_dim, heads=heads, seeds=1)
        self.base_projection = nn.Sequential(
            nn.LayerNorm(model_dim), nn.Linear(model_dim, model_dim), nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        self.opening_projection = nn.Sequential(
            nn.LayerNorm(2 * model_dim), nn.Linear(2 * model_dim, 2 * model_dim),
            nn.GELU(), nn.Linear(2 * model_dim, model_dim),
        )
        # One global temperature is shared by D/M/S and cannot encode a boundary.
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        nn.init.normal_(self.layer_embedding, std=0.02)

    def _topk_vector(self, spatial, response, projection):
        values, indices = response(spatial).flatten(1).topk(self.topk, dim=-1)
        selected = torch.gather(
            spatial.flatten(2), 2,
            indices.unsqueeze(1).expand(-1, spatial.shape[1], -1),
        ).transpose(1, 2)
        return projection(
            torch.cat((selected, values.unsqueeze(-1)), dim=-1).flatten(1),
        )

    def encode(self, opening):
        state, observed = opening['deep_state'], opening['observed']
        innovation = opening['innovation']
        if state.ndim != 5 or observed.shape != state.shape:
            raise ValueError('invalid predictive opening shape')
        frames, added, channels, height, width = state.shape
        residual = self.innovation_encoder(
            innovation.reshape(frames * added, 1, height, width),
        )
        flat_state = state.reshape(frames * added, channels, height, width)
        flat_observed = observed.reshape(frames * added, channels, height, width)
        relation = self.relation(torch.cat((
            flat_state, flat_observed, residual, flat_state * flat_observed,
        ), dim=1))
        layers = self._topk_vector(
            relation, self.response, self.patch_projection,
        ).reshape(frames, added, self.model_dim)
        layer_ids = torch.as_tensor(
            opening['added_layers'], device=state.device, dtype=torch.long,
        )
        layers = layers + self.layer_embedding.index_select(0, layer_ids).view(
            1, added, -1,
        )
        transition = self.layer_pool(layers)
        base = self._topk_vector(
            opening['base_state'], self.state_response, self.state_projection,
        )
        candidate = self._topk_vector(
            opening['candidate_state'], self.state_response, self.state_projection,
        )
        base = F.normalize(self.base_projection(base), dim=-1)
        opening_token = F.normalize(
            self.opening_projection(torch.cat((candidate, transition), dim=-1)), dim=-1,
        )
        return base, opening_token

    def logits(self, opening):
        base, candidate = self.encode(opening)
        scale = self.logit_scale.clamp(max=math.log(100.0)).exp()
        return scale * base @ candidate.transpose(0, 1)

    def cross_opening_logits(self, first_opening, second_opening):
        """Retrieve the same frame across two pseudo views using opening tokens only."""
        _, first = self.encode(first_opening)
        _, second = self.encode(second_opening)
        scale = self.logit_scale.clamp(max=math.log(100.0)).exp()
        return scale * first @ second.transpose(0, 1)


def symmetric_retrieval_loss(logits):
    if logits.ndim != 2 or logits.shape[0] != logits.shape[1] or len(logits) < 2:
        raise ValueError('retrieval logits must be a square matrix with at least two frames')
    target = torch.arange(len(logits), device=logits.device)
    return 0.5 * (
        F.cross_entropy(logits, target) + F.cross_entropy(logits.transpose(0, 1), target)
    )


@torch.no_grad()
def retrieval_metrics(logits):
    target = torch.arange(len(logits), device=logits.device)
    metrics = []
    for matrix in (logits, logits.transpose(0, 1)):
        order = matrix.argsort(dim=1, descending=True)
        rank = torch.nonzero(order == target.unsqueeze(1), as_tuple=False)[:, 1] + 1
        metrics.append((
            (rank == 1).float().mean(), rank.float().reciprocal().mean(),
        ))
    loss = symmetric_retrieval_loss(logits)
    return {
        'loss': float(loss),
        'top1': float(torch.stack([value[0] for value in metrics]).mean()),
        'mrr': float(torch.stack([value[1] for value in metrics]).mean()),
        'normalized_information': float((math.log(len(logits)) - loss) / math.log(len(logits))),
    }


def boundary_from_retrieval(evidence):
    if set(evidence) != set(SCALE_TRANSITIONS):
        raise ValueError('missing D/M/S retrieval evidence')
    values = torch.as_tensor([evidence[name] for name in SCALE_TRANSITIONS])
    if not torch.isfinite(values).all():
        raise ValueError('retrieval evidence must be finite')
    winner = tuple(SCALE_TRANSITIONS)[int(values.argmax())]
    return SCALE_BOUNDARIES[winner], values
