"""Raw-patch dual-axis DLBM for continuous opening transport."""
import math

import torch
import torch.nn as nn

from experiment_DLBM.episode_innovation_boundary import SeedAttentionPool
from experiment_DLBM.predictive_order_dlbm import PredictiveOrderDLBM


class RawTransportDualAxisDLBM(nn.Module):
    """Learn normal order transport and pseudo transport nuisance from raw maps.

    Top-k selects the same hard patch support used by Probe training, but all
    subsequent patch, layer, pair, and branch relations are learned.
    """

    def __init__(self, topk=5, channels=16, model_dim=32, heads=4,
                 num_layers=12, video_consistency=False):
        super().__init__()
        self.topk = int(topk)
        self.model_dim = int(model_dim)
        self.patch_encoder = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1), nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.GELU(),
        )
        self.patch_projection = nn.Sequential(
            nn.LayerNorm(self.topk * (channels + 1)),
            nn.Linear(self.topk * (channels + 1), model_dim), nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        self.layer_embedding = nn.Parameter(torch.empty(num_layers, model_dim))
        self.layer_pool = SeedAttentionPool(model_dim, heads=heads, seeds=1)
        self.base_pair = nn.Sequential(
            nn.LayerNorm(3 * model_dim), nn.Linear(3 * model_dim, model_dim),
            nn.GELU(), nn.Linear(model_dim, model_dim),
        )
        self.open_residual = nn.Sequential(
            nn.LayerNorm(4 * model_dim), nn.Linear(4 * model_dim, 2 * model_dim),
            nn.GELU(), nn.Linear(2 * model_dim, model_dim),
        )
        self.order_head = nn.Sequential(
            nn.LayerNorm(model_dim), nn.Linear(model_dim, model_dim), nn.GELU(),
            nn.Linear(model_dim, 1),
        )
        self.transport_projection = nn.Sequential(
            nn.LayerNorm(4 * model_dim), nn.Linear(4 * model_dim, 2 * model_dim),
            nn.GELU(), nn.Linear(2 * model_dim, model_dim),
        )
        self.branch_critic = nn.Sequential(
            nn.LayerNorm(model_dim), nn.Linear(model_dim, model_dim), nn.GELU(),
            nn.Linear(model_dim, 1),
        )
        self.axis_log_variance = nn.Parameter(torch.zeros(2))
        self.consistency_log_variance = (
            nn.Parameter(torch.zeros(())) if video_consistency else None
        )
        nn.init.normal_(self.layer_embedding, std=0.02)

    def layer_tokens(self, maps, layers):
        if maps.ndim != 3:
            raise ValueError('expected raw maps (frames,layers,patches)')
        layer_ids = torch.as_tensor(layers, dtype=torch.long, device=maps.device)
        selected = maps.index_select(1, layer_ids)
        frames, count, patches = selected.shape
        side = math.isqrt(int(patches))
        if side * side != patches or count < 2:
            raise ValueError('layers need a square patch grid and at least two members')
        flat = selected.reshape(frames * count, patches)
        values, indices = flat.topk(self.topk, dim=-1)
        encoded = self.patch_encoder(
            flat.reshape(frames * count, 1, side, side),
        ).flatten(2)
        gathered = torch.gather(
            encoded, 2,
            indices.unsqueeze(1).expand(-1, encoded.shape[1], -1),
        ).transpose(1, 2)
        tokens = self.patch_projection(torch.cat((
            gathered, values.unsqueeze(-1),
        ), dim=-1).flatten(1)).reshape(frames, count, self.model_dim)
        return tokens + self.layer_embedding.index_select(
            0, layer_ids,
        ).view(1, count, -1)

    def group_tokens(self, maps, layers):
        return self.layer_pool(self.layer_tokens(maps, layers))

    @staticmethod
    def _delta(tokens, first, second):
        return tokens.index_select(0, first) - tokens.index_select(0, second)

    def order_logits(self, maps, base_layers, candidate_layers, first, second):
        base = self.group_tokens(maps, base_layers)
        candidate = self.group_tokens(maps, candidate_layers)
        base_delta = self._delta(base, first, second)
        candidate_delta = self._delta(candidate, first, second)
        base_pair = self.base_pair(torch.cat((
            base_delta, torch.abs(base_delta),
            base.index_select(0, first) * base.index_select(0, second),
        ), dim=-1))
        residual = self.open_residual(torch.cat((
            base_pair, candidate_delta, torch.abs(candidate_delta),
            base_pair * candidate_delta,
        ), dim=-1))
        return (
            self.order_head(base_pair).squeeze(-1),
            self.order_head(base_pair + residual).squeeze(-1),
        )

    def transport_tokens(self, maps, base_layers, candidate_layers):
        base = self.group_tokens(maps, base_layers)
        candidate = self.group_tokens(maps, candidate_layers)
        return self.transport_projection(torch.cat((
            base, candidate, torch.abs(candidate - base), base * candidate,
        ), dim=-1))

    def critic_logits(self, maps, base_layers, candidate_layers):
        return self.branch_critic(self.transport_tokens(
            maps, base_layers, candidate_layers,
        )).squeeze(-1)

    def multitask_loss(self, order_loss, critic_loss, consistency_loss=None):
        losses = torch.stack((order_loss, critic_loss))
        precision = torch.exp(-self.axis_log_variance)
        weighted = precision * losses + self.axis_log_variance
        if consistency_loss is None:
            return weighted.mean()
        if self.consistency_log_variance is None:
            raise ValueError('video consistency was not enabled')
        consistency = (
            torch.exp(-self.consistency_log_variance) * consistency_loss
            + self.consistency_log_variance
        )
        return torch.cat((weighted, consistency.reshape(1))).mean()

    def axis_precision(self):
        return torch.exp(-self.axis_log_variance)

    def dual_boundary_distribution(self, middle_axes, shallow_axes):
        precision = self.axis_precision().detach()
        middle = precision[0] * middle_axes[0] - precision[1] * middle_axes[1]
        shallow = precision[0] * shallow_axes[0] - precision[1] * shallow_axes[1]
        boundary, probabilities, utilities = PredictiveOrderDLBM.boundary_distribution(
            middle, shallow,
        )
        return boundary, probabilities, utilities, torch.stack((middle, shallow))
