"""DLBM that learns opening value as held-out order predictability."""
import torch
import torch.nn as nn

from experiment_DLBM.episode_innovation_boundary import SeedAttentionPool


class PredictiveOrderDLBM(nn.Module):
    """Predict candidate frame order before and after observing an opening.

    The base and opened predictions share the same order head.  The opened path
    differs only by a residual computed from spatial opening tokens, so a lower
    opened NLL has a direct interpretation: the added layers explain order
    structure that was absent from the current suffix.
    """

    def __init__(self, predictor_channels=16, topk=5, model_dim=32, heads=4):
        super().__init__()
        channels = int(predictor_channels)
        self.topk = int(topk)
        self.model_dim = int(model_dim)
        self.relation = nn.Sequential(
            nn.Conv2d(4 * channels + 1, 2 * channels, 3, padding=1), nn.GELU(),
            nn.Conv2d(2 * channels, channels, 3, padding=1), nn.GELU(),
        )
        self.response = nn.Conv2d(channels, 1, 1)
        self.open_projection = nn.Sequential(
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
        # Attention is invariant to the number and identity of added layers.
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

    def _topk_vector(self, spatial, response, projection):
        values, indices = response(spatial).flatten(1).topk(self.topk, dim=-1)
        selected = torch.gather(
            spatial.flatten(2), 2,
            indices.unsqueeze(1).expand(-1, spatial.shape[1], -1),
        ).transpose(1, 2)
        return projection(
            torch.cat((selected, values.unsqueeze(-1)), dim=-1).flatten(1),
        )

    def frame_tokens(self, opening):
        state, observed, innovation = (
            opening['deep_state'], opening['observed'], opening['innovation'],
        )
        if state.ndim != 5 or observed.shape != state.shape:
            raise ValueError('invalid predictive opening features')
        frames, added, channels, height, width = state.shape
        flat_state = state.reshape(frames * added, channels, height, width)
        flat_observed = observed.reshape(frames * added, channels, height, width)
        flat_innovation = innovation.reshape(frames * added, 1, height, width)
        relation = self.relation(torch.cat((
            flat_state, flat_observed, flat_state - flat_observed,
            flat_state * flat_observed, flat_innovation,
        ), dim=1))
        per_layer = self._topk_vector(
            relation, self.response, self.open_projection,
        ).reshape(frames, added, self.model_dim)
        opened = self.layer_pool(per_layer)
        base = self._topk_vector(
            opening['base_state'], self.state_response, self.state_projection,
        )
        return base, opened

    def pair_logits(self, opening, first, second):
        base, opened = self.frame_tokens(opening)
        base_delta = base.index_select(0, first) - base.index_select(0, second)
        open_delta = opened.index_select(0, first) - opened.index_select(0, second)
        base_pair = self.base_pair(torch.cat((
            base_delta, torch.abs(base_delta),
            base.index_select(0, first) * base.index_select(0, second),
        ), dim=-1))
        residual = self.open_residual(torch.cat((
            base_pair, open_delta, torch.abs(open_delta), base_pair * open_delta,
        ), dim=-1))
        return (
            self.order_head(base_pair).squeeze(-1),
            self.order_head(base_pair + residual).squeeze(-1),
        )

    @staticmethod
    def boundary_distribution(middle_gain, shallow_gain):
        """Cumulative utilities for D, D+M and D+M+S."""
        utilities = torch.stack((
            middle_gain.new_zeros(()), middle_gain, middle_gain + shallow_gain,
        ))
        probabilities = torch.softmax(utilities, dim=0)
        boundaries = (8, 4, 0)
        return boundaries[int(utilities.argmax().item())], probabilities, utilities
