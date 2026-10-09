"""Episode order-transport DLBM on frozen predictive innovation features."""
import torch
import torch.nn as nn

from experiment_DLBM.episode_innovation_boundary import SeedAttentionPool


class OrderTransportDLBM(nn.Module):
    """Learn opening-induced frame-order transport relative to structural nulls."""

    def __init__(self, predictor_channels=16, topk=5, model_dim=32, heads=4):
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
        # No layer id enters this pool.  D/M/S share all parameters and differ
        # only through their observed predictive relations.
        self.layer_pool = SeedAttentionPool(model_dim, heads=heads, seeds=1)
        self.order_head = nn.Linear(model_dim, 1)
        self.pair_encoder = nn.Sequential(
            nn.LayerNorm(4 * model_dim), nn.Linear(4 * model_dim, 2 * model_dim),
            nn.GELU(), nn.Linear(2 * model_dim, model_dim),
        )
        self.pair_pool = SeedAttentionPool(model_dim, heads=heads, seeds=2)
        pair_dim = 2 * model_dim
        self.branch_projection = nn.Sequential(
            nn.LayerNorm(pair_dim), nn.Linear(pair_dim, model_dim), nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        self.branch_fusion = nn.Sequential(
            nn.LayerNorm(4 * model_dim), nn.Linear(4 * model_dim, 2 * model_dim),
            nn.GELU(), nn.Linear(2 * model_dim, model_dim), nn.LayerNorm(model_dim),
        )
        self.null_attention = nn.MultiheadAttention(
            model_dim, heads, batch_first=True,
        )
        self.gain_head = nn.Sequential(
            nn.LayerNorm(4 * model_dim), nn.Linear(4 * model_dim, 2 * model_dim),
            nn.GELU(), nn.Linear(2 * model_dim, 1),
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

    def encode_branch(self, opening, first, second):
        state, observed = opening['deep_state'], opening['observed']
        innovation = opening['innovation']
        if state.ndim != 5 or observed.shape != state.shape:
            raise ValueError('invalid predictive opening features')
        frames, added, channels, height, width = state.shape
        residual = self.innovation_encoder(
            innovation.reshape(frames * added, 1, height, width),
        )
        flat_state = state.reshape(frames * added, channels, height, width)
        flat_observed = observed.reshape(frames * added, channels, height, width)
        relation = self.relation(torch.cat((
            flat_state, flat_observed, residual, flat_state * flat_observed,
        ), dim=1))
        layer = self._topk_vector(
            relation, self.response, self.patch_projection,
        ).reshape(frames, added, self.model_dim)
        transition = self.layer_pool(layer)
        base = self._topk_vector(
            opening['base_state'], self.state_response, self.state_projection,
        )
        candidate = self._topk_vector(
            opening['candidate_state'], self.state_response, self.state_projection,
        )
        base_delta = base.index_select(0, first) - base.index_select(0, second)
        candidate_delta = (
            candidate.index_select(0, first) - candidate.index_select(0, second)
        )
        transition_delta = (
            transition.index_select(0, first) - transition.index_select(0, second)
        )
        pair = self.pair_encoder(torch.cat((
            base_delta, candidate_delta, torch.abs(candidate_delta - base_delta),
            transition_delta,
        ), dim=-1))
        episode = self.branch_projection(self.pair_pool(pair.unsqueeze(0)).squeeze(0))
        return {
            'episode': episode,
            'base_score': self.order_head(base).squeeze(-1),
            'candidate_score': self.order_head(candidate).squeeze(-1),
        }

    def fuse_branches(self, normal, pseudo):
        return self.branch_fusion(torch.cat((
            normal, pseudo, torch.abs(normal - pseudo), normal * pseudo,
        ), dim=-1))

    def gain_logit(self, real, nulls):
        if nulls.ndim != 2 or not len(nulls):
            raise ValueError('expected at least one structural-null token')
        query = real.view(1, 1, -1)
        values = nulls.unsqueeze(0)
        reference, _ = self.null_attention(query, values, values, need_weights=False)
        reference = reference[0, 0]
        return self.gain_head(torch.cat((
            real, reference, torch.abs(real - reference), real * reference,
        ), dim=-1)).squeeze(-1)

    @staticmethod
    def boundary_distribution(middle_logit, shallow_logit):
        """Ordered continuation distribution over D, MD and ALL."""
        middle = torch.sigmoid(middle_logit)
        shallow = torch.sigmoid(shallow_logit)
        probabilities = torch.stack((
            1.0 - middle, middle * (1.0 - shallow), middle * shallow,
        ))
        boundaries = (8, 4, 0)
        return boundaries[int(probabilities.argmax())], probabilities
