"""Scene-episode boundary head on frozen predictive innovation maps.

The head compares three incremental scales with one shared network:
L11 -> D, D -> MD, and MD -> ALL.  It receives full spatial predictive
relations and learns episode evidence from real same-frame openings versus
frame-permuted openings that preserve every per-layer marginal distribution.
"""
import torch
import torch.nn as nn


SCALE_TRANSITIONS = {
    'D': ((11,), tuple(range(8, 12))),
    'M': (tuple(range(8, 12)), tuple(range(4, 12))),
    'S': (tuple(range(4, 12)), tuple(range(12))),
}
SCALE_BOUNDARIES = {'D': 8, 'M': 4, 'S': 0}


class SeedAttentionPool(nn.Module):
    """Permutation-invariant learned pooling with multiple summary seeds."""

    def __init__(self, model_dim, heads=4, seeds=2):
        super().__init__()
        if model_dim % heads:
            raise ValueError('model_dim must be divisible by heads')
        self.seeds = nn.Parameter(torch.empty(1, seeds, model_dim))
        self.attention = nn.MultiheadAttention(model_dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(model_dim)
        nn.init.normal_(self.seeds, std=0.02)

    def forward(self, tokens):
        if tokens.ndim != 3 or tokens.shape[1] < 2:
            raise ValueError('expected episode tokens (episodes,frames,dim)')
        query = self.seeds.expand(len(tokens), -1, -1)
        pooled, _ = self.attention(query, tokens, tokens, need_weights=False)
        return self.norm(pooled).flatten(1)


class EpisodeInnovationBoundary(nn.Module):
    """Encode predictive openings and score their scene-level coherence."""

    def __init__(self, predictor_channels=16, topk=5, model_dim=32, heads=4,
                 num_layers=12, pool_seeds=2):
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
        self.layer_embedding = nn.Parameter(torch.empty(num_layers, model_dim))
        # Attention pooling across added layers avoids a scale-dependent recurrent
        # accumulation when D contains three additions and M/S contain four.
        self.layer_pool = SeedAttentionPool(model_dim, heads=heads, seeds=1)
        self.state_response = nn.Conv2d(channels, 1, 1)
        self.state_projection = nn.Sequential(
            nn.LayerNorm(topk * (channels + 1)),
            nn.Linear(topk * (channels + 1), model_dim), nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        self.frame_projection = nn.Sequential(
            nn.LayerNorm(5 * model_dim), nn.Linear(5 * model_dim, 2 * model_dim),
            nn.GELU(), nn.Linear(2 * model_dim, model_dim), nn.LayerNorm(model_dim),
        )
        frame_layer = nn.TransformerEncoderLayer(
            model_dim, heads, 2 * model_dim, dropout=0.0, activation='gelu',
            batch_first=True, norm_first=True,
        )
        self.frame_context = nn.TransformerEncoder(frame_layer, num_layers=1)
        self.episode_pool = SeedAttentionPool(
            model_dim, heads=heads, seeds=pool_seeds,
        )
        episode_dim = pool_seeds * model_dim
        self.branch_score = nn.Sequential(
            nn.LayerNorm(episode_dim), nn.Linear(episode_dim, model_dim),
            nn.GELU(), nn.Linear(model_dim, 1),
        )
        self.pair_score = nn.Sequential(
            nn.LayerNorm(4 * episode_dim), nn.Linear(4 * episode_dim, 2 * model_dim),
            nn.GELU(), nn.Linear(2 * model_dim, 1),
        )
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

    def frame_tokens(self, opening):
        state, observed = opening['deep_state'], opening['observed']
        innovation = opening['innovation']
        if state.ndim != 5 or observed.shape != state.shape:
            raise ValueError('invalid predictive opening feature shape')
        frames, added, channels, height, width = state.shape
        residual = self.innovation_encoder(
            innovation.reshape(frames * added, 1, height, width),
        )
        flat_state = state.reshape(frames * added, channels, height, width)
        flat_observed = observed.reshape(frames * added, channels, height, width)
        relation = self.relation(torch.cat((
            flat_state, flat_observed, residual, flat_state * flat_observed,
        ), dim=1))
        layer_tokens = self._topk_vector(
            relation, self.response, self.patch_projection,
        ).reshape(frames, added, self.model_dim)
        layer_ids = torch.as_tensor(
            opening['added_layers'], dtype=torch.long, device=state.device,
        )
        layer_tokens = layer_tokens + self.layer_embedding.index_select(
            0, layer_ids,
        ).view(1, added, -1)
        transition = self.layer_pool(layer_tokens)
        base = self._topk_vector(
            opening['base_state'], self.state_response, self.state_projection,
        )
        candidate = self._topk_vector(
            opening['candidate_state'], self.state_response, self.state_projection,
        )
        return self.frame_projection(torch.cat((
            base, candidate, transition, torch.abs(candidate - base),
            candidate * base,
        ), dim=-1))

    def encode_episode(self, opening):
        frames = self.frame_tokens(opening).unsqueeze(0)
        return self.episode_pool(self.frame_context(frames)).squeeze(0)

    def scores(self, normal_opening, pseudo_opening):
        normal = self.encode_episode(normal_opening)
        pseudo = self.encode_episode(pseudo_opening)
        return {
            'normal': self.branch_score(normal).squeeze(-1),
            'pseudo': self.branch_score(pseudo).squeeze(-1),
            'joint': self.pair_score(torch.cat((
                normal, pseudo, torch.abs(normal - pseudo), normal * pseudo,
            ), dim=-1)).squeeze(-1),
        }


def permute_added_layers(maps, base_layers, candidate_layers, permutation):
    """Destroy same-frame opening coherence while preserving layer marginals."""
    if maps.ndim != 3:
        raise ValueError('expected discrepancy maps (frames,layers,patches)')
    base, candidate = set(base_layers), set(candidate_layers)
    if not base < candidate:
        raise ValueError('base must be a strict subset of candidate')
    permutation = torch.as_tensor(permutation, dtype=torch.long, device=maps.device)
    if permutation.shape != (len(maps),):
        raise ValueError('permutation must contain one entry per frame')
    output = maps.clone()
    added = torch.as_tensor(sorted(candidate - base), dtype=torch.long, device=maps.device)
    output[:, added] = maps.index_select(0, permutation).index_select(1, added)
    return output


def boundary_from_scale_margins(margins):
    if set(margins) != set(SCALE_TRANSITIONS):
        raise ValueError(f'expected margins for {tuple(SCALE_TRANSITIONS)}')
    values = torch.stack([margins[name] for name in SCALE_TRANSITIONS])
    if values.ndim != 1 or not torch.isfinite(values).all():
        raise ValueError('scale margins must be finite scalars')
    name = tuple(SCALE_TRANSITIONS)[int(values.argmax())]
    return SCALE_BOUNDARIES[name], values
