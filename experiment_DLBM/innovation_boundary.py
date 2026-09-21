"""Innovation boundary head for the frozen cross-layer predictive representation."""
import torch
import torch.nn as nn


class InnovationOpeningHead(nn.Module):
    def __init__(self, predictor_channels=16, topk=5, model_dim=32, num_layers=12):
        super().__init__()
        self.topk = int(topk)
        self.model_dim = int(model_dim)
        channels = int(predictor_channels)
        self.innovation_encoder = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1), nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.GELU(),
        )
        self.relation = nn.Sequential(
            nn.Conv2d(4 * channels, 2 * channels, 3, padding=1), nn.GELU(),
            nn.Conv2d(2 * channels, channels, 3, padding=1), nn.GELU(),
        )
        self.response = nn.Conv2d(channels, 1, 1)
        # Top-k features are kept separately and concatenated in response order.
        self.patch_projection = nn.Sequential(
            nn.LayerNorm(topk * (channels + 1)),
            nn.Linear(topk * (channels + 1), model_dim), nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        self.layer_embedding = nn.Parameter(torch.empty(num_layers, model_dim))
        self.layer_sequence = nn.GRU(model_dim, model_dim, batch_first=True)
        self.group_response = nn.Conv2d(channels, 1, 1)
        self.group_projection = nn.Sequential(
            nn.LayerNorm(topk * (channels + 1)),
            nn.Linear(topk * (channels + 1), model_dim), nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        self.order_head = nn.Linear(model_dim, 1)
        self.flip_head = nn.Sequential(
            nn.LayerNorm(5 * model_dim), nn.Linear(5 * model_dim, 2 * model_dim),
            nn.GELU(), nn.Linear(2 * model_dim, 1),
        )
        nn.init.normal_(self.layer_embedding, std=0.02)

    def _topk_vector(self, spatial, response_head, projection):
        response = response_head(spatial).flatten(1)
        values, indices = response.topk(self.topk, dim=-1)
        channels = spatial.shape[1]
        features = spatial.flatten(2)
        selected = torch.gather(
            features, 2, indices.unsqueeze(1).expand(-1, channels, -1),
        ).transpose(1, 2)
        return projection(
            torch.cat((selected, values.unsqueeze(-1)), dim=-1).flatten(1),
        )

    def encode(self, opening):
        state = opening['deep_state']
        observed = opening['observed']
        innovation = opening['innovation']
        if state.ndim != 5 or observed.shape != state.shape:
            raise ValueError('expected aligned (frames,added,channels,height,width) features')
        frames, added, channels, height, width = state.shape
        residual = self.innovation_encoder(innovation.reshape(frames * added, 1, height, width))
        state = state.reshape(frames * added, channels, height, width)
        observed = observed.reshape(frames * added, channels, height, width)
        relation = self.relation(torch.cat((state, observed, residual, state * observed), dim=1))
        response = self.response(relation).flatten(1)
        if response.shape[1] < self.topk:
            raise ValueError('not enough spatial patches for top-k')
        values, indices = response.topk(self.topk, dim=-1)
        features = relation.flatten(2)
        selected = torch.gather(
            features, 2, indices.unsqueeze(1).expand(-1, channels, -1),
        ).transpose(1, 2)
        tokens = torch.cat((selected, values.unsqueeze(-1)), dim=-1).flatten(1)
        tokens = self.patch_projection(tokens).reshape(frames, added, self.model_dim)
        layers = torch.as_tensor(
            opening['added_layers'], dtype=torch.long, device=tokens.device,
        )
        tokens = tokens + self.layer_embedding.index_select(0, layers).view(1, added, -1)
        _, hidden = self.layer_sequence(tokens)
        base = self._topk_vector(
            opening['base_state'], self.group_response, self.group_projection,
        )
        candidate = self._topk_vector(
            opening['candidate_state'], self.group_response, self.group_projection,
        )
        return {'base': base, 'candidate': candidate, 'transition': hidden[-1]}

    def order_scores(self, embedding):
        return (self.order_head(embedding['base']).squeeze(-1),
                self.order_head(embedding['candidate']).squeeze(-1))

    def flip_logits(self, embedding, first, second):
        base_first = embedding['base'].index_select(0, first)
        base_second = embedding['base'].index_select(0, second)
        candidate_first = embedding['candidate'].index_select(0, first)
        candidate_second = embedding['candidate'].index_select(0, second)
        base_delta = base_first - base_second
        candidate_delta = candidate_first - candidate_second
        transition_delta = (
            embedding['transition'].index_select(0, first)
            - embedding['transition'].index_select(0, second)
        )
        features = torch.cat((
            base_delta, candidate_delta, torch.abs(base_delta - candidate_delta),
            base_delta * candidate_delta, transition_delta,
        ), dim=-1)
        return self.flip_head(features).squeeze(-1)
