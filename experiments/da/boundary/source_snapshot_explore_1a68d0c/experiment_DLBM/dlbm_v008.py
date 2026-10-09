"""DLBM v008: neural estimator of the manual Gate's opening-order signal."""
import torch
import torch.nn as nn


class NeuralOpeningOrder(nn.Module):
    """Encode raw top-k patch tokens and predict group order changes between frames."""

    def __init__(self, num_layers=12, topk=5, model_dim=32, heads=4):
        super().__init__()
        if num_layers < 2 or topk < 1 or model_dim % heads:
            raise ValueError('invalid layer/top-k/model dimensions')
        self.num_layers = int(num_layers)
        self.topk = int(topk)
        self.model_dim = int(model_dim)
        self.token_encoder = nn.Sequential(
            nn.Linear(4, model_dim), nn.GELU(), nn.Linear(model_dim, model_dim),
        )
        self.layer_embedding = nn.Parameter(torch.empty(num_layers, model_dim))
        self.patch_weight = nn.Sequential(
            nn.LayerNorm(model_dim), nn.Linear(model_dim, 1),
        )
        layer = nn.TransformerEncoderLayer(
            model_dim, heads, 2 * model_dim, dropout=0.0,
            activation='gelu', batch_first=True, norm_first=True,
        )
        self.layer_context = nn.TransformerEncoder(layer, num_layers=1)
        self.layer_weight = nn.Sequential(
            nn.LayerNorm(model_dim), nn.Linear(model_dim, 1),
        )
        self.group_projection = nn.Sequential(
            nn.LayerNorm(model_dim), nn.Linear(model_dim, model_dim), nn.GELU(),
            nn.Linear(model_dim, model_dim), nn.LayerNorm(model_dim),
        )
        # The shared head learns the two score axes without receiving a scene statistic.
        self.order_head = nn.Linear(model_dim, 1)
        self.flip_head = nn.Sequential(
            nn.LayerNorm(4 * model_dim), nn.Linear(4 * model_dim, 2 * model_dim),
            nn.GELU(), nn.Linear(2 * model_dim, 1),
        )
        nn.init.normal_(self.layer_embedding, std=0.02)

    def topk_tokens(self, maps):
        if maps.ndim != 3 or maps.shape[1] != self.num_layers:
            raise ValueError(
                f'expected maps (frames,{self.num_layers},patches), got {tuple(maps.shape)}')
        if maps.shape[0] < 2 or maps.shape[-1] < self.topk:
            raise ValueError('at least two frames and top-k patches are required')
        if not torch.isfinite(maps).all():
            raise FloatingPointError('non-finite discrepancy maps')
        side = int(round(maps.shape[-1] ** 0.5))
        if side * side != maps.shape[-1]:
            raise ValueError('patch count must form a square grid')
        values, indices = maps.topk(self.topk, dim=-1)
        denom = float(max(side - 1, 1))
        x = (indices % side).to(values.dtype) / denom
        y = torch.div(indices, side, rounding_mode='floor').to(values.dtype) / denom
        rank = torch.arange(self.topk, device=maps.device, dtype=values.dtype)
        rank = rank.view(1, 1, -1).expand_as(values) / float(max(self.topk - 1, 1))
        return self.token_encoder(torch.stack((values, x, y, rank), dim=-1))

    def encode_group_from_tokens(self, tokens, layers):
        layers = tuple(int(layer) for layer in layers)
        if not layers or len(set(layers)) != len(layers):
            raise ValueError('group layers must be non-empty and unique')
        if min(layers) < 0 or max(layers) >= self.num_layers:
            raise ValueError('group layer outside valid range')
        index = torch.as_tensor(layers, device=tokens.device, dtype=torch.long)
        selected = tokens.index_select(1, index)
        selected = selected + self.layer_embedding.index_select(0, index).view(
            1, len(layers), 1, self.model_dim,
        )
        patch_weight = torch.softmax(self.patch_weight(selected).squeeze(-1), dim=-1)
        layer_tokens = (patch_weight.unsqueeze(-1) * selected).sum(dim=2)
        layer_tokens = self.layer_context(layer_tokens)
        layer_weight = torch.softmax(self.layer_weight(layer_tokens).squeeze(-1), dim=-1)
        group = (layer_weight.unsqueeze(-1) * layer_tokens).sum(dim=1)
        return self.group_projection(group)

    def encode_transition(self, maps, base_layers, candidate_layers):
        base = set(int(x) for x in base_layers)
        candidate = set(int(x) for x in candidate_layers)
        if not base < candidate:
            raise ValueError('base must be a strict subset of candidate')
        tokens = self.topk_tokens(maps)
        return (
            self.encode_group_from_tokens(tokens, sorted(base)),
            self.encode_group_from_tokens(tokens, sorted(candidate)),
        )

    def order_scores(self, base_embedding, candidate_embedding):
        return (
            self.order_head(base_embedding).squeeze(-1),
            self.order_head(candidate_embedding).squeeze(-1),
        )

    def flip_logits(self, base_embedding, candidate_embedding, first, second):
        db = base_embedding.index_select(0, first) - base_embedding.index_select(0, second)
        dc = candidate_embedding.index_select(0, first) - candidate_embedding.index_select(0, second)
        features = torch.cat((db, dc, torch.abs(db - dc), db * dc), dim=-1)
        return self.flip_head(features).squeeze(-1)


def cumulative_coarse_policy(middle_gain, shallow_gain):
    """Sequential opening utility for D, D+M, and D+M+S."""
    values = torch.stack((middle_gain.new_zeros(()), middle_gain,
                          middle_gain + shallow_gain))
    boundaries = (8, 4, 0)
    return boundaries[int(values.argmax().item())], values
