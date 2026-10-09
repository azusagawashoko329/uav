"""DLBM v002: shared Top-k sequence encoder for exact-suffix adjacent gains."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FrameLayerEncoder(nn.Module):
    """Encode one frame's ordered per-layer Top-k means without hand-crafted deltas."""

    def __init__(self, num_layers=12, model_dim=32, depth=1, heads=4, dropout=0.0):
        super().__init__()
        if model_dim % heads:
            raise ValueError('model_dim must be divisible by heads')
        self.num_layers = int(num_layers)
        self.scalar_projection = nn.Sequential(
            nn.Linear(1, model_dim), nn.GELU(), nn.Linear(model_dim, model_dim),
        )
        self.layer_embedding = nn.Parameter(torch.empty(num_layers, model_dim))
        self.cls_token = nn.Parameter(torch.empty(1, 1, model_dim))
        block = nn.TransformerEncoderLayer(
            model_dim, heads, 4 * model_dim, dropout=dropout, activation='gelu',
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=depth)
        self.norm = nn.LayerNorm(model_dim)
        nn.init.normal_(self.layer_embedding, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, scores):
        if scores.ndim != 2 or scores.shape[1] != self.num_layers:
            raise ValueError(f'expected scores (B,{self.num_layers}), got {tuple(scores.shape)}')
        if not torch.isfinite(scores).all():
            raise FloatingPointError('non-finite DLBM input')
        tokens = self.scalar_projection(scores.unsqueeze(-1)) + self.layer_embedding.unsqueeze(0)
        cls = self.cls_token.expand(len(scores), -1, -1)
        return self.norm(self.encoder(torch.cat((cls, tokens), dim=1))[:, 0])


class SetAttentionPool(nn.Module):
    """Permutation-invariant learned pooling over scene frames."""

    def __init__(self, model_dim, heads):
        super().__init__()
        self.query = nn.Parameter(torch.empty(1, 1, model_dim))
        self.attention = nn.MultiheadAttention(model_dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(model_dim)
        nn.init.normal_(self.query, std=0.02)

    def forward(self, frames):
        if frames.ndim != 2 or len(frames) == 0:
            raise ValueError(f'expected non-empty frame set (B,D), got {tuple(frames.shape)}')
        query = self.query.expand(1, -1, -1)
        pooled, _ = self.attention(query, frames.unsqueeze(0), frames.unsqueeze(0),
                                   need_weights=False)
        return self.norm(pooled[:, 0]).squeeze(0)


class ExactSuffixGainPredictor(nn.Module):
    """Predict eleven signed opening gains from separate Normal/Pseudo Top-k frame sets."""

    def __init__(self, num_layers=12, model_dim=32, layer_depth=1, pair_depth=1, heads=4,
                 dropout=0.0):
        super().__init__()
        if num_layers < 2:
            raise ValueError('num_layers must be at least two')
        self.num_layers = int(num_layers)
        self.frame_encoder = FrameLayerEncoder(
            num_layers, model_dim, layer_depth, heads, dropout,
        )
        self.frame_pool = SetAttentionPool(model_dim, heads)
        self.branch_embedding = nn.Parameter(torch.empty(2, model_dim))
        pair_block = nn.TransformerEncoderLayer(
            model_dim, heads, 4 * model_dim, dropout=dropout, activation='gelu',
            batch_first=True, norm_first=True,
        )
        self.pair_encoder = nn.TransformerEncoder(pair_block, num_layers=pair_depth)
        self.head = nn.Sequential(
            nn.LayerNorm(2 * model_dim),
            nn.Linear(2 * model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, num_layers - 1),
        )
        nn.init.normal_(self.branch_embedding, std=0.02)

    def forward(self, normal_scores, pseudo_scores):
        if normal_scores.shape != pseudo_scores.shape:
            raise ValueError(
                f'normal/pseudo shape mismatch: {tuple(normal_scores.shape)} vs {tuple(pseudo_scores.shape)}')
        normal = self.frame_pool(self.frame_encoder(normal_scores)) + self.branch_embedding[0]
        pseudo = self.frame_pool(self.frame_encoder(pseudo_scores)) + self.branch_embedding[1]
        pair = self.pair_encoder(torch.stack((normal, pseudo), dim=0).unsqueeze(0)).squeeze(0)
        gains = self.head(pair.reshape(-1))
        if not torch.isfinite(gains).all():
            raise FloatingPointError('non-finite predicted gains')
        return gains


def exact_suffix_risks(normal_boundary_scores, pseudo_boundary_scores):
    """Training/readout-matched pairwise risk for each exact suffix boundary."""
    if normal_boundary_scores.shape != pseudo_boundary_scores.shape:
        raise ValueError('normal/pseudo boundary-score shape mismatch')
    if normal_boundary_scores.ndim != 2 or normal_boundary_scores.shape[1] < 2:
        raise ValueError(f'expected boundary scores (B,L), got {tuple(normal_boundary_scores.shape)}')
    return F.softplus(normal_boundary_scores - pseudo_boundary_scores).mean(dim=0)


def exact_suffix_adjacent_gains(normal_boundary_scores, pseudo_boundary_scores):
    risks = exact_suffix_risks(normal_boundary_scores, pseudo_boundary_scores)
    return risks[1:] - risks[:-1]

