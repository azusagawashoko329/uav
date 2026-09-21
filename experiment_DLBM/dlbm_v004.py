"""Learned patch-localization evaluator for DLBM opening-target validation.

The evaluator sees raw per-patch Teacher/Student cosine discrepancies.  For every spatial patch it
encodes the available absolute layers as a set/sequence, then predicts whether that patch belongs to
the known synthetic perturbation region.  It is shared by all candidate suffix boundaries.
"""
import torch
import torch.nn as nn


class SharedLayerPatchLocalizer(nn.Module):
    """One shared cross-layer encoder followed by learned pooling and a patch logit head."""

    def __init__(self, num_layers=12, model_dim=16, depth=1, heads=4, dropout=0.0):
        super().__init__()
        if num_layers < 1:
            raise ValueError('num_layers must be positive')
        if model_dim % heads:
            raise ValueError('model_dim must be divisible by heads')
        self.num_layers = int(num_layers)
        self.scalar_projection = nn.Sequential(
            nn.Linear(1, model_dim), nn.GELU(), nn.Linear(model_dim, model_dim),
        )
        self.layer_embedding = nn.Parameter(torch.empty(num_layers, model_dim))
        block = nn.TransformerEncoderLayer(
            model_dim, heads, 4 * model_dim, dropout=dropout, activation='gelu',
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=depth)
        self.pool_score = nn.Linear(model_dim, 1)
        self.head = nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, 1))
        nn.init.normal_(self.layer_embedding, std=0.02)

    def forward(self, discrepancy_maps, valid_layers):
        """Return patch logits.

        Args:
            discrepancy_maps: ``(B,L,P)`` cosine discrepancy maps, zero-padded outside the suffix.
            valid_layers: ``(B,L)`` bool mask identifying layers belonging to each suffix.
        """
        if discrepancy_maps.ndim != 3:
            raise ValueError(f'expected maps (B,L,P), got {tuple(discrepancy_maps.shape)}')
        batch, layers, patches = discrepancy_maps.shape
        if layers != self.num_layers or valid_layers.shape != (batch, layers):
            raise ValueError(
                f'expected valid mask {(batch, self.num_layers)}, got {tuple(valid_layers.shape)}')
        if valid_layers.dtype != torch.bool or not valid_layers.any(dim=1).all():
            raise ValueError('every sample must contain at least one valid layer')
        if not torch.isfinite(discrepancy_maps).all():
            raise FloatingPointError('non-finite localization input')

        # Each image patch becomes an independent cross-layer sequence.  There is only one feature
        # encoder: scalar projection -> shared Transformer.  Learned pooling is a reduction, not a
        # second branch-specific encoder.
        values = discrepancy_maps.transpose(1, 2).reshape(batch * patches, layers, 1)
        tokens = self.scalar_projection(values) + self.layer_embedding.unsqueeze(0)
        padding = (~valid_layers).unsqueeze(1).expand(batch, patches, layers).reshape(
            batch * patches, layers,
        )
        tokens = self.encoder(tokens, src_key_padding_mask=padding)
        weights = self.pool_score(tokens).squeeze(-1).masked_fill(padding, -torch.inf).softmax(dim=1)
        pooled = (tokens * weights.unsqueeze(-1)).sum(dim=1)
        logits = self.head(pooled).reshape(batch, patches)
        if not torch.isfinite(logits).all():
            raise FloatingPointError('non-finite localization output')
        return logits

