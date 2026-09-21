"""Distillation Layer Boundary Module (DLBM).

The module reads the teacher/student feature-map pair at every teacher layer and predicts one
ordered boundary ``k``.  Boundary ``k`` means that the final distillation run uses the contiguous
suffix ``[k, ..., L-1]``.  It deliberately does not consume hand-crafted GAP/z-score features.

The categorical boundary distribution is converted to differentiable suffix gates by a cumulative
sum.  Consequently ``g[0] <= ... <= g[L-1] == 1`` for every input, including during training.
"""
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class BoundaryOutput:
    logits: torch.Tensor
    probabilities: torch.Tensor
    gates: torch.Tensor
    expected_boundary: torch.Tensor
    hard_boundary: torch.Tensor


class DistillationLayerBoundaryModule(nn.Module):
    """Encode direct T/S feature-map relations and predict an ordered suffix boundary."""

    def __init__(self, num_layers=12, feature_dim=768, model_dim=96, depth=2, heads=4,
                 dropout=0.0, temperature=1.0):
        super().__init__()
        if num_layers < 1:
            raise ValueError('num_layers must be positive')
        if temperature <= 0:
            raise ValueError('temperature must be positive')
        self.num_layers = num_layers
        self.feature_dim = feature_dim
        self.model_dim = model_dim
        self.temperature = temperature

        # Shared across depth: layer identity is supplied separately below.  LayerNorm is learned
        # feature normalization, not a compressed hand-crafted statistic.
        # T/S live in the same teacher target space, so sharing this projector keeps their
        # subtraction/product meaningful instead of letting two transforms manufacture a gap.
        self.feature_proj = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, model_dim))
        self.pair_proj = nn.Sequential(
            nn.Linear(model_dim * 4, model_dim),
            nn.GELU(),
            nn.LayerNorm(model_dim),
        )
        self.patch_attention = nn.Linear(model_dim, 1)
        self.layer_embedding = nn.Parameter(torch.zeros(1, num_layers, model_dim))
        nn.init.trunc_normal_(self.layer_embedding, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=heads, dim_feedforward=model_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
        )
        self.cross_layer = nn.TransformerEncoder(enc_layer, num_layers=depth,
                                                 norm=nn.LayerNorm(model_dim))
        # One shared scalar head is evaluated at every ordered layer token.  Its position determines
        # the candidate boundary, avoiding twelve unrelated binary gates.
        self.boundary_head = nn.Linear(model_dim, 1)

    @staticmethod
    def probabilities_to_gates(probabilities):
        """Boundary probabilities ``p(k)`` -> monotonic inclusion probability for every layer."""
        if probabilities.ndim != 2:
            raise ValueError(f'expected (B,L) probabilities, got {tuple(probabilities.shape)}')
        return probabilities.cumsum(dim=-1)

    def _validate(self, teacher_features, student_features):
        if len(teacher_features) != self.num_layers or len(student_features) != self.num_layers:
            raise ValueError(
                f'DLBM expects {self.num_layers} T/S layers, got '
                f'{len(teacher_features)}/{len(student_features)}')
        ref = teacher_features[0].shape
        if len(ref) != 3 or ref[-1] != self.feature_dim:
            raise ValueError(f'expected feature maps (B,P,{self.feature_dim}), got {tuple(ref)}')
        for i, (teacher, student) in enumerate(zip(teacher_features, student_features)):
            if teacher.shape != ref or student.shape != ref:
                raise ValueError(
                    f'layer {i} shape mismatch: teacher={tuple(teacher.shape)} '
                    f'student={tuple(student.shape)}, reference={tuple(ref)}')

    def encode(self, teacher_features, student_features):
        """Return one learned token per layer while retaining patch-map information until pooling."""
        self._validate(teacher_features, student_features)
        layer_tokens = []
        for teacher, student in zip(teacher_features, student_features):
            t = self.feature_proj(teacher)
            s = self.feature_proj(student)
            pair = self.pair_proj(torch.cat((t, s, t - s, t * s), dim=-1))
            attn = self.patch_attention(pair).squeeze(-1).softmax(dim=-1)
            layer_tokens.append((pair * attn.unsqueeze(-1)).sum(dim=1))
        tokens = torch.stack(layer_tokens, dim=1) + self.layer_embedding
        return self.cross_layer(tokens)

    def forward(self, teacher_features, student_features, reduce='none'):
        """Predict sample policies or one scene/batch policy obtained by mean-logit aggregation."""
        logits = self.boundary_head(self.encode(teacher_features, student_features)).squeeze(-1)
        if reduce == 'scene':
            logits = logits.mean(dim=0, keepdim=True)
        elif reduce != 'none':
            raise ValueError(f'unknown reduce={reduce}')
        probabilities = (logits / self.temperature).softmax(dim=-1)
        gates = self.probabilities_to_gates(probabilities)
        indices = torch.arange(self.num_layers, device=logits.device, dtype=probabilities.dtype)
        return BoundaryOutput(
            logits=logits,
            probabilities=probabilities,
            gates=gates,
            expected_boundary=(probabilities * indices).sum(dim=-1),
            hard_boundary=probabilities.argmax(dim=-1),
        )

    @staticmethod
    def soft_suffix_score(per_layer_scores, gates, eps=1e-6):
        """Gate-weighted mean score; normalization removes the trivial benefit of opening more taps."""
        if per_layer_scores.shape != gates.shape:
            raise ValueError(
                f'layer score/gate mismatch: {tuple(per_layer_scores.shape)} vs {tuple(gates.shape)}')
        return (per_layer_scores * gates).sum(dim=-1) / gates.sum(dim=-1).clamp_min(eps)


def build_dlbm(**kwargs):
    return DistillationLayerBoundaryModule(**kwargs)
