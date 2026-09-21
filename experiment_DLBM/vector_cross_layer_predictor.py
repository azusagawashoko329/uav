"""Cross-layer predictor that preserves Teacher/Student channel direction."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiment_DLBM.cross_layer_predictor import ConvGRUCell


class VectorCrossLayerPredictor(nn.Module):
    """Predict shallower normalized T-S vectors from deeper T/S patch pairs.

    Unlike the scalar discrepancy predictor, the shared stem receives the original
    Teacher and Student patch directions.  The decoder reconstructs the raw 768-D
    residual, preventing a collapsed learned relation encoder.
    """

    def __init__(self, feature_dim=768, num_layers=12, channels=32):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_layers = int(num_layers)
        self.channels = int(channels)
        self.pair_projection = nn.Sequential(
            nn.LayerNorm(2 * feature_dim),
            nn.Linear(2 * feature_dim, channels), nn.GELU(),
            nn.LayerNorm(channels),
        )
        self.spatial_stem = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.GELU(),
        )
        self.observed_layer_embedding = nn.Parameter(torch.empty(num_layers, channels))
        self.target_layer_embedding = nn.Parameter(torch.empty(num_layers, channels))
        self.recurrent = ConvGRUCell(channels, kernel_size=3)
        self.decoder = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.GELU(),
            nn.Conv2d(channels, feature_dim, 1),
        )
        nn.init.normal_(self.observed_layer_embedding, std=0.02)
        nn.init.normal_(self.target_layer_embedding, std=0.02)

    def _validate(self, teacher, student):
        if teacher.shape != student.shape or teacher.ndim != 4:
            raise ValueError('expected matched Teacher/Student (frames,layers,patches,dim)')
        if teacher.shape[1] != self.num_layers or teacher.shape[-1] != self.feature_dim:
            raise ValueError('Teacher/Student feature shape does not match predictor')
        side = math.isqrt(int(teacher.shape[2]))
        if side * side != teacher.shape[2]:
            raise ValueError('patch count must form a square grid')
        if not torch.isfinite(teacher).all() or not torch.isfinite(student).all():
            raise FloatingPointError('non-finite Teacher/Student features')
        return side

    def encode_pairs(self, teacher, student):
        side = self._validate(teacher, student)
        teacher = F.normalize(teacher.float(), dim=-1)
        student = F.normalize(student.float(), dim=-1)
        pair = torch.cat((teacher, student), dim=-1)
        encoded = self.pair_projection(pair)
        encoded = encoded.permute(0, 1, 3, 2).reshape(
            len(teacher), self.num_layers, self.channels, side, side,
        )
        encoded = self.spatial_stem(
            encoded.reshape(-1, self.channels, side, side),
        ).reshape(len(teacher), self.num_layers, self.channels, side, side)
        # Direction, not the very small raw magnitude, carries channel semantics.
        # Unit normalization also removes the all-zero SmoothL1 optimum.
        residual = F.normalize(teacher - student, dim=-1)
        return encoded, residual

    def reconstruct(self, teacher, student):
        encoded, target = self.encode_pairs(teacher, student)
        frames, layers, _, side, _ = encoded.shape
        prediction = self.decoder(
            encoded.reshape(frames * layers, self.channels, side, side),
        ).flatten(2).transpose(1, 2).reshape(
            frames, layers, side * side, self.feature_dim,
        )
        return prediction, target

    def forward(self, teacher, student):
        encoded, target_residual = self.encode_pairs(teacher, student)
        frames, _, _, side, _ = encoded.shape
        hidden = encoded.new_zeros((frames, self.channels, side, side))
        deepest = encoded[:, -1] + self.observed_layer_embedding[-1].view(1, -1, 1, 1)
        hidden = self.recurrent(deepest, hidden)
        predictions, states = [], []
        for layer in range(self.num_layers - 2, -1, -1):
            target_state = hidden + self.target_layer_embedding[layer].view(1, -1, 1, 1)
            prediction = self.decoder(target_state).flatten(2).transpose(1, 2)
            predictions.append(prediction)
            states.append(hidden)
            observed = encoded[:, layer]
            observed = observed + self.observed_layer_embedding[layer].view(1, -1, 1, 1)
            hidden = self.recurrent(observed, hidden)
        return {
            'prediction': torch.stack(predictions[::-1], dim=1),
            'target': target_residual[:, :-1],
            'deep_state': torch.stack(states[::-1], dim=1),
            'encoded_relation': encoded,
        }


def vector_prediction_loss(model, teacher, student):
    output = model(teacher, student)
    loss = (1.0 - F.cosine_similarity(
        output['prediction'], output['target'], dim=-1,
    )).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError('non-finite vector prediction loss')
    return loss, output


def vector_reconstruction_loss(model, teacher, student):
    prediction, target = model.reconstruct(teacher, student)
    loss = (1.0 - F.cosine_similarity(prediction, target, dim=-1)).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError('non-finite vector reconstruction loss')
    return loss
