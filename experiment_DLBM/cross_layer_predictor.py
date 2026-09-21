"""Bottom-up study: spatially aligned deep-to-shallow discrepancy prediction."""
import math

import torch
import torch.nn as nn


class ConvGRUCell(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.channels = int(channels)
        if kernel_size not in (1, 3):
            raise ValueError('kernel_size must be 1 or 3')
        padding = kernel_size // 2
        self.gates = nn.Conv2d(
            2 * channels, 2 * channels, kernel_size, padding=padding,
        )
        self.candidate = nn.Conv2d(
            2 * channels, channels, kernel_size, padding=padding,
        )

    def forward(self, value, hidden):
        if value.shape != hidden.shape:
            raise ValueError('ConvGRU value/hidden shape mismatch')
        reset, update = self.gates(torch.cat((value, hidden), dim=1)).chunk(2, dim=1)
        reset, update = torch.sigmoid(reset), torch.sigmoid(update)
        candidate = torch.tanh(self.candidate(torch.cat((value, reset * hidden), dim=1)))
        return (1.0 - update) * hidden + update * candidate


class ResidualConvCell(nn.Module):
    """Fixed-update recurrent control for testing whether learned GRU gates matter."""

    def __init__(self, channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.candidate = nn.Conv2d(
            2 * channels, channels, kernel_size, padding=padding,
        )

    def forward(self, value, hidden):
        if value.shape != hidden.shape:
            raise ValueError('residual value/hidden shape mismatch')
        candidate = torch.tanh(self.candidate(torch.cat((value, hidden), dim=1)))
        return 0.5 * hidden + 0.5 * candidate


class CrossLayerPredictor(nn.Module):
    """Predict each shallower map from all already-observed deeper maps.

    The recurrent state stays on the native square patch grid.  No spatial or layer
    pooling occurs, so the innovation maps can later feed a boundary head.
    """

    def __init__(self, num_layers=12, channels=16, kernel_size=3, cell='gru'):
        super().__init__()
        self.num_layers = int(num_layers)
        self.channels = int(channels)
        if kernel_size not in (1, 3):
            raise ValueError('kernel_size must be 1 or 3')
        padding = kernel_size // 2
        self.spatial_stem = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size, padding=padding), nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size, padding=padding), nn.GELU(),
        )
        self.observed_layer_embedding = nn.Parameter(torch.empty(num_layers, channels))
        self.target_layer_embedding = nn.Parameter(torch.empty(num_layers, channels))
        if cell == 'gru':
            self.recurrent = ConvGRUCell(channels, kernel_size=kernel_size)
        elif cell == 'residual':
            self.recurrent = ResidualConvCell(channels, kernel_size=kernel_size)
        else:
            raise ValueError('cell must be gru or residual')
        self.decoder = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=padding), nn.GELU(),
            nn.Conv2d(channels, 1, 1),
        )
        nn.init.normal_(self.observed_layer_embedding, std=0.02)
        nn.init.normal_(self.target_layer_embedding, std=0.02)

    def forward(self, maps):
        if maps.ndim != 3 or maps.shape[1] != self.num_layers:
            raise ValueError(
                f'expected (frames,{self.num_layers},patches), got {tuple(maps.shape)}')
        side = math.isqrt(int(maps.shape[-1]))
        if side * side != maps.shape[-1]:
            raise ValueError('patch count must form a square grid')
        if not torch.isfinite(maps).all():
            raise FloatingPointError('non-finite discrepancy maps')
        grid = maps.reshape(len(maps), self.num_layers, side, side)
        hidden = maps.new_zeros((len(maps), self.channels, side, side))
        deepest = self.spatial_stem(grid[:, -1:].contiguous())
        deepest = deepest + self.observed_layer_embedding[-1].view(1, -1, 1, 1)
        hidden = self.recurrent(deepest, hidden)
        predictions, innovations, states = [], [], []
        for layer in range(self.num_layers - 2, -1, -1):
            target_state = hidden + self.target_layer_embedding[layer].view(1, -1, 1, 1)
            prediction = self.decoder(target_state).squeeze(1)
            target = grid[:, layer]
            innovation = target - prediction
            predictions.append(prediction)
            innovations.append(innovation)
            states.append(hidden)
            observed = self.spatial_stem(target.unsqueeze(1))
            observed = observed + self.observed_layer_embedding[layer].view(1, -1, 1, 1)
            hidden = self.recurrent(observed, hidden)
        # Restore ascending L0..L10 order.
        return {
            'prediction': torch.stack(predictions[::-1], dim=1),
            'innovation': torch.stack(innovations[::-1], dim=1),
            'deep_state': torch.stack(states[::-1], dim=1),
        }

    def opening_features(self, maps, base_layers, candidate_layers):
        """Return spatial innovations while extending an arbitrary ordered layer subset."""
        if maps.ndim != 3 or maps.shape[1] != self.num_layers:
            raise ValueError('invalid discrepancy map shape')
        base = set(int(layer) for layer in base_layers)
        candidate = set(int(layer) for layer in candidate_layers)
        if not base < candidate or min(candidate) < 0 or max(candidate) >= self.num_layers:
            raise ValueError('base must be a valid strict subset of candidate')
        side = math.isqrt(int(maps.shape[-1]))
        if side * side != maps.shape[-1]:
            raise ValueError('patch count must form a square grid')
        grid = maps.reshape(len(maps), self.num_layers, side, side)
        hidden = maps.new_zeros((len(maps), self.channels, side, side))
        for layer in sorted(base, reverse=True):
            observed = self.spatial_stem(grid[:, layer:layer + 1].contiguous())
            observed = observed + self.observed_layer_embedding[layer].view(1, -1, 1, 1)
            hidden = self.recurrent(observed, hidden)
        base_state = hidden
        states, observed_values, innovations = [], [], []
        added_layers = sorted(candidate - base, reverse=True)
        for layer in added_layers:
            prediction = self.decoder(
                hidden + self.target_layer_embedding[layer].view(1, -1, 1, 1),
            ).squeeze(1)
            target = grid[:, layer]
            observed = self.spatial_stem(target.unsqueeze(1))
            observed = observed + self.observed_layer_embedding[layer].view(1, -1, 1, 1)
            states.append(hidden)
            observed_values.append(observed)
            innovations.append(target - prediction)
            hidden = self.recurrent(observed, hidden)
        return {
            'base_state': base_state,
            'candidate_state': hidden,
            'deep_state': torch.stack(states, dim=1),
            'observed': torch.stack(observed_values, dim=1),
            'innovation': torch.stack(innovations, dim=1),
            'added_layers': added_layers,
        }
