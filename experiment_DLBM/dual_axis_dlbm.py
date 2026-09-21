"""Two-axis DLBM: normal order novelty and pseudo perturbation nuisance."""
import torch
import torch.nn as nn

from experiment_DLBM.predictive_order_dlbm import PredictiveOrderDLBM


class DualAxisDLBM(PredictiveOrderDLBM):
    """Keep normal and pseudo evidence in separate learned spatial branches."""

    def __init__(self, predictor_channels=16, topk=5, model_dim=32, heads=4):
        super().__init__(predictor_channels, topk, model_dim, heads)
        channels = int(predictor_channels)
        self.pseudo_relation = nn.Sequential(
            nn.Conv2d(4 * channels + 1, 2 * channels, 3, padding=1), nn.GELU(),
            nn.Conv2d(2 * channels, channels, 3, padding=1), nn.GELU(),
        )
        self.pseudo_spatial = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.GELU(),
        )
        self.pseudo_layer_attention = nn.Conv2d(channels, 1, 1)
        self.pseudo_mask_head = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.GELU(),
            nn.Conv2d(channels, 1, 1),
        )
        # Kendall-style homoscedastic task uncertainty.  It learns the relative
        # scale of two NLL axes instead of introducing a hand-tuned loss weight.
        self.axis_log_variance = nn.Parameter(torch.zeros(2))

    @staticmethod
    def _spatial_relation(opening, encoder):
        state, observed, innovation = (
            opening['deep_state'], opening['observed'], opening['innovation'],
        )
        if state.ndim != 5 or observed.shape != state.shape:
            raise ValueError('invalid predictive opening features')
        frames, added, channels, height, width = state.shape
        flat_state = state.reshape(frames * added, channels, height, width)
        flat_observed = observed.reshape(frames * added, channels, height, width)
        flat_innovation = innovation.reshape(frames * added, 1, height, width)
        relation = encoder(torch.cat((
            flat_state, flat_observed, flat_state - flat_observed,
            flat_state * flat_observed, flat_innovation,
        ), dim=1))
        return relation.reshape(frames, added, relation.shape[1], height, width)

    def nuisance_logits(self, opening):
        relation = self._spatial_relation(opening, self.pseudo_relation)
        frames, added, channels, height, width = relation.shape
        encoded = self.pseudo_spatial(
            relation.reshape(frames * added, channels, height, width),
        ).reshape(frames, added, channels, height, width)
        attention = self.pseudo_layer_attention(
            encoded.reshape(frames * added, channels, height, width),
        ).reshape(frames, added, 1, height, width).softmax(dim=1)
        fused = (attention * encoded).sum(dim=1)
        return self.pseudo_mask_head(fused).squeeze(1)

    def multitask_loss(self, normal_order_loss, pseudo_mask_loss):
        losses = torch.stack((normal_order_loss, pseudo_mask_loss))
        precision = torch.exp(-self.axis_log_variance)
        return (precision * losses + self.axis_log_variance).mean()

    def axis_precision(self):
        return torch.exp(-self.axis_log_variance)

    def dual_boundary_distribution(self, middle_axes, shallow_axes):
        """Combine learned NLL evidence with learned task precision."""
        precision = self.axis_precision().detach()
        middle = precision[0] * middle_axes[0] - precision[1] * middle_axes[1]
        shallow = precision[0] * shallow_axes[0] - precision[1] * shallow_axes[1]
        boundary, probabilities, utilities = self.boundary_distribution(middle, shallow)
        return boundary, probabilities, utilities, torch.stack((middle, shallow))
