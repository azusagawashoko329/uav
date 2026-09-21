"""Dual order-transport axes for normal structure and pseudo nuisance."""
import torch
import torch.nn as nn

from experiment_DLBM.predictive_order_dlbm import PredictiveOrderDLBM


class DualOrderAxisDLBM(nn.Module):
    """Two independent order predictors with learned NLL scale calibration."""

    def __init__(self, predictor_channels=16, topk=5, model_dim=32, heads=4):
        super().__init__()
        kwargs = dict(
            predictor_channels=predictor_channels, topk=topk,
            model_dim=model_dim, heads=heads,
        )
        self.normal_axis = PredictiveOrderDLBM(**kwargs)
        self.pseudo_axis = PredictiveOrderDLBM(**kwargs)
        self.axis_log_variance = nn.Parameter(torch.zeros(2))

    def multitask_loss(self, normal_loss, pseudo_loss):
        losses = torch.stack((normal_loss, pseudo_loss))
        precision = torch.exp(-self.axis_log_variance)
        return (precision * losses + self.axis_log_variance).mean()

    def axis_precision(self):
        return torch.exp(-self.axis_log_variance)

    def dual_boundary_distribution(self, middle_axes, shallow_axes):
        precision = self.axis_precision().detach()
        middle = precision[0] * middle_axes[0] - precision[1] * middle_axes[1]
        shallow = precision[0] * shallow_axes[0] - precision[1] * shallow_axes[1]
        boundary, probabilities, utilities = PredictiveOrderDLBM.boundary_distribution(
            middle, shallow,
        )
        return boundary, probabilities, utilities, torch.stack((middle, shallow))
