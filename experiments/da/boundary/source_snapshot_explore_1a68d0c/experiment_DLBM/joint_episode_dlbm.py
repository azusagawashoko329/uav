"""Joint normal/pseudo episode DLBM.

The model does not reduce normal and pseudo openings to two independently
trained scalar axes.  It encodes both branches with one shared patch encoder,
builds joint pair tokens, and learns the video-level opening event directly.
"""
import torch
import torch.nn as nn

from experiment_DLBM.episode_innovation_boundary import SeedAttentionPool
from experiment_DLBM.predictive_order_dlbm import PredictiveOrderDLBM


class JointEpisodeDLBM(nn.Module):
    """Learn joint order-migration evidence from a set of frame pairs."""

    def __init__(self, predictor_channels=16, topk=5, model_dim=32, heads=4,
                 pool_seeds=2):
        super().__init__()
        self.model_dim = int(model_dim)
        # One encoder deliberately shares patch/layer semantics across normal
        # and pseudo inputs.  The two branches differ only in their data.
        self.opening_encoder = PredictiveOrderDLBM(
            predictor_channels=predictor_channels, topk=topk,
            model_dim=model_dim, heads=heads,
        )
        self.pair_encoder = nn.Sequential(
            nn.LayerNorm(8 * model_dim),
            nn.Linear(8 * model_dim, 2 * model_dim), nn.GELU(),
            nn.Linear(2 * model_dim, model_dim), nn.LayerNorm(model_dim),
        )
        self.normal_flip_head = nn.Linear(model_dim, 1)
        self.pseudo_flip_head = nn.Linear(model_dim, 1)
        self.joint_pair_head = nn.Sequential(
            nn.Linear(model_dim, model_dim), nn.GELU(), nn.Linear(model_dim, 1),
        )
        self.pair_pool = SeedAttentionPool(
            model_dim, heads=heads, seeds=pool_seeds,
        )
        self.episode_head = nn.Sequential(
            nn.LayerNorm(pool_seeds * model_dim),
            nn.Linear(pool_seeds * model_dim, model_dim), nn.GELU(),
            nn.Linear(model_dim, 1),
        )
        # Learned homoscedastic uncertainty replaces four hand-tuned loss
        # coefficients.  It changes optimization balance, not target meaning.
        self.task_log_variance = nn.Parameter(torch.zeros(4))

    @staticmethod
    def _pair_delta(tokens, first, second):
        return tokens.index_select(0, first) - tokens.index_select(0, second)

    def pair_tokens(self, normal_opening, pseudo_opening, first, second):
        normal_base, normal_open = self.opening_encoder.frame_tokens(normal_opening)
        pseudo_base, pseudo_open = self.opening_encoder.frame_tokens(pseudo_opening)
        nb = self._pair_delta(normal_base, first, second)
        no = self._pair_delta(normal_open, first, second)
        pb = self._pair_delta(pseudo_base, first, second)
        po = self._pair_delta(pseudo_open, first, second)
        normal_change, pseudo_change = no - nb, po - pb
        return self.pair_encoder(torch.cat((
            nb, no, pb, po,
            torch.abs(normal_change), torch.abs(pseudo_change),
            normal_change - pseudo_change, normal_change * pseudo_change,
        ), dim=-1))

    def forward(self, normal_opening, pseudo_opening, first, second):
        pairs = self.pair_tokens(
            normal_opening, pseudo_opening, first, second,
        )
        if len(pairs) < 2:
            raise ValueError('joint episode requires at least two frame pairs')
        episode = self.pair_pool(pairs.unsqueeze(0)).squeeze(0)
        return {
            'normal_flip': self.normal_flip_head(pairs).squeeze(-1),
            'pseudo_flip': self.pseudo_flip_head(pairs).squeeze(-1),
            'joint_pair': self.joint_pair_head(pairs).squeeze(-1),
            'episode': self.episode_head(episode).squeeze(-1),
        }

    def multitask_loss(self, losses):
        if len(losses) != len(self.task_log_variance):
            raise ValueError('expected four joint-episode losses')
        losses = torch.stack(tuple(losses))
        precision = torch.exp(-self.task_log_variance)
        return (precision * losses + self.task_log_variance).mean()

    def task_precision(self):
        return torch.exp(-self.task_log_variance)

    @staticmethod
    def joint_target(normal_flip, pseudo_flip):
        """Normal migration that is not reproduced inside the pseudo branch."""
        return normal_flip * (1.0 - pseudo_flip)

    @staticmethod
    def boundary_distribution(middle_gain, shallow_gain):
        return PredictiveOrderDLBM.boundary_distribution(
            middle_gain, shallow_gain,
        )
