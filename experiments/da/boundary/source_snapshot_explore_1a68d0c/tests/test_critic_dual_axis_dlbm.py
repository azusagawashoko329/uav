import unittest

import torch

from experiment_DLBM.critic_dual_axis_dlbm import CriticDualAxisDLBM


class CriticDualAxisDLBMTest(unittest.TestCase):
    @staticmethod
    def opening(frames=5, added=4, channels=8, side=4):
        return {
            'base_state': torch.randn(frames, channels, side, side),
            'candidate_state': torch.randn(frames, channels, side, side),
            'deep_state': torch.randn(frames, added, channels, side, side),
            'observed': torch.randn(frames, added, channels, side, side),
            'innovation': torch.randn(frames, added, side, side),
        }

    def test_critic_and_normal_axis_backpropagate(self):
        model = CriticDualAxisDLBM(
            predictor_channels=8, topk=3, model_dim=16, heads=4,
        )
        first, second = torch.triu_indices(5, 5, offset=1)
        base, opened = model.normal_axis.pair_logits(self.opening(), first, second)
        nuisance = model.nuisance_logits(self.opening())
        loss = model.multitask_loss(
            base.square().mean() + opened.square().mean(), nuisance.square().mean(),
        )
        loss.backward()
        self.assertIsNotNone(model.normal_axis.response.weight.grad)
        self.assertIsNotNone(model.nuisance_axis.response.weight.grad)
        self.assertIsNotNone(model.nuisance_head[-1].weight.grad)


if __name__ == '__main__':
    unittest.main()
