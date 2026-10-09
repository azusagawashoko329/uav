import unittest

import torch

from experiment_DLBM.dual_axis_dlbm import DualAxisDLBM


class DualAxisDLBMTest(unittest.TestCase):
    @staticmethod
    def opening(frames=5, added=4, channels=8, side=4):
        return {
            'base_state': torch.randn(frames, channels, side, side),
            'candidate_state': torch.randn(frames, channels, side, side),
            'deep_state': torch.randn(frames, added, channels, side, side),
            'observed': torch.randn(frames, added, channels, side, side),
            'innovation': torch.randn(frames, added, side, side),
        }

    def test_independent_axes_backpropagate_and_preserve_mask_grid(self):
        model = DualAxisDLBM(
            predictor_channels=8, topk=3, model_dim=16, heads=4,
        )
        opening = self.opening()
        first, second = torch.triu_indices(5, 5, offset=1)
        base, opened = model.pair_logits(opening, first, second)
        nuisance = model.nuisance_logits(opening)
        self.assertEqual(nuisance.shape, (5, 4, 4))
        loss = model.multitask_loss(
            base.square().mean() + opened.square().mean(), nuisance.square().mean(),
        )
        loss.backward()
        self.assertIsNotNone(model.response.weight.grad)
        self.assertIsNotNone(model.pseudo_mask_head[-1].weight.grad)
        self.assertIsNotNone(model.axis_log_variance.grad)

    def test_dual_evidence_can_suppress_nuisance_and_keep_shallow_bridge(self):
        model = DualAxisDLBM(
            predictor_channels=8, topk=3, model_dim=16, heads=4,
        )
        boundary, _, _, evidence = model.dual_boundary_distribution(
            (torch.tensor(0.2), torch.tensor(0.5)),
            (torch.tensor(-0.1), torch.tensor(0.0)),
        )
        self.assertEqual(boundary, 8)
        self.assertLess(float(evidence[0]), 0)
        boundary, _, _, _ = model.dual_boundary_distribution(
            (torch.tensor(-0.1), torch.tensor(0.0)),
            (torch.tensor(0.3), torch.tensor(0.0)),
        )
        self.assertEqual(boundary, 0)


if __name__ == '__main__':
    unittest.main()
