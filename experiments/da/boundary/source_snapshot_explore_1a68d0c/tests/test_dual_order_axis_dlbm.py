import unittest

import torch

from experiment_DLBM.dual_order_axis_dlbm import DualOrderAxisDLBM


class DualOrderAxisDLBMTest(unittest.TestCase):
    @staticmethod
    def opening(frames=5, added=4, channels=8, side=4):
        return {
            'base_state': torch.randn(frames, channels, side, side),
            'candidate_state': torch.randn(frames, channels, side, side),
            'deep_state': torch.randn(frames, added, channels, side, side),
            'observed': torch.randn(frames, added, channels, side, side),
            'innovation': torch.randn(frames, added, side, side),
        }

    def test_axes_are_independent_and_both_backpropagate(self):
        model = DualOrderAxisDLBM(
            predictor_channels=8, topk=3, model_dim=16, heads=4,
        )
        first, second = torch.triu_indices(5, 5, offset=1)
        normal = model.normal_axis.pair_logits(self.opening(), first, second)
        pseudo = model.pseudo_axis.pair_logits(self.opening(), first, second)
        loss = model.multitask_loss(
            normal[0].square().mean() + normal[1].square().mean(),
            pseudo[0].square().mean() + pseudo[1].square().mean(),
        )
        loss.backward()
        self.assertIsNotNone(model.normal_axis.response.weight.grad)
        self.assertIsNotNone(model.pseudo_axis.response.weight.grad)
        self.assertIsNotNone(model.axis_log_variance.grad)

    def test_signed_pseudo_axis_can_veto_or_support_an_opening(self):
        model = DualOrderAxisDLBM(
            predictor_channels=8, topk=3, model_dim=16, heads=4,
        )
        boundary, _, _, evidence = model.dual_boundary_distribution(
            (torch.tensor(0.1), torch.tensor(0.2)),
            (torch.tensor(0.0), torch.tensor(0.0)),
        )
        self.assertEqual(boundary, 8)
        self.assertLess(float(evidence[0]), 0)
        boundary, _, _, evidence = model.dual_boundary_distribution(
            (torch.tensor(-0.1), torch.tensor(-0.2)),
            (torch.tensor(0.2), torch.tensor(-0.1)),
        )
        self.assertEqual(boundary, 0)
        self.assertGreater(float(evidence[1]), 0)


if __name__ == '__main__':
    unittest.main()
