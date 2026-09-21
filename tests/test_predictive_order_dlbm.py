import unittest

import torch

from experiment_DLBM.predictive_order_dlbm import PredictiveOrderDLBM


class PredictiveOrderDLBMTest(unittest.TestCase):
    @staticmethod
    def opening(frames=5, added=4, channels=8, side=4):
        return {
            'base_state': torch.randn(frames, channels, side, side),
            'candidate_state': torch.randn(frames, channels, side, side),
            'deep_state': torch.randn(frames, added, channels, side, side),
            'observed': torch.randn(frames, added, channels, side, side),
            'innovation': torch.randn(frames, added, side, side),
        }

    def test_shared_order_head_learns_through_base_and_opening(self):
        model = PredictiveOrderDLBM(
            predictor_channels=8, topk=3, model_dim=16, heads=4,
        )
        first, second = torch.triu_indices(5, 5, offset=1)
        base, opened = model.pair_logits(self.opening(), first, second)
        self.assertEqual(base.shape, opened.shape)
        (base.mean() + opened.mean()).backward()
        self.assertIsNotNone(model.response.weight.grad)
        self.assertIsNotNone(model.state_response.weight.grad)
        self.assertIsNotNone(model.order_head[-1].weight.grad)

    def test_cumulative_gain_selects_all_three_boundaries(self):
        cases = [((-1.0, -1.0), 8), ((1.0, -2.0), 4), ((-1.0, 2.0), 0)]
        for gains, expected in cases:
            boundary, probabilities, utilities = (
                PredictiveOrderDLBM.boundary_distribution(
                    torch.tensor(gains[0]), torch.tensor(gains[1]),
                )
            )
            self.assertEqual(boundary, expected)
            self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=6)
            self.assertEqual(int(utilities.argmax()), (8, 4, 0).index(expected))


if __name__ == '__main__':
    unittest.main()
