import unittest

import torch

from experiment_DLBM.cross_layer_predictor import CrossLayerPredictor
from experiment_DLBM.order_transport_dlbm import OrderTransportDLBM


class OrderTransportDLBMTest(unittest.TestCase):
    def test_episode_transport_and_gain_backpropagate(self):
        predictor = CrossLayerPredictor(channels=8).eval().requires_grad_(False)
        model = OrderTransportDLBM(predictor_channels=8, model_dim=16, heads=4)
        maps = torch.rand(7, 12, 256)
        first, second = torch.triu_indices(7, 7, offset=1)
        opening = predictor.opening_features(maps, range(8, 12), range(4, 12))
        encoded = model.encode_branch(opening, first, second)
        self.assertEqual(encoded['episode'].shape, (16,))
        real = model.fuse_branches(encoded['episode'], encoded['episode'])
        logit = model.gain_logit(real, torch.stack((real + 0.1, real - 0.1)))
        logit.backward()
        self.assertGreater(float(model.relation[0].weight.grad.abs().sum()), 0.0)
        self.assertIsNone(predictor.spatial_stem[0].weight.grad)

    def test_ordered_distribution_is_normalized_and_can_select_each_boundary(self):
        cases = [((-8.0, 0.0), 8), ((8.0, -8.0), 4), ((8.0, 8.0), 0)]
        for logits, expected in cases:
            boundary, probability = OrderTransportDLBM.boundary_distribution(
                *map(torch.tensor, logits),
            )
            self.assertEqual(boundary, expected)
            self.assertAlmostEqual(float(probability.sum()), 1.0, places=6)


if __name__ == '__main__':
    unittest.main()
