import unittest

import torch

from experiment_DLBM.dlbm_v002 import (ExactSuffixGainPredictor,
                                       exact_suffix_adjacent_gains, exact_suffix_risks)


class DLBMV002Test(unittest.TestCase):
    def test_predictor_is_frame_permutation_invariant_and_backpropagates(self):
        torch.manual_seed(4)
        model = ExactSuffixGainPredictor(
            num_layers=6, model_dim=16, layer_depth=1, pair_depth=1, heads=4,
        ).eval()
        normal = torch.rand(5, 6)
        pseudo = torch.rand(5, 6)
        permutation = torch.tensor([3, 0, 4, 1, 2])
        actual = model(normal, pseudo)
        permuted = model(normal[permutation], pseudo[permutation])
        self.assertEqual(actual.shape, (5,))
        self.assertTrue(torch.allclose(actual, permuted, atol=1e-6, rtol=1e-5))
        actual.square().mean().backward()
        gradient = model.frame_encoder.scalar_projection[0].weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0)

    def test_exact_suffix_risk_and_gain_sign(self):
        normal = torch.tensor([[0.2, 0.2, 0.2], [0.2, 0.2, 0.2]])
        pseudo = torch.tensor([[0.8, 0.5, 0.3], [0.8, 0.5, 0.3]])
        risks = exact_suffix_risks(normal, pseudo)
        gains = exact_suffix_adjacent_gains(normal, pseudo)
        self.assertTrue(torch.allclose(gains, risks[1:] - risks[:-1]))
        # L0 separates pseudo most strongly, so closing it raises risk: opening gain is positive.
        self.assertGreater(float(gains[0]), 0)

    def test_predictor_rejects_explicit_branch_shape_mismatch(self):
        model = ExactSuffixGainPredictor(
            num_layers=6, model_dim=16, layer_depth=1, pair_depth=1, heads=4,
        )
        with self.assertRaisesRegex(ValueError, 'shape mismatch'):
            model(torch.rand(3, 6), torch.rand(2, 6))


if __name__ == '__main__':
    unittest.main()
