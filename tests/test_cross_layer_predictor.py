import unittest

import torch

from experiment_DLBM.cross_layer_predictor import CrossLayerPredictor


class CrossLayerPredictorTest(unittest.TestCase):
    def test_full_grid_is_preserved_and_backpropagates(self):
        torch.manual_seed(9)
        model = CrossLayerPredictor(channels=8)
        maps = torch.rand(3, 12, 196)
        output = model(maps)
        self.assertEqual(output['prediction'].shape, (3, 11, 14, 14))
        self.assertEqual(output['innovation'].shape, (3, 11, 14, 14))
        self.assertEqual(output['deep_state'].shape, (3, 11, 8, 14, 14))
        output['prediction'].square().mean().backward()
        self.assertGreater(float(model.spatial_stem[0].weight.grad.abs().sum()), 0.0)

    def test_invalid_patch_grid_is_rejected(self):
        with self.assertRaises(ValueError):
            CrossLayerPredictor(channels=8)(torch.rand(2, 12, 195))

    def test_one_by_one_recurrent_variant_is_valid(self):
        output = CrossLayerPredictor(channels=8, kernel_size=1)(torch.rand(2, 12, 256))
        self.assertEqual(output['prediction'].shape, (2, 11, 16, 16))

    def test_residual_recurrent_control_is_valid(self):
        output = CrossLayerPredictor(channels=8, cell='residual')(torch.rand(2, 12, 256))
        self.assertEqual(output['innovation'].shape, (2, 11, 16, 16))

    def test_arbitrary_subset_opening_preserves_added_layer_maps(self):
        model = CrossLayerPredictor(channels=8)
        output = model.opening_features(
            torch.rand(2, 12, 256), [8, 10], [8, 9, 10, 11],
        )
        self.assertEqual(output['added_layers'], [11, 9])
        self.assertEqual(output['innovation'].shape, (2, 2, 16, 16))
        self.assertEqual(output['deep_state'].shape, (2, 2, 8, 16, 16))
        self.assertEqual(output['base_state'].shape, (2, 8, 16, 16))
        self.assertEqual(output['candidate_state'].shape, (2, 8, 16, 16))


if __name__ == '__main__':
    unittest.main()
