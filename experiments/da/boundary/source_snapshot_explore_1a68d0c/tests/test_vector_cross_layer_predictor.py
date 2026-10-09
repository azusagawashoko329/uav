import unittest

import torch

from experiment_DLBM.vector_cross_layer_predictor import (
    VectorCrossLayerPredictor, vector_prediction_loss, vector_reconstruction_loss,
)


class VectorCrossLayerPredictorTest(unittest.TestCase):
    def test_reconstructs_vector_residual_and_backpropagates(self):
        model = VectorCrossLayerPredictor(feature_dim=24, channels=8)
        teacher = torch.rand(2, 12, 16, 24)
        student = torch.rand(2, 12, 16, 24)
        loss, output = vector_prediction_loss(model, teacher, student)
        self.assertEqual(output['prediction'].shape, (2, 11, 16, 24))
        self.assertEqual(output['deep_state'].shape, (2, 11, 8, 4, 4))
        loss.backward()
        self.assertGreater(float(model.pair_projection[1].weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.decoder[-1].weight.grad.abs().sum()), 0.0)

    def test_same_layer_reconstruction_prevents_zero_bottleneck(self):
        model = VectorCrossLayerPredictor(feature_dim=24, channels=8)
        teacher = torch.rand(2, 12, 16, 24)
        student = torch.rand(2, 12, 16, 24)
        loss = vector_reconstruction_loss(model, teacher, student)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(float(model.pair_projection[1].weight.grad.abs().sum()), 0.0)

    def test_mismatched_pair_is_rejected(self):
        model = VectorCrossLayerPredictor(feature_dim=24, channels=8)
        with self.assertRaises(ValueError):
            model(torch.rand(2, 12, 16, 24), torch.rand(2, 11, 16, 24))


if __name__ == '__main__':
    unittest.main()
