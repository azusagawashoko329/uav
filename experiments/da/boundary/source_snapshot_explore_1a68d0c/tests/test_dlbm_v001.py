import unittest

import torch

from experiment_DLBM.dlbm_v001 import (TopKSequenceEvaluator, boundary_from_adjacent_gains,
                                       detached_gradients, first_order_adjacent_gains,
                                       suffix_gradient_alignments, suffix_means,
                                       video_balanced_mean)
from experiment_DLBM.run_boundary_v001 import within_frame_cutpaste


class DLBMV001Test(unittest.TestCase):
    def test_evaluator_forward_and_backward(self):
        model = TopKSequenceEvaluator(num_layers=6, model_dim=16, depth=1, heads=4)
        scores = torch.rand(4, 6)
        logits = model(scores)
        self.assertEqual(logits.shape, (4,))
        logits.square().mean().backward()
        self.assertIsNotNone(model.scalar_projection[0].weight.grad)
        self.assertGreater(float(model.scalar_projection[0].weight.grad.abs().sum()), 0)

    def test_suffix_gain_math_allows_a_shallow_bridge(self):
        # Opening k=2 and k=1 hurts locally, but opening k=0 compensates enough to select L0.
        gains = torch.tensor([5.0, -1.0, -1.0])
        decision = boundary_from_adjacent_gains(gains)
        self.assertEqual(decision.boundary, 0)
        self.assertTrue(torch.allclose(decision.cumulative_benefits, torch.tensor([3., -2., -1., 0.])))

    def test_first_order_gain_matches_manual_gradient_alignment(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        risk = 0.5 * parameter.square()
        risk_gradient = detached_gradients(risk, (parameter,))
        layer_losses = torch.stack((0.5 * (parameter - 2).square(),
                                    0.5 * (parameter + 1).square()))
        alignments = suffix_gradient_alignments(layer_losses, (parameter,), risk_gradient)
        # grad R=1; suffix gradients are 0.5 and 2.0.
        self.assertTrue(torch.allclose(alignments, torch.tensor([0.5, 2.0])))
        gains = first_order_adjacent_gains(alignments, inner_lr=0.1)
        self.assertTrue(torch.allclose(gains, torch.tensor([-0.15])))

    def test_within_frame_cutpaste_never_uses_another_batch_item(self):
        torch.manual_seed(9)
        base = torch.arange(3 * 32 * 32, dtype=torch.float32).reshape(1, 3, 32, 32)
        batch = torch.cat((base, base + 10000), dim=0)
        output = within_frame_cutpaste(batch)
        self.assertEqual(output.shape, batch.shape)
        self.assertGreaterEqual(float(output[0].min()), float(batch[0].min()))
        self.assertLessEqual(float(output[0].max()), float(batch[0].max()))
        self.assertGreaterEqual(float(output[1].min()), float(batch[1].min()))
        self.assertLessEqual(float(output[1].max()), float(batch[1].max()))
        self.assertFalse(torch.equal(output, batch))

    def test_video_balanced_mean_does_not_overweight_long_video(self):
        values = torch.tensor([0., 0., 0., 10.])
        self.assertEqual(float(video_balanced_mean(values, ['a', 'a', 'a', 'b'])), 5.0)

    def test_suffix_means(self):
        values = suffix_means(torch.tensor([1., 2., 6.]))
        self.assertTrue(torch.allclose(values, torch.tensor([3., 4., 6.])))


if __name__ == '__main__':
    unittest.main()
