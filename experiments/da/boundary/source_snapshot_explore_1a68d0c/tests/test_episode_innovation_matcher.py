import unittest

import torch

from experiment_DLBM.cross_layer_predictor import CrossLayerPredictor
from experiment_DLBM.episode_innovation_matcher import (
    EpisodeInnovationMatcher, boundary_from_retrieval, retrieval_metrics,
    symmetric_retrieval_loss,
)


class EpisodeInnovationMatcherTest(unittest.TestCase):
    def test_matching_backpropagates_only_to_matcher(self):
        predictor = CrossLayerPredictor(channels=8).eval().requires_grad_(False)
        matcher = EpisodeInnovationMatcher(
            predictor_channels=8, model_dim=16, heads=4, topk=5,
        )
        opening = predictor.opening_features(torch.rand(6, 12, 256), (11,), range(8, 12))
        logits = matcher.logits(opening)
        self.assertEqual(logits.shape, (6, 6))
        symmetric_retrieval_loss(logits).backward()
        self.assertGreater(float(matcher.relation[0].weight.grad.abs().sum()), 0.0)
        self.assertIsNone(predictor.spatial_stem[0].weight.grad)

    def test_two_pseudo_openings_form_square_retrieval(self):
        predictor = CrossLayerPredictor(channels=8).eval()
        matcher = EpisodeInnovationMatcher(
            predictor_channels=8, model_dim=16, heads=4, topk=5,
        )
        first = predictor.opening_features(torch.rand(5, 12, 256), (11,), range(8, 12))
        second = predictor.opening_features(torch.rand(5, 12, 256), (11,), range(8, 12))
        self.assertEqual(matcher.cross_opening_logits(first, second).shape, (5, 5))

    def test_identity_logits_have_perfect_retrieval(self):
        metrics = retrieval_metrics(torch.eye(5) * 10)
        self.assertEqual(metrics['top1'], 1.0)
        self.assertEqual(metrics['mrr'], 1.0)

    def test_each_scale_can_select_boundary(self):
        for scale, boundary in (('D', 8), ('M', 4), ('S', 0)):
            evidence = {'D': 0.0, 'M': 0.0, 'S': 0.0}
            evidence[scale] = 1.0
            self.assertEqual(boundary_from_retrieval(evidence)[0], boundary)


if __name__ == '__main__':
    unittest.main()
