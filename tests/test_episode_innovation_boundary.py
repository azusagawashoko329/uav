import unittest

import torch

from experiment_DLBM.cross_layer_predictor import CrossLayerPredictor
from experiment_DLBM.episode_innovation_boundary import (
    EpisodeInnovationBoundary, boundary_from_scale_margins, permute_added_layers,
)


class EpisodeInnovationBoundaryTest(unittest.TestCase):
    def test_episode_head_backpropagates_without_pooling_frames_early(self):
        torch.manual_seed(23)
        predictor = CrossLayerPredictor(channels=8).eval().requires_grad_(False)
        head = EpisodeInnovationBoundary(
            predictor_channels=8, model_dim=16, heads=4, topk=5,
        )
        normal = torch.rand(7, 12, 256)
        pseudo = torch.rand(7, 12, 256)
        opening_n = predictor.opening_features(normal, range(8, 12), range(4, 12))
        opening_p = predictor.opening_features(pseudo, range(8, 12), range(4, 12))
        scores = head.scores(opening_n, opening_p)
        self.assertEqual(set(scores), {'normal', 'pseudo', 'joint'})
        sum(scores.values()).backward()
        self.assertGreater(float(head.relation[0].weight.grad.abs().sum()), 0.0)
        self.assertIsNone(predictor.spatial_stem[0].weight.grad)

    def test_permutation_preserves_marginals_and_base(self):
        maps = torch.arange(5 * 12 * 4).reshape(5, 12, 4).float()
        permutation = torch.tensor([1, 2, 3, 4, 0])
        output = permute_added_layers(maps, range(8, 12), range(4, 12), permutation)
        self.assertTrue(torch.equal(output[:, 8:], maps[:, 8:]))
        self.assertTrue(torch.equal(output[:, 4:8], maps[permutation, 4:8]))
        self.assertTrue(torch.equal(
            output[:, 4:8].sort(dim=0).values, maps[:, 4:8].sort(dim=0).values,
        ))

    def test_each_legal_boundary_can_win(self):
        for expected, winner in ((8, 'D'), (4, 'M'), (0, 'S')):
            margins = {name: torch.tensor(0.0) for name in ('D', 'M', 'S')}
            margins[winner] = torch.tensor(1.0)
            boundary, _ = boundary_from_scale_margins(margins)
            self.assertEqual(boundary, expected)

    def test_episode_scores_are_scalar_logits(self):
        predictor = CrossLayerPredictor(channels=8).eval()
        head = EpisodeInnovationBoundary(
            predictor_channels=8, model_dim=16, heads=4, topk=5,
        )
        maps = torch.rand(5, 12, 256)
        opening = predictor.opening_features(maps, (11,), range(8, 12))
        scores = head.scores(opening, opening)
        self.assertTrue(all(value.shape == torch.Size([]) for value in scores.values()))


if __name__ == '__main__':
    unittest.main()
