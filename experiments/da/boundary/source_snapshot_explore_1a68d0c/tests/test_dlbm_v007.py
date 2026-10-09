import unittest

import torch

from experiment_DLBM.dlbm_v007 import LearnedOpeningTransition, boundary_from_evidence
from experiment_DLBM.train_boundary_v007 import (derangement, opening_scores,
                                                  pair_flip_rate)


class DLBMV007Test(unittest.TestCase):
    def test_real_and_shuffled_openings_share_shape_and_backpropagate(self):
        torch.manual_seed(7)
        model = LearnedOpeningTransition(
            num_layers=4, topk=2, model_dim=16, heads=4,
        )
        normal = torch.rand(6, 4, 16)
        pseudo = torch.rand(6, 4, 16)
        permutation = derangement(6, normal.device)
        real_n, real_p = model.encode(normal, pseudo, boundary=1)
        null_n, null_p = model.encode(normal, pseudo, boundary=1,
                                      opened_permutation=permutation)
        logits = torch.cat((model.realness_logits(real_n, real_p),
                            model.realness_logits(null_n, null_p)))
        logits.sum().backward()
        self.assertEqual(real_n.shape, null_n.shape)
        self.assertIsNotNone(model.cross_relation.in_proj_weight.grad)
        self.assertGreater(float(model.cross_relation.in_proj_weight.grad.abs().sum()), 0)

    def test_derangement_has_no_fixed_points(self):
        for size in (2, 3, 8):
            permutation = derangement(size, torch.device('cpu'))
            self.assertTrue(torch.all(permutation != torch.arange(size)))

    def test_opening_scores_and_flip_rate_detect_order_reversal(self):
        maps = torch.zeros(2, 3, 4)
        maps[0, 2] = 2.0; maps[1, 2] = 1.0
        maps[0, 1] = 0.0; maps[1, 1] = 4.0
        base, opened = opening_scores(maps, boundary=1, topk=2)
        self.assertEqual(float(pair_flip_rate(base, opened, maximum=4)), 1.0)

    def test_order_heads_backpropagate(self):
        model = LearnedOpeningTransition(num_layers=4, topk=2, model_dim=16, heads=4)
        embedding = torch.randn(5, 16)
        base, opened = model.order_scores(embedding)
        (base.sum() + opened.sum()).backward()
        self.assertIsNotNone(model.base_order_head.weight.grad)
        self.assertIsNotNone(model.open_order_head.weight.grad)

    def test_boundary_uses_evidence_argmax_with_l11_baseline(self):
        boundary, complete = boundary_from_evidence(torch.tensor([-1.0, 0.2, 0.1]))
        self.assertEqual(boundary, 1)
        self.assertTrue(torch.allclose(complete, torch.tensor([-1.0, 0.2, 0.1, 0.0])))


if __name__ == '__main__':
    unittest.main()
