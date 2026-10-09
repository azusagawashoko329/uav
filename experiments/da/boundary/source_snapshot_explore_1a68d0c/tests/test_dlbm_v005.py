import unittest

import torch
import torch.nn as nn

from experiment_DLBM.dlbm_v005 import (SceneBoundaryDistribution, area_rank_loss,
                                       boundary_policy, ExactBoundarySearchSupernet)
from model import all_boundary_topk_scores, build_bottleneck, build_student


class DLBMV005Test(unittest.TestCase):
    def test_hard_gumbel_gate_has_straight_through_gradient(self):
        torch.manual_seed(3)
        model = SceneBoundaryDistribution((0, 4, 8, 11))
        gate = model.sample(temperature=1.0)
        self.assertEqual(float(gate.sum()), 1.0)
        self.assertTrue(bool(((gate == 0) | (gate == 1)).all()))
        (gate * torch.tensor([0.1, 0.2, 0.4, 0.8])).sum().backward()
        self.assertTrue(torch.isfinite(model.logits.grad).all())
        self.assertGreater(float(model.logits.grad.abs().sum()), 0)

    def test_area_rank_loss_prefers_larger_area_higher_score(self):
        areas_a = torch.tensor([0.02, 0.12])
        areas_b = torch.tensor([0.10, 0.03])
        good = area_rank_loss(torch.tensor([0.0, 1.0]), torch.tensor([1.0, 0.0]),
                              areas_a, areas_b)
        bad = area_rank_loss(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]),
                             areas_a, areas_b)
        self.assertLess(float(good), float(bad))

    def test_policy_serializes_legal_suffix(self):
        result = boundary_policy(torch.tensor([-2.0, 3.0, 1.0, 0.0]), (0, 4, 8, 11))
        self.assertEqual(result['boundary'], 4)
        self.assertEqual(result['selected_taps'], list(range(4, 12)))
        self.assertAlmostEqual(sum(result['probabilities'].values()), 1.0, places=6)

    def test_virtual_update_can_send_outer_gradient_to_gate(self):
        # Tiny analytic analogue of the full functional one-step update.
        logits = nn.Parameter(torch.zeros(2))
        weight = nn.Parameter(torch.tensor(1.0))
        gate = torch.softmax(logits, dim=0)
        inner = gate[0] * (weight - 1) ** 2 + gate[1] * (weight + 2) ** 2
        grad = torch.autograd.grad(inner, weight, create_graph=True)[0]
        updated = weight - 0.1 * grad
        outer = (updated - 0.5) ** 2
        outer.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()), 0)

    def test_selected_search_scores_equal_existing_exact_supernet(self):
        torch.manual_seed(7)
        ocbe = build_bottleneck(n_taps=4, embed_dim=8, code_dim=3, fuse_hidden=7)
        student = build_student(n_taps=4, embed_dim=8, code_dim=3, depth=4, heads=2,
                                tap_indices=list(range(4)), tap_align='absolute', teacher_depth=4)
        taps = [torch.randn(2, 6, 8) for _ in range(4)]
        expected, _ = all_boundary_topk_scores(ocbe, student, taps, topk=2)
        search = ExactBoundarySearchSupernet(ocbe, student, boundaries=(0, 2, 3), topk=2)
        actual = search(taps)
        torch.testing.assert_close(actual, expected[:, [0, 2, 3]], atol=1e-6, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
