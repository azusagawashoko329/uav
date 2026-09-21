import unittest

import torch

from experiment_DLBM.dlbm_v006 import (PairedIncrementalRelationNetwork,
                                       policy_from_adjacent_logits)
from experiment_DLBM.train_boundary_v006 import adjacent_rollout_gains, discrepancy_maps


class DLBMV006Test(unittest.TestCase):
    def test_policy_uses_contiguous_cumulative_open_utility(self):
        policy = policy_from_adjacent_logits(torch.tensor([-2.0, 1.0, 1.0]))
        self.assertEqual(policy.boundary, 1)
        self.assertEqual(policy.cumulative_utilities.tolist(), [0.0, 2.0, 1.0, 0.0])

    def test_relation_network_preserves_topk_patch_structure_and_backpropagates(self):
        torch.manual_seed(3)
        model = PairedIncrementalRelationNetwork(
            num_layers=4, topk=3, model_dim=16, heads=4, frame_depth=1,
        )
        normal = torch.rand(5, 4, 16)
        pseudo = torch.rand(5, 4, 16)
        logits = model(normal, pseudo)
        self.assertEqual(logits.shape, (3,))
        logits.sum().backward()
        self.assertIsNotNone(model.token_encoder[0].weight.grad)
        self.assertGreater(float(model.token_encoder[0].weight.grad.abs().sum()), 0.0)
        self.assertIsNotNone(model.cross_relation.in_proj_weight.grad)
        self.assertGreater(float(model.cross_relation.in_proj_weight.grad.abs().sum()), 0.0)

    def test_candidate_pairing_permutation_is_supported(self):
        torch.manual_seed(4)
        model = PairedIncrementalRelationNetwork(
            num_layers=4, topk=2, model_dim=16, heads=4, frame_depth=1,
        ).eval()
        normal = torch.rand(5, 4, 16)
        pseudo = torch.rand(5, 4, 16)
        direct = model(normal, pseudo)
        permuted = model(normal, pseudo, candidate_permutation=torch.arange(4, -1, -1))
        self.assertEqual(direct.shape, permuted.shape)
        self.assertFalse(torch.equal(direct, permuted))

    def test_discrepancy_maps_keep_every_patch(self):
        teacher = [torch.randn(2, 9, 8) for _ in range(3)]
        student = [value + 0.1 * torch.randn_like(value) for value in teacher]
        maps = discrepancy_maps(teacher, student)
        self.assertEqual(maps.shape, (2, 3, 9))
        self.assertTrue(torch.isfinite(maps).all())

    def test_rollout_gain_uses_only_the_common_anchor(self):
        improvements = {
            1: torch.tensor([9.0, 0.30, 0.50, 0.70]),
            2: torch.tensor([0.10, 0.20, 0.30]),
            3: torch.tensor([0.20, 0.40]),
        }
        gains = adjacent_rollout_gains(improvements, (1, 2))
        # The opened layer's own 9.0 improvement must not credit the L1 action.
        self.assertTrue(torch.allclose(gains, torch.tensor([0.30, -0.05])))

    def test_rejects_non_square_patch_grid(self):
        model = PairedIncrementalRelationNetwork(
            num_layers=4, topk=2, model_dim=16, heads=4,
        )
        maps = torch.rand(3, 4, 15)
        with self.assertRaisesRegex(ValueError, 'square grid'):
            model(maps, maps)


if __name__ == '__main__':
    unittest.main()
