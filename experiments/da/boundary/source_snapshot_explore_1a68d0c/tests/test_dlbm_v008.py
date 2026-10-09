import unittest

import torch

from experiment_DLBM.dlbm_v008 import NeuralOpeningOrder, cumulative_coarse_policy


class DLBMV008Test(unittest.TestCase):
    def test_group_transition_preserves_separate_axes_and_backpropagates(self):
        torch.manual_seed(3)
        model = NeuralOpeningOrder(model_dim=16, heads=4, topk=5)
        maps = torch.rand(8, 12, 196)
        base, candidate = model.encode_transition(maps, range(8, 12), range(4, 12))
        self.assertEqual(base.shape, (8, 16))
        self.assertEqual(candidate.shape, (8, 16))
        first = torch.tensor([0, 1, 2, 3])
        second = torch.tensor([4, 5, 6, 7])
        order = model.order_scores(base, candidate)
        flip = model.flip_logits(base, candidate, first, second)
        (order[0].mean() + order[1].mean() + flip.mean()).backward()
        self.assertGreater(float(model.token_encoder[0].weight.grad.abs().sum()), 0.0)

    def test_base_must_be_strict_candidate_subset(self):
        model = NeuralOpeningOrder(model_dim=16, heads=4)
        with self.assertRaises(ValueError):
            model.encode_transition(torch.rand(4, 12, 196), [8, 9], [8, 10])

    def test_cumulative_policy_can_choose_each_legal_coarse_boundary(self):
        self.assertEqual(cumulative_coarse_policy(torch.tensor(-1.), torch.tensor(-1.))[0], 8)
        self.assertEqual(cumulative_coarse_policy(torch.tensor(1.), torch.tensor(-2.))[0], 4)
        self.assertEqual(cumulative_coarse_policy(torch.tensor(-1.), torch.tensor(3.))[0], 0)


if __name__ == '__main__':
    unittest.main()
