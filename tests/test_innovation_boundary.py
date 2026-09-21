import unittest

import torch

from experiment_DLBM.innovation_boundary import InnovationOpeningHead


class InnovationOpeningHeadTest(unittest.TestCase):
    def test_topk_features_remain_separate_and_backpropagate(self):
        torch.manual_seed(12)
        model = InnovationOpeningHead(predictor_channels=8, model_dim=16, topk=5)
        opening = {
            'base_state': torch.rand(4, 8, 16, 16),
            'candidate_state': torch.rand(4, 8, 16, 16),
            'deep_state': torch.rand(4, 3, 8, 16, 16),
            'observed': torch.rand(4, 3, 8, 16, 16),
            'innovation': torch.rand(4, 3, 16, 16),
            'added_layers': [7, 6, 5],
        }
        embedding = model.encode(opening)
        self.assertEqual(embedding['base'].shape, (4, 16))
        self.assertEqual(embedding['candidate'].shape, (4, 16))
        self.assertEqual(embedding['transition'].shape, (4, 16))
        first, second = torch.tensor([0, 1]), torch.tensor([2, 3])
        base, candidate = model.order_scores(embedding)
        loss = base.mean() + candidate.mean() + model.flip_logits(
            embedding, first, second,
        ).mean()
        loss.backward()
        self.assertGreater(float(model.relation[0].weight.grad.abs().sum()), 0.0)


if __name__ == '__main__':
    unittest.main()
