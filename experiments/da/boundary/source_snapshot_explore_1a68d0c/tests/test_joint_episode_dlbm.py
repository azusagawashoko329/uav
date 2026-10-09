import unittest

import torch

from experiment_DLBM.joint_episode_dlbm import JointEpisodeDLBM


class JointEpisodeDLBMTest(unittest.TestCase):
    @staticmethod
    def opening(frames=5, added=4, channels=8, side=4):
        return {
            'base_state': torch.randn(frames, channels, side, side),
            'candidate_state': torch.randn(frames, channels, side, side),
            'deep_state': torch.randn(frames, added, channels, side, side),
            'observed': torch.randn(frames, added, channels, side, side),
            'innovation': torch.randn(frames, added, side, side),
        }

    def test_joint_pair_and_episode_paths_backpropagate(self):
        model = JointEpisodeDLBM(
            predictor_channels=8, topk=3, model_dim=16, heads=4,
        )
        first, second = torch.triu_indices(5, 5, offset=1)
        output = model(self.opening(), self.opening(), first, second)
        self.assertEqual(output['normal_flip'].shape, first.shape)
        self.assertEqual(output['pseudo_flip'].shape, first.shape)
        self.assertEqual(output['joint_pair'].shape, first.shape)
        self.assertEqual(output['episode'].shape, torch.Size([]))
        losses = [value.square().mean() for value in output.values()]
        model.multitask_loss(losses).backward()
        self.assertIsNotNone(model.opening_encoder.response.weight.grad)
        self.assertIsNotNone(model.pair_encoder[1].weight.grad)
        self.assertIsNotNone(model.joint_pair_head[-1].weight.grad)
        self.assertIsNotNone(model.episode_head[-1].weight.grad)
        self.assertIsNotNone(model.task_log_variance.grad)

    def test_joint_target_is_normal_only_migration(self):
        normal = torch.tensor([0.0, 1.0, 1.0, 0.0])
        pseudo = torch.tensor([0.0, 0.0, 1.0, 1.0])
        expected = torch.tensor([0.0, 1.0, 0.0, 0.0])
        torch.testing.assert_close(
            JointEpisodeDLBM.joint_target(normal, pseudo), expected,
        )

    def test_cumulative_joint_evidence_selects_all_boundaries(self):
        for gains, expected in (
            ((-1.0, -1.0), 8), ((1.0, -2.0), 4), ((-1.0, 2.0), 0),
        ):
            boundary, probability, _ = JointEpisodeDLBM.boundary_distribution(
                torch.tensor(gains[0]), torch.tensor(gains[1]),
            )
            self.assertEqual(boundary, expected)
            self.assertAlmostEqual(float(probability.sum()), 1.0, places=6)


if __name__ == '__main__':
    unittest.main()
