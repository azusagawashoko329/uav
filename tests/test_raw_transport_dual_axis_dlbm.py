import unittest

import torch

from experiment_DLBM.raw_transport_dual_axis_dlbm import RawTransportDualAxisDLBM


class RawTransportDualAxisDLBMTest(unittest.TestCase):
    def test_raw_patch_order_and_critic_paths_backpropagate(self):
        model = RawTransportDualAxisDLBM(
            topk=3, channels=8, model_dim=16, heads=4, num_layers=6,
        )
        normal = torch.rand(5, 6, 16)
        pseudo = torch.rand(5, 6, 16)
        first, second = torch.triu_indices(5, 5, offset=1)
        base, opened = model.order_logits(
            normal, (4, 5), (2, 3, 4, 5), first, second,
        )
        normal_critic = model.critic_logits(normal, (4, 5), (2, 3, 4, 5))
        pseudo_critic = model.critic_logits(pseudo, (4, 5), (2, 3, 4, 5))
        loss = model.multitask_loss(
            base.square().mean() + opened.square().mean(),
            normal_critic.square().mean() + pseudo_critic.square().mean(),
        )
        loss.backward()
        self.assertEqual(base.shape, first.shape)
        self.assertEqual(normal_critic.shape, (5,))
        self.assertIsNotNone(model.patch_encoder[0].weight.grad)
        self.assertIsNotNone(model.order_head[-1].weight.grad)
        self.assertIsNotNone(model.branch_critic[-1].weight.grad)
        self.assertIsNotNone(model.axis_log_variance.grad)

    def test_layer_pool_keeps_explicit_raw_topk_path(self):
        model = RawTransportDualAxisDLBM(
            topk=2, channels=4, model_dim=8, heads=2, num_layers=4,
        )
        maps = torch.rand(3, 4, 9, requires_grad=True)
        # A LayerNorm output has an exactly/approximately zero channel sum;
        # use a non-degenerate scalar objective for the gradient-path check.
        model.group_tokens(maps, (1, 2, 3)).square().mean().backward()
        self.assertGreater(float(maps.grad.abs().sum()), 0.0)

    def test_video_consistency_has_a_learned_task_scale(self):
        model = RawTransportDualAxisDLBM(
            topk=2, channels=4, model_dim=8, heads=2, num_layers=4,
            video_consistency=True,
        )
        loss = model.multitask_loss(
            torch.tensor(0.4, requires_grad=True),
            torch.tensor(0.5, requires_grad=True),
            torch.tensor(0.2, requires_grad=True),
        )
        loss.backward()
        self.assertIsNotNone(model.consistency_log_variance.grad)


if __name__ == '__main__':
    unittest.main()
