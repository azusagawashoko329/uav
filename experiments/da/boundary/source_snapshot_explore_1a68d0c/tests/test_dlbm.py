import unittest

import torch
import torch.nn.functional as F

from experiment_DLBM.common import cross_frame_cutpaste, policy_from_logits
from model.dlbm import DistillationLayerBoundaryModule
from model.student import StudentDecoder


def feature_pairs(batch=3, layers=6, patches=9, dim=16):
    teacher = [torch.randn(batch, patches, dim) for _ in range(layers)]
    student = [t + 0.2 * torch.randn_like(t) for t in teacher]
    return teacher, student


class DLBMTest(unittest.TestCase):
    def test_boundary_output_is_a_legal_monotonic_suffix(self):
        model = DistillationLayerBoundaryModule(
            num_layers=6, feature_dim=16, model_dim=16, depth=1, heads=4,
        )
        teacher, student = feature_pairs()
        output = model(teacher, student)
        self.assertEqual(output.logits.shape, (3, 6))
        self.assertEqual(output.gates.shape, (3, 6))
        self.assertTrue(torch.all(output.gates[:, 1:] >= output.gates[:, :-1]))
        self.assertTrue(torch.allclose(output.gates[:, -1], torch.ones(3), atol=1e-6))
        self.assertTrue(torch.all((0 <= output.hard_boundary) & (output.hard_boundary < 6)))

    def test_scene_reduction_and_ranking_objective_backpropagate_to_map_encoder(self):
        model = DistillationLayerBoundaryModule(
            num_layers=6, feature_dim=16, model_dim=16, depth=1, heads=4,
        )
        teacher, student = feature_pairs()
        output = model(teacher, student, reduce='scene')
        normal = torch.tensor([[0.30, 0.28, 0.25, 0.22, 0.20, 0.18]])
        counterfactual = torch.tensor([[0.31, 0.30, 0.32, 0.40, 0.55, 0.75]])
        loss = F.softplus(
            model.soft_suffix_score(normal, output.gates)
            - model.soft_suffix_score(counterfactual, output.gates)
        ).mean()
        loss.backward()
        self.assertIsNotNone(model.boundary_head.weight.grad)
        self.assertGreater(float(model.boundary_head.weight.grad.abs().sum()), 0)
        self.assertIsNotNone(model.feature_proj[1].weight.grad)
        self.assertGreater(float(model.feature_proj[1].weight.grad.abs().sum()), 0)

    def test_policy_serialization_uses_argmax_boundary_and_suffix(self):
        logits = torch.tensor([-3.0, -2.0, 4.0, 0.0])
        policy = policy_from_logits(logits)
        self.assertEqual(policy['boundary'], 2)
        self.assertEqual(policy['selected_taps'], [2, 3])
        self.assertAlmostEqual(policy['soft_gates'][-1], 1.0, places=6)
        self.assertTrue(0 <= policy['normalized_entropy'] <= 1)

    def test_legacy_relative_student_forward_semantics_are_unchanged(self):
        model = StudentDecoder(
            n_taps=3, code_dim=8, embed_dim=16, depth=6, heads=4,
            tap_indices=[5, 8, 11], tap_align='relative',
        ).eval()
        code = torch.randn(2, 7, 8)
        actual = model(code)
        x = model.proj_in(code)
        expected = []
        head = 0
        for block_index, block in enumerate(model.blocks):
            x = block(x)
            if block_index in {1, 3, 5}:
                expected.append(model.heads[head](x))
                head += 1
        expected = expected[::-1]
        self.assertEqual(model.recon_at, [5, 3, 1])
        self.assertTrue(all(torch.allclose(a, b) for a, b in zip(actual, expected)))

    def test_absolute_alignment_preserves_layer_positions_across_suffixes(self):
        full = StudentDecoder(
            n_taps=12, code_dim=8, embed_dim=16, depth=12, heads=4,
            tap_indices=list(range(12)), tap_align='absolute',
        )
        suffix = StudentDecoder(
            n_taps=4, code_dim=8, embed_dim=16, depth=12, heads=4,
            tap_indices=[8, 9, 10, 11], tap_align='absolute',
        )
        self.assertEqual(suffix.recon_at, full.recon_at[8:])
        self.assertEqual(suffix.recon_at, [3, 2, 1, 0])

        late_block_was_called = False
        def mark_late_block(_module, _inputs, _output):
            nonlocal late_block_was_called
            late_block_was_called = True
        handle = suffix.blocks[4].register_forward_hook(mark_late_block)
        outputs = suffix(torch.randn(1, 7, 8))
        handle.remove()
        self.assertEqual(len(outputs), 4)
        self.assertFalse(late_block_was_called)

    def test_cutpaste_is_shape_preserving_and_changes_pixels(self):
        torch.manual_seed(3)
        batch = torch.arange(2 * 3 * 16 * 16, dtype=torch.float32).reshape(2, 3, 16, 16)
        output = cross_frame_cutpaste(batch)
        self.assertEqual(output.shape, batch.shape)
        self.assertFalse(torch.equal(output, batch))

    def test_dlbm_rejects_wrong_layer_count(self):
        model = DistillationLayerBoundaryModule(
            num_layers=6, feature_dim=16, model_dim=16, depth=1, heads=4,
        )
        teacher, student = feature_pairs(layers=5)
        with self.assertRaisesRegex(ValueError, 'expects 6'):
            model(teacher, student)


if __name__ == '__main__':
    unittest.main()
