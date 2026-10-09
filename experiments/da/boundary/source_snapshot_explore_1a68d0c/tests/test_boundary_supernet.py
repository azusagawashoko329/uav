import unittest

import torch

from model.boundary_supernet import (all_boundary_topk_scores, build_fixed_suffix_ocbe,
                                     build_fixed_suffix_student, suffix_code_from_all_layer_ocbe)
from model.bottleneck import build_bottleneck
from model.loss import per_layer_topk_dist
from model.student import build_student


class BoundarySupernetTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        self.layers = 6
        self.taps = [torch.randn(2, 9, 16) for _ in range(self.layers)]
        self.ocbe = build_bottleneck(
            n_taps=self.layers, code_dim=4, embed_dim=16, fuse_hidden=12,
        ).eval()
        self.student = build_student(
            n_taps=self.layers, code_dim=4, embed_dim=16, depth=6, heads=4,
            tap_indices=list(range(self.layers)), tap_align='absolute', teacher_depth=self.layers,
        ).eval()

    def test_sliced_ocbe_matches_all_layer_column_restriction(self):
        for boundary in (0, 2, 5):
            fixed = build_fixed_suffix_ocbe(self.ocbe, boundary).eval()
            expected = fixed(self.taps[boundary:])
            actual = suffix_code_from_all_layer_ocbe(self.ocbe, self.taps, boundary)
            self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-5))

    def test_fixed_suffix_student_matches_selected_supernet_outputs(self):
        for boundary in (0, 2, 5):
            code = suffix_code_from_all_layer_ocbe(self.ocbe, self.taps, boundary)
            full_outputs = self.student(code)[boundary:]
            fixed = build_fixed_suffix_student(self.student, boundary).eval()
            fixed_outputs = fixed(code)
            self.assertEqual(len(full_outputs), len(fixed_outputs))
            self.assertTrue(all(torch.allclose(a, b, atol=1e-6, rtol=1e-5)
                                for a, b in zip(full_outputs, fixed_outputs)))

    def test_batched_boundary_scores_match_individual_fixed_suffixes(self):
        actual, all_layer = all_boundary_topk_scores(self.ocbe, self.student, self.taps, topk=3)
        expected = []
        for boundary in range(self.layers):
            fixed_ocbe = build_fixed_suffix_ocbe(self.ocbe, boundary).eval()
            fixed_student = build_fixed_suffix_student(self.student, boundary).eval()
            reconstructions = fixed_student(fixed_ocbe(self.taps[boundary:]))
            layer_scores = per_layer_topk_dist(
                self.taps[boundary:], reconstructions, k=3,
            )
            expected.append(layer_scores.mean(dim=1))
            if boundary == 0:
                self.assertTrue(torch.allclose(all_layer, layer_scores, atol=1e-6, rtol=1e-5))
        self.assertTrue(torch.allclose(actual, torch.stack(expected, dim=1), atol=1e-6, rtol=1e-5))

    def test_invalid_relative_student_is_rejected(self):
        relative = build_student(
            n_taps=self.layers, code_dim=4, embed_dim=16, depth=6, heads=4,
            tap_indices=list(range(self.layers)), tap_align='relative', teacher_depth=self.layers,
        )
        with self.assertRaisesRegex(ValueError, 'absolute'):
            all_boundary_topk_scores(self.ocbe, relative, self.taps, topk=3)


if __name__ == '__main__':
    unittest.main()
