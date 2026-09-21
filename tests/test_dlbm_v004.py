import unittest

import torch

from experiment_DLBM.dlbm_v004 import SharedLayerPatchLocalizer
from experiment_DLBM.mask_localization_targets_v004 import (cutpaste_with_mask,
                                                             binary_auc,
                                                             patch_discrepancy_maps,
                                                             pixel_to_patch_mask)


class DLBMV004Test(unittest.TestCase):
    def test_shared_localizer_supports_mixed_suffixes_and_backpropagates(self):
        model = SharedLayerPatchLocalizer(num_layers=12, model_dim=8, heads=2)
        maps = torch.rand(3, 12, 16, requires_grad=True)
        valid = torch.zeros(3, 12, dtype=torch.bool)
        valid[0, 0:] = True; valid[1, 4:] = True; valid[2, 11:] = True
        logits = model(maps, valid)
        self.assertEqual(tuple(logits.shape), (3, 16))
        logits.mean().backward()
        self.assertTrue(torch.isfinite(maps.grad).all())
        self.assertGreater(float(maps.grad.abs().sum()), 0)

    def test_padded_invalid_values_do_not_affect_output(self):
        torch.manual_seed(1)
        model = SharedLayerPatchLocalizer(num_layers=12, model_dim=8, heads=2).eval()
        maps = torch.rand(2, 12, 9)
        valid = torch.zeros(2, 12, dtype=torch.bool); valid[:, 7:] = True
        changed = maps.clone(); changed[:, :7] = 1000
        torch.testing.assert_close(model(maps, valid), model(changed, valid), atol=1e-6, rtol=1e-6)

    def test_cutpaste_mask_and_patch_projection_are_valid(self):
        torch.manual_seed(2)
        images = torch.rand(2, 3, 32, 32)
        pseudo, mask = cutpaste_with_mask(images, 0.05, 0.1)
        self.assertEqual(pseudo.shape, images.shape)
        self.assertEqual(tuple(mask.shape), (2, 1, 32, 32))
        self.assertTrue(bool((mask.sum(dim=(1, 2, 3)) > 0).all()))
        patch_mask = pixel_to_patch_mask(mask, 16)
        self.assertEqual(tuple(patch_mask.shape), (2, 16))
        self.assertTrue(bool(((patch_mask >= 0) & (patch_mask <= 1)).all()))

    def test_discrepancy_maps_preserve_absolute_suffix_positions(self):
        teacher = [torch.randn(2, 9, 4) for _ in range(4)]
        student = [value + 0.1 * torch.randn_like(value) for value in teacher]
        maps, valid = patch_discrepancy_maps(teacher, student, boundary=8)
        self.assertEqual(tuple(maps.shape), (2, 12, 9))
        self.assertEqual(tuple(valid.shape), (2, 12))
        self.assertEqual(float(maps[:, :8].abs().sum()), 0.0)
        self.assertTrue(bool(valid[:, 8:].all()))
        self.assertFalse(bool(valid[:, :8].any()))

    def test_binary_auc_flattens_frame_patch_axes(self):
        labels = torch.tensor([[False, True], [False, True]])
        scores = torch.tensor([[0.0, 1.0], [0.1, 0.9]])
        self.assertEqual(binary_auc(labels, scores), 1.0)


if __name__ == '__main__':
    unittest.main()
