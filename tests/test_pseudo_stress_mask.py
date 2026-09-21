import random
import unittest

import torch

from experiment_DLBM.common import multi_mode_pseudo_stress


class PseudoStressMaskTest(unittest.TestCase):
    def test_soft_mask_matches_changed_region_and_legacy_return(self):
        random.seed(42); torch.manual_seed(42)
        source = torch.randn(3, 3, 64, 64)
        stressed, mask = multi_mode_pseudo_stress(source, return_mask=True)
        self.assertEqual(mask.shape, (3, 64, 64))
        self.assertTrue(bool((mask >= 0).all() and (mask <= 1).all()))
        self.assertTrue(bool((mask.flatten(1).sum(1) > 0).all()))
        changed = (stressed - source).abs().sum(dim=1) > 1e-7
        self.assertTrue(bool((changed <= (mask > 0)).all()))
        random.seed(42); torch.manual_seed(42)
        legacy = multi_mode_pseudo_stress(source)
        self.assertTrue(torch.allclose(stressed, legacy))


if __name__ == '__main__':
    unittest.main()
