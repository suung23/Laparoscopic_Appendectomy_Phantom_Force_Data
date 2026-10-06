import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "newton_cell"))
from scene import effective_contact_mu


class NewtonContactMaterialTests(unittest.TestCase):
    def test_default_uses_authored_tissue_friction(self):
        self.assertEqual(effective_contact_mu(None, 0.6), 0.6)

    def test_zero_override_sets_zero_for_both_soft_and_shape_sides(self):
        self.assertEqual(effective_contact_mu(0.0, 0.6), 0.0)

    def test_nonzero_override_is_finite_and_nonnegative(self):
        self.assertEqual(effective_contact_mu(0.25, 0.6), 0.25)

    def test_invalid_override_fails_closed(self):
        for value in (-0.1, np.nan, np.inf, -np.inf):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "contact mu"):
                effective_contact_mu(value, 0.6)


if __name__ == "__main__":
    unittest.main()
