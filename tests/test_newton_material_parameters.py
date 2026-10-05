import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ct_appendectomy"))
from newton_contact_probe import isotropic_lame_parameters


class NewtonMaterialParameterTests(unittest.TestCase):
    def test_lame_parameters_match_isotropic_mapping(self):
        mu, lam = isotropic_lame_parameters(3294.4, 0.45)
        self.assertAlmostEqual(mu, 1136.0, places=1)
        self.assertAlmostEqual(lam, 10224.0, places=0)

    def test_rejects_nonphysical_values(self):
        for young, poisson in ((0, 0.45), (-1, 0.3), (1000, 0.5),
                               (1000, -0.1), (float("nan"), 0.3)):
            with self.subTest(young=young, poisson=poisson):
                with self.assertRaises(ValueError):
                    isotropic_lame_parameters(young, poisson)


if __name__ == "__main__":
    unittest.main()
