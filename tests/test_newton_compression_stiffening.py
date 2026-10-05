import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "local_validation"))
from newton_compression_stiffening_audit import (
    DATA,
    audit,
    lame_from_shear_storage_modulus,
    shear_tangent_at_compression,
)


class NewtonCompressionStiffeningTests(unittest.TestCase):
    def test_uncompressed_tangent_recovers_input_shear_modulus(self):
        for values in DATA.values():
            mu, lame = lame_from_shear_storage_modulus(values["storage_modulus_pa"], .45)
            row = shear_tangent_at_compression(0., mu, lame)
            self.assertAlmostEqual(row["current_cauchy_shear_tangent_pa"], mu, places=5)
            self.assertAlmostEqual(row["tangent_ratio_to_uncompressed"], 1., places=8)

    def test_lateral_free_stress_and_current_shear_tangent_are_reported(self):
        mu, lame = lame_from_shear_storage_modulus(1136., .45)
        rows = [shear_tangent_at_compression(c, mu, lame) for c in (0., .2, .4)]
        self.assertTrue(all(abs(row["free_lateral_piola_residual_pa"]) < 1e-5 for row in rows))
        self.assertAlmostEqual(rows[1]["tangent_ratio_to_uncompressed"], 1.25, places=7)
        self.assertAlmostEqual(rows[2]["tangent_ratio_to_uncompressed"], 1. / .6, places=7)

    def test_current_newton_law_underpredicts_published_40pct_ratios(self):
        report = audit()
        self.assertTrue(report["status"].startswith("compression_stiffening_gate_failed"))
        for group in report["groups"].values():
            self.assertAlmostEqual(group["newton_stable_nh_predicted_40pct_factor"], 1. / .6, places=7)
            self.assertLess(group["newton_stable_nh_predicted_40pct_factor"],
                            group["reported_40pct_modulus_factor"])
            self.assertGreater(group["relative_underprediction_of_40pct_factor"], .4)

    def test_invalid_material_and_deformation_inputs_fail_closed(self):
        with self.assertRaises(ValueError):
            lame_from_shear_storage_modulus(0., .45)
        mu, lame = lame_from_shear_storage_modulus(1000., .45)
        with self.assertRaises(ValueError):
            shear_tangent_at_compression(1., mu, lame)


if __name__ == "__main__":
    unittest.main()
