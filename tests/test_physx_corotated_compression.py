import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "local_validation"))
from physx_corotated_compression_audit import audit, lame_from_g_nu, shear_tangent


class PhysXCorotatedCompressionTests(unittest.TestCase):
    def test_uncompressed_shear_tangent_matches_baseline_modulus(self):
        mu, lame = lame_from_g_nu(1136., .45)
        row = shear_tangent(0., mu, lame)
        self.assertAlmostEqual(row["current_cauchy_shear_tangent_pa"], 1136., places=4)
        self.assertAlmostEqual(row["ratio_to_uncompressed"], 1., places=7)

    def test_free_lateral_condition_and_compression_curve_are_consistent(self):
        mu, lame = lame_from_g_nu(1136., .45)
        rows = [shear_tangent(c, mu, lame) for c in (0., .2, .4)]
        self.assertTrue(all(abs(row["free_lateral_piola_residual_pa"]) < 1e-5 for row in rows))
        self.assertGreater(rows[2]["free_lateral_stretch"], rows[1]["free_lateral_stretch"])
        self.assertGreater(rows[2]["ratio_to_uncompressed"], rows[1]["ratio_to_uncompressed"])
        self.assertAlmostEqual(rows[2]["ratio_to_uncompressed"], 1.99284, places=4)

    def test_physics_analytical_law_still_underpredicts_appendix_group_endpoints(self):
        report = audit()
        self.assertFalse(report["runtime"]["physx_simulation_executed"])
        for group in report["groups"].values():
            self.assertLess(group["analytical_physx_law_factor_at_40pct"],
                            group["reported_factor_at_40pct"])
            self.assertGreater(group["relative_underprediction_of_factor"], .3)

    def test_invalid_inputs_fail_closed(self):
        with self.assertRaises(ValueError):
            lame_from_g_nu(0., .45)
        mu, lame = lame_from_g_nu(1000., .45)
        with self.assertRaises(ValueError):
            shear_tangent(1., mu, lame)


if __name__ == "__main__":
    unittest.main()
