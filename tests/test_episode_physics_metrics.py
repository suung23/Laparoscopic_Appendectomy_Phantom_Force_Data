import math
import unittest

from ct_appendectomy.episode_physics_metrics import EpisodePhysicsMetrics


def sample(pin, tracking, displacement, ratio, p99, inverted):
    return {
        "pin_drift_m": pin,
        "tracking_error_m": tracking,
        "maximum_displacement_m": displacement,
        "tissue_health": {
            "minimum_signed_volume_ratio": ratio,
            "p99_absolute_volume_change": p99,
            "inverted_tetrahedra": inverted,
        },
    }


class EpisodePhysicsMetricsTests(unittest.TestCase):
    def test_aggregates_worst_values_and_keeps_force_unmeasured(self):
        metrics = EpisodePhysicsMetrics()
        metrics.update(sample(.0002, .001, .002, .91, .08, 0))
        metrics.update(sample(.0007, .0004, .013, -.2, .3, 2))
        result = metrics.as_dict()
        self.assertEqual(result["samples"], 2)
        self.assertAlmostEqual(result["max_pin_drift_m"], .0007)
        self.assertAlmostEqual(result["max_tracking_error_m"], .001)
        self.assertAlmostEqual(result["max_node_displacement_m"], .013)
        self.assertAlmostEqual(result["minimum_signed_volume_ratio"], -.2)
        self.assertEqual(result["max_inverted_tetrahedra"], 2)
        self.assertEqual(result["inversion_steps"], 1)
        self.assertFalse(result["force_readback_available"])

    def test_rejects_missing_or_nonfinite_measurements(self):
        metrics = EpisodePhysicsMetrics()
        bad = sample(0, 0, 0, 1, 0, 0)
        bad["tracking_error_m"] = math.nan
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            metrics.update(bad)
        self.assertEqual(metrics.samples, 0)


if __name__ == "__main__":
    unittest.main()
