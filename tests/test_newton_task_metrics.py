import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "newton_cell"))
from task_metrics import deformation_health, resection_observations, tissue_probe


class NewtonTaskMetricTests(unittest.TestCase):
    def test_reports_midbody_grasp_and_live_resection_segment_without_claiming_cut(self):
        handles = np.array([[x, 0.0, 0.0] for x in (.00, .02, .04, .06, .08)])
        vertices = np.array([[x, y, 0.0]
                             for x in np.linspace(.04, .08, 9)
                             for y in (-.002, 0.0, .002)])
        out = resection_observations(
            vertices, handles, 2, np.array([.08, 0., 0.]),
            np.array([.05, 0., 0.]))

        self.assertEqual(out["grasp_frac"], .5)
        self.assertTrue(out["grasp_in_band"])
        self.assertGreater(out["resect_seg_n"], 0)
        self.assertAlmostEqual(out["right_tip_to_resect_m"], 0.)
        self.assertFalse(out["cut_done"])
        self.assertFalse(out["appendix_severed"])

    def test_probe_requires_tissue_to_cross_both_jaw_sides(self):
        axis = np.array([0., 0., -1.])
        jaw_axis = np.array([1., 1., 0.]) / np.sqrt(2.)
        flange = np.eye(3)
        n = jaw_axis
        u = np.cross(axis, n)
        points = np.array([
            .010 * (-axis) + su * u
            for su in (-.003, -.002, -.001, .001, .002, .003)
        ])

        inside, thickness, _ = tissue_probe(
            points, np.zeros(3), axis, flange, .55, jaw_axis)
        one_side, _, _ = tissue_probe(
            points[3:], np.zeros(3), axis, flange, .55, jaw_axis)

        self.assertTrue(inside)
        self.assertGreater(thickness, 0.)
        self.assertFalse(one_side)

    def test_deformation_health_detects_inversion_and_nonfinite_state(self):
        rest = np.array([[0., 0., 0.], [1., 0., 0.],
                         [0., 1., 0.], [0., 0., 1.]])
        tets = np.array([[0, 1, 2, 3]])
        velocity = np.zeros_like(rest)
        inverted = rest.copy()
        inverted[3, 2] = -1.

        result = deformation_health(inverted, velocity, rest, tets)
        self.assertTrue(result["physics_state_finite"])
        self.assertEqual(result["tissue_inverted_tets"], 1)
        self.assertAlmostEqual(result["tissue_min_signed_volume_ratio"], -1.)

        broken = rest.copy()
        broken[2, 0] = np.nan
        result = deformation_health(broken, velocity, rest, tets)
        self.assertFalse(result["physics_state_finite"])
        self.assertEqual(result["tissue_nonfinite_tets"], 1)


if __name__ == "__main__":
    unittest.main()
