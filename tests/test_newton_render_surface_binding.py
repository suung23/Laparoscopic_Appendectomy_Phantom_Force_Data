import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "newton_cell"))
from appearance import bind_surface_vertices_to_fem_nodes


class RenderSurfaceBindingTests(unittest.TestCase):
    def setUp(self):
        self.fem = np.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
        ])

    def test_binds_a_unique_noisy_render_surface_subset(self):
        surface = self.fem[[3, 0, 2]] + np.array([2e-8, -1e-8, 1e-8])

        indices, report = bind_surface_vertices_to_fem_nodes(
            surface, self.fem, max_error_m=1e-6)

        np.testing.assert_array_equal(indices, [3, 0, 2])
        self.assertTrue(report["unique_binding"])
        self.assertEqual(report["render_vertices"], 3)
        self.assertLess(report["max_rest_error_m"], 1e-6)

    def test_rejects_render_surface_outside_rest_tolerance(self):
        surface = self.fem[:3].copy()
        surface[1, 2] = 2e-4

        with self.assertRaisesRegex(ValueError, "max error"):
            bind_surface_vertices_to_fem_nodes(surface, self.fem,
                                               max_error_m=1e-5)

    def test_rejects_non_unique_nearest_node_correspondence(self):
        surface = np.array([[0.0, 0.0, 0.0], [1e-9, 0.0, 0.0]])

        with self.assertRaisesRegex(ValueError, "duplicate"):
            bind_surface_vertices_to_fem_nodes(surface, self.fem,
                                               max_error_m=1e-5)

    def test_rejects_invalid_shape_and_nonfinite_coordinates(self):
        with self.assertRaisesRegex(ValueError, "arrays"):
            bind_surface_vertices_to_fem_nodes(np.zeros((3, 2)), self.fem)
        bad = self.fem.copy()
        bad[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            bind_surface_vertices_to_fem_nodes(self.fem, bad)


if __name__ == "__main__":
    unittest.main()
