import sys
import unittest
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ct_appendectomy"))
from newton_contact_probe import (inverted_tetrahedron_indices,
                                  contact_probe_tool_positions,
                                  offset_contact_probe_pose,
                                  isotropic_lame_parameters)
from tetra_cut import signed_volumes


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

    def test_reports_original_tetrahedron_id_when_orientation_flips(self):
        rest = [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
                [2., 0., 0.], [2., 1., 0.], [2., 0., 1.]]
        cells = [[0, 1, 2, 3], [1, 4, 5, 6]]
        current = [row[:] for row in rest]
        current[3] = [0., 0., -1.]
        self.assertEqual(inverted_tetrahedron_indices(rest, current, cells).tolist(), [0])

    def test_tool_pose_has_requested_surface_clearance(self):
        center, approach = offset_contact_probe_pose([1., 2., 3.], [0., 0., 4.])
        np.testing.assert_allclose(center, [1., 2., 3.008])
        np.testing.assert_allclose(approach, [0., 0., -1.])

    def test_rejects_invalid_surface_normal(self):
        with self.assertRaises(ValueError):
            offset_contact_probe_pose([0., 0., 0.], [0., 0., 0.])

    def test_contact_probe_positions_preserve_clearance_from_whole_surface(self):
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.1)
        mesh.apply_translation([0.15, 0., 0.])
        points = np.vstack((mesh.vertices, mesh.vertices.mean(axis=0)))
        center_id = len(points) - 1
        tetrahedra = np.column_stack((mesh.faces, np.full(len(mesh.faces), center_id)))
        reversed_tetrahedra = signed_volumes(points, tetrahedra) < 0
        tetrahedra[reversed_tetrahedra, :2] = tetrahedra[reversed_tetrahedra, 1::-1]
        centers, _, clearances = contact_probe_tool_positions(
            points, [0., 0., 0.], tetrahedra, band_tolerance=0.04)
        self.assertEqual(centers.shape, (2, 3))
        self.assertTrue(np.all(clearances >= 0.005 - 1e-5))


if __name__ == "__main__":
    unittest.main()
