import sys
import unittest
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "newton_cell"))
from appearance import loop_subdivide_surface


class LoopSurfaceTests(unittest.TestCase):
    def setUp(self):
        mesh = trimesh.creation.icosphere(subdivisions=1, radius=.01)
        self.vertices = np.asarray(mesh.vertices, np.float64)
        self.faces = np.asarray(mesh.faces, np.int32)

    def test_one_level_refines_closed_surface_and_preserves_affine_stencils(self):
        v, f, indices, weights, report = loop_subdivide_surface(
            self.vertices, self.faces, levels=1)
        rebuilt = np.einsum("nk,nkj->nj", weights,
                             self.vertices[indices], optimize=True)
        np.testing.assert_allclose(v, rebuilt, atol=1e-12)
        self.assertEqual(len(f), 4 * len(self.faces))
        self.assertEqual(len(v), len(self.vertices) + len(self.faces) // 2 * 3)
        self.assertTrue(trimesh.Trimesh(vertices=v, faces=f,
                                        process=False).is_watertight)
        np.testing.assert_allclose(weights.sum(axis=1), 1., atol=1e-12)
        self.assertFalse(report["simulation_geometry_changed"])
        self.assertFalse(report["contact_geometry_changed"])

    def test_zero_level_is_identity_and_keeps_existing_topology(self):
        v, f, indices, weights, report = loop_subdivide_surface(
            self.vertices, self.faces, levels=0)
        np.testing.assert_array_equal(v, self.vertices)
        np.testing.assert_array_equal(f, self.faces)
        np.testing.assert_array_equal(indices[:, 0], np.arange(len(v)))
        np.testing.assert_array_equal(weights[:, 0], np.ones(len(v)))
        self.assertEqual(report["render_faces"], len(self.faces))

    def test_rejects_open_nonmanifold_and_invalid_level(self):
        with self.assertRaisesRegex(ValueError, "closed 2-manifold"):
            loop_subdivide_surface(self.vertices, self.faces[:-1], levels=1)
        with self.assertRaisesRegex(ValueError, "integer in \\[0, 3\\]"):
            loop_subdivide_surface(self.vertices, self.faces, levels=4)


if __name__ == "__main__":
    unittest.main()
