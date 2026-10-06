import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "newton_cell"))
from render import vertex_normals


class VertexNormalTests(unittest.TestCase):
    def test_planar_mesh_returns_consistent_unit_normals(self):
        vertices = np.array([[0., 0., 0.], [1., 0., 0.],
                             [1., 1., 0.], [0., 1., 0.]])
        faces = np.array([[0, 1, 2], [0, 2, 3]])

        normals = vertex_normals(vertices, faces)

        np.testing.assert_allclose(normals, np.tile([0., 0., 1.], (4, 1)))

    def test_curved_mesh_averages_adjacent_face_directions(self):
        vertices = np.array([[0., 0., 0.], [1., 0., 0.],
                             [0., 1., 0.], [0., 0., 1.]])
        faces = np.array([[0, 1, 2], [0, 3, 1]])

        normals = vertex_normals(vertices, faces)

        self.assertGreater(normals[0, 1], 0.0)
        self.assertGreater(normals[0, 2], 0.0)
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0)


if __name__ == "__main__":
    unittest.main()
