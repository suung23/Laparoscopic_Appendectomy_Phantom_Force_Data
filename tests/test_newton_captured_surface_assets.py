import sys
import unittest
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "newton_cell"))
from appearance import bind_surface_vertices_to_fem_nodes, loop_subdivide_surface
from layout import CellLayout
from scene import load_tet
from view import tet_surface


class CapturedSurfaceAssetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.layout = CellLayout()
        cls.organs = ("cecum", "bowel1", "bowel2", "bowel3",
                      "stomach", "appendix")

    def test_all_captured_render_surfaces_bind_to_fem_rest_nodes(self):
        for name in self.organs:
            with self.subTest(organ=name):
                captured = trimesh.load(ROOT / "mesh" / f"organ_{name}.ply",
                                        process=False)
                fem_vertices, tets = load_tet(ROOT / "tet" / f"{name}.npz")
                render_world = self.layout.organs_to_world(
                    np.asarray(captured.vertices, np.float64) * 0.001)
                fem_world = self.layout.organs_to_world(fem_vertices)

                indices, report = bind_surface_vertices_to_fem_nodes(
                    render_world, fem_world, max_error_m=1e-5)

                self.assertEqual(len(np.unique(indices)), len(captured.vertices))
                self.assertTrue(captured.is_watertight)
                self.assertTrue(captured.is_winding_consistent)
                self.assertLess(report["max_rest_error_m"], 1e-5)
                tet_facets = {tuple(sorted(face))
                              for face in tet_surface(tets).tolist()}
                mapped_faces = indices[np.asarray(captured.faces, np.int32)]
                matching = sum(tuple(sorted(face)) in tet_facets
                               for face in mapped_faces.tolist())
                self.assertGreaterEqual(matching / len(mapped_faces), 0.95)

                smooth_v, smooth_f, stencil_i, stencil_w, smooth_report = \
                    loop_subdivide_surface(fem_world[indices], captured.faces, 1)
                smooth_mesh = trimesh.Trimesh(vertices=smooth_v, faces=smooth_f,
                                              process=False)
                self.assertTrue(smooth_mesh.is_watertight)
                self.assertTrue(smooth_mesh.is_winding_consistent)
                self.assertEqual(len(smooth_f), len(captured.faces) * 4)
                self.assertEqual(stencil_i.shape, stencil_w.shape)
                self.assertFalse(smooth_report["simulation_geometry_changed"])
                self.assertFalse(smooth_report["contact_geometry_changed"])
                self.assertLess(smooth_report["max_rest_smoothing_displacement_m"], .002)


if __name__ == "__main__":
    unittest.main()
