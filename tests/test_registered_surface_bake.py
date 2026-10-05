import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ct_appendectomy"))
from registered_surface_bake import RegisteredView, bake_registered_surface


def fixture(mask=True, depth=1.0, reverse=False, color=(64, 128, 192)):
    vertices = np.array([
        [-0.5, -0.5, 1.0],
        [-0.5, 0.5, 1.0],
        [0.5, 0.5, 1.0],
    ])
    face = np.array([[0, 1, 2]])
    if reverse:
        face = face[:, ::-1]
    uv = np.array([[[0.1, 0.1], [0.1, 0.9], [0.9, 0.9]]])
    image = np.empty((8, 8, 3), dtype=np.uint8)
    image[:] = color
    depth_image = np.full((8, 8), depth, dtype=np.float32)
    tissue_mask = np.full((8, 8), mask, dtype=bool)
    view = RegisteredView(image, depth_image, tissue_mask,
                          np.array([[4., 0., 3.5], [0., 4., 3.5], [0., 0., 1.]]),
                          np.eye(4), "synthetic")
    return vertices, face, uv, view


class RegisteredSurfaceBakeTests(unittest.TestCase):
    def bake(self, args):
        vertices, faces, uv, view = args
        return bake_registered_surface(vertices, faces, uv, [view], atlas_size=64)

    def test_depth_and_mask_registered_observation_is_baked(self):
        result = self.bake(fixture())
        covered = result["coverage"]
        self.assertGreater(covered.sum(), 200)
        expected = np.array([64, 128, 192], dtype=np.float32) / 255.
        np.testing.assert_allclose(result["rgb"][covered], np.broadcast_to(expected, (covered.sum(), 3)), atol=2e-3)
        self.assertEqual(result["report"]["accepted_projection_count_by_view"]["synthetic"], int(covered.sum()))
        self.assertIn("not intrinsic albedo", result["report"]["source_semantics"])

    def test_depth_occlusion_and_non_target_mask_are_rejected(self):
        self.assertEqual(self.bake(fixture(depth=0.5))["report"]["covered_texels"], 0)
        self.assertEqual(self.bake(fixture(mask=False))["report"]["covered_texels"], 0)

    def test_back_faces_are_not_projected(self):
        self.assertEqual(self.bake(fixture(reverse=True))["report"]["covered_texels"], 0)

    def test_two_views_blend_in_linear_light_and_report_overlap(self):
        vertices, faces, uv, first = fixture(color=(0, 0, 0))
        second = RegisteredView(np.full((8, 8, 3), 255, dtype=np.uint8),
                                first.depth_z_m.copy(), first.tissue_mask.copy(),
                                first.intrinsics.copy(), first.camera_from_world.copy(), "white")
        result = bake_registered_surface(vertices, faces, uv, [first, second], atlas_size=64)
        covered = result["coverage"]
        np.testing.assert_allclose(result["rgb"][covered], 0.735357, atol=2e-3)
        self.assertGreater(result["report"]["multi_view_texel_fraction"], 0.05)

    def test_missing_registered_inputs_fail_closed(self):
        vertices, faces, uv, _ = fixture()
        with self.assertRaisesRegex(ValueError, "registered view"):
            bake_registered_surface(vertices, faces, uv, [], atlas_size=64)

    def test_nonrigid_camera_calibration_is_rejected(self):
        vertices, faces, uv, view = fixture()
        bad_transform = view.camera_from_world.copy()
        bad_transform[0, 0] = 1.2
        bad_view = RegisteredView(view.rgb, view.depth_z_m, view.tissue_mask,
                                  view.intrinsics, bad_transform, "bad-camera")
        with self.assertRaisesRegex(ValueError, "orthonormal rotation"):
            bake_registered_surface(vertices, faces, uv, [bad_view], atlas_size=64)


if __name__ == "__main__":
    unittest.main()
