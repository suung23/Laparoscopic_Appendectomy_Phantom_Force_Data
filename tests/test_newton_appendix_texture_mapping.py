import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "newton_cell"))
from appearance import (appendix300_texture_provenance, appendix_texture_uv,
                        appendix_texture_vertex_colors)


class NewtonAppendixTextureMappingTests(unittest.TestCase):
    def setUp(self):
        # A short cylindrical FEM rest surface aligned with the principal axis.
        points = []
        for z in np.linspace(-0.02, 0.02, 7):
            for theta in np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False):
                points.append([0.004 * np.cos(theta), 0.004 * np.sin(theta), z])
        self.rest = np.asarray(points, np.float64)
        yy, xx = np.mgrid[0:32, 0:64]
        # Horizontal red gradient makes circumferential wrap/sampling observable.
        self.rgb = np.stack([xx * 4, yy * 8, np.full_like(xx, 80)], axis=-1).clip(0, 255).astype(np.uint8)
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "candidate.png"
        Image.fromarray(self.rgb).save(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_maps_local_rgb_candidate_to_one_color_per_fem_node(self):
        mapped = appendix_texture_vertex_colors(self.rest, self.path)
        self.assertEqual(mapped.shape, self.rest.shape)
        self.assertTrue(np.isfinite(mapped).all())
        self.assertGreater(float(mapped[:, 0].std()), 0.05)
        self.assertGreater(float(mapped[:, 1].std()), 0.05)
        self.assertTrue((mapped >= 0.0).all() and (mapped <= 1.0).all())

    def test_zero_mix_preserves_captured_linear_vertex_colors(self):
        base = np.tile([0.12, 0.24, 0.36], (len(self.rest), 1)).astype(np.float32)
        mapped = appendix_texture_vertex_colors(self.rest, self.path,
                                                base_colors=base, mix=0.0)
        np.testing.assert_allclose(mapped, base, atol=1e-7)

    def test_colors_are_computed_from_rest_points_and_invalid_mix_fails(self):
        first = appendix_texture_vertex_colors(self.rest, self.path)
        moved = self.rest.copy()
        moved[:, 0] += np.linspace(0.0, 0.01, len(moved))
        second = appendix_texture_vertex_colors(moved, self.path)
        self.assertFalse(np.allclose(first, second))
        with self.assertRaisesRegex(ValueError, "mix"):
            appendix_texture_vertex_colors(self.rest, self.path, mix=1.1)

    def test_rest_uvs_cover_the_circumference_and_longitudinal_axis(self):
        uv = appendix_texture_uv(self.rest)
        self.assertEqual(uv.shape, (len(self.rest), 2))
        self.assertTrue(np.isfinite(uv).all())
        self.assertTrue(((uv >= 0.0) & (uv <= 1.0)).all())
        self.assertGreater(float(uv[:, 0].std()), 0.2)
        self.assertGreater(float(uv[:, 1].std()), 0.2)
        np.testing.assert_allclose(uv, appendix_texture_uv(self.rest.copy()))

    def test_uv_mapping_rejects_degenerate_or_nonfinite_rest_surface(self):
        with self.assertRaisesRegex(ValueError, "longitudinal"):
            appendix_texture_uv(np.zeros((8, 3)))
        points = self.rest.copy()
        points[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            appendix_texture_uv(points)

    def test_texture_requires_matching_appendix300_provenance_hash(self):
        import hashlib
        import json

        digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        manifest = {
            "status": "provisional_prompt_mask_texture_candidate_not_promoted",
            "dataset": "Appendix300", "license": "CC BY 4.0",
            "doi": "10.25532/OPARA-1173",
            "source_video_sha256": "video-hash",
            "source_frame_sha256": "frame-hash",
            "mask": {"mask_sha256": "mask-hash"},
            "texture": {"texture_sha256": digest},
        }
        (self.path.parent / "appearance.json").write_text(json.dumps(manifest),
                                                           encoding="utf-8")
        verified = appendix300_texture_provenance(self.path)
        self.assertEqual(verified["source_doi"], "10.25532/OPARA-1173")
        self.assertEqual(verified["source_texture_sha256"], digest)
        self.assertEqual(verified["source_mask_sha256"], "mask-hash")
        self.assertEqual(verified["source_status"],
                         "provisional_prompt_mask_texture_candidate_not_promoted")
        manifest["doi"] = "10.0000/not-appendix300"
        (self.path.parent / "appearance.json").write_text(json.dumps(manifest),
                                                           encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "DOI"):
            appendix300_texture_provenance(self.path)
        manifest["doi"] = "10.25532/OPARA-1173"
        (self.path.parent / "appearance.json").write_text(json.dumps(manifest),
                                                           encoding="utf-8")
        self.path.write_bytes(self.path.read_bytes() + b"tampered")
        with self.assertRaisesRegex(ValueError, "hash"):
            appendix300_texture_provenance(self.path)


if __name__ == "__main__":
    unittest.main()
