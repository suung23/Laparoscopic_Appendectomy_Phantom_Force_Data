import sys
import json
import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ct_appendectomy"))
from appendix300_video_texture import build_from_verified_frame, suppress_local_specular
from appendix300_temporal_texture import (
    align_dense, deduplicate_odd_window, phase_translation, rebind_reviewed_uv_layer,
    translate,
)


class Appendix300TextureTests(unittest.TestCase):
    def test_phase_correlation_recovers_small_integer_camera_shift(self):
        rng = np.random.default_rng(27)
        source = rng.integers(20, 230, size=(96, 112, 3), dtype=np.uint8)
        reference = Image.fromarray(source)
        moving = Image.fromarray(np.roll(source, shift=(5, -7), axis=(0, 1)))

        dx, dy, ratio = phase_translation(reference, moving)
        recovered = np.asarray(translate(moving, dx, dy))

        self.assertEqual((dx, dy), (7, -5))
        self.assertGreater(ratio, 1.)
        self.assertTrue(np.array_equal(recovered[10:-10, 10:-10], source[10:-10, 10:-10]))

    def test_phase_correlation_rejects_unbounded_alignment(self):
        rng = np.random.default_rng(28)
        source = Image.fromarray(rng.integers(20, 230, size=(96, 112, 3), dtype=np.uint8))
        moving = Image.fromarray(np.roll(np.asarray(source), shift=(31, 0), axis=(0, 1)))
        with self.assertRaisesRegex(ValueError, "exceeds 24px"):
            phase_translation(source, moving)

    def test_duplicate_video_decodes_are_removed_and_window_stays_odd(self):
        records = [
            {"time_seconds": 1.00, "requested_time_seconds": 1.00, "sha256": "a"},
            {"time_seconds": 1.00, "requested_time_seconds": 1.02, "sha256": "a"},
            {"time_seconds": 1.04, "requested_time_seconds": 1.04, "sha256": "b"},
            {"time_seconds": 1.08, "requested_time_seconds": 1.08, "sha256": "c"},
            {"time_seconds": 1.12, "requested_time_seconds": 1.12, "sha256": "d"},
        ]

        selected, report = deduplicate_odd_window(records, 1.06)

        self.assertEqual(len(selected), 3)
        self.assertEqual(len(report["duplicate_decodes"]), 1)
        self.assertEqual(len(report["trimmed_for_odd_window"]), 1)
        self.assertEqual(len({record["sha256"] for record in selected}), 3)

    @unittest.skipUnless(importlib.util.find_spec("scipy") and
                         importlib.util.find_spec("skimage"),
                         "dense optical-flow dependencies are optional")
    def test_dense_flow_reduces_known_image_translation(self):
        from scipy.ndimage import gaussian_filter

        rng = np.random.default_rng(3)
        source = gaussian_filter(rng.normal(size=(128, 128, 3)), sigma=(2, 2, 0))
        source = (255 * (source - source.min()) / (source.max() - source.min())).astype(np.uint8)
        moving = np.roll(source, shift=(3, -5), axis=(0, 1))
        reference = Image.fromarray(source)
        moved = Image.fromarray(moving)

        aligned, valid, record = align_dense(reference, moved)
        before = np.abs(source.astype(float).mean(2) - moving.astype(float).mean(2)).mean()
        after = np.abs(source.astype(float).mean(2) -
                       np.asarray(aligned).astype(float).mean(2))[valid].mean()

        self.assertGreater(float(valid.mean()), 0.75)
        self.assertLess(after, before * 0.1)
        self.assertEqual(record["method"], "skimage_TVL1_dense_flow")

    def test_rebinds_only_reviewed_usd_texture_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = root / "previous"
            out = root / "candidate"
            scene = root / "ct.usda"
            scene.write_text("# scene", encoding="utf-8")
            (root / "fat").mkdir()
            fat_scene = root / "fat" / "review.usda"
            fat_scene.write_text("# fat", encoding="utf-8")
            layer = parent / "appearance.usda"
            parent.mkdir()
            layer.write_text(
                '#usda 1.0\n( subLayers = [ @../ct.usda@, @../fat/review.usda@ ] )\n'
                'inputs:file = @appendix_appendix300_rgb.png@\n'
                'inputs:file = @mesoappendix_appendix300_rgb.png@\n',
                encoding="utf-8")
            report = {
                "dataset": "Appendix300", "license": "CC BY 4.0",
                "geometry_unchanged": True,
                "authored_stage_sha256": hashlib.sha256(layer.read_bytes()).hexdigest(),
                "mapped_meshes": [
                    {"mesh": "/World/Atlas/appendix/render"},
                    {"mesh": "/World/Atlas/mesoappendix/render"},
                ],
                "source_scene_layers": [
                    {"file": "ct.usda", "sha256": hashlib.sha256(scene.read_bytes()).hexdigest()},
                    {"file": "review.usda", "sha256": hashlib.sha256(fat_scene.read_bytes()).hexdigest()},
                ],
            }
            (parent / "appearance.json").write_text(json.dumps(report), encoding="utf-8")
            out.mkdir()
            for name in ("appendix_appendix300_temporal_rgb.png",
                         "mesoappendix_appendix300_temporal_rgb.png"):
                Image.new("RGB", (8, 8), (100, 80, 70)).save(out / name)

            result = rebind_reviewed_uv_layer(layer, out)
            authored = (out / "appearance.usda").read_text(encoding="utf-8")

            self.assertTrue(result["uvs_reused_unchanged"])
            self.assertIn("@appendix_appendix300_temporal_rgb.png@", authored)
            self.assertIn("@mesoappendix_appendix300_temporal_rgb.png@", authored)
            self.assertNotIn("@appendix_appendix300_rgb.png@", authored)

    def test_repairs_compact_neutral_glint_and_preserves_flat_tissue(self):
        rgb = np.empty((48, 48, 3), dtype=np.uint8)
        rgb[:] = (154, 72, 83)
        rgb[21:25, 21:25] = (250, 247, 242)

        repaired, flagged = suppress_local_specular(Image.fromarray(rgb, "RGB"))
        result = np.asarray(repaired)

        self.assertGreater(flagged, 0.0)
        self.assertLess(float(result[22:24, 22:24].mean()), 180.0)
        self.assertTrue(np.array_equal(result[5:15, 5:15], rgb[5:15, 5:15]))

    def test_does_not_remove_uniform_bright_tissue_without_local_contrast(self):
        rgb = np.empty((48, 48, 3), dtype=np.uint8)
        rgb[:] = (210, 125, 135)
        source = Image.fromarray(rgb, "RGB")

        repaired, flagged = suppress_local_specular(source)

        self.assertEqual(flagged, 0.0)
        self.assertTrue(np.array_equal(np.asarray(repaired), rgb))

    def test_rejects_frame_that_does_not_match_parent_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame = root / "frame.png"
            Image.new("RGB", (8, 8), (100, 60, 70)).save(frame)
            report = root / "parent.json"
            report.write_text(json.dumps({
                "dataset": "Appendix300",
                "license": "CC BY 4.0",
                "source_frame_sha256": "0" * 64,
            }), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "frame hash"):
                build_from_verified_frame(frame, report, root / "out")


if __name__ == "__main__":
    unittest.main()
