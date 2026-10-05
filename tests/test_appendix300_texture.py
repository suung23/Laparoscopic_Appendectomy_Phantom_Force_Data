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
    align_dense, copy_reviewed_temporal_uv_layer, deduplicate_odd_window,
    fuse_temporal_detail, phase_translation, rebind_reviewed_uv_layer, translate,
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

    @unittest.skipUnless(importlib.util.find_spec("scipy"),
                         "temporal detail fusion requires scipy")
    def test_detail_fusion_restores_supported_reference_detail_and_rejects_transient_glint(self):
        from scipy.ndimage import gaussian_filter

        yy, xx = np.mgrid[:96, :96]
        base = np.stack((120 + xx * .12, 95 + yy * .10, 78 + (xx + yy) * .06), axis=-1)
        detail = 12 * np.sin(xx * .85) + 8 * np.cos(yy * .72)
        sharp = base + detail[..., None]
        frames = np.stack([base + gaussian_filter(detail, 1.8)[..., None]
                           for _ in range(5)]).astype(np.float32)
        frames[2] = sharp
        frames[2, 43:51, 43:51] += 100.0  # Transient specular-like glint in the reference.
        valid = np.ones((5, 96, 96), dtype=bool)

        fused, report = fuse_temporal_detail(frames, valid, 2)
        median = np.median(frames, axis=0)
        median_high = median - gaussian_filter(median, sigma=(2., 2., 0), mode="reflect")
        fused_float = fused.astype(np.float32)
        fused_high = fused_float - gaussian_filter(fused_float, sigma=(2., 2., 0), mode="reflect")
        stable_region = np.ones((96, 96), dtype=bool)
        stable_region[38:56, 38:56] = False
        self.assertGreater(float(np.sqrt(np.mean(fused_high[stable_region] ** 2))),
                           float(np.sqrt(np.mean(median_high[stable_region] ** 2))))
        self.assertLess(float(fused[45:49, 45:49].mean()), float(frames[2, 45:49, 45:49].mean()) - 30)
        self.assertEqual(report["method"],
                         "temporal_median_low_frequency_with_consensus_gated_reference_detail")
        self.assertEqual(report["supported_fraction"], 1.0)

    def test_detail_fusion_rejects_bad_inputs(self):
        with self.assertRaisesRegex(ValueError, "Invalid temporal detail-fusion inputs"):
            fuse_temporal_detail(np.zeros((2, 8, 8, 3)), np.ones((2, 8, 8), bool), 0)

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

    def test_copies_only_hash_verified_temporal_uv_layer_for_detail_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent, out = root / "parent", root / "candidate"
            parent.mkdir()
            out.mkdir()
            for path in (root / "ct.usda", root / "fat" / "review.usda"):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("scene-layer", encoding="utf-8")
            for name in ("appendix_appendix300_temporal_rgb.png",
                         "mesoappendix_appendix300_temporal_rgb.png"):
                Image.new("RGB", (4, 4), (80, 90, 100)).save(out / name)
            layer = parent / "appearance.usda"
            layer.write_text(
                '#usda 1.0\n( subLayers = [ @../ct.usda@, @../fat/review.usda@ ] )\n'
                'over "appendix" { over "render" {} }\n'
                'over "mesoappendix" { over "render" {} }\n'
                'asset inputs:file = @appendix_appendix300_temporal_rgb.png@\n'
                'asset inputs:file = @mesoappendix_appendix300_temporal_rgb.png@\n',
                encoding="utf-8")
            source_records = [
                {"path": "../ct.usda", "sha256": hashlib.sha256((root / "ct.usda").read_bytes()).hexdigest()},
                {"path": "../fat/review.usda", "sha256": hashlib.sha256((root / "fat/review.usda").read_bytes()).hexdigest()},
            ]
            report = {
                "dataset": "Appendix300", "license": "CC BY 4.0", "geometry_unchanged": True,
                "static_usd_layer": {
                    "sha256": hashlib.sha256(layer.read_bytes()).hexdigest(),
                    "referenced_sublayers": ["../ct.usda", "../fat/review.usda"],
                    "source_scene_layers": source_records,
                    "uvs_reused_unchanged": True,
                    "geometry_unchanged_by_layer_operation": True,
                    "physics_unchanged_by_layer_operation": True,
                },
            }
            (parent / "appearance.json").write_text(json.dumps(report), encoding="utf-8")

            result = copy_reviewed_temporal_uv_layer(layer, out)

            self.assertTrue(result["geometry_unchanged_by_layer_operation"])
            self.assertEqual((out / "appearance.usda").read_bytes(), layer.read_bytes())
            self.assertEqual(result["changed_properties"], [])

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
