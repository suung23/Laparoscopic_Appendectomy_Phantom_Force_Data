import sys
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ct_appendectomy"))
from appendix300_video_texture import build_from_verified_frame, suppress_local_specular


class Appendix300TextureTests(unittest.TestCase):
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
