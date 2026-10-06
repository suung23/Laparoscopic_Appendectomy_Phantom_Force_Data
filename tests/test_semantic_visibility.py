import sys
import unittest
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ct_appendectomy"))
from semantic_visibility import class_pixel_fraction
from bimanual_env import BimanualEnv


class SemanticVisibilityTests(unittest.TestCase):
    def test_class_fraction_combines_ids_without_counting_background(self):
        image = np.array([[0, 1], [2, 2]], dtype=np.uint32)
        labels = {"0": {"class": "background"},
                  "1": {"class": "appendix"},
                  "2": {"class": "appendix"}}
        self.assertEqual(class_pixel_fraction(image, labels, "appendix"), .75)

    def test_singleton_channel_is_accepted(self):
        image = np.array([[[1], [0]]], dtype=np.uint16)
        labels = {"1": {"class": "appendix"}}
        self.assertEqual(class_pixel_fraction(image, labels, "appendix"), .5)

    def test_missing_or_malformed_ground_truth_fails_closed(self):
        with self.assertRaises(ValueError):
            class_pixel_fraction(np.zeros((2, 2), dtype=np.uint8), {}, "appendix")
        with self.assertRaises(ValueError):
            class_pixel_fraction(np.zeros((2, 2), dtype=float), {}, "appendix")

    def test_rgb_and_semantics_share_one_render_tick(self):
        calls = []
        rep = types.ModuleType("omni.replicator.core")
        rep.orchestrator = types.SimpleNamespace(
            step=lambda **kwargs: calls.append(("render", kwargs)))
        omni = types.ModuleType("omni")
        replicator = types.ModuleType("omni.replicator")
        env = object.__new__(BimanualEnv)
        env.control = object()
        env.product = types.SimpleNamespace(path="/Render/Product")
        env.rgb = types.SimpleNamespace(get_data=lambda: np.zeros((360, 640, 4), np.uint8))
        semantic = np.zeros((360, 640), np.uint32)
        semantic[100:200, 100:300] = 7
        env.semantic = types.SimpleNamespace(get_data=lambda: {
            "data": semantic,
            "info": {"idToLabels": {"7": {"class": "appendix"}}}})
        env.sim = types.SimpleNamespace(
            is_playing=lambda: True,
            pause=lambda: calls.append(("pause", None)),
            play=lambda: calls.append(("play", None)))
        with patch.dict(sys.modules, {"omni": omni,
                                      "omni.replicator": replicator,
                                      "omni.replicator.core": rep}):
            rgb, fraction = env.render_with_class_fraction("appendix")
        self.assertEqual(rgb.shape, (360, 640, 3))
        self.assertAlmostEqual(fraction, 20000 / (360 * 640))
        self.assertEqual([call[0] for call in calls], ["pause", "render", "play"])


if __name__ == "__main__":
    unittest.main()
