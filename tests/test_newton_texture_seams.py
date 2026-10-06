import sys
import unittest
from pathlib import Path

import numpy as np
import warp as wp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "newton_cell"))
from render import _triangle_uv_wrapped


@wp.kernel
def _interpolate_uv(uvs: wp.array(dtype=wp.vec2),
                    barycentric: wp.array(dtype=wp.vec2),
                    result: wp.array(dtype=wp.vec2)):
    uv = _triangle_uv_wrapped(uvs[0], uvs[1], uvs[2],
                              barycentric[0][0], barycentric[0][1])
    result[0] = uv


class TextureSeamTests(unittest.TestCase):
    def _interpolate(self, uv, bu, bv):
        source = wp.array(np.asarray(uv, np.float32), dtype=wp.vec2, device="cpu")
        weights = wp.array(np.asarray([[bu, bv]], np.float32), dtype=wp.vec2,
                           device="cpu")
        result = wp.zeros(1, dtype=wp.vec2, device="cpu")
        wp.launch(_interpolate_uv, dim=1, inputs=[source, weights, result],
                  device="cpu")
        return result.numpy()[0]

    def test_triangle_crossing_wrapped_u_seam_uses_short_arc(self):
        got = self._interpolate([[0.98, 0.1], [0.02, 0.1], [0.04, 0.9]],
                                bu=0.5, bv=0.25)
        # The short periodic arc is near U=0; naive interpolation would be 0.26.
        np.testing.assert_allclose(got, [0.015, 0.3], atol=1e-6)

    def test_triangle_away_from_seam_keeps_ordinary_barycentric_uv(self):
        got = self._interpolate([[0.1, 0.2], [0.3, 0.4], [0.6, 0.8]],
                                bu=0.5, bv=0.25)
        np.testing.assert_allclose(got, [0.325, 0.45], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
