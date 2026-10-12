import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "newton_cell"))
from scene import close_pins_over_tets


class NewtonPinClosureTests(unittest.TestCase):
    def test_tet_hinged_on_pinned_face_gets_its_free_corner_pinned(self):
        tets = np.array([0, 1, 2, 3], np.int32)

        self.assertEqual(close_pins_over_tets([0, 1, 2], tets).tolist(),
                         [0, 1, 2, 3])

    def test_tets_with_two_free_corners_are_left_free(self):
        tets = np.array([0, 1, 2, 3], np.int32)

        self.assertEqual(close_pins_over_tets([0, 1], tets).tolist(), [0, 1])

    def test_closure_is_a_single_pass(self):
        # Pinning node 3 hinges the second tet on 1, 2, 3; a fixed point would
        # keep growing and, on the organ bands, pinned every node.
        tets = np.array([0, 1, 2, 3,
                         1, 2, 3, 4], np.int32)

        self.assertEqual(close_pins_over_tets([0, 1, 2], tets).tolist(),
                         [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
