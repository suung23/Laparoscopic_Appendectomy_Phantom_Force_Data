import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ct_appendectomy'))
from trocar_scope import TrocarScope


class ScopeTests(unittest.TestCase):
    def test_optical_pose_and_pivot_are_coupled(self):
        initial=np.eye(4);initial[:3,3]=[.1,.2,.3]
        scope=TrocarScope(initial)
        np.testing.assert_allclose(scope.pose,initial)
        port=scope.instrument.port.copy()
        for _ in range(1000):
            M=scope.step([.4,-.3,.1,.2],1/480)
            forward=-M[:3,2]
            self.assertLess(np.linalg.norm(np.cross(M[:3,3]-port,forward)),1e-10)
            self.assertTrue(scope.metrics()['aperture_clear'])
            self.assertAlmostEqual(np.linalg.det(M[:3,:3]),1)
        self.assertGreater(np.linalg.norm(scope.pose[:3,3]-initial[:3,3]),.01)
        np.testing.assert_array_equal(scope.instrument.port,port)

    def test_invalid_command_does_not_move_eye(self):
        scope=TrocarScope(np.eye(4));before=scope.pose.copy()
        for action in ([0,0,0], [1.1,0,0,0], [np.nan,0,0,0]):
            with self.assertRaises(ValueError):scope.step(action,.01)
            np.testing.assert_array_equal(scope.pose,before)


if __name__=='__main__':unittest.main()
