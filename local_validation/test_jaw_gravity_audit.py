import sys
import unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ct_appendectomy'))
from jaw_gravity_audit import support_torque


class GravityConvention(unittest.TestCase):
    def test_known_horizontal_lever(self):
        # A 10 g jaw, COM 20 mm along +Z, gravity +X produces +Y
        # external torque, requiring negative support torque.
        self.assertAlmostEqual(support_torque(.01,[0,0,.02],np.eye(3),[9.81,0,0]),-.001962)

    def test_world_coordinate_rotation_does_not_change_joint_torque(self):
        body=Rotation.from_euler('xyz',[.3,-.8,.7]).as_matrix()
        world=Rotation.from_euler('xyz',[-.2,.4,1.1]).as_matrix()
        g=np.array([0,-9.81,0])
        expected=support_torque(.004,[.002,0,.008],body,g)
        self.assertAlmostEqual(expected,support_torque(.004,[.002,0,.008],world@body,world@g))

    def test_gravity_parallel_to_hinge_has_no_hinge_torque(self):
        self.assertEqual(support_torque(.004,[.003,.002,.01],np.eye(3),[0,-9.81,0]),0.)


if __name__=='__main__':unittest.main()
