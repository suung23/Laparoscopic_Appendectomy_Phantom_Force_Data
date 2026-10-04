import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ct_appendectomy'))
from static_load_check import point_jacobian,assess_torque


class StaticLoadTests(unittest.TestCase):
    def test_lever_arm_and_virtual_work(self):
        J=point_jacobian([[0,0,0],[1,0,0]],[[0,0,1],[0,1,0]],[2,0,0])
        np.testing.assert_allclose(J,[[0,0],[2,0],[0,-1]])
        dq=np.array([.2,-.3]);f=np.array([1,2,3])
        self.assertAlmostEqual(float(f@(J@dq)),float((J.T@f)@dq))

    def test_wrong_sign_and_release_load_fail(self):
        tau=np.array([[.01,.02,-.03]])
        self.assertEqual(assess_torque(tau,tau,np.zeros_like(tau))['status'],'passed')
        self.assertEqual(assess_torque(-tau,tau,np.zeros_like(tau))['status'],'failed')
        self.assertEqual(assess_torque(tau,tau,tau)['status'],'failed')

    def test_nonunit_axis_is_not_silently_rescaled(self):
        with self.assertRaises(ValueError):point_jacobian([[0,0,0]],[[0,0,2]],[1,0,0])

    def test_common_rigid_frame_change_preserves_torque(self):
        origins=np.array([[0.,0,0],[1,0,0]]);axes=np.array([[0.,0,1],[0,1,0]])
        point=np.array([2.,1,0]);force=np.array([1.,2,3])
        R=np.array([[0.,-1,0],[1,0,0],[0,0,1]]);shift=np.array([4.,-2,1])
        original=point_jacobian(origins,axes,point).T@force
        changed=point_jacobian(origins@R.T+shift,axes@R.T,R@point+shift).T@(R@force)
        np.testing.assert_allclose(original,changed)


if __name__=='__main__':unittest.main()
