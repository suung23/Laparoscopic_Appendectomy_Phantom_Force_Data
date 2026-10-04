import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ct_appendectomy'))
from mount_sensor_probe import evaluate_response


class MountCalibrationTests(unittest.TestCase):
    def test_consistent_opposite_force_convention_is_recorded(self):
        f=np.eye(3)*.1
        r=evaluate_response(-f,f,[0]*3,[1e-6]*3)
        self.assertEqual(r['status'],'passed');self.assertEqual(r['response_sign'],-1)

    def test_magnitude_and_axis_errors_are_rejected(self):
        f=np.eye(3)*.1
        self.assertEqual(evaluate_response(f*.94,f,[0]*3,[0]*3)['status'],'failed')
        self.assertEqual(evaluate_response(f[[1,0,2]],f,[0]*3,[0]*3)['status'],'failed')

    def test_release_offset_is_not_ignored(self):
        f=np.eye(3)*.1
        self.assertEqual(evaluate_response(f,f,[.01,0,0],[0]*3)['status'],'failed')

    def test_invalid_uncertainty_and_tolerance_are_rejected(self):
        f=np.eye(3)*.1
        for release,noise,tol in (([0,0,float('nan')],[0]*3,.005),
                                   ([0]*3,[-1,0,0],.005),([0]*3,[0]*3,float('inf'))):
            with self.assertRaises(ValueError):evaluate_response(f,f,release,noise,tol)


if __name__=='__main__':unittest.main()
