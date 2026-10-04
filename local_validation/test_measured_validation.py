import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ct_appendectomy'))
from measured_validation import surface_error, force_error


class MeasuredValidationTests(unittest.TestCase):
    def test_surface_keeps_calibration_error(self):
        a=np.array([[0.,0,0],[.01,0,0],[0,.01,0]])
        self.assertEqual(surface_error(a,a)['f_score'],1)
        r=surface_error(a+[0,0,.002],a)
        self.assertAlmostEqual(r['symmetric_rms_m'],.002)
        self.assertEqual(r['f_score'],0)

    def test_missing_surface_is_penalized(self):
        a=np.array([[0.,0,0],[.01,0,0],[0,.01,0]])
        b=np.concatenate([a,a+[0,0,.02]])
        r=surface_error(a,b)
        self.assertEqual(r['precision'],1)
        self.assertEqual(r['recall'],.5)

    def test_force_interpolation_preserves_bias(self):
        p=np.array([[0.,0,0],[2,0,0]])
        m=np.array([[1.,0,0],[2,0,0],[3,0,0]])
        r=force_error([0,2],p,[0,1,2],m)
        self.assertAlmostEqual(r['vector_rmse_n'],1)
        self.assertEqual(r['bias_xyz_n'],[-1.,0.,0.])

    def test_extrapolation_and_bad_clock_rejected(self):
        f=np.zeros((2,3))
        with self.assertRaises(ValueError):force_error([0,1],f,[0,2],f)
        with self.assertRaises(ValueError):force_error([0,0],f,[0,1],f)


if __name__=='__main__':unittest.main()
