import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ct_appendectomy'))
from deformation_metrics import DeformationMeasure


class ObjectiveDeformation(unittest.TestCase):
    def setUp(self):
        self.p=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]])
        self.m=DeformationMeasure(self.p,[[0,1,2,3]])

    def test_rigid_motion_does_not_count_as_deformation(self):
        R=np.array([[0,-1.,0],[1,0,0],[0,0,1]])
        d=self.m.measure(self.p@R+[4,5,6])
        self.assertLess(d['nonrigid_rms_m'],1e-12)
        self.assertAlmostEqual(d['stretch_max'],1)

    def test_known_stretch_and_inversion(self):
        d=self.m.measure(self.p*[2,1,1])
        self.assertAlmostEqual(d['stretch_max'],2)
        self.assertAlmostEqual(d['minimum_jacobian'],2)
        self.assertGreater(d['nonrigid_rms_m'],.1)
        self.assertEqual(self.m.measure(self.p*[-1,1,1])['inverted_tetrahedra'],1)

    def test_bad_states_fail(self):
        with self.assertRaises(ValueError):self.m.measure(self.p*np.nan)
        with self.assertRaises(ValueError):DeformationMeasure(self.p*0,[[0,1,2,3]])


if __name__=='__main__':unittest.main()
