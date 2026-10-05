import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ct_appendectomy'))
from determinant_step import positive_volume_step

class StepTests(unittest.TestCase):
    def setUp(self):
        self.x=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]])
        self.t=np.array([[0,1,2,3]])
    def test_translation_unrestricted(self):
        self.assertEqual(positive_volume_step(self.x,self.x,np.ones_like(self.x),self.t)['alpha'],1.)
    def test_collapse_bound(self):
        d=np.zeros_like(self.x);d[3,2]=-2
        self.assertAlmostEqual(positive_volume_step(self.x,self.x,d,self.t)['alpha'],.9*.495)
    def test_positive_endpoints_can_have_invalid_interior(self):
        target=self.x*np.array([-1,-1,1])
        self.assertAlmostEqual(positive_volume_step(self.x,self.x,target-self.x,self.t)['alpha'],.9*.45)
    def test_invalid_initial_state_rejected(self):
        with self.assertRaises(ValueError):positive_volume_step(self.x,self.x*np.array([-1,1,1]),self.x*0,self.t)

if __name__=='__main__':unittest.main()
