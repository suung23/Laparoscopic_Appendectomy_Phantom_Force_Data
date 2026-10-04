import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ct_appendectomy'))
from reset_checkpoint import checkpoint_points


class Checkpoint(unittest.TestCase):
    def setUp(self):
        self.rest=np.array([[0.,0,0],[1.,0,0],[0.,1,0],[0.,0,1]])
        self.tets=np.array([[0,1,2,3]])
        self.data=dict(rest=self.rest.copy(),tissue=self.rest*.9,tets=self.tets.copy())

    def test_restore_does_not_change_rest_shape(self):
        points,health=checkpoint_points(self.data,self.rest,self.tets)
        np.testing.assert_array_equal(points,self.data['tissue'])
        self.assertEqual(self.rest[1,0],1.)
        self.assertAlmostEqual(health['median_volume_ratio'],.9**3)

    def test_inversion_rejected(self):
        self.data['tissue'][3,2]=-1
        with self.assertRaises(ValueError):checkpoint_points(self.data,self.rest,self.tets)

    def test_different_anatomy_rejected(self):
        other=self.rest.copy();other[1,0]=2
        with self.assertRaises(ValueError):checkpoint_points(self.data,other,self.tets)


if __name__=='__main__':unittest.main()
