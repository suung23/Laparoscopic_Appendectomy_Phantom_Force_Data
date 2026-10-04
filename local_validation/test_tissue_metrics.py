import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ct_appendectomy'))
from tissue_metrics import tetra_health


class TissueMetricsTests(unittest.TestCase):
    def test_translation_preserves_volume(self):
        points=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]])
        result=tetra_health(points,points+[2,3,4],[[0,1,2,3]])
        self.assertEqual(result['total_absolute_volume_ratio'],1)
        self.assertEqual(result['inverted_tetrahedra'],0)

    def test_inversion_is_detected(self):
        points=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]])
        mirrored=points.copy();mirrored[:,2]*=-1
        result=tetra_health(points,mirrored,[[0,1,2,3]])
        self.assertEqual(result['inverted_tetrahedra'],1)
        self.assertEqual(result['minimum_signed_volume_ratio'],-1)


if __name__=='__main__':unittest.main()
