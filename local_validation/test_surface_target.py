import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ct_appendectomy'))
from surface_target import TissueSurface


class SurfaceTargets(unittest.TestCase):
    def test_face_interior_and_outward_normal_independent_of_winding(self):
        p=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],float)
        for tet in ([[0,1,2,3]],[[1,0,2,3]]):
            r=TissueSurface(tet).closest(p,[[.2,.2,-.1]])[0]
            np.testing.assert_allclose(r['point_m'],[.2,.2,0])
            np.testing.assert_allclose(r['outward_normal'],[0,0,-1])
            self.assertAlmostEqual(r['distance_m'],.1)

    def test_shared_face_is_not_exterior(self):
        surface=TissueSurface([[0,1,2,3],[0,2,1,4]])
        self.assertEqual(len(surface.faces),6)
        self.assertFalse(any(set(f)=={0,1,2} for f in surface.faces))


if __name__=='__main__':unittest.main()
