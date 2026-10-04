import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ct_appendectomy'))
from node_correspondence import node_permutation


class NodeCorrespondence(unittest.TestCase):
    def setUp(self):
        self.p=np.array([[0.,0,0],[1.,0,0],[0.,1,0],[0.,0,1],[2.,0,0]])
        self.t=np.array([[0,1,2,3],[1,2,3,4]])

    def test_permutation_preserves_pinned_coordinates_and_connectivity(self):
        order=np.array([4,2,0,3,1]);inverse=np.argsort(order)
        m=node_permutation(self.p,self.t,self.p[order],inverse[self.t])
        np.testing.assert_array_equal(m,inverse)
        np.testing.assert_array_equal(self.p[order][m[[0,1]]],self.p[[0,1]])

    def test_geometry_change_rejected(self):
        q=self.p.copy();q[0,0]=.001
        with self.assertRaises(ValueError):node_permutation(self.p,self.t,q,self.t)

    def test_topology_change_rejected(self):
        with self.assertRaises(ValueError):node_permutation(self.p,self.t,self.p,self.t[:1])


if __name__=='__main__':unittest.main()
