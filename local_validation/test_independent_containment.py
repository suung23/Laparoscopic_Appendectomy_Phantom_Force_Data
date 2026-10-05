import sys,unittest
from pathlib import Path
import numpy as np
import trimesh
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ct_appendectomy'))
from independent_containment import contains

class IndependentContainmentTests(unittest.TestCase):
    def test_box_across_scales_and_rotation(self):
        rng=np.random.default_rng(19);p=rng.uniform(-1,1,(2000,3));expected=(np.abs(p)<.5).all(axis=1)
        rotation=trimesh.transformations.rotation_matrix(.7,[1,2,3])[:3,:3]
        for scale in [1e-4,1,1000]:
            m=trimesh.creation.box();m.vertices=(m.vertices@rotation.T+np.array([2,3,4]))*scale
            q=(p@rotation.T+np.array([2,3,4]))*scale
            np.testing.assert_array_equal(contains(m,q),expected)
            m.invert();np.testing.assert_array_equal(contains(m,q),expected)
    def test_sphere_away_from_discretization_boundary(self):
        m=trimesh.creation.icosphere(subdivisions=3);p=np.random.default_rng(23).uniform(-1.5,1.5,(2000,3));r=np.linalg.norm(p,axis=1);keep=(r<.98)|(r>1.02)
        np.testing.assert_array_equal(contains(m,p[keep]),r[keep]<1)
    def test_open_mesh_rejected(self):
        m=trimesh.Trimesh(vertices=[[0,0,0],[1,0,0],[0,1,0]],faces=[[0,1,2]])
        with self.assertRaises(ValueError):contains(m,[[.2,.2,.1]])

if __name__=='__main__':unittest.main()
