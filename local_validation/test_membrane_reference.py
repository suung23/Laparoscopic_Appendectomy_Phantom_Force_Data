import sys,unittest
from pathlib import Path
import numpy as np
import trimesh
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ct_appendectomy'))
from membrane_reference import MembraneReference
class MembraneTests(unittest.TestCase):
    def test_objectivity_and_force_gradient(self):
        p=np.array([[0.,0,0],[.01,0,0],[0,.01,0],[.01,.01,0]])
        m=MembraneReference(p,[[0,1,2],[1,3,2]],12000,.45,.0002)
        rotation=trimesh.transformations.rotation_matrix(.8,[1,2,3])[:3,:3]
        e,f=m.energy_forces(p@rotation.T+[.03,.02,-.01]);self.assertLess(abs(e),1e-14);self.assertLess(np.linalg.norm(f),1e-10)
        q=p.copy();q[:,0]*=1.1;e,f=m.energy_forces(q);self.assertGreater(e,0)
        delta=np.random.default_rng(3).normal(size=p.shape);h=1e-8
        numerical=(m.energy_forces(q+h*delta)[0]-m.energy_forces(q-h*delta)[0])/(2*h)
        self.assertAlmostEqual(numerical,-float(np.sum(f*delta)),places=8)
        np.testing.assert_allclose(f.sum(axis=0),0,atol=1e-12)
        np.testing.assert_allclose(np.cross(q,f).sum(axis=0),0,atol=1e-12)
    def test_collapse_rejected(self):
        p=np.array([[0.,0,0],[1.,0,0],[0,1.,0]])
        m=MembraneReference(p,[[0,1,2]],1000,.3,.001)
        with self.assertRaises(ValueError):m.energy_forces(np.zeros_like(p))
if __name__=='__main__':unittest.main()
