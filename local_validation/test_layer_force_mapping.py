import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ct_appendectomy'))
from layer_force_mapping import pullback_forces
from build_fat_layer import evaluate
class ForceMappingTests(unittest.TestCase):
    def test_resultant_torque_and_virtual_work(self):
        p=np.array([[0.,0.,0.],[.012,.001,0.],[.001,.01,.002],[.012,.011,.001]])
        nodes=np.array([[0,1,2],[1,3,2],[0,1,2]])
        weights=np.array([[.2,.3,.5],[.3,.2,.5],[.1,.8,.1]])
        offset=np.array([[.0002,.0001,.001],[-.0003,0.,.0007],[0.,0.,-.0005]])
        force=np.array([[.1,.2,-.3],[-.2,.1,.15],[.12,-.1,.02]])
        parent_force=pullback_forces(p,nodes,weights,offset,force);x=evaluate(p,nodes,weights,offset)
        np.testing.assert_allclose(parent_force.sum(axis=0),force.sum(axis=0),atol=1e-12)
        np.testing.assert_allclose(np.cross(p,parent_force).sum(axis=0),np.cross(x,force).sum(axis=0),atol=1e-12)
        delta=np.random.default_rng(4).normal(size=p.shape);eps=1e-8
        derivative=(evaluate(p+eps*delta,nodes,weights,offset)-evaluate(p-eps*delta,nodes,weights,offset))/(2*eps)
        self.assertAlmostEqual(float(np.sum(derivative*force)),float(np.sum(delta*parent_force)),places=8)
    def test_degenerate_rejected(self):
        with self.assertRaises(ValueError):pullback_forces(np.zeros((3,3)),[[0,1,2]],[[1.,0,0]],[[0,0,.001]],[[1,0,0]])
if __name__=='__main__':unittest.main()
