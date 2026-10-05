import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ct_appendectomy'))
from membrane_reference import MembraneReference
from membrane_dynamics import MembraneDynamics
class DynamicsTests(unittest.TestCase):
    def make(self,fixed):
        p=np.array([[0.,0,0],[.01,0,0],[0,.01,0],[.01,.01,0]])
        return MembraneDynamics(MembraneReference(p,[[0,1,2],[1,3,2]],12000,.45,.0002,1e-8),p,1000,fixed)
    def test_uniform_acceleration(self):
        d=self.make([]);rest=d.points.copy();a=np.array([0,0,-9.81]);dt=.001
        for _ in range(3):d.step(dt,d.mass[:,None]*a,tolerance=1e-9)
        np.testing.assert_allclose(d.points,rest+6*dt**2*a,atol=1e-10)
        np.testing.assert_allclose(d.velocity,np.tile(3*dt*a,(4,1)),atol=1e-8)
    def test_fixed_points(self):
        d=self.make([0,1]);rest=d.points.copy();f=np.zeros_like(rest);f[2:,2]=.0001;d.step(.001,f)
        np.testing.assert_array_equal(d.points[:2],rest[:2]);self.assertGreater(d.points[2:,2].max(),0)
if __name__=='__main__':unittest.main()
