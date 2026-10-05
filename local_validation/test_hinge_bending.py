import sys,unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ct_appendectomy'))
from hinge_bending import HingeBending
from membrane_reference import MembraneReference
class BendingTests(unittest.TestCase):
    def test_isometric_fold_and_force_gradient(self):
        p=np.array([[0.,0,0],[.01,0,0],[0,.01,0],[0,-.01,0]])
        faces=np.array([[0,1,2],[1,0,3]])
        b=HingeBending(p,faces,1e-8);q=p.copy();q[3]=[0,-.01*np.cos(.4),.01*np.sin(.4)]
        m=MembraneReference(p,faces,12000,.45,.0002)
        self.assertLess(abs(m.energy_forces(q)[0]),1e-14)
        x=torch.tensor(q,dtype=torch.float64,requires_grad=True);e=b.energy(x);g=torch.autograd.grad(e,x)[0].numpy();self.assertGreater(float(e.detach()),0)
        np.testing.assert_allclose(g.sum(axis=0),0,atol=1e-12);np.testing.assert_allclose(np.cross(q,g).sum(axis=0),0,atol=1e-12)
        d=np.random.default_rng(2).normal(size=q.shape);h=1e-8
        derivative=(float(b.energy(torch.tensor(q+h*d)))-float(b.energy(torch.tensor(q-h*d))))/(2*h)
        self.assertAlmostEqual(derivative,float(np.sum(g*d)),places=10)
    def test_curved_rest_zero_force(self):
        p=np.array([[0.,0,0],[.01,0,0],[0,.01,0],[0,-.009,.004]])
        b=HingeBending(p,[[0,1,2],[1,0,3]],1e-8)
        x=torch.tensor(p,dtype=torch.float64,requires_grad=True);e=b.energy(x);g=torch.autograd.grad(e,x)[0]
        self.assertLess(abs(float(e.detach())),1e-20);self.assertLess(float(torch.linalg.vector_norm(g)),1e-12)
if __name__=='__main__':unittest.main()
