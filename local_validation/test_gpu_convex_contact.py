import sys,unittest
from pathlib import Path
import numpy as np
import torch
import trimesh
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ct_appendectomy'))
from gpu_convex_contact import ConvexContact

class PlaneContactTests(unittest.TestCase):
    def test_analytic_contact_rotation_and_energy_gradient(self):
        contact=ConvexContact(trimesh.creation.box(),device='cpu')
        rotation=trimesh.transformations.rotation_matrix(.6,[1,2,3])[:3,:3]
        pose=np.eye(4);pose[:3,:3]=rotation;pose[:3,3]=[.2,.3,.4]
        local=np.array([[.49,0,0],[.51,0,0]])
        points=torch.tensor((local@rotation.T+pose[:3,3])[None],dtype=torch.float64)
        poses=torch.tensor(pose[None],dtype=torch.float64);area=torch.tensor([.002,.002],dtype=torch.float64)
        force,report=contact.penalty(points,poses,area,1000)
        np.testing.assert_allclose(force.numpy()[0],np.array([[.02,0,0],[0,0,0]])@rotation.T,atol=1e-12)
        self.assertAlmostEqual(float(report['energy_J'][0]),.0001)
        direction=torch.tensor(rotation[:,0],dtype=torch.float64);plus=points.clone();minus=points.clone()
        plus[0,0]+=1e-6*direction;minus[0,0]-=1e-6*direction
        derivative=(contact.penalty(plus,poses,area,1000)[1]['energy_J']-contact.penalty(minus,poses,area,1000)[1]['energy_J'])/2e-6
        self.assertAlmostEqual(float(derivative[0]),-float(force[0,0]@direction),places=9)

if __name__=='__main__':unittest.main()
