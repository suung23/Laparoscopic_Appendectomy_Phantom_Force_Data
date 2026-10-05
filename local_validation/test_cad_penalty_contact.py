import sys,unittest
from pathlib import Path
import numpy as np
import trimesh
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ct_appendectomy'))
from cad_penalty_contact import penalty,compound_penalty
class ContactTests(unittest.TestCase):
    def test_compound_energy_gradient_and_reaction(self):
        left=trimesh.creation.box();right=left.copy();right.apply_translation([.8,0,0])
        parts={'left':left,'right':right};p=np.array([[.49,0,0],[.31,0,0]]);area=np.array([.002,.003])
        force,report=compound_penalty(parts,p,area,1000)
        np.testing.assert_allclose(report['reaction_N'],-force.sum(axis=0))
        self.assertEqual(report['active_nodes'],2)
        for i in range(2):
            plus=p.copy();minus=p.copy();plus[i,0]+=1e-6;minus[i,0]-=1e-6
            derivative=(compound_penalty(parts,plus,area,1000)[1]['energy_J']-compound_penalty(parts,minus,area,1000)[1]['energy_J'])/2e-6
            self.assertAlmostEqual(derivative,-force[i,0],places=8)
    def test_cube_force_energy_and_separation(self):
        mesh=trimesh.creation.box();p=np.array([[.49,0,0],[.51,0,0]]);a=np.array([.002,.002]);f,r=penalty(mesh,p,a,1000)
        np.testing.assert_allclose(f,[[.02,0,0],[0,0,0]],atol=1e-12)
        self.assertAlmostEqual(r['energy_J'],.0001)
        h=1e-6;plus=p.copy();minus=p.copy();plus[0,0]+=h;minus[0,0]-=h
        derivative=(penalty(mesh,plus,a,1000)[1]['energy_J']-penalty(mesh,minus,a,1000)[1]['energy_J'])/(2*h)
        self.assertAlmostEqual(derivative,-f[0,0],places=10)
if __name__=='__main__':unittest.main()
