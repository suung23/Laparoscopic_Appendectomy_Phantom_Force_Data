import sys
import unittest
from pathlib import Path
import numpy as np
import trimesh
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ct_appendectomy'))
from boundary_surface import SurfaceBinding,project_boundary,coons_surface


class BoundaryTests(unittest.TestCase):
    def test_affine_motion_virtual_work_and_force_moment_conservation(self):
        nodes=np.array([[0,0,0],[.02,0,0],[0,.03,0],[0,0,.04]])
        binding=SurfaceBinding([[0,1,2],[1,2,3]],[[.2,.3,.5],[.6,.1,.3]],4)
        rng=np.random.default_rng(7);force=rng.normal(size=(2,3));delta=rng.normal(size=(4,3))
        output=binding.parent_forces(force);p=binding.positions(nodes)
        np.testing.assert_allclose(output.sum(0),force.sum(0),atol=1e-12)
        np.testing.assert_allclose(np.cross(nodes,output).sum(0),np.cross(p,force).sum(0),atol=1e-12)
        self.assertAlmostEqual(float(np.sum(output*delta)),float(np.sum(force*binding.positions(delta))),places=12)
        a=rng.normal(size=(3,3));translation=rng.normal(size=3)
        np.testing.assert_allclose(binding.positions(nodes@a.T+translation),p@a.T+translation,atol=1e-12)

    def test_projection_rejects_remote_landmark(self):
        mesh=trimesh.creation.box(extents=[.01]*3)
        with self.assertRaisesRegex(ValueError,'projection limit'):project_boundary(mesh,[[1,0,0]],.001)
        binding,error=project_boundary(mesh,[[.005,.001,.002]],.001)
        np.testing.assert_allclose(binding.positions(mesh.vertices),[[.005,.001,.002]],atol=1e-10)

    def test_fold_preserves_curved_boundaries_and_single_surface(self):
        u=np.linspace(0,1,31);v=np.linspace(0,1,17)
        bottom=np.c_[u*.04,u*0,.003*np.sin(np.pi*u)]
        top=bottom+np.array([0,.025,.005])
        left=(1-v[:,None])*bottom[0]+v[:,None]*top[0]
        right=(1-v[:,None])*bottom[-1]+v[:,None]*top[-1]
        offset=np.zeros((len(v),len(u),3));offset[:,:,2]=.001*np.sin(4*np.pi*u)[None]*np.sin(np.pi*v)[:,None]
        mesh=coons_surface(bottom,top,left,right,offset);p=mesh.vertices.reshape(len(v),len(u),3)
        for actual,expected in ((p[0],bottom),(p[-1],top),(p[:,0],left),(p[:,-1],right)):
            np.testing.assert_allclose(actual,expected,atol=1e-12)
        self.assertEqual(len(mesh.split(only_watertight=False)),1)
        self.assertTrue(mesh.is_winding_consistent)
        offset[0,0,2]=.001
        with self.assertRaisesRegex(ValueError,'attachment boundaries'):coons_surface(bottom,top,left,right,offset)


if __name__=='__main__':unittest.main()
