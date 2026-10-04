import sys
import unittest
from pathlib import Path
import numpy as np
import trimesh
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ct_appendectomy'))
from tetra_cut import cut_plane,signed_volumes,boundary_faces


class TetraCutTests(unittest.TestCase):
    def test_volume_connectivity_and_velocity_transfer(self):
        points=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]])
        cells=np.array([[0,1,2,3]])
        result=cut_plane(points,cells,[1,1,0],.4,points*2+[1,2,3])
        self.assertAlmostEqual(signed_volumes(result['points'],result['tetrahedra']).sum(),1/6)
        np.testing.assert_allclose(result['velocities'],result['points']*2+[1,2,3])
        negative=set(result['tetrahedra'][result['side']<0].ravel())
        positive=set(result['tetrahedra'][result['side']>0].ravel())
        self.assertFalse(negative&positive)
        for side in (-1,1):
            triangles=boundary_faces(result['points'],result['tetrahedra'][result['side']==side])
            mesh=trimesh.Trimesh(result['points'],triangles,process=False)
            self.assertTrue(mesh.is_watertight)

    def test_coincident_plane_is_rejected(self):
        points=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]])
        with self.assertRaises(ValueError):cut_plane(points,[[0,1,2,3]],[1,0,0],0)


if __name__=='__main__':unittest.main()
