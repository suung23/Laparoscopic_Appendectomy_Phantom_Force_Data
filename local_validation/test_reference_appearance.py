import sys
import unittest
from pathlib import Path
import numpy as np
from pxr import Usd, UsdGeom, UsdShade, Vt
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'ct_appendectomy'))
from reference_appearance import author_texture, require_nondegenerate_uv


class ReferenceAppearanceTests(unittest.TestCase):
    def test_texture_preserves_surface_topology_and_points(self):
        s=Usd.Stage.CreateInMemory()
        m=UsdGeom.Mesh.Define(s,'/World/Liver')
        points=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],np.float32)
        faces=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]])
        m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
        m.CreateFaceVertexCountsAttr([3]*4)
        m.CreateFaceVertexIndicesAttr(faces.ravel().tolist())
        author_texture(s,'/World/Liver','reference.png')
        np.testing.assert_array_equal(m.GetPointsAttr().Get(),points)
        np.testing.assert_array_equal(m.GetFaceVertexIndicesAttr().Get(),faces.ravel())
        uv=UsdGeom.PrimvarsAPI(m).GetPrimvar('st')
        self.assertEqual(uv.GetInterpolation(),'faceVarying')
        self.assertEqual(len(uv.Get()),12)
        self.assertTrue(np.isfinite(np.asarray(uv.Get())).all())
        bound,_=UsdShade.MaterialBindingAPI(m).ComputeBoundMaterial()
        self.assertEqual(str(bound.GetPath()),'/World/ReferenceLiverMaterial')
        shader=UsdShade.Shader(s.GetPrimAtPath('/World/ReferenceLiverMaterial/Surface'))
        self.assertTrue(shader.GetInput('diffuseColor').HasConnectedSource())

    def test_invalid_parameters_fail_before_mutation(self):
        s=Usd.Stage.CreateInMemory()
        with self.assertRaises(ValueError):author_texture(s,'/missing','x.png',repeat=np.nan)
        self.assertFalse(s.GetPrimAtPath('/World/ReferenceLiverMaterial'))

    def test_submillimetre_triangles_keep_valid_uvs(self):
        s=Usd.Stage.CreateInMemory()
        m=UsdGeom.Mesh.Define(s,'/World/SmallSurface')
        points=np.array([[0,0,0],[.0002,0,0],[0,.0002,0],[0,0,.0002]],np.float32)
        faces=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]])
        m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
        m.CreateFaceVertexCountsAttr([3]*4)
        m.CreateFaceVertexIndicesAttr(faces.ravel().tolist())
        author_texture(s,'/World/SmallSurface','reference.png',repeat=1.)
        uv=np.asarray(UsdGeom.PrimvarsAPI(m).GetPrimvar('st').Get()).reshape(-1,3,2)
        require_nondegenerate_uv(uv)
        np.testing.assert_array_equal(m.GetPointsAttr().Get(),points)


if __name__=='__main__':unittest.main()
