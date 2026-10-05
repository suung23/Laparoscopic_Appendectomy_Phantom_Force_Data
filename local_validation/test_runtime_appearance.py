import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
from pxr import Usd, UsdGeom, UsdShade, Vt, Sdf
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ct_appendectomy'))
from runtime_appearance import apply_appearance


def mesh(stage):
    m=UsdGeom.Mesh.Define(stage,'/World/Organ')
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.array([[0,0,0],[1,0,0],[0,1,0]],np.float32)))
    m.CreateFaceVertexCountsAttr([3]);m.CreateFaceVertexIndicesAttr([0,1,2])
    return m


class TransferTests(unittest.TestCase):
    def source(self,path):
        s=Usd.Stage.CreateNew(str(path));m=mesh(s)
        mat=UsdShade.Material.Define(s,'/World/Appendix300Look/Tissue')
        UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)
        UsdGeom.PrimvarsAPI(m).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,
            UsdGeom.Tokens.faceVarying).Set([(0,0),(1,0),(0,1)])
        s.GetRootLayer().Save()

    def test_transfer_preserves_physics_attribute_and_points(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'source.usda';self.source(path)
            target=Usd.Stage.CreateInMemory();m=mesh(target)
            m.GetPrim().CreateAttribute('testPhysicsStiffness',Sdf.ValueTypeNames.Float).Set(123.)
            before=np.asarray(m.GetPointsAttr().Get()).copy()
            report=apply_appearance(target,path)
            np.testing.assert_array_equal(before,m.GetPointsAttr().Get())
            self.assertEqual(m.GetPrim().GetAttribute('testPhysicsStiffness').Get(),123.)
            self.assertEqual(report['meshes'],['/World/Organ'])

    def test_mismatched_rest_surface_fails_without_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'source.usda';self.source(path)
            target=Usd.Stage.CreateInMemory();m=mesh(target)
            m.GetPointsAttr().Set([(0,0,0),(2,0,0),(0,1,0)])
            before=target.GetRootLayer().ExportToString()
            with self.assertRaisesRegex(ValueError,'correspondence mismatch'):apply_appearance(target,path)
            self.assertEqual(before,target.GetRootLayer().ExportToString())


if __name__=='__main__':unittest.main()
