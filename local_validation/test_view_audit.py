"""Analytic occlusion/transform checks independent of the anatomy assets."""
import sys
import tempfile
import unittest
from pathlib import Path

from pxr import Usd, UsdGeom

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ct_appendectomy'))
from audit_view import audit


class ViewAuditTests(unittest.TestCase):
    def scene(self, path, hide_blocker=False):
        stage = Usd.Stage.CreateNew(str(path))
        UsdGeom.SetStageMetersPerUnit(stage, 1)
        camera = UsdGeom.Camera.Define(stage, '/World/Endoscope')
        camera.CreateFocalLengthAttr(20)
        camera.CreateHorizontalApertureAttr(20)
        camera.CreateVerticalApertureAttr(20)
        camera.CreateClippingRangeAttr((.01, 10))
        for name, depth in (('/World/Atlas/appendix/render', -2), ('/World/blocker', -1)):
            plane = UsdGeom.Mesh.Define(stage, name)
            plane.CreatePointsAttr([(-2,-2,0), (2,-2,0), (2,2,0), (-2,2,0)])
            plane.CreateFaceVertexCountsAttr([3,3])
            plane.CreateFaceVertexIndicesAttr([0,1,2,0,2,3])
            plane.AddTranslateOp().Set((0,0,depth))
            if hide_blocker and name == '/World/blocker':
                plane.CreateVisibilityAttr('invisible')
        stage.GetRootLayer().Save()

    def test_front_mesh_occludes_target(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'scene.usda'
            self.scene(path)
            result = audit(path, width=8, height=8)
            self.assertEqual(result['first_hit_frame_fraction']['/World/blocker'], 1)
            self.assertFalse(result['appendix_visible_in_sample'])
            self.assertEqual(result['background_fraction'], 0)

    def test_hidden_blocker_reveals_target(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'scene.usda'
            self.scene(path, hide_blocker=True)
            result = audit(path, width=8, height=8)
            self.assertEqual(result['first_hit_frame_fraction']['/World/Atlas/appendix/render'], 1)
            self.assertTrue(result['appendix_visible_in_sample'])


if __name__ == '__main__':
    unittest.main()
