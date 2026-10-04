"""Geometry safety invariants for interactive camera motion."""
import sys
import unittest
from pathlib import Path
import numpy as np
from pxr import Usd, UsdGeom, UsdLux, Gf
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ct_appendectomy'))
from endoscope_control import EndoscopeControl


class EndoscopeTests(unittest.TestCase):
    def test_camera_light_bounded_and_reversible(self):
        stage=Usd.Stage.CreateInMemory()
        camera=UsdGeom.Camera.Define(stage,'/World/Endoscope')
        pose=Gf.Matrix4d().SetLookAt(Gf.Vec3d(.1,.05,.02),Gf.Vec3d(.12,0,0),Gf.Vec3d(0,0,1)).GetInverse()
        camera.AddTransformOp().Set(pose)
        UsdLux.SphereLight.Define(stage,'/World/ScopeLight').AddTranslateOp().Set(Gf.Vec3d(.1,.05,.02))
        controller=EndoscopeControl(stage,.008)
        for fraction in np.linspace(0,1,91):
            matrix=controller.scan(fraction)
            self.assertLessEqual(np.linalg.norm(matrix[3,:3]-controller.origin),.008+1e-12)
            np.testing.assert_allclose(matrix[:3,:3]@matrix[:3,:3].T,np.eye(3),atol=1e-12)
            np.testing.assert_allclose(np.asarray(controller.light.Get()),matrix[3,:3])
        np.testing.assert_allclose(controller.apply(0,0,0,0,0,0),np.asarray(pose),atol=1e-12)
        before=np.asarray(controller.camera.Get()).copy()
        with self.assertRaises(ValueError):controller.apply(float('nan'),0,0,0,0,0)
        np.testing.assert_array_equal(np.asarray(controller.camera.Get()),before)


if __name__=='__main__':unittest.main()
