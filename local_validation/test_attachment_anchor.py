import sys
import unittest
from pathlib import Path
import numpy as np
from pxr import Usd,UsdPhysics
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ct_appendectomy'))
from attachments import author_retraction


class AnchorAuthoring(unittest.TestCase):
    def test_fixed_rigid_anchor_preserves_pin_rest_coordinates(self):
        stage=Usd.Stage.CreateInMemory()
        points=np.column_stack((np.linspace(0,.05,101),np.zeros((101,2))))
        result=author_retraction(stage,'/Tissue/sim',points,np.zeros(3))
        anchor=stage.GetPrimAtPath('/World/BaseAnchor')
        self.assertTrue(anchor.HasAPI(UsdPhysics.RigidBodyAPI))
        self.assertTrue(UsdPhysics.RigidBodyAPI(anchor).GetKinematicEnabledAttr().Get())
        attachment=stage.GetPrimAtPath('/World/BaseAttachment')
        local=np.asarray(attachment.GetAttribute('omniphysics:localPositionsSrc1').Get())
        np.testing.assert_allclose(local,points[result['pin']],atol=1e-9)
        self.assertEqual(str(attachment.GetRelationship('omniphysics:src1').GetTargets()[0]),'/World/BaseAnchor')


if __name__=='__main__':unittest.main()
