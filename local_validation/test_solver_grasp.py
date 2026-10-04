import sys
import unittest
from pathlib import Path
import numpy as np
from pxr import Usd,UsdGeom,UsdPhysics,Sdf
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ct_appendectomy'))
from solver_grasp import SolverTissueGrasp,bind_links


class SolverGraspTests(unittest.TestCase):
    def setUp(self):
        self.stage=Usd.Stage.CreateInMemory()
        self.body=UsdGeom.Xform.Define(self.stage,'/Robot/shaft')
        UsdPhysics.RigidBodyAPI.Apply(self.body.GetPrim()).CreateKinematicEnabledAttr(False)
        UsdGeom.Xform.Define(self.stage,'/patch').AddTransformOp()
        self.attachment=self.stage.DefinePrim('/patchAttachment')
        self.attachment.CreateAttribute('omniphysics:attachmentEnabled',Sdf.ValueTypeNames.Bool)
        self.attachment.CreateAttribute('omniphysics:localPositionsSrc1',Sdf.ValueTypeNames.Point3fArray)
        self.attachment.CreateRelationship('omniphysics:src1')
        self.clusters=[{'path':'/patch','indices':[0,1,2]}]
        self.links={'left':(0,'/Robot/shaft')}

    def test_live_local_coordinates_without_kinematic_writes(self):
        bind_links(self.stage,self.clusters,self.links)
        R=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
        p=np.array([.1,.2,.3]);q=np.array([[.11,.2,.3],[.1,.21,.3],[.1,.2,.31]])
        g=SolverTissueGrasp(self.stage,self.clusters,self.links,lambda h:(R,p))
        g.capture('left',0,q,p,R,evidence='unit test')
        local=np.asarray(self.attachment.GetAttribute('omniphysics:localPositionsSrc1').Get())
        np.testing.assert_allclose(local@R.T+p,q,atol=1e-8)
        before_follow=self.stage.GetRootLayer().ExportToString()
        g.follow('left',p+[1,2,3],np.eye(3))
        self.assertEqual(before_follow,self.stage.GetRootLayer().ExportToString())
        self.assertTrue(g.enabled(self.clusters[0]).Get())
        g.release('left');self.assertFalse(g.enabled(self.clusters[0]).Get())

    def test_rejects_kinematic_target(self):
        UsdPhysics.RigidBodyAPI(self.body.GetPrim()).CreateKinematicEnabledAttr(True)
        with self.assertRaises(ValueError):bind_links(self.stage,self.clusters,self.links)

    def test_rejects_changed_binding(self):
        bind_links(self.stage,self.clusters,self.links)
        g=SolverTissueGrasp(self.stage,self.clusters,self.links,lambda h:(np.eye(3),np.zeros(3)))
        self.attachment.GetRelationship('omniphysics:src1').SetTargets(['/patch'])
        with self.assertRaises(RuntimeError):g.capture('left',0,np.zeros((3,3)),np.zeros(3),np.eye(3),evidence='test')

    def test_invalid_second_binding_does_not_modify_first(self):
        before=self.stage.GetRootLayer().ExportToString()
        with self.assertRaises(ValueError):
            bind_links(self.stage,self.clusters,{'left':(0,'/Robot/shaft'),'right':(1,'/missing')})
        self.assertEqual(before,self.stage.GetRootLayer().ExportToString())

    def test_nonrigid_capture_pose_is_rejected_without_enable(self):
        bind_links(self.stage,self.clusters,self.links)
        g=SolverTissueGrasp(self.stage,self.clusters,self.links,lambda h:(np.eye(3)*2,np.zeros(3)))
        with self.assertRaises(ValueError):g.capture('left',0,np.zeros((3,3)),np.zeros(3),np.eye(3),evidence='test')
        self.assertFalse(g.enabled(self.clusters[0]).Get())
        self.assertFalse(g.held)


if __name__=='__main__':unittest.main()
