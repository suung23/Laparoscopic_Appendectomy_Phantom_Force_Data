"""Attachment transitions and geometry; does not certify contact physics."""
import sys
import unittest
from pathlib import Path
import numpy as np
from pxr import Usd, UsdGeom, Sdf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ct_appendectomy'))
from tissue_grasp import TissueGrasp


class GraspTransitions(unittest.TestCase):
    def setUp(self):
        self.stage = Usd.Stage.CreateInMemory()
        self.points = np.array([[-.001, 0, 0], [.001, 0, 0],
                                [0, -.001, 0], [0, .001, 0]])
        self.clusters = [dict(path='/patch', indices=[0, 1, 2, 3])]
        UsdGeom.Xform.Define(self.stage, '/patch').AddTransformOp()
        prim = self.stage.DefinePrim('/patchAttachment')
        prim.CreateAttribute('omniphysics:attachmentEnabled', Sdf.ValueTypeNames.Bool)
        prim.CreateAttribute('omniphysics:localPositionsSrc1', Sdf.ValueTypeNames.Point3fArray)
        self.grasp = TissueGrasp(self.stage, self.clusters)

    def test_live_rebase_and_measured_rigid_follow(self):
        p = self.points + [.1, .2, .3]
        tip = np.array([.1, .2, .3])
        self.grasp.capture('left', 0, p, tip, np.eye(3), evidence='test')
        offsets = self.stage.GetPrimAtPath('/patchAttachment').GetAttribute(
            'omniphysics:localPositionsSrc1').Get()
        np.testing.assert_allclose(offsets, self.points, atol=1e-9)
        self.grasp.follow('left', tip + [0, .002, 0], np.eye(3))
        M = np.asarray(UsdGeom.Xformable(self.stage.GetPrimAtPath('/patch')).ComputeLocalToWorldTransform(0))
        np.testing.assert_allclose(M[3, :3], tip + [0, .002, 0])
        np.testing.assert_allclose(p, self.points + tip)
        with self.assertRaises(ValueError):
            self.grasp.capture('right', 0, p, tip, np.eye(3), evidence='test')
        self.grasp.release('left')
        self.assertFalse(self.grasp.enabled(self.clusters[0]).Get())

    def test_outside_mouth_cannot_capture_even_with_contact_signal(self):
        self.grasp.update('left', self.points, np.array([.02, 0, 0]),
                          np.eye(3), 0, True, 0)
        self.assertFalse(self.grasp.held)
        self.grasp.preferred = {'left': 1}
        self.assertIsNone(self.grasp.probe('left', self.points, np.zeros(3), np.eye(3)))

    def test_patch_rotates_about_measured_tool_tip_without_node_edit(self):
        tip=np.array([.002,0.,0.])
        self.grasp.capture('left',0,self.points,tip,np.eye(3),evidence='test')
        R=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
        self.grasp.follow('left',tip,R)
        M=np.asarray(UsdGeom.Xformable(self.stage.GetPrimAtPath('/patch')).ComputeLocalToWorldTransform(0))
        offsets=np.asarray(self.stage.GetPrimAtPath('/patchAttachment').GetAttribute(
            'omniphysics:localPositionsSrc1').Get())
        np.testing.assert_allclose(offsets@M[:3,:3]+M[3,:3],
            [[.002,-.003,0],[.002,-.001,0],[.003,-.002,0],[.001,-.002,0]],atol=1e-9)

    def test_physics_pose_is_written_after_target_and_before_enable(self):
        calls=[]
        def write(c,R,p):
            M=np.asarray(UsdGeom.Xformable(self.stage.GetPrimAtPath(c['path'])).ComputeLocalToWorldTransform(0))
            np.testing.assert_allclose(M[3,:3],p)
            self.assertFalse(self.grasp.enabled(c).Get())
            calls.append(p.copy())
        self.grasp.pose_writer=write
        self.grasp.capture('left',0,self.points+[.01,.02,.03],np.zeros(3),np.eye(3),evidence='test')
        self.assertEqual(len(calls),1)
        self.assertTrue(self.grasp.enabled(self.clusters[0]).Get())

    def test_close_capture_open_release(self):
        hit = self.grasp.probe('left', self.points, np.zeros(3), np.eye(3))
        self.assertIsNotNone(hit)
        self.grasp.update('left', self.points, np.zeros(3), np.eye(3), 0, False, 74)
        self.assertFalse(self.grasp.held)
        self.grasp.update('left', self.points, np.zeros(3), np.eye(3), 0, True, 74)
        self.assertFalse(self.grasp.held)
        self.grasp.update('left', self.points, np.zeros(3), np.eye(3), 0, False, hit[1])
        self.assertIn('left', self.grasp.held)
        self.grasp.update('left', self.points, np.zeros(3), np.eye(3), 100, False, hit[1])
        self.assertFalse(self.grasp.held)


if __name__ == '__main__':
    unittest.main()
