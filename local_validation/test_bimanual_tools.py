import sys
import unittest
from pathlib import Path
import numpy as np
from pxr import Usd, UsdGeom, Sdf, Gf
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ct_appendectomy'))
from bimanual_tools import BimanualControl, ROOT, transform


class BimanualTests(unittest.TestCase):
    def make_controller(self):
        stage=Usd.Stage.CreateInMemory()
        offsets=np.array(np.meshgrid(*([[-.001,.001]]*3))).reshape(3,-1).T
        centers=np.array([[0,0,.1],[.025,0,.1]])
        points=np.vstack([offsets+c for c in centers])
        mesh=UsdGeom.Mesh.Define(stage,'/World/Tissue');mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in points])
        clusters=[]
        for i,center in enumerate(centers):
            path=ROOT+'/Grip'+str(i)
            UsdGeom.Xform.Define(stage,path).AddTransformOp().Set(transform(np.eye(3),center))
            prim=stage.DefinePrim(path+'Attachment')
            prim.CreateAttribute('omniphysics:attachmentEnabled',Sdf.ValueTypeNames.Bool).Set(True)
            prim.CreateAttribute('omniphysics:localPositionsSrc1',Sdf.ValueTypeNames.Point3fArray).Set([Gf.Vec3f(*p) for p in offsets])
            clusters.append({'indices':np.arange(i*8,(i+1)*8),'path':path,'origin':center})
        arms={}
        for h,c in zip(('Left','Right'),centers):
            for part in ('left','right','shaft'):UsdGeom.Xform.Define(stage,ROOT+'/'+h+'/'+part).AddTransformOp()
            arms[h]={'port':c-[0,0,.1],'axes':np.eye(3),'state':np.array([0,0,.1,0,.1])}
        data={'sim_path':'/World/Tissue','clusters':clusters,'axis':np.array([0,1,0]),'arms':arms,'pin':np.array([],int)}
        control=BimanualControl(stage,data,points)
        for h in arms:control.targets[h][4]=0.;control.grasp_requested[h]=True
        return control

    def test_independent_grasps_release_and_no_snap_target(self):
        c=self.make_controller();before=c.points().copy();c.update_grasps()
        self.assertEqual({v['cluster'] for v in c.held.values()},{0,1})
        np.testing.assert_array_equal(c.points(),before)
        for h,held in c.held.items():
            cluster=c.data['clusters'][held['cluster']]
            matrix=np.asarray(UsdGeom.Xformable(c.stage.GetPrimAtPath(cluster['path'])).GetOrderedXformOps()[0].Get())
            local=np.asarray(c.stage.GetPrimAtPath(cluster['path']+'Attachment').GetAttribute('omniphysics:localPositionsSrc1').Get())
            np.testing.assert_allclose(local+matrix[3,:3],before[cluster['indices']],atol=1e-8)
            c.arms[h].state[4]=.6
            c.grasp_requested[h]=False
        c.update_grasps();self.assertFalse(c.held)

    def test_empty_space_and_exclusive_ownership(self):
        c=self.make_controller();c.arms['Left'].state[2]=.05;c.arms['Right'].state[2]=.05
        c.update_grasps();self.assertFalse(c.held)
        c.arms['Left'].state[2]=.1;c.arms['Right'].port=c.arms['Left'].port.copy();c.arms['Right'].state[2]=.1
        c.update_grasps();self.assertEqual(len(c.held),1)

    def test_invalid_action_is_atomic_and_port_inverse(self):
        c=self.make_controller();before=c.observation().copy()
        with self.assertRaises(ValueError):c.step_action(np.r_[np.zeros(9),np.nan])
        np.testing.assert_array_equal(c.observation(),before)
        for h in c.arms:
            target=c.arms[h].port+np.array([.01,-.015,.11]);c.target_position(h,target)
            c.arms[h].state=c.targets[h].copy()
            pose=c.arms[h].pose();np.testing.assert_allclose(pose['position_m'],target,atol=1e-12)
            residual=np.cross(pose['position_m']-c.arms[h].port,pose['axes'][:,2])
            np.testing.assert_allclose(residual,0,atol=1e-12)

    def test_velocity_actions_can_grasp_and_release(self):
        c=self.make_controller();c.grasp_requested={'Left':False,'Right':False}
        action=np.zeros(10);action[[4,9]]=-1
        for _ in range(6):c.step_action(action)
        self.assertEqual(len(c.held),2)
        self.assertGreater(c.observation()[4],0.)
        for _ in range(20):c.step_action(action)
        held_angles={h:c.held[h]['angle'] for h in c.arms}
        for h in c.arms:self.assertAlmostEqual(c.arms[h].state[4],held_angles[h])
        action[[4,9]]=1;c.step_action(action)
        self.assertFalse(c.held)
        for h in c.arms:self.assertGreater(c.arms[h].state[4],held_angles[h])

    def test_roll_wrap_does_not_spin_at_pi(self):
        c=self.make_controller()
        for h in c.arms:
            c.arms[h].state[3]=-np.pi;c.targets[h][3]=np.pi
            c.targets[h][4]=c.arms[h].state[4]
        c.tick()
        for h in c.arms:self.assertAlmostEqual(c.arms[h].state[3],-np.pi)


if __name__=='__main__':unittest.main()
