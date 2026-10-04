"""Geometry-gated attachment directly to dynamic robot links.

This remains idealized adhesion, not frictional grasp. The solver owns both
the rigid link and deformable nodes; no kinematic handle follows the robot.
"""
import numpy as np
from pxr import Gf, Sdf, UsdPhysics
from tissue_grasp import TissueGrasp


def bind_links(stage, clusters, links):
    """Call before physics initialization, with hand -> (patch index, body path)."""
    if len({index for index,_ in links.values()}) != len(links):
        raise ValueError('Each hand needs an exclusive patch')
    validated=[]
    for hand,(index,path) in links.items():
        if not isinstance(index,(int,np.integer)) or not 0<=index<len(clusters):
            raise ValueError('Invalid patch index')
        prim=stage.GetPrimAtPath(path)
        if not prim or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError('Attachment target must be a rigid body')
        if UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get():
            raise ValueError('Two-way grasp requires a dynamic target')
        if UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get() is False:
            raise ValueError('Rigid body simulation must be enabled')
        attachment=stage.GetPrimAtPath(clusters[index]['path']+'Attachment')
        if not attachment or not attachment.GetAttribute('omniphysics:attachmentEnabled'):
            raise ValueError('Missing authored attachment')
        validated.append((attachment,path))
    for attachment,path in validated:
        attachment.GetAttribute('omniphysics:attachmentEnabled').Set(False)
        attachment.GetRelationship('omniphysics:src1').SetTargets([Sdf.Path(path)])


class SolverTissueGrasp(TissueGrasp):
    def __init__(self,stage,clusters,links,body_pose_reader):
        super().__init__(stage,clusters,preferred={h:i for h,(i,_) in links.items()})
        self.links,self.body_pose_reader=links,body_pose_reader

    def capture(self,hand,index,points,tip,axes,*,evidence):
        if hand in self.held or index in [v['cluster'] for v in self.held.values()]:
            raise ValueError('Patch already owned')
        expected,path=self.links[hand]
        if index!=expected:raise ValueError('Wrong solver-bound patch')
        c=self.clusters[index]
        attachment=self.stage.GetPrimAtPath(c['path']+'Attachment')
        if list(attachment.GetRelationship('omniphysics:src1').GetTargets())!=[Sdf.Path(path)]:
            raise RuntimeError('Attachment target changed after initialization')
        R,p=self.body_pose_reader(hand)
        R,p=np.asarray(R,float),np.asarray(p,float)
        if R.shape!=(3,3) or p.shape!=(3,) or not np.isfinite(R).all() or not np.isfinite(p).all():
            raise ValueError('Invalid actual rigid body pose')
        if not np.allclose(R.T@R,np.eye(3),atol=1e-5) or not np.isclose(np.linalg.det(R),1,atol=1e-5):
            raise ValueError('Rigid body rotation required')
        q=np.asarray(points,float)[c['indices']]
        if not np.isfinite(q).all():raise ValueError('Invalid live tissue')
        attachment.GetAttribute('omniphysics:localPositionsSrc1').Set(
            [Gf.Vec3f(*v) for v in (q-p)@R])
        self.held[hand]=dict(cluster=index,center=q.mean(0),tip=np.asarray(tip).copy(),
                             axes=np.asarray(axes).copy())
        self.enabled(c).Set(True)
        self.events.append(dict(event='capture',hand=hand,cluster=index,evidence=evidence,
            constraint_target=path,coupling='solver-owned dynamic rigid link',frictional_grasp=False))

    def follow(self,hand,tip,axes):
        # Deliberately no USD or physics pose writes: the coupled solver owns
        # rigid motion and constraint reaction throughout the attached phase.
        if hand not in self.held:raise ValueError('No captured patch')
