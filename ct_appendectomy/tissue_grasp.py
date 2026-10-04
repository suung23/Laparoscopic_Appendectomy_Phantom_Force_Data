"""Explicit Rosota-style FEM patch assistance, driven by measured tool poses.

This is an attachment approximation, not friction-only grasp physics. No node
positions are assigned: PhysX solves the deformation. Live rebasing avoids
snapping nodes back to the authored rest positions when an attachment starts.
"""
import numpy as np


class TissueGrasp:
    def __init__(self,stage,clusters,preferred=None,pose_writer=None):
        self.stage=stage;self.clusters=clusters;self.held={};self.events=[]
        self.pose_writer=pose_writer
        self.preferred=preferred or {}
        for c in clusters:self.enabled(c).Set(False)

    def enabled(self,c):
        return self.stage.GetPrimAtPath(c['path']+'Attachment').GetAttribute('omniphysics:attachmentEnabled')

    def set_handle(self,c,R,p):
        from pxr import UsdGeom,Gf
        M=np.eye(4);M[:3,:3]=R.T;M[3,:3]=p
        UsdGeom.Xformable(self.stage.GetPrimAtPath(c['path'])).GetOrderedXformOps()[0].Set(Gf.Matrix4d(M.tolist()))
        # Keep the queued kinematic target and immediate physics pose aligned.
        # Updating only one either lags activation or snaps back next tick.
        if self.pose_writer is not None:self.pose_writer(c,R,p)

    def capture(self,hand,index,points,tip,axes,*,evidence):
        from pxr import Gf
        if hand in self.held or index in [v['cluster'] for v in self.held.values()]:raise ValueError('Patch already owned')
        c=self.clusters[index];q=np.asarray(points)[c['indices']];center=q.mean(0)
        self.set_handle(c,np.eye(3),center)
        self.stage.GetPrimAtPath(c['path']+'Attachment').GetAttribute('omniphysics:localPositionsSrc1').Set(
            [Gf.Vec3f(*v) for v in q-center])
        self.held[hand]=dict(cluster=index,center=center.copy(),tip=np.asarray(tip).copy(),axes=np.asarray(axes).copy())
        self.enabled(c).Set(True)
        self.events.append(dict(event='capture',hand=hand,cluster=index,evidence=evidence))

    def release(self,hand):
        held=self.held.pop(hand,None)
        if held is not None:
            self.enabled(self.clusters[held['cluster']]).Set(False)
            self.events.append(dict(event='release',hand=hand,cluster=held['cluster']))

    def follow(self,hand,tip,axes):
        held=self.held[hand];R=axes@held['axes'].T
        p=tip+R@(held['center']-held['tip'])
        self.set_handle(self.clusters[held['cluster']],R,p)

    def probe(self,hand,points,tip,axes):
        local=(points-tip)@axes
        mask=(np.abs(local[:,0])<.014)&(np.abs(local[:,1])<.0035)&(np.abs(local[:,2])<.005)
        if mask.sum()<4:return None
        owned={v['cluster'] for v in self.held.values()};candidates=[]
        for i,c in enumerate(self.clusters):
            if i in owned or (hand in self.preferred and i!=self.preferred[hand]):continue
            centre=points[c['indices']].mean(0);v=(centre-tip)@axes
            if abs(v[0])>.004 or abs(v[1])>.0035 or abs(v[2])>.005:continue
            candidates.append((float(np.linalg.norm(v)),i))
        if not candidates:return None
        mouth=local[mask]
        angle=np.degrees(2*np.quantile(np.arctan2(np.abs(mouth[:,0])+.0005,np.maximum(.004,mouth[:,2]+.014)),.98))
        if angle>=70:return None
        return min(candidates)[1],float(np.clip(angle,1,70))

    def update(self,hand,points,tip,axes,opening,contact_latched,measured_deg):
        if hand in self.held:
            if opening>0:self.release(hand)
            else:self.follow(hand,tip,axes)
            return
        if opening>0:return
        hit=self.probe(hand,points,tip,axes)
        if hit is None:return
        index,floor=hit
        # A residual from an earlier collision cannot latch a still-open jaw.
        if measured_deg>floor+1.5:return
        self.capture(hand,index,points,tip,axes,evidence='live mouth geometry with contact-envelope floor; explicit attachment approximation')
        self.events[-1]['jaw_contact_latched']=bool(contact_latched)
        self.held[hand]['jaw_deg']=measured_deg
