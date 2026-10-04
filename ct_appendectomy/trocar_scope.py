"""Rigid scope pivot: camera orientation and position share a fixed port."""
import numpy as np
from instrument_kinematics import PortInstrument, InstrumentLimits


class TrocarScope:
    def __init__(self, camera_pose, insertion=.12):
        T=np.asarray(camera_pose,float)
        if T.shape!=(4,4) or not np.isfinite(T).all():raise ValueError('Camera SE3 required')
        self.initial=T.copy()
        # Input is conventional column SE3, USD optical axis is local -z.
        axes=T[:3,:3]@np.diag([1.,-1.,-1.])
        self.instrument=PortInstrument(T[:3,3]-axes[:,2]*insertion,axes,
            InstrumentLimits(insertion_min_m=.10,insertion_max_m=.16))
        self.instrument.state[2]=insertion
        self.pose=self.matrix()

    def matrix(self):
        p=self.instrument.pose();T=np.eye(4)
        T[:3,:3]=p['axes']@np.diag([1.,-1.,-1.]);T[:3,3]=p['position_m']
        return T

    def step(self, velocity, dt):
        a=np.asarray(velocity,float)
        if a.shape!=(4,) or not np.isfinite(a).all() or np.any(np.abs(a)>1):
            raise ValueError('Four finite normalized scope velocities required')
        self.instrument.step([a[0],a[1],a[2],a[3],0],dt)
        self.pose=self.matrix();return self.pose

    def metrics(self):
        m=self.instrument.trocar_metrics();m['controller']='rigid scope geometric RCM'
        m['normal_source']='engineering scope port; not CT wall calibration'
        return m

    def apply_usd(self,stage):
        from pxr import Gf,UsdGeom
        UsdGeom.Xformable(stage.GetPrimAtPath('/World/Endoscope')).GetOrderedXformOps()[0].Set(Gf.Matrix4d(self.pose.T.tolist()))
        UsdGeom.Xformable(stage.GetPrimAtPath('/World/ScopeLight')).GetOrderedXformOps()[0].Set(Gf.Vec3d(*self.pose[:3,3]))
