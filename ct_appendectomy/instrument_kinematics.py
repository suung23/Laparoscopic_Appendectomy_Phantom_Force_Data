"""Port-constrained instrument commands in metres/radians.

The mathematical pivot is exact; port anatomy and trocar compliance are separate.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class InstrumentLimits:
    tilt_rad: float = np.pi/3
    insertion_min_m: float = .03
    insertion_max_m: float = .25
    rotation_speed_rad_s: float = .4
    insertion_speed_m_s: float = .02
    jaw_speed_rad_s: float = .6
    jaw_max_rad: float = np.pi/4

    def __post_init__(self):
        if not all(np.isfinite(value) for value in self.__dict__.values()):raise ValueError('Finite limits required')
        if not 0<self.tilt_rad<np.pi/2 or not 0<self.insertion_min_m<self.insertion_max_m:raise ValueError('Invalid workspace limits')
        if min(self.rotation_speed_rad_s,self.insertion_speed_m_s,self.jaw_speed_rad_s,self.jaw_max_rad)<=0:raise ValueError('Positive speed/jaw limits required')


class PortInstrument:
    def __init__(self, port_m, port_axes=None, limits=None):
        self.port=np.asarray(port_m,float).copy()
        self.axes=np.eye(3) if port_axes is None else np.asarray(port_axes,float).copy()
        if self.port.shape!=(3,) or not np.isfinite(self.port).all():
            raise ValueError('Finite 3D port required')
        if self.axes.shape!=(3,3) or not np.isfinite(self.axes).all() or not np.allclose(self.axes.T@self.axes,np.eye(3),atol=1e-8) or not np.isclose(np.linalg.det(self.axes),1):
            raise ValueError('Proper orthonormal port frame required')
        self.limits=limits or InstrumentLimits()
        self.reset()

    def reset(self):
        # yaw, pitch, insertion, roll, jaw opening; no virtual port displacement.
        self.state=np.array([0.,0.,self.limits.insertion_min_m,0.,self.limits.jaw_max_rad])
        return self.pose()

    def step(self, action, dt):
        action=np.asarray(action,float)
        if action.shape!=(5,) or not np.isfinite(action).all() or np.any(np.abs(action)>1):
            raise ValueError('Five finite normalized actions required')
        if not np.isfinite(dt) or dt<=0 or dt>.1:
            raise ValueError('Control dt must be in (0, 0.1] seconds')
        lim=self.limits
        speed=np.array([lim.rotation_speed_rad_s,lim.rotation_speed_rad_s,
                        lim.insertion_speed_m_s,lim.rotation_speed_rad_s,lim.jaw_speed_rad_s])
        candidate=self.state+action*speed*dt
        candidate[:2]=np.clip(candidate[:2],-lim.tilt_rad,lim.tilt_rad)
        # Enforce the total cone, rather than just two independent angle limits.
        total=np.arccos(np.clip(np.cos(candidate[0])*np.cos(candidate[1]),-1,1))
        if total>lim.tilt_rad:
            low,high=0.,1.
            for _ in range(32):
                factor=(low+high)/2
                angle=np.arccos(np.clip(np.cos(candidate[0]*factor)*np.cos(candidate[1]*factor),-1,1))
                if angle>lim.tilt_rad: high=factor
                else: low=factor
            candidate[:2]*=low
        candidate[2]=np.clip(candidate[2],lim.insertion_min_m,lim.insertion_max_m)
        candidate[3]=(candidate[3]+np.pi)%(2*np.pi)-np.pi
        candidate[4]=np.clip(candidate[4],0,lim.jaw_max_rad)
        self.state=candidate
        return self.pose()

    def pose(self):
        yaw,pitch,insertion,roll,jaw=self.state
        cy,sy=np.cos(yaw),np.sin(yaw)
        cp,sp=np.cos(pitch),np.sin(pitch)
        cr,sr=np.cos(roll),np.sin(roll)
        rotation=self.axes@np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])@np.array([[1,0,0],[0,cp,-sp],[0,sp,cp]])@np.array([[cr,-sr,0],[sr,cr,0],[0,0,1]])
        return {'position_m':self.port+rotation[:,2]*insertion,'axes':rotation,
                'jaw_angle_rad':float(jaw),'insertion_m':float(insertion)}
