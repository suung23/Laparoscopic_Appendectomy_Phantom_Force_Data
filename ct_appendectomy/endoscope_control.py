"""Bounded camera pan/tilt/roll and translation within a prevalidated free ball."""
import math
import numpy as np
from pxr import Gf, UsdGeom


class EndoscopeControl:
    def __init__(self,stage,travel):
        self.camera=UsdGeom.Xformable(stage.GetPrimAtPath('/World/Endoscope')).GetOrderedXformOps()[0]
        self.light=UsdGeom.Xformable(stage.GetPrimAtPath('/World/ScopeLight')).GetOrderedXformOps()[0]
        matrix=np.asarray(self.camera.Get())
        self.origin=matrix[3,:3].copy()
        self.basis=matrix[:3,:3].T.copy()
        self.travel=float(travel)
        self.apply(0,0,0,0,0,0)

    def apply(self,pan,tilt,roll,right,up,insertion):
        values=np.array([pan,tilt,roll,right,up,insertion],float)
        if not np.isfinite(values).all():raise ValueError('Non-finite camera command')
        pan,tilt,roll=np.deg2rad(np.clip(values[:3],[-35,-30,-90],[35,30,90]))
        def rotation(axis,a):
            c,s=math.cos(a),math.sin(a)
            if axis==0:return np.array([[1,0,0],[0,c,-s],[0,s,c]])
            if axis==1:return np.array([[c,0,s],[0,1,0],[-s,0,c]])
            return np.array([[c,-s,0],[s,c,0],[0,0,1]])
        offset=np.clip(values[3:],-1,1)*self.travel
        norm=np.linalg.norm(offset)
        if norm>self.travel and norm:offset*=self.travel/norm
        offset[2]*=-1
        eye=self.origin+self.basis@offset
        orientation=self.basis@rotation(1,pan)@rotation(0,tilt)@rotation(2,roll)
        matrix=np.eye(4);matrix[:3,:3]=orientation.T;matrix[3,:3]=eye
        self.camera.Set(Gf.Matrix4d(matrix.tolist()))
        self.light.Set(Gf.Vec3d(*eye))
        self.pose=matrix
        return matrix

    def scan(self,fraction):
        phase=2*math.pi*fraction
        return self.apply(18*math.sin(phase),10*math.sin(2*phase),0,
                          .6*math.sin(phase),.3*math.sin(2*phase),.4*math.cos(phase))

    def create_ui(self):
        import omni.ui as ui
        self.window=ui.Window('Endoscope exploration',width=360,height=420,position_x=15,position_y=80)
        self.sliders=[]
        with self.window.frame:
            with ui.VStack(spacing=5):
                ui.Label('Camera and light move together')
                for title,lo,hi in [('Pan (deg)',-35,35),('Tilt (deg)',-30,30),('Roll (deg)',-90,90),
                                    ('Right / left',-1,1),('Up / down',-1,1),('Insertion / withdrawal',-1,1)]:
                    ui.Label(title);slider=ui.FloatSlider(min=lo,max=hi)
                    slider.model.set_value(0.);self.sliders.append(slider)
                ui.Label('Translation confined to collision-checked free space')
                ui.Button('Reset view',clicked_fn=self.reset_ui)

    def reset_ui(self):
        for slider in self.sliders:slider.model.set_value(0.)

    def update_ui(self):
        self.apply(*(slider.model.get_value_as_float() for slider in self.sliders))
