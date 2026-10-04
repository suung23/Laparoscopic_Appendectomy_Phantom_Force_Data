"""Rosota clean-export CAD, corrected positive-opening hinge convention.

Triangle visuals with convex-hull collision approximation, not measured contact.
"""
import math
import numpy as np
from pxr import UsdGeom, UsdPhysics, PhysxSchema, UsdShade
from forceps import author_forceps, transform


def author(stage,sim_path,points,base,cad_file):
    data=author_forceps(stage,sim_path,points,base,disabled=True)
    stage.RemovePrim('/World/Forceps')
    data.update(cad=True,length=.021,open_angle=math.radians(37),closed_angle=0.)
    for name in ('left','right','shaft'):
        root=UsdGeom.Xform.Define(stage,'/World/Forceps/'+name)
        root.AddTransformOp()
        UsdPhysics.RigidBodyAPI.Apply(root.GetPrim()).CreateKinematicEnabledAttr(True)
        UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(.004 if name!='shaft' else .04)
        parts=['gripper_tip_'+name] if name!='shaft' else ['laparo_shaft','part_13','part_15','gripper_shaft_inner','gripper_shaft_outer']
        for part in parts:
            prim=stage.DefinePrim(str(root.GetPath())+'/'+part)
            prim.GetReferences().AddReference(str(cad_file.resolve()),'/Tool/'+part)
            prim=stage.GetPrimAtPath(str(prim.GetPath())+'/mesh')
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('convexHull')
            api=PhysxSchema.PhysxCollisionAPI.Apply(prim)
            api.CreateContactOffsetAttr(.0002);api.CreateRestOffsetAttr(0.)
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(UsdShade.Material.Get(stage,'/World/JawContactMaterial'),UsdShade.Tokens.weakerThanDescendants,'physics')
    # Park open jaws anterior of the tissue while gravity settles it onto support.
    data['origin']=data['origin']+np.array([0,.025,0])
    move(stage,data,0.,0.)
    return data


def move(stage,data,closure,lift):
    if not np.isfinite([closure,lift]).all() or not 0<=closure<=1:raise ValueError('Invalid tool command')
    angle=(1-closure)*data['open_angle']
    pivot=data['origin']+np.array([0,lift,0])-data['axes'][:,2]*(.75*data['length'])
    for name,sign in [('left',1),('right',-1),('shaft',0)]:
        theta=sign*angle;c,s=math.cos(theta),math.sin(theta)
        rotation=np.array([[c,0,s],[0,1,0],[-s,0,c]])
        op=UsdGeom.Xformable(stage.GetPrimAtPath('/World/Forceps/'+name)).GetOrderedXformOps()[0]
        op.Set(transform(data['axes']@rotation,pivot))
