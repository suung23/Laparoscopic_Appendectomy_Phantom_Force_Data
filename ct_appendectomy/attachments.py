"""Explicit FEM attachment benchmark; not a contact-based grasp model."""
import numpy as np
from pxr import Gf,Sdf,UsdGeom,UsdPhysics


def attach(stage,path,sim_path,handle_path,points,indices,origin):
    prim=stage.DefinePrim(path,'OmniPhysicsVtxXformAttachment')
    prim.CreateAttribute('omniphysics:attachmentEnabled',Sdf.ValueTypeNames.Bool).Set(True)
    prim.CreateAttribute('omniphysics:vtxIndicesSrc0',Sdf.ValueTypeNames.IntArray).Set([int(i) for i in indices])
    prim.CreateAttribute('omniphysics:localPositionsSrc1',Sdf.ValueTypeNames.Point3fArray).Set(
        [Gf.Vec3f(*(points[i]-origin)) for i in indices])
    prim.CreateRelationship('omniphysics:src0').SetTargets([Sdf.Path(sim_path)])
    prim.CreateRelationship('omniphysics:src1').SetTargets([Sdf.Path(handle_path)])


def author_retraction(stage,sim_path,points,base):
    points=np.asarray(points,float)
    base=np.asarray(base,float)
    distances=np.linalg.norm(points-base,axis=1)
    pin=np.flatnonzero(distances<=max(.004,float(distances.min())+.002))
    direction=points[np.argmax(distances)]-base
    projection=(points-base) @ (direction/np.linalg.norm(direction))
    desired=points[np.argmin(np.abs(projection-.65*projection.max()))]
    grip=np.flatnonzero(np.linalg.norm(points-desired,axis=1)<.005)
    grip=np.setdiff1d(grip,pin)
    if not len(pin) or not len(grip): raise RuntimeError('Empty attachment cluster')
    origin=points[grip].mean(axis=0)
    fixed=UsdGeom.Xform.Define(stage,'/World/BaseAnchor')
    fixed.AddTranslateOp().Set(Gf.Vec3d(0,0,0))
    handle=UsdGeom.Xform.Define(stage,'/World/RetractionHandle')
    op=handle.AddTranslateOp()
    op.Set(Gf.Vec3d(*origin))
    body=UsdPhysics.RigidBodyAPI.Apply(handle.GetPrim())
    body.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(handle.GetPrim()).CreateMassAttr(.01)
    marker=UsdGeom.Sphere.Define(stage,'/World/RetractionHandle/marker')
    marker.CreateRadiusAttr(.0015)
    marker.CreateDisplayColorAttr([Gf.Vec3f(.05,.5,.8)])
    attach(stage,'/World/BaseAttachment',sim_path,str(fixed.GetPath()),points,pin,np.zeros(3))
    attach(stage,'/World/GripAttachment',sim_path,str(handle.GetPath()),points,grip,origin)
    return {'pin':pin,'grip':grip,'origin':origin,'translation':op}
