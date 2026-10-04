"""Kinematic blunt-tip contact with a base-pinned FEM appendix."""
import numpy as np
from pxr import Gf,UsdGeom,UsdPhysics,PhysxSchema
from attachments import author_retraction

def author_contact(stage,sim_path,points,base,disabled=False,ct_contact=False):
    data=author_retraction(stage,sim_path,points,base)
    stage.RemovePrim('/World/GripAttachment')
    stage.RemovePrim('/World/RetractionHandle')
    points=np.asarray(points)
    distance=np.linalg.norm(points-np.asarray(base),axis=1)
    free=np.flatnonzero(distance>np.percentile(distance,65))
    selected=free[np.argmax(points[free,1])]
    origin=points[selected]+np.array([0,.009,0])
    tip=UsdGeom.Sphere.Define(stage,'/World/BluntProbe')
    tip.CreateRadiusAttr(.005)
    tip.CreateDisplayColorAttr([Gf.Vec3f(.5,.6,.7)])
    op=tip.AddTranslateOp();op.Set(Gf.Vec3d(*origin))
    body=UsdPhysics.RigidBodyAPI.Apply(tip.GetPrim());body.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(tip.GetPrim()).CreateMassAttr(.05)
    collision=UsdPhysics.CollisionAPI.Apply(tip.GetPrim())
    collision.CreateCollisionEnabledAttr(not disabled)
    api=PhysxSchema.PhysxCollisionAPI.Apply(tip.GetPrim())
    api.CreateContactOffsetAttr(.0005);api.CreateRestOffsetAttr(0.)
    colliders=[]
    for prim in stage.GetPrimAtPath('/World/CT').GetChildren():
        if prim.IsA(UsdGeom.Mesh):
            UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(ct_contact)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('none')
            colliders.append(str(prim.GetPath()))
    return dict(pin=data['pin'],origin=origin,translation=op,colliders=colliders)
