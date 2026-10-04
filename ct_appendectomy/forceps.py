"""Two hinged kinematic jaws contacting FEM tissue, without a grasp attachment.

Engineering contact probe: estimated dimensions/friction, no clinical CAD claim.
"""
import math
import numpy as np
from pxr import Gf, UsdGeom, UsdPhysics, UsdShade, PhysxSchema
from attachments import author_retraction


def transform(axes, position):
    matrix=np.eye(4)
    matrix[:3,:3]=axes.T
    matrix[3,:3]=position
    return Gf.Matrix4d(matrix.tolist())


def author_forceps(stage,sim_path,points,base,disabled=False):
    data=author_retraction(stage,sim_path,points,base)
    stage.RemovePrim('/World/GripAttachment')
    stage.RemovePrim('/World/RetractionHandle')
    points=np.asarray(points)
    distance=np.linalg.norm(points-np.asarray(base),axis=1)
    tissue_axis=points[np.argmax(distance)]-np.asarray(base)
    tissue_axis/=np.linalg.norm(tissue_axis)
    projection=(points-np.asarray(base))@tissue_axis
    band=np.abs(projection-.65*projection.max())<.0025
    if np.count_nonzero(band)<4:raise RuntimeError('Insufficient jaw cross-section nodes')
    origin=points[band].mean(axis=0)
    sensor=np.flatnonzero(np.linalg.norm(points-origin,axis=1)<.004)
    sensor=np.setdiff1d(sensor,data['pin'])
    if not len(sensor):raise RuntimeError('No tissue nodes in jaw centre')
    forward=np.array([0.,-1.,-.15])
    forward/=np.linalg.norm(forward)
    close=np.cross(tissue_axis,forward)
    close/=np.linalg.norm(close)
    up=np.cross(forward,close)
    axes=np.column_stack((close,up,forward))
    near=points[np.linalg.norm(points-origin,axis=1)<.008]
    diameter=float(np.ptp((near-origin)@close))
    length=.018
    thickness=.0015
    closed=math.asin(min(.75,(.75*diameter+thickness)/(1.5*length)))
    material=UsdShade.Material.Define(stage,'/World/JawContactMaterial')
    physical=UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physical.CreateStaticFrictionAttr(1.0)
    physical.CreateDynamicFrictionAttr(.8)
    physical.CreateRestitutionAttr(0.)
    for name in ('left','right'):
        root=UsdGeom.Xform.Define(stage,'/World/Forceps/'+name)
        root.AddTransformOp()
        rigid=UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
        rigid.CreateKinematicEnabledAttr(True)
        UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(.01)
        jaw=UsdGeom.Cube.Define(stage,str(root.GetPath())+'/jaw')
        jaw.CreateSizeAttr(1.)
        jaw.AddScaleOp().Set(Gf.Vec3f(thickness,.003,length))
        jaw.CreateDisplayColorAttr([Gf.Vec3f(.28,.31,.34)])
        collision=UsdPhysics.CollisionAPI.Apply(jaw.GetPrim())
        collision.CreateCollisionEnabledAttr(not disabled)
        api=PhysxSchema.PhysxCollisionAPI.Apply(jaw.GetPrim())
        api.CreateContactOffsetAttr(.0003)
        api.CreateRestOffsetAttr(0.)
        UsdShade.MaterialBindingAPI.Apply(jaw.GetPrim()).Bind(material,UsdShade.Tokens.weakerThanDescendants,'physics')
        # Explicit inner ridges provide geometric purchase rather than a weld.
        inner=1 if name=='left' else -1
        for number,z in enumerate(np.linspace(-.006,.006,7)):
            ridge=UsdGeom.Cube.Define(stage,str(root.GetPath())+'/ridge_'+str(number))
            ridge.CreateSizeAttr(1.)
            ridge.AddTranslateOp().Set(Gf.Vec3d(inner*(thickness/2+.00025),0,float(z)))
            ridge.AddScaleOp().Set(Gf.Vec3f(.0005,.003,.0008))
            ridge.CreateDisplayColorAttr([Gf.Vec3f(.28,.31,.34)])
            UsdPhysics.CollisionAPI.Apply(ridge.GetPrim()).CreateCollisionEnabledAttr(not disabled)
            ridge_api=PhysxSchema.PhysxCollisionAPI.Apply(ridge.GetPrim())
            ridge_api.CreateContactOffsetAttr(.00015);ridge_api.CreateRestOffsetAttr(0.)
            UsdShade.MaterialBindingAPI.Apply(ridge.GetPrim()).Bind(material,UsdShade.Tokens.weakerThanDescendants,'physics')
    result={'pin':data['pin'],'grip':sensor,'origin':origin,'axes':axes,
            'length':length,'closed_angle':closed,'open_angle':math.radians(45),
            'diameter_m':diameter,'disabled':disabled}
    move_forceps(stage,result,0.,0.)
    return result


def move_forceps(stage,data,closure,lift):
    if not 0<=closure<=1 or not np.isfinite(lift):
        raise ValueError('Invalid forceps command')
    angle=(1-closure)*data['open_angle']+closure*data['closed_angle']
    center=data['origin']+np.array([0.,0.,lift])
    pivot=center-data['axes'][:,2]*(.75*data['length'])
    for name,side in (('left',-1),('right',1)):
        theta=angle*side
        rotation=np.array([[math.cos(theta),0,math.sin(theta)],[0,1,0],[-math.sin(theta),0,math.cos(theta)]])
        axes=data['axes']@rotation
        position=pivot+axes[:,2]*(data['length']*.5)
        UsdGeom.Xformable(stage.GetPrimAtPath('/World/Forceps/'+name)).GetOrderedXformOps()[0].Set(transform(axes,position))


def probe_command(fraction):
    """Close, hold, lift 8 mm, then open; no point attachment follows the tool."""
    if fraction<.3: return fraction/.3,0.
    if fraction<.45: return 1.,0.
    if fraction<.75: return 1.,.008*(fraction-.45)/.3
    return max(0.,1-(fraction-.75)/.25),.008
