"""Author dynamic FR3 chains around the existing CT tool start poses.

Engineering fixture placement, not measured robot-to-patient calibration.
URDF inertias and joint geometry are preserved. Full arm CAD/collision is absent;
the existing Rosota distal CAD is retained on both hands (right is a grasper).
"""
import math
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from pxr import Gf,Sdf,UsdGeom,UsdPhysics,PhysxSchema
from probe_rosota_controller import build_chain
from laparo_ik import rpy_to_R
from rosota_controller import STRICT_CFG


def quat(R):
    # Gf uses row-vector matrices.
    return Gf.Quatf(Gf.Matrix3d(np.asarray(R).T.tolist()).ExtractRotation().GetQuat())


def pose(stage,path,T):
    x=UsdGeom.Xformable(stage.GetPrimAtPath(path));x.ClearXformOpOrder()
    x.AddTranslateOp().Set(Gf.Vec3d(*T[:3,3]))
    x.AddOrientOp().Set(Gf.Quatf(quat(T[:3,:3])))


def body(stage,path,T):
    x=UsdGeom.Xform.Define(stage,path);pose(stage,path,T)
    UsdPhysics.RigidBodyAPI.Apply(x.GetPrim()).CreateKinematicEnabledAttr(False)
    return x.GetPrim()


def joint(stage,path,parent,child,Tlocal,fixed=False,lo=-180,hi=180,axis='Z',kp=5e5,kd=16650):
    j=(UsdPhysics.FixedJoint if fixed else UsdPhysics.RevoluteJoint).Define(stage,path)
    if parent:j.CreateBody0Rel().SetTargets([parent])
    j.CreateBody1Rel().SetTargets([child])
    j.CreateLocalPos0Attr(Gf.Vec3f(*Tlocal[:3,3]));j.CreateLocalRot0Attr(quat(Tlocal[:3,:3]))
    j.CreateLocalPos1Attr(Gf.Vec3f(0));j.CreateLocalRot1Attr(Gf.Quatf(1))
    if not fixed:
        j.CreateAxisAttr(axis);j.CreateLowerLimitAttr(lo);j.CreateUpperLimitAttr(hi)
        drive=UsdPhysics.DriveAPI.Apply(j.GetPrim(),'angular')
        drive.CreateTypeAttr('force');drive.CreateStiffnessAttr(kp);drive.CreateDampingAttr(kd)
        drive.CreateMaxForceAttr(10000.)
    return j


def author(stage,config,fr3_urdf,out):
    source=ET.parse(fr3_urdf).getroot();specs={}
    for hand,label in [('left','Left'),('right','Right')]:
        # Both current CAD tools are graspers. Do not label the right as a cutter.
        chain=build_chain(fr3_urdf,'left',out/(hand+'_chain.urdf'))
        seed=np.array([0.,-.785,0.,-2.356,0.,1.571,.785])
        if hand=='right':seed*=np.array([-1,1,-1,1,-1,1,-1])
        a=config['arms'][label];from instrument_kinematics import PortInstrument
        old=PortInstrument(a['port'],a['axes']);old.state=np.array(a['state'])
        desired=old.pose()
        # Free-space controller commissioning starts 15 mm further withdrawn.
        # This is explicit placement, not a collision toggle or successful grasp.
        desired['position_m']=desired['position_m']-.015*desired['axes'][:,2]
        p,R=chain.fk(seed);Rtip=R@chain.R_f2t
        target_R=desired['axes']@np.diag([-1.,1.,-1.])
        base=np.eye(4);base[:3,:3]=target_R@Rtip.T
        base[:3,3]=desired['position_m']-base[:3,:3]@p
        port=desired['position_m']-.12*desired['axes'][:,2]
        root='/World/FR3/'+label
        UsdGeom.Xform.Define(stage,root)
        b=body(stage,root+'/base',base)
        UsdPhysics.MassAPI.Apply(b).CreateMassAttr(3.)
        fixed=joint(stage,root+'/fixed_base',None,root+'/base',base,True)
        UsdPhysics.ArticulationRootAPI.Apply(fixed.GetPrim())
        art=PhysxSchema.PhysxArticulationAPI.Apply(fixed.GetPrim())
        art.CreateEnabledSelfCollisionsAttr(False);art.CreateSolverPositionIterationCountAttr(64)
        art.CreateSolverVelocityIterationCountAttr(8)
        T=base.copy();parent=root+'/base'
        frames=[]
        for i,(Tf,ax) in enumerate(chain.steps[:7]):
            assert np.allclose(ax,[0,0,1]),'Unsupported joint axis'
            angle=seed[i];c,s=np.cos(angle),np.sin(angle)
            Rj=np.eye(4);Rj[:3,:3]=[[c,-s,0],[s,c,0],[0,0,1]]
            T=T@Tf@Rj;path=root+f'/link{i+1}';prim=body(stage,path,T)
            inertial=source.find(f"link[@name='fr3_link{i+1}']/inertial")
            mass=UsdPhysics.MassAPI.Apply(prim)
            mass.CreateMassAttr(float(inertial.find('mass').get('value')))
            origin=inertial.find('origin');mass.CreateCenterOfMassAttr(Gf.Vec3f(*map(float,origin.get('xyz').split())))
            v=inertial.find('inertia').attrib
            I=np.array([[float(v['ixx']),float(v['ixy']),float(v['ixz'])],
                        [float(v['ixy']),float(v['iyy']),float(v['iyz'])],
                        [float(v['ixz']),float(v['iyz']),float(v['izz'])]])
            Ri=rpy_to_R(*map(float,origin.get('rpy','0 0 0').split()));I=Ri@I@Ri.T
            values,vectors=np.linalg.eigh(I)
            if np.linalg.det(vectors)<0:vectors[:,0]*=-1
            mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*values));mass.CreatePrincipalAxesAttr(quat(vectors))
            j=joint(stage,root+f'/fr3_joint{i+1}',parent,path,Tf,
                    lo=float(np.rad2deg(chain.lo[i])),hi=float(np.rad2deg(chain.hi[i])),kd=STRICT_CFG.kd)
            UsdPhysics.DriveAPI(j.GetPrim(),'angular').CreateTargetPositionAttr(float(np.rad2deg(seed[i])))
            parent=path;frames.append(T.copy())
        # CAD is authored around the jaw hinge, 14 mm behind the virtual mouth.
        mount=np.eye(4);mount[:3,:3]=desired['axes'];mount[:3,3]=desired['position_m']-.014*desired['axes'][:,2]
        for part in ('shaft','left','right'):
            old_path='/World/Bimanual/'+label+'/'+part;new_path=root+'/'+part
            Sdf.CopySpec(stage.GetRootLayer(),old_path,stage.GetRootLayer(),new_path)
            prim=stage.GetPrimAtPath(new_path)
            UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr(False)
            half=float(old.state[4]);sign=1 if part=='left' else -1
            M=mount.copy()
            if part!='shaft':M[:3,:3]=mount[:3,:3]@rpy_to_R(0,sign*half,0)
            pose(stage,new_path,M)
            if part=='shaft':joint(stage,root+'/tool_mount',parent,new_path,np.linalg.inv(T)@mount,True)
            else:
                j=joint(stage,root+'/jaw_'+part,root+'/shaft',new_path,np.eye(4),axis='Y',
                        lo=0 if sign>0 else -37,hi=37 if sign>0 else 0,kp=200.,kd=10.)
                UsdPhysics.DriveAPI(j.GetPrim(),'angular').CreateTargetPositionAttr(float(sign*np.rad2deg(half)))
                # N.m cap acts in the physics solver, independent of policy or
                # sensor latency. Engineering limit, not a tissue-force claim.
                UsdPhysics.DriveAPI(j.GetPrim(),'angular').CreateMaxForceAttr(.00045)
            stage.RemovePrim(old_path)
        specs[hand]=dict(root=root,chain=chain,base=base,port=port,
                         normal=desired['axes'][:,2],seed=seed,mount_to_tip=np.array([0,0,.014]))
    return specs
