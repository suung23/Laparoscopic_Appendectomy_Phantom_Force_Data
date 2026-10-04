"""Two independent port-constrained CAD instruments and explicit grasp assist.

The grasp is a geometry-gated FEM attachment, not measured friction or force.
Adapted in concept from Rosota's registered grip clusters, not a tissue teleport.
"""
import math
import numpy as np
from pxr import Gf, UsdGeom, UsdPhysics, UsdShade
from attachments import author_retraction, attach
from instrument_kinematics import PortInstrument, InstrumentLimits

ROOT='/World/Bimanual'
SPEED=np.array([.4,.4,.02,.4,.6])


def transform(axes,position):
    matrix=np.eye(4);matrix[:3,:3]=axes.T;matrix[3,:3]=position
    return Gf.Matrix4d(matrix.tolist())


def frame(forward,up):
    z=np.asarray(forward,float);z/=np.linalg.norm(z)
    x=np.cross(up,z);x/=np.linalg.norm(x)
    y=np.cross(z,x)
    return np.column_stack((x,y,z))


def author(stage,sim_path,points,base,cad):
    from pxr import PhysxSchema
    points=np.asarray(points,float)
    data=author_retraction(stage,sim_path,points,base)
    stage.RemovePrim('/World/GripAttachment');stage.RemovePrim('/World/RetractionHandle')
    axis=points[np.argmax(np.linalg.norm(points-base,axis=1))]-base
    axis/=np.linalg.norm(axis)
    axial=(points-base)@axis
    clusters=[];used=set(int(i) for i in data['pin'])
    for fraction in np.linspace(.25,.9,7):
        band=np.abs(axial-fraction*axial.max())<.003
        if not band.any():continue
        centre=points[band].mean(0)
        candidates=np.argsort(np.linalg.norm(points-centre,axis=1))
        available=[i for i in candidates if int(i) not in used]
        local=[i for i in available if np.linalg.norm(points[i]-centre)<.0025]
        indices=np.array(local if len(local)>=8 else available[:8],dtype=int)
        if len(indices)<4:continue
        used.update(int(i) for i in indices)
        centre=points[indices].mean(0)
        index=len(clusters);path=ROOT+'/Grip'+str(index)
        handle=UsdGeom.Xform.Define(stage,path)
        handle.AddTransformOp().Set(transform(np.eye(3),centre))
        UsdPhysics.RigidBodyAPI.Apply(handle.GetPrim()).CreateKinematicEnabledAttr(True)
        UsdPhysics.MassAPI.Apply(handle.GetPrim()).CreateMassAttr(.001)
        attach(stage,path+'Attachment',sim_path,path,points,indices,centre)
        clusters.append({'indices':indices,'origin':centre,'path':path})
    if len(clusters)<4:raise RuntimeError('Insufficient disjoint grasp patches')
    material=UsdShade.Material.Define(stage,ROOT+'/ContactMaterial')
    api=UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(.8);api.CreateDynamicFrictionAttr(.6);api.CreateRestitutionAttr(0.)
    for hand in ('Left','Right'):
        for part in ('left','right','shaft'):
            path=ROOT+'/'+hand+'/'+part
            body=UsdGeom.Xform.Define(stage,path);body.AddTransformOp()
            UsdPhysics.RigidBodyAPI.Apply(body.GetPrim()).CreateKinematicEnabledAttr(True)
            UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(.004 if part!='shaft' else .04)
            meshes=['gripper_tip_'+part] if part!='shaft' else ['laparo_shaft','part_13','part_15','gripper_shaft_inner','gripper_shaft_outer']
            for mesh in meshes:
                prim=stage.DefinePrim(path+'/'+mesh)
                prim.GetReferences().AddReference(str(cad.resolve()),'/Tool/'+mesh)
                geom=stage.GetPrimAtPath(path+'/'+mesh+'/mesh')
                UsdPhysics.CollisionAPI.Apply(geom)
                UsdPhysics.MeshCollisionAPI.Apply(geom).CreateApproximationAttr('convexHull')
                collision=PhysxSchema.PhysxCollisionAPI.Apply(geom)
                collision.CreateContactOffsetAttr(.0002);collision.CreateRestOffsetAttr(0.)
                UsdShade.MaterialBindingAPI.Apply(geom).Bind(material,UsdShade.Tokens.weakerThanDescendants,'physics')
    result={'pin':data['pin'],'clusters':clusters,'axis':axis,'sim_path':sim_path}
    # Place before PhysX registration; do not create all bodies at world origin.
    controller=BimanualControl(stage,result,points,registering=True)
    result['arms']={name:{'port':arm.port,'axes':arm.axes,'state':arm.state.copy()} for name,arm in controller.arms.items()}
    return result


class BimanualControl:
    def __init__(self,stage,data,points,registering=False,assist=True):
        self.stage=stage;self.data=data;self.assist=assist
        self.arms={};self.targets={};self.held={};self.events=[];self.status_labels={}
        self.demo_frame=None;self.demo_lift=None
        self.grasp_requested={'Left':False,'Right':False}
        self.sim_mesh=stage.GetPrimAtPath(data['sim_path'])
        limits=InstrumentLimits(jaw_max_rad=math.radians(37))
        for hand,index,sign in [('Left',1,-1),('Right',5,1)]:
            if 'arms' in data:
                spec=data['arms'][hand];arm=PortInstrument(spec['port'],spec['axes'],limits);arm.state=np.array(spec['state']).copy()
            else:
                target=points[data['clusters'][index]['indices']].mean(0)
                port=target+np.array([sign*.045,.09,.025])
                arm=PortInstrument(port,frame(target-port,data['axis']),limits)
                arm.state[2]=np.linalg.norm(target-port)-.025
            self.arms[hand]=arm;self.targets[hand]=arm.state.copy()
        if not registering:
            for cluster in data['clusters']:self.enabled(cluster).Set(False)
        self.render_tools()

    def enabled(self,cluster):
        return self.stage.GetPrimAtPath(cluster['path']+'Attachment').GetAttribute('omniphysics:attachmentEnabled')

    def points(self):return np.asarray(self.sim_mesh.GetAttribute('points').Get(),dtype=float)

    def centers(self):
        q=self.points()
        return np.array([q[c['indices']].mean(0) for c in self.data['clusters']])

    def target_position(self,hand,position,jaw=None):
        arm=self.arms[hand];v=arm.axes.T@(np.asarray(position)-arm.port)
        length=np.linalg.norm(v)
        if not np.isfinite(v).all() or length<1e-6:raise ValueError('Invalid tool target')
        target=self.targets[hand].copy()
        target[:3]=[math.atan2(v[0],v[2]),-math.asin(np.clip(v[1]/length,-1,1)),length]
        if jaw is not None:target[4]=jaw
        self.targets[hand]=target

    def step_action(self,action,dt=1/60):
        """10 actions: left then right yaw, pitch, insertion, roll, opening."""
        a=np.asarray(action,float)
        if a.shape!=(10,) or not np.isfinite(a).all() or np.any(np.abs(a)>1):raise ValueError('10 finite actions in [-1,1] required')
        if not np.isfinite(dt) or not 0<dt<=.1:raise ValueError('Invalid control timestep')
        for i,hand in enumerate(('Left','Right')):
            if a[i*5+4]<-1e-6:self.grasp_requested[hand]=True
            elif a[i*5+4]>1e-6:self.grasp_requested[hand]=False
            self.arms[hand].step(a[i*5:i*5+5],dt)
        self.update_grasps();self.render_tools()

    def tick(self,dt=1/60):
        actions=[]
        for h in ('Left','Right'):
            delta=self.targets[h]-self.arms[h].state
            delta[3]=(delta[3]+np.pi)%(2*np.pi)-np.pi
            actions.append(np.clip(delta/(SPEED*dt),-1,1))
        action=np.concatenate(actions)
        self.step_action(action,dt)

    def release(self,hand,reason='opened'):
        held=self.held.pop(hand,None)
        if held is not None:
            self.enabled(self.data['clusters'][held['cluster']]).Set(False)
            self.events.append({'event':'release','hand':hand,'cluster':held['cluster'],'reason':reason})

    def update_grasps(self):
        q=self.points()
        for hand,arm in self.arms.items():
            pose=arm.pose();position=pose['position_m'];axes=pose['axes']
            held=self.held.get(hand)
            if held is not None:
                if not self.grasp_requested[hand]:self.release(hand);continue
                # The executed jaw stops on tissue. Do not wind the command
                # state down to zero and snap the mesh closed when releasing.
                arm.state[4]=held['angle']
                cluster=self.data['clusters'][held['cluster']]
                # Follow incremental translation, preserving the live grasp location.
                relative=axes@held['axes'].T
                target=position+relative@(held['capture_center']-held['tool_origin'])
                op=UsdGeom.Xformable(self.stage.GetPrimAtPath(cluster['path'])).GetOrderedXformOps()[0]
                op.Set(transform(relative,target))
                continue
            if not self.assist or arm.state[4]>.64 or not self.grasp_requested[hand]:continue
            owned={h['cluster'] for h in self.held.values()}
            candidates=[]
            for index,cluster in enumerate(self.data['clusters']):
                if index in owned:continue
                centre=q[cluster['indices']].mean(0)
                local=(centre-position)@axes
                # A bounded patch in the actual mouth, not nearest-organ snapping.
                if abs(local[0])>.004 or abs(local[1])>.0035 or abs(local[2])>.005:continue
                tissue=(q-position)@axes
                mouth=tissue[(np.abs(tissue[:,1])<.0035)&(np.abs(tissue[:,2])<.005)]
                if len(mouth)<4 or not (mouth[:,0].min()<0<mouth[:,0].max()):continue
                candidates.append((np.linalg.norm(local),index,centre,mouth))
            if not candidates:continue
            _,index,centre,mouth=min(candidates,key=lambda x:x[0])
            surface_angles=np.arctan2(np.abs(mouth[:,0])+.0005,np.maximum(.004,mouth[:,2]+.014))
            contact_angle=float(np.clip(np.quantile(surface_angles,.98),.10,math.radians(37)))
            if arm.state[4]>contact_angle+.005:continue
            cluster=self.data['clusters'][index]
            # Re-anchor offsets to LIVE positions before enabling: no rest-pose snap.
            attachment=self.stage.GetPrimAtPath(cluster['path']+'Attachment')
            attachment.GetAttribute('omniphysics:localPositionsSrc1').Set([Gf.Vec3f(*(p-centre)) for p in q[cluster['indices']]])
            UsdGeom.Xformable(self.stage.GetPrimAtPath(cluster['path'])).GetOrderedXformOps()[0].Set(transform(np.eye(3),centre))
            hold_angle=float(arm.state[4])
            self.held[hand]={'cluster':index,'capture_center':centre.copy(),'tool_origin':position.copy(),'angle':hold_angle,'axes':axes.copy()}
            self.enabled(cluster).Set(True)
            self.events.append({'event':'grasp','hand':hand,'cluster':index,'mouth_error_m':float(np.linalg.norm(centre-position))})

    def render_tools(self):
        for hand,arm in self.arms.items():
            pose=arm.pose();axes=pose['axes'];pivot=pose['position_m']-axes[:,2]*.014
            angle=self.held[hand]['angle'] if hand in self.held else arm.state[4]
            for part,sign in [('left',1),('right',-1),('shaft',0)]:
                theta=sign*angle;c,s=math.cos(theta),math.sin(theta)
                rotation=np.array([[c,0,s],[0,1,0],[-s,0,c]])
                op=UsdGeom.Xformable(self.stage.GetPrimAtPath(ROOT+'/'+hand+'/'+part)).GetOrderedXformOps()[0]
                op.Set(transform(axes@rotation,pivot))
            if hand in self.status_labels:
                self.status_labels[hand].text='GRASP ASSIST: patch '+str(self.held[hand]['cluster']) if hand in self.held else 'Open / no tissue held'

    def observation(self):
        observations=[]
        for h in ('Left','Right'):
            actual=self.arms[h].state.copy()
            if h in self.held:actual[4]=self.held[h]['angle']
            observations.append(np.r_[actual,float(h in self.held)])
        return np.concatenate(observations).astype(np.float32)

    def create_ui(self,scope=None):
        import omni.ui as ui
        self.windows=[];self.sliders={}
        for hand,y in [('Left',70),('Right',480)]:
            window=ui.Window(hand+' laparoscopic instrument',width=365,height=420,position_x=1015,position_y=y)
            self.windows.append(window);self.sliders[hand]=[]
            with window.frame:
                with ui.VStack(spacing=5):
                    ui.Label('Independent port pivot / Rosota CAD')
                    for name,lo,hi,val in zip(['Yaw (deg)','Pitch (deg)','Insertion (mm)','Roll (deg)','Jaw command (deg)'],
                        [-45,-45,30,-180,0],[45,45,250,180,37],self.arms[hand].state*np.array([180/np.pi,180/np.pi,1000,180/np.pi,180/np.pi])):
                        ui.Label(name);slider=ui.FloatSlider(min=lo,max=hi);slider.model.set_value(float(val));self.sliders[hand].append(slider)
                    self.status_labels[hand]=ui.Label('Open / no tissue held')
                    ui.Label('Gravity: anatomical -Y, 9.81 m/s2')
                    ui.Button('Both hands: approach / grasp / lift / release',clicked_fn=self.begin_demo)
                    ui.Button('Release both / cancel demo',clicked_fn=self.release_all)
                    if scope:ui.Button('Show / hide camera controls',clicked_fn=lambda:setattr(scope.window,'visible',not scope.window.visible))

    def update_ui(self):
        if self.demo_frame is not None:
            self.demo_tick()
            self.sync_ui()
            return
        for hand,sliders in self.sliders.items():
            self.targets[hand]=np.array([s.model.get_value_as_float() for s in sliders])*np.array([np.pi/180,np.pi/180,.001,np.pi/180,np.pi/180])
        self.tick()

    def sync_ui(self):
        for h,sliders in self.sliders.items():
            for slider,value in zip(sliders,self.arms[h].state*np.array([180/np.pi,180/np.pi,1000,180/np.pi,180/np.pi])):
                slider.model.set_value(float(value))

    def begin_demo(self):
        self.release_all();self.demo_frame=0;self.demo_lift=None

    def release_all(self):
        self.demo_frame=None
        for h in self.arms:
            self.release(h,'user release');self.grasp_requested[h]=False;self.targets[h][4]=math.radians(37)
            if hasattr(self,'sliders'):self.sliders[h][4].model.set_value(37.)

    def demo_tick(self):
        f=self.demo_frame;cents=self.centers()
        for h,i in [('Left',1),('Right',5)]:
            if f<180:self.target_position(h,cents[i],math.radians(37))
            elif f<360 and h not in self.held:self.target_position(h,cents[i],0.)
        if f==360:
            if len(self.held)!=2:self.release_all();return
            for h in self.arms:self.target_position(h,self.arms[h].pose()['position_m']+np.array([0,.008,0]),0.)
        if f==570:
            for h in self.arms:self.targets[h][4]=math.radians(37)
        self.tick()
        self.demo_frame=None if f>=690 else f+1
