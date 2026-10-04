"""CT scene with measured PhysX FR3 joints and the original Rosota controller.

Engineering port/base placement; both distal tools are currently graspers.
Scope has a geometric RCM but no shaft-organ collision body. No clinical claim.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from rosota_controller import RosotaArmController, check_chunk, aperture_metrics
from trocar_scope import TrocarScope
from bimanual_env import BimanualEnv


class RosotaCTEnv(gym.Env):
    metadata={'render_modes':['rgb_array'],'render_fps':30}
    def __init__(self,app,assets,fr3_urdf,out,max_steps=300,grasp_mode='none',workspace='legacy',settle_policy='fixed',diagnostic_no_tool_collision=False,diagnostic_zero_gravity=False,enhanced_determinism=False):
        if settle_policy not in ('fixed','adaptive'):raise ValueError('Unknown settlement policy')
        self.settle_policy=settle_policy
        self.enhanced_determinism=bool(enhanced_determinism)
        self.diagnostic_no_tool_collision=bool(diagnostic_no_tool_collision)
        self.diagnostic_zero_gravity=bool(diagnostic_zero_gravity)
        if self.diagnostic_zero_gravity and not self.diagnostic_no_tool_collision:raise ValueError('Zero gravity is restricted to the isolated sensor diagnostic')
        if self.diagnostic_no_tool_collision and grasp_mode!='none':raise ValueError('Collision-free sensor fixture cannot be a grasp task')
        if grasp_mode not in ('none','rosota_attachment','solver_attachment'):raise ValueError('Unknown grasp model')
        self.grasp_mode=grasp_mode
        if workspace not in ('legacy','appendix_patches'):raise ValueError('Unknown workspace')
        self.workspace=workspace;self.target_patches=[1,4]
        self.app=app;self.assets=Path(assets);self.fr3_urdf=Path(fr3_urdf);self.out=Path(out)
        self.out.mkdir(parents=True,exist_ok=True)
        self.config=json.loads((self.assets/'bimanual_config.json').read_text())
        stage=self.assets/'attached.usda'
        if hashlib.sha256(stage.read_bytes()).hexdigest()!=self.config['stage_sha256']:raise ValueError('Stage hash mismatch')
        lower=np.full((2,1,7),-np.inf,np.float32);upper=-lower;lower[:,:,6]=0;upper[:,:,6]=100
        self.action_space=spaces.Dict({'tools':spaces.Box(lower,upper,dtype=np.float32),
                                       'scope':spaces.Box(-1,1,(4,),np.float32)})
        self.observation_space=spaces.Dict({'joint_position':spaces.Box(-np.inf,np.inf,(2,7),np.float32),
            'tool_pose':spaces.Box(-np.inf,np.inf,(2,4,4),np.float32),
            'scope_pose':spaces.Box(-np.inf,np.inf,(4,4),np.float32),
            'contact_residual':spaces.Box(-np.inf,np.inf,(2,6),np.float32),
            'jaw_position':spaces.Box(-np.inf,np.inf,(2,2),np.float32),
            'jaw_effort':spaces.Box(-np.inf,np.inf,(2,2),np.float32),
            'jaw_contact_hold':spaces.MultiBinary(2),
            'contact_stop':spaces.MultiBinary(2),
            'tissue_held':spaces.MultiBinary(2),
            'goal_position':spaces.Box(-np.inf,np.inf,(2,3),np.float32)})
        self.max_steps=max_steps;self.sim=None;self.rgb=self.product=None;self.done=True

    def reset(self,*,seed=None,options=None):
        if options:raise ValueError('No reset options')
        super().reset(seed=seed)
        self.done=True
        self.reset_sequence=getattr(self,'reset_sequence',0)+1
        self.config=json.loads((self.assets/'bimanual_config.json').read_text())
        import omni.usd
        from pxr import UsdGeom,UsdPhysics,PhysxSchema,Gf,Sdf
        from isaacsim.core.api import SimulationContext
        from isaacsim.core.prims import SingleArticulation
        from fr3_ct_scene import author
        self._destroy_render()
        if self.sim:self.sim.stop()
        SimulationContext.clear_instance()
        # USD layers survive stage closure while old prim/view references exist.
        # Force a disk reload so prior cooking, attachment and physics edits
        # cannot become the next episode's authored initial condition.
        source_path=str((self.assets/'attached.usda').resolve())
        omni.usd.get_context().close_stage()
        source_layer=Sdf.Layer.FindOrOpen(source_path)
        if source_layer is None:raise RuntimeError('Missing source USD layer')
        source_layer.Reload(force=True)
        omni.usd.get_context().open_stage(source_path)
        for _ in range(20):self.app.update()
        self.stage=omni.usd.get_context().get_stage()
        self.stage.Reload()  # Reload referenced layers as well as the root.
        # Older authored stages used a bare Xform; modern PhysX attachments
        # require a rigid actor for this stationary attachment target.
        anchor=self.stage.GetPrimAtPath('/World/BaseAnchor')
        if not anchor:raise ValueError('Missing appendix base anchor')
        UsdPhysics.RigidBodyAPI.Apply(anchor).CreateKinematicEnabledAttr(True)
        UsdPhysics.MassAPI.Apply(anchor).CreateMassAttr(.01)
        self.specs=author(self.stage,self.config['data'],self.fr3_urdf,self.out,
            patches=self.target_patches if self.workspace=='appendix_patches' else None)
        if self.diagnostic_no_tool_collision:
            for prim in self.stage.Traverse():
                if str(prim.GetPath()).startswith('/World/FR3/') and prim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
        for c in self.config['data']['clusters']:
            self.stage.GetPrimAtPath(c['path']+'Attachment').GetAttribute('omniphysics:attachmentEnabled').Set(False)
        if self.grasp_mode=='solver_attachment':
            from solver_grasp import bind_links
            self.grasp_links={h:(i,s['root']+'/shaft') for (h,s),i in zip(self.specs.items(),self.target_patches)}
            bind_links(self.stage,self.config['data']['clusters'],self.grasp_links)
        camera=np.asarray(UsdGeom.Xformable(self.stage.GetPrimAtPath('/World/Endoscope')).GetOrderedXformOps()[0].Get()).T
        camera[:3,3]+=np.mean([s['workspace_shift'] for s in self.specs.values()],axis=0)
        self.scope=TrocarScope(camera);self.scope.apply_usd(self.stage)
        self.mesh=self.stage.GetPrimAtPath(self.config['data']['sim_path'])
        self.rest=self.points().copy();self.tets=np.asarray(self.mesh.GetAttribute('tetVertexIndices').Get(),int)
        self.pin=np.asarray(self.config['data']['pin'],int)
        self.sim=SimulationContext(stage_units_in_meters=1.,physics_dt=1/480,rendering_dt=1/30)
        for prim in self.stage.Traverse():
            if prim.IsA(UsdPhysics.Scene):
                scene=UsdPhysics.Scene(prim);scene.CreateGravityDirectionAttr(Gf.Vec3f(0,-1,0));scene.CreateGravityMagnitudeAttr(0. if self.diagnostic_zero_gravity else 9.81)
                ps=PhysxSchema.PhysxSceneAPI.Apply(prim);ps.CreateEnableGPUDynamicsAttr(True);ps.CreateBroadphaseTypeAttr('GPU');ps.CreateEnableEnhancedDeterminismAttr(self.enhanced_determinism)
        self.stage.GetRootLayer().Export(str(self.out/'scene.usda'))
        self._audit_tissue('before_initialize')
        from omni.physx import get_physx_cooking_interface
        if not get_physx_cooking_interface().cook_auto_deformable_body('/World/Atlas/appendix'):
            raise RuntimeError('Explicit pre-initialization tissue cooking failed')
        self._audit_tissue('after_cooking')
        # Cooking can permute nodes; attachment indices must follow the exact
        # rest-geometry/topology permutation, never nearest-point guesses.
        from node_correspondence import node_permutation
        cooked_rest=np.asarray(self.mesh.GetAttribute('omniphysics:restShapePoints').Get(),float)
        cooked_tets=np.asarray(self.mesh.GetAttribute('tetVertexIndices').Get(),int)
        authored=np.load(self.out/'tissue_before_initialize.npz')
        permutation=node_permutation(authored['rest'],authored['tets'],cooked_rest,cooked_tets)
        for prim in self.stage.Traverse():
            rel=prim.GetRelationship('omniphysics:src0')
            if rel and self.mesh.GetPath() in rel.GetTargets():
                attr=prim.GetAttribute('omniphysics:vtxIndicesSrc0')
                if attr and attr.Get() is not None:
                    attr.Set(permutation[np.asarray(attr.Get(),int)].tolist())
        self.pin=permutation[self.pin]
        for cluster in self.config['data']['clusters']:
            cluster['indices']=permutation[np.asarray(cluster['indices'],int)].tolist()
        self.rest=cooked_rest.copy();self.tets=cooked_tets
        (self.out/'node_correspondence.json').write_text(json.dumps(dict(
            exact_geometry_and_topology=True,nodes=len(permutation),
            changed_indices=int(np.sum(permutation!=np.arange(len(permutation))))),indent=2),encoding='utf-8')
        self.sim.initialize_physics()
        self._audit_tissue('after_initialize')
        if not np.isclose(self.sim.get_physics_dt(),1/480):raise RuntimeError('Wrong physics interval')
        if not np.array_equal(self.mesh.GetAttribute('omniphysics:restShapePoints').Get(),self.rest):
            raise RuntimeError('Tissue rest nodes changed again after explicit cooking')
        if not np.array_equal(self.mesh.GetAttribute('tetVertexIndices').Get(),self.tets):
            raise RuntimeError('Tissue connectivity changed again after explicit cooking')
        self.sim.play()
        from isaacsim.core.utils.types import ArticulationAction
        self.robots={};self.indices={};self.jaw_indices={}
        for hand,s in self.specs.items():
            robot=SingleArticulation(prim_path=s['root'],name=hand,reset_xform_properties=False)
            robot.initialize();self.robots[hand]=robot
            self.indices[hand]=np.array([robot.get_dof_index('fr3_joint'+str(i)) for i in range(1,8)])
            self.jaw_indices[hand]=np.array([robot.get_dof_index('jaw_left'),robot.get_dof_index('jaw_right')])
            robot.set_joint_positions(s['seed'],joint_indices=self.indices[hand])
            robot.set_joint_positions(np.deg2rad([37.,-37.]),joint_indices=self.jaw_indices[hand])
            robot.set_joint_velocities(np.zeros(robot.num_dof))
            robot.apply_action(ArticulationAction(joint_positions=np.deg2rad([37.,-37.]),
                joint_velocities=np.zeros(2),joint_indices=self.jaw_indices[hand]))
        from isaacsim.core.utils.types import ArticulationAction
        self._audit_tissue('before_warmup')
        for warmup_tick in range(480):
            for hand,s in self.specs.items():
                self.robots[hand].apply_action(ArticulationAction(joint_positions=s['seed'],joint_velocities=np.zeros(7),joint_indices=self.indices[hand]))
            self.sim.step(render=False)
            if warmup_tick in (0,59,479):self._audit_tissue('warmup_'+str(warmup_tick+1))
        self.settlement={'policy':self.settle_policy,'measured_equilibrium':False,'physics_ticks':480}
        if self.settle_policy=='adaptive':
            from settlement import SettlementGate
            gate=SettlementGate(self.points())
            settled=False
            for window in range(56):
                for _ in range(60):self.sim.step(render=False)
                settled=gate.update(self.points())
                (self.out/'settlement_progress.json').write_text(json.dumps(dict(
                    window=window+1,physics_ticks=480+60*(window+1),latest=gate.rows[-1])),encoding='utf-8')
                if window>=7 and settled:break
            self.settlement.update(measured_equilibrium=settled,physics_ticks=480+60*len(gate.rows),
                limits={'rms_speed_m_s':gate.rms_limit,'p99_speed_m_s':gate.p99_limit,'consecutive_windows':gate.required},
                interval_s=gate.dt,trace=gate.rows)
            (self.out/'settlement.json').write_text(json.dumps(self.settlement,indent=2),encoding='utf-8')
            if not settled:raise RuntimeError('Tissue did not reach the measured reset equilibrium gate')
        self.sim.render()
        self.controllers={h:RosotaArmController(s['chain'],s['base'],s['port'],s['normal'],self.q(h),hand='left') for h,s in self.specs.items()}
        self.control=self.controllers  # Shared RGB renderer's initialized-state guard.
        self.reference=self.points().copy();self.steps=0
        self.goal=np.array([self.controllers[h].pose(self.q(h))[0] for h in self.specs])+np.array([[.003,0,0],[-.003,0,0]])
        self.trace=[];self.jaw_open={h:74. for h in self.specs}
        self.initial_q=np.array([self.q(h) for h in self.specs]);self.initial_scope=self.scope.pose.copy()
        from contact_control import ContactGuard
        self.force_rows={};samples={h:[] for h in self.specs}
        for h,robot in self.robots.items():
            self.force_rows[h]=int(robot._articulation_view._metadata.joint_indices['tool_mount'])+1
        for _ in range(64):
            self.sim.step(render=False)
            for h in self.specs:samples[h].append(self.sensor(h)[0])
        self.guards={};self.hold_q={};calibration={}
        for h in self.specs:
            values=np.asarray(samples[h]);baseline=values.mean(axis=0)
            noise=float(np.linalg.norm(values[:,:3]-baseline[:3],axis=1).max())
            self.guards[h]=ContactGuard(baseline,noise)
            self.guards[h].update(*self.sensor(h))
            calibration[h]=dict(baseline=baseline.tolist(),max_force_residual_n=noise,
                samples=64,frame='local child joint frame at tool_mount',
                model='static tare only; no orientation or inertia compensation')
        (self.out/'contact_calibration.json').write_text(json.dumps(calibration,indent=2),encoding='utf-8')
        from tissue_grasp import TissueGrasp
        from deformation_metrics import DeformationMeasure
        from isaacsim.core.prims import SingleRigidPrim
        self.grip_bodies={}
        for i,c in enumerate(self.config['data']['clusters']):
            body=SingleRigidPrim(prim_path=c['path'],name='grip_handle_'+str(i),reset_xform_properties=False)
            body.initialize();self.grip_bodies[c['path']]=body
        def write_grip_pose(c,R,p):
            # A USD edit is a next-tick kinematic target. Activation needs the
            # actual PhysX pose in place NOW, before enabling its constraint.
            rotation=Gf.Matrix3d(np.asarray(R).T.tolist()).ExtractRotation().GetQuat()
            quaternion=np.array([rotation.GetReal(),*rotation.GetImaginary()])
            self.grip_bodies[c['path']].set_world_pose(position=np.asarray(p),orientation=quaternion)
        self.grasp=TissueGrasp(self.stage,self.config['data']['clusters'],
            preferred=dict(zip(self.specs,self.target_patches)) if self.workspace=='appendix_patches' else None,
            pose_writer=write_grip_pose)
        if self.grasp_mode=='solver_attachment':
            from solver_grasp import SolverTissueGrasp
            self.grasp_link_bodies={}
            for hand,(_,path) in self.grasp_links.items():
                body=SingleRigidPrim(prim_path=path,name=hand+'_grasp_link',reset_xform_properties=False)
                body.initialize();self.grasp_link_bodies[hand]=body
            def read_link(hand):
                from scipy.spatial.transform import Rotation
                p,q=self.grasp_link_bodies[hand].get_world_pose()
                return Rotation.from_quat(np.asarray(q)[[1,2,3,0]]).as_matrix(),np.asarray(p)
            self.grasp=SolverTissueGrasp(self.stage,self.config['data']['clusters'],self.grasp_links,read_link)
        self.deformation=DeformationMeasure(self.points(),self.tets)
        self._audit_tissue('after_settlement')
        info=self.measure()
        (self.out/'reset_diagnostics.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
        if info['failure']:self.done=True;raise RuntimeError('Invalid reset: '+info['failure'])
        self.done=False
        return self.observe(),info

    def q(self,h):return self.robots[h].get_joint_positions(joint_indices=self.indices[h]).astype(float)
    def sensor(self,h):
        robot=self.robots[h]
        wrench=np.asarray(robot.get_measured_joint_forces(),float)[self.force_rows[h]]
        efforts=np.asarray(robot.get_measured_joint_efforts(joint_indices=self.jaw_indices[h]),float)
        if not np.isfinite(wrench).all() or not np.isfinite(efforts).all():raise RuntimeError('Invalid contact sensor')
        return wrench,efforts
    def _audit_tissue(self,label):
        """Persist actual buffers and attachment coordinates for runtime migration diagnosis."""
        import omni.usd
        from pxr import UsdGeom
        prim=self.stage.GetPrimAtPath('/World/BaseAttachment')
        indices=np.asarray(prim.GetAttribute('omniphysics:vtxIndicesSrc0').Get(),int)
        targets=np.asarray(prim.GetAttribute('omniphysics:localPositionsSrc1').Get(),float)
        anchor=prim.GetRelationship('omniphysics:src1').GetTargets()[0]
        transform=np.asarray(UsdGeom.Xformable(self.stage.GetPrimAtPath(anchor)).ComputeLocalToWorldTransform(0)).T
        world_targets=targets@transform[:3,:3].T+transform[:3,3]
        points=self.points().copy()
        rest=np.asarray(self.mesh.GetAttribute('omniphysics:restShapePoints').Get(),float)
        tets=np.asarray(self.mesh.GetAttribute('tetVertexIndices').Get(),int)
        np.savez_compressed(self.out/('tissue_'+label+'.npz'),points=points,rest=rest,tets=tets,
            pin=indices,target_world=world_targets,anchor_world=transform)
        summary=dict(label=label,nodes=len(points),tetrahedra=len(tets),
            pin_target_error_m=float(np.linalg.norm(points[indices]-world_targets,axis=1).max()),
            points_sha256=hashlib.sha256(points.tobytes()).hexdigest(),
            tets_sha256=hashlib.sha256(tets.tobytes()).hexdigest())
        (self.out/('tissue_'+label+'.json')).write_text(json.dumps(summary,indent=2),encoding='utf-8')
        import shutil
        for suffix in ('.npz','.json'):
            shutil.copyfile(self.out/('tissue_'+label+suffix),
                self.out/(f'reset_{self.reset_sequence}_tissue_'+label+suffix))

    def points(self):return np.asarray(self.mesh.GetAttribute('points').Get(),float)
    def observe(self):
        poses=[];qs=[]
        for h in self.specs:
            q=self.q(h);p,R,_=self.controllers[h].pose(q);T=np.eye(4);T[:3,:3]=R;T[:3,3]=p
            poses.append(T);qs.append(q)
        return {'joint_position':np.asarray(qs,np.float32),'tool_pose':np.asarray(poses,np.float32),
                'contact_residual':np.asarray([g.filtered for g in self.guards.values()],np.float32),
                'jaw_position':np.asarray([self.robots[h].get_joint_positions(joint_indices=self.jaw_indices[h]) for h in self.specs],np.float32),
                'jaw_effort':np.asarray([g.last['jaw_effort_nm'] for g in self.guards.values()],np.float32),
                'jaw_contact_hold':np.asarray([g.pinch_latched for g in self.guards.values()],np.int8),
                'contact_stop':np.asarray([g.latched for g in self.guards.values()],np.int8),
                'tissue_held':np.asarray([h in self.grasp.held for h in self.specs],np.int8),
                'scope_pose':self.scope.pose.astype(np.float32),'goal_position':self.goal.astype(np.float32)}

    def measure(self):
        from pxr import UsdGeom
        from tissue_metrics import tetra_health
        ports={};mismatch=[];tool_frames={}
        for h,s in self.specs.items():
            # Compare actual rigid-body USD readback against FK of measured q.
            M=np.asarray(UsdGeom.Xformable(self.stage.GetPrimAtPath(s['root']+'/shaft')).ComputeLocalToWorldTransform(0)).T
            tip=M[:3,3]+M[:3,:3]@s['mount_to_tip'];axis=M[:3,2]
            tool_frames[h]=(tip,M[:3,:3])
            ports[h]=aperture_metrics(s['port'],s['normal'],tip,axis)
            fk=self.controllers[h].pose(self.q(h))[0];mismatch.append(float(np.linalg.norm(tip-fk)))
        tissue=self.points();health=tetra_health(self.rest,tissue,self.tets)
        drift=float(np.linalg.norm(tissue[self.pin]-self.rest[self.pin],axis=1).max())
        scope=self.scope.metrics();failure=''
        if any(not x['aperture_clear'] or x['insertion_m']<.1-1e-6 for x in ports.values()):failure='trocar_violation'
        if not scope['aperture_clear']:failure='scope_trocar_violation'
        if max(mismatch)>.0005:failure='joint_body_pose_mismatch'
        if health['inverted_tetrahedra']:failure='tet_inversion'
        if drift>.001:failure='pin_drift'
        grasp_geometry={}
        if self.workspace=='appendix_patches':
            for h,index in zip(self.specs,self.target_patches):
                tip,axes=tool_frames[h]
                centre=tissue[self.config['data']['clusters'][index]['indices']].mean(0)
                grasp_geometry[h]=dict(target_patch=index,centre_in_shaft_frame_m=((centre-tip)@axes).tolist(),
                    distance_m=float(np.linalg.norm(centre-tip)))
        return dict(failure=failure,trocar=ports,scope_trocar=scope,body_fk_error_m=mismatch,
            grasp_geometry=grasp_geometry,
            contact={h:g.last.copy() for h,g in self.guards.items()},
            jaw_position_rad={h:self.robots[h].get_joint_positions(joint_indices=self.jaw_indices[h]).tolist() for h in self.specs},
            tissue_health=health,pin_drift_m=drift,controller='original Rosota TipController strict aperture',
            plant='PhysX dynamic articulations with measured joint feedback',
            calibrated_ports=False,right_tool='grasper; cutter pending',scope_shaft_collision=False,
            commissioning_retreat_m=.015 if self.workspace=='legacy' else .018,workspace=self.workspace,
            grasp_assist=self.grasp_mode!='none' or bool(self.grasp.held),
            grasp_mode=self.grasp_mode,held_patches={h:v['cluster'] for h,v in self.grasp.held.items()},
            settlement_verified=self.settlement['measured_equilibrium'],
            diagnostic_no_tool_collision=self.diagnostic_no_tool_collision,
            rosota_full_contract=False)

    def step(self,action):
        if self.done:raise RuntimeError('Reset required')
        if not isinstance(action,dict) or set(action)!={'tools','scope'}:raise ValueError('tools and scope required')
        tools=check_chunk(action['tools'],1);scope=np.asarray(action['scope'],float)
        if scope.shape!=(4,) or not np.isfinite(scope).all() or np.any(np.abs(scope)>1):raise ValueError('Invalid scope action')
        # Contact commissioning profile: cap requested travel to 3.6 mm/s.
        for row in tools[:,0]:
            distance=np.linalg.norm(row[:3])
            if distance>.00012:row[:3]*=.00012/distance
        from isaacsim.core.utils.types import ArticulationAction
        for i,h in enumerate(self.specs):self.controllers[h].begin_substep(tools[i,0],self.q(h))
        rejected={h:np.zeros(2) for h in self.specs}
        try:
            for tick in range(16):
                # Both hands act on the same pre-step state. Reading USD's
                # FEM buffer once avoids four identical host conversions.
                grasp_points=self.points() if self.grasp_mode!='none' else None
                for h in self.specs:
                    guard=self.guards[h]
                    if guard.latched:
                        if h not in self.hold_q:self.hold_q[h]=self.q(h)
                        self.controllers[h].remaining-=1
                        result=dict(joint_target=self.hold_q[h],joint_velocity_ff=np.zeros(7),
                            gripper_target_deg=self.controllers[h].row[6]*.74,rot_rejected=np.zeros(2))
                    else:result=self.controllers[h].tick(self.q(h))
                    rejected[h]+=result['rot_rejected']
                    self.robots[h].apply_action(ArticulationAction(joint_positions=result['joint_target'],
                        joint_velocities=result['joint_velocity_ff'],joint_indices=self.indices[h]))
                    jaws=self.robots[h].get_joint_positions(joint_indices=self.jaw_indices[h])
                    measured=float(np.rad2deg(jaws[0]-jaws[1]))
                    if self.grasp_mode!='none':
                        from pxr import UsdGeom
                        s=self.specs[h]
                        M=np.asarray(UsdGeom.Xformable(self.stage.GetPrimAtPath(s['root']+'/shaft')).ComputeLocalToWorldTransform(0)).T
                        tip=M[:3,3]+M[:3,:3]@s['mount_to_tip']
                        hit=self.grasp.probe(h,grasp_points,tip,M[:3,:3])
                        if hit and self.controllers[h].row[6]<=0:result['gripper_target_deg']=max(result['gripper_target_deg'],hit[1])
                        self.grasp.update(h,grasp_points,tip,M[:3,:3],self.controllers[h].row[6],guard.pinch_latched,measured)
                    if h in self.grasp.held and self.controllers[h].row[6]<=0:
                        self.jaw_open[h]=self.grasp.held[h]['jaw_deg']
                    else:self.jaw_open[h]=guard.jaw_target(result['gripper_target_deg'],self.jaw_open[h],measured,1/480)
                    half=np.deg2rad(self.jaw_open[h]/2)
                    self.robots[h].apply_action(ArticulationAction(joint_positions=np.array([half,-half]),joint_indices=self.jaw_indices[h]))
                self.scope.step(scope,1/480);self.scope.apply_usd(self.stage)
                self.sim.step(render=False)
                for h in self.specs:self.guards[h].update(*self.sensor(h))
                info=self.measure();self.trace.append(info)
                if info['failure']:break
            self.sim.render()
        except Exception:
            self.done=True;raise
        self.steps+=1;obs=self.observe()
        distance=np.linalg.norm(obs['tool_pose'][:,:3,3]-self.goal,axis=1)
        terminated=bool(info['failure']);truncated=self.steps>=self.max_steps and not terminated
        self.done=terminated or truncated
        info.update(rot_rejected={h:x.tolist() for h,x in rejected.items()},distance_m=distance.tolist())
        info['deformation']=self.deformation.measure(self.points())
        reward=-float(distance.mean())/.01-(5. if terminated else 0.)
        return obs,reward,terminated,truncated,info

    render=BimanualEnv.render
    _destroy_render=BimanualEnv._destroy_render
    def close(self):
        if hasattr(self,'grasp'):
            for hand in list(self.grasp.held):self.grasp.release(hand)
        self._destroy_render()
        if self.sim:self.sim.stop()
        self.done=True
