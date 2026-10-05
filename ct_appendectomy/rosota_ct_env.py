"""CT scene with measured PhysX FR3 joints and the original Rosota controller.

Engineering port/base placement; right cutter is optional and cutting is unimplemented.
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
    def __init__(self,app,assets,fr3_urdf,out,max_steps=300,grasp_mode='none',workspace='legacy',settle_policy='fixed',diagnostic_no_tool_collision=False,diagnostic_zero_gravity=False,enhanced_determinism=False,reset_checkpoint=None,fine_settlement=False,external_every_iteration=False,cutter_cad=None,jaw_gravity_compensation=False,gpu_tissue_observation=False,physics_substeps=1):
        if settle_policy not in ('fixed','adaptive'):raise ValueError('Unknown settlement policy')
        self.cutter_cad=Path(cutter_cad) if cutter_cad else None
        if self.cutter_cad and grasp_mode!='none':raise ValueError('Cutter cannot use assisted tissue grasp')
        if self.cutter_cad:
            manifest=json.loads(self.cutter_cad.with_name('manifest.json').read_text())
            if hashlib.sha256(self.cutter_cad.read_bytes()).hexdigest()!=manifest['usd_sha256']:
                raise ValueError('Cutter asset hash mismatch')
            if manifest['units']!='metres' or manifest['open_total_deg']!=50 or not np.allclose(manifest['hinge_to_tip_m'],[0,0,.021],atol=1e-7):
                raise ValueError('Unsupported cutter geometry contract')
        if isinstance(physics_substeps,bool) or physics_substeps not in (1,2,4):raise ValueError('Supported physics substeps: 1, 2, 4')
        self.physics_substeps=physics_substeps
        self.physics_dt=1/(480*physics_substeps)
        self.gpu_tissue_observation=bool(gpu_tissue_observation)
        self._tissue_observer=None
        if self.gpu_tissue_observation and grasp_mode!='none':raise ValueError('GPU observation grasp path requires separate validation')
        self.jaw_gravity_compensation=bool(jaw_gravity_compensation)
        self.settle_policy=settle_policy
        self.enhanced_determinism=bool(enhanced_determinism)
        self.reset_checkpoint=Path(reset_checkpoint) if reset_checkpoint else None
        self.fine_settlement=bool(fine_settlement)
        self.external_every_iteration=bool(external_every_iteration)
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
        if self._tissue_observer:self._tissue_observer.close();self._tissue_observer=None
        self.shaft_bodies={}
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
        from deformable_parameters import validate_deformable_parameters
        validate_deformable_parameters(self.stage.GetPrimAtPath('/World/Atlas/appendix'))
        # Older authored stages used a bare Xform; modern PhysX attachments
        # require a rigid actor for this stationary attachment target.
        anchor=self.stage.GetPrimAtPath('/World/BaseAnchor')
        if not anchor:raise ValueError('Missing appendix base anchor')
        UsdPhysics.RigidBodyAPI.Apply(anchor).CreateKinematicEnabledAttr(True)
        UsdPhysics.MassAPI.Apply(anchor).CreateMassAttr(.01)
        self.specs=author(self.stage,self.config['data'],self.fr3_urdf,self.out,
            patches=self.target_patches if self.workspace=='appendix_patches' else None,cutter_cad=self.cutter_cad)
        self.disabled_robot_colliders=[]
        if self.diagnostic_no_tool_collision:
            for prim in self.stage.Traverse():
                if str(prim.GetPath()).startswith('/World/FR3/') and prim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
                    self.disabled_robot_colliders.append(str(prim.GetPath()))
            if not self.disabled_robot_colliders:raise RuntimeError('Collision diagnostic disabled no robot colliders')
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
        pipeline=dict(backend='torch',device='cuda:0') if self.gpu_tissue_observation else {}
        self.sim=SimulationContext(stage_units_in_meters=1.,physics_dt=self.physics_dt,rendering_dt=1/30,**pipeline)
        for prim in self.stage.Traverse():
            if prim.IsA(UsdPhysics.Scene):
                scene=UsdPhysics.Scene(prim);scene.CreateGravityDirectionAttr(Gf.Vec3f(0,-1,0));scene.CreateGravityMagnitudeAttr(0. if self.diagnostic_zero_gravity else 9.81)
                ps=PhysxSchema.PhysxSceneAPI.Apply(prim);ps.CreateEnableGPUDynamicsAttr(True);ps.CreateBroadphaseTypeAttr('GPU');ps.CreateEnableEnhancedDeterminismAttr(self.enhanced_determinism);ps.CreateEnableExternalForcesEveryIterationAttr(self.external_every_iteration)
        self.stage.GetRootLayer().Export(str(self.out/'scene.usda'))
        self._audit_tissue('before_initialize')
        from omni.physx import get_physx_cooking_interface
        tissue_root=self.stage.GetPrimAtPath('/World/Atlas/appendix')
        self.tissue_generation='automatic' if 'PhysxAutoDeformableBodyAPI' in tissue_root.GetAppliedSchemas() else 'explicit'
        if self.tissue_generation=='automatic':
            if not get_physx_cooking_interface().cook_auto_deformable_body('/World/Atlas/appendix'):
                raise RuntimeError('Explicit pre-initialization tissue cooking failed')
        else:
            from surface_target import TissueSurface
            from tissue_metrics import tetra_health
            TissueSurface(self.tets)
            if len(np.unique(np.sort(self.tets,axis=1),axis=0))!=len(self.tets):
                raise RuntimeError('Duplicate explicit tetrahedra')
            manual_health=tetra_health(self.rest,self.points(),self.tets)
            if manual_health['degenerate_rest_tetrahedra'] or manual_health['inverted_tetrahedra']:
                raise RuntimeError('Invalid explicit tissue geometry')
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
        if self.reset_checkpoint:
            from reset_checkpoint import checkpoint_points
            from pxr import Vt
            with np.load(self.reset_checkpoint) as checkpoint:
                initial_points,health=checkpoint_points(checkpoint,self.rest,self.tets)
            self.mesh.GetAttribute('points').Set(Vt.Vec3fArray.FromNumpy(initial_points.astype(np.float32)))
            (self.out/'reset_checkpoint.json').write_text(json.dumps(dict(
                sha256=hashlib.sha256(self.reset_checkpoint.read_bytes()).hexdigest(),
                health=health,scope='Initial nodal configuration only; not full solver-state serialization',
                rest_shape_unchanged=True,settlement_required=self.settle_policy=='adaptive'),indent=2),encoding='utf-8')
        (self.out/'node_correspondence.json').write_text(json.dumps(dict(
            exact_geometry_and_topology=True,nodes=len(permutation),
            changed_indices=int(np.sum(permutation!=np.arange(len(permutation))))),indent=2),encoding='utf-8')
        self.sim.initialize_physics()
        if self.gpu_tissue_observation:
            from tissue_observer import TissueTensorObserver
            self._tissue_observer=TissueTensorObserver.bind(self.stage,'/World/Atlas/appendix',self.config['data']['sim_path'],omni.usd.get_context().get_stage_id())
        self._audit_tissue('after_initialize')
        if not np.isclose(self.sim.get_physics_dt(),self.physics_dt):raise RuntimeError('Wrong physics interval')
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
            self.indices[hand]=self._input([robot.get_dof_index('fr3_joint'+str(i)) for i in range(1,8)],indices=True)
            self.jaw_indices[hand]=self._input([robot.get_dof_index('jaw_left'),robot.get_dof_index('jaw_right')],indices=True)
            robot.set_joint_positions(self._input(s['seed']),joint_indices=self.indices[hand])
            robot.set_joint_positions(self._input(np.deg2rad([s['open_deg']/2,-s['open_deg']/2])),joint_indices=self.jaw_indices[hand])
            robot.set_joint_velocities(self._input(np.zeros(robot.num_dof)))
            robot.apply_action(ArticulationAction(joint_positions=np.deg2rad([s['open_deg']/2,-s['open_deg']/2]),
                joint_velocities=np.zeros(2),joint_indices=self.jaw_indices[hand]))
        from isaacsim.core.utils.types import ArticulationAction
        if self.gpu_tissue_observation:
            from isaacsim.core.prims import SingleRigidPrim
            for hand,s in self.specs.items():
                body=SingleRigidPrim(s['root']+'/shaft',name=hand+'_live_shaft',reset_xform_properties=False)
                body.initialize();self.shaft_bodies[hand]=body
        self._audit_tissue('before_warmup')
        for warmup_tick in range(480):
            for hand,s in self.specs.items():
                self.robots[hand].apply_action(ArticulationAction(joint_positions=s['seed'],joint_velocities=np.zeros(7),joint_indices=self.indices[hand]))
            self._physics_step()
            if warmup_tick in (0,59,479):self._audit_tissue('warmup_'+str(warmup_tick+1))
        self.settlement={'policy':self.settle_policy,'measured_equilibrium':False,'physics_ticks':480*self.physics_substeps,'control_ticks':480,'physics_dt_s':self.physics_dt}
        if self.settle_policy=='adaptive':
            from settlement import SettlementGate
            gate=SettlementGate(self.points(),rms_limit_m_s=1e-6 if self.fine_settlement else 1e-4,
                p99_limit_m_s=5e-6 if self.fine_settlement else 5e-4,
                require_velocity_peak=self._tissue_observer is not None)
            settled=False
            for window in range(120 if self.fine_settlement else 56):
                peak_speed=np.zeros(len(self.rest)) if self._tissue_observer else None
                sample_substep=None
                state=None
                if self._tissue_observer:
                    from tissue_metrics import step_velocity_comparison
                    previous=self.points();displacement_peak=np.zeros(len(self.rest));velocity_gap=0.
                    def sample_substep():
                        nonlocal peak_speed,previous,displacement_peak,velocity_gap,state
                        state=self._tissue_observer.snapshot(omni.usd.get_context().get_stage_id())
                        peak_speed=np.maximum(peak_speed,np.linalg.norm(state['velocity_m_s'],axis=1))
                        mean_velocity,gap=step_velocity_comparison(previous,state['position_m'],state['velocity_m_s'],self.physics_dt)
                        displacement_peak=np.maximum(displacement_peak,np.linalg.norm(mean_velocity,axis=1))
                        velocity_gap=max(velocity_gap,gap);previous=state['position_m']
                for _ in range(60):self._physics_step(sample_substep)
                settled=gate.update(state['position_m'] if self._tissue_observer else self.points(),peak_speed)
                if self._tissue_observer:
                    gate.rows[-1].update(step_displacement_peak_rms_m_s=float(np.sqrt(np.mean(displacement_peak**2))),
                        step_displacement_peak_p99_m_s=float(np.percentile(displacement_peak,99)),
                        maximum_endpoint_vs_step_mean_velocity_rms_gap_m_s=velocity_gap)
                (self.out/'settlement_progress.json').write_text(json.dumps(dict(
                    window=window+1,physics_ticks=(480+60*(window+1))*self.physics_substeps,control_ticks=480+60*(window+1),latest=gate.rows[-1])),encoding='utf-8')
                if window>=7 and settled:break
            self.settlement.update(measured_equilibrium=settled,physics_ticks=(480+60*len(gate.rows))*self.physics_substeps,control_ticks=480+60*len(gate.rows),
                limits={'rms_speed_m_s':gate.rms_limit,'p99_speed_m_s':gate.p99_limit,'consecutive_windows':gate.required},
                interval_s=gate.dt,velocity_sampling_hz=480*self.physics_substeps if self._tissue_observer else None,trace=gate.rows)
            (self.out/'settlement.json').write_text(json.dumps(self.settlement,indent=2),encoding='utf-8')
            if not settled:
                self._audit_tissue('settlement_timeout')
                raise RuntimeError('Tissue did not reach the measured reset equilibrium gate')
        self.sim.render()
        self.controllers={h:RosotaArmController(s['chain'],s['base'],s['port'],s['normal'],self.q(h),hand=s['tool_hand']) for h,s in self.specs.items()}
        self.control=self.controllers  # Shared RGB renderer's initialized-state guard.
        self.reference=self.points().copy();self.steps=0
        self.goal=np.array([self.controllers[h].pose(self.q(h))[0] for h in self.specs])+np.array([[.003,0,0],[-.003,0,0]])
        self.trace=[];self.jaw_open={h:s['open_deg'] for h,s in self.specs.items()}
        self.initial_q=np.array([self.q(h) for h in self.specs]);self.initial_scope=self.scope.pose.copy()
        from contact_control import ContactGuard
        self.force_rows={};samples={h:[] for h in self.specs};jaw_samples={h:[] for h in self.specs}
        for h,robot in self.robots.items():
            self.force_rows[h]=int(robot._articulation_view._metadata.joint_indices['tool_mount'])+1
        for _ in range(64):
            self._physics_step()
            for h in self.specs:
                wrench,effort=self.sensor(h)
                samples[h].append(wrench);jaw_samples[h].append(effort)
        self.jaw_gravity_model=None
        if self.jaw_gravity_compensation:
            from jaw_gravity_audit import JawGravityModel
            self.jaw_gravity_model=JawGravityModel(self)
        self.guards={};self.hold_q={};calibration={}
        for h in self.specs:
            values=np.asarray(samples[h]);baseline=values.mean(axis=0)
            noise=float(np.linalg.norm(values[:,:3]-baseline[:3],axis=1).max())
            self.guards[h]=ContactGuard(baseline,noise,max_open_deg=self.specs[h]['open_deg'])
            self.update_guard(h)
            calibration[h]=dict(baseline=baseline.tolist(),max_force_residual_n=noise,
                jaw_effort_mean_nm=np.mean(jaw_samples[h],axis=0).tolist(),
                jaw_effort_min_nm=np.min(jaw_samples[h],axis=0).tolist(),
                jaw_effort_max_nm=np.max(jaw_samples[h],axis=0).tolist(),
                jaw_effort_tare_applied=False,
                samples=64,frame='local child joint frame at tool_mount',
                model='static tare only; no orientation or inertia compensation')
        (self.out/'contact_calibration.json').write_text(json.dumps(calibration,indent=2),encoding='utf-8')
        if self.diagnostic_no_tool_collision:
            from jaw_gravity_audit import audit
            (self.out/'jaw_gravity_audit.json').write_text(json.dumps(audit(self),indent=2),encoding='utf-8')
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

    def _physics_step(self,after_substep=None):
        # One controller interval remains 1/480 s at every solver resolution.
        for _ in range(self.physics_substeps):
            self.sim.step(render=False)
            if after_substep is not None:after_substep()

    @staticmethod
    def _array(value):
        if hasattr(value,'detach'):value=value.detach().cpu()
        return np.asarray(value.numpy() if hasattr(value,'numpy') else value,float)

    def _input(self,value,indices=False):
        if self.gpu_tissue_observation:
            import torch
            return torch.as_tensor(value,dtype=torch.int64 if indices else torch.float32,device='cuda:0')
        return np.asarray(value,dtype=int if indices else float)

    def q(self,h):return self._array(self.robots[h].get_joint_positions(joint_indices=self.indices[h]))
    def update_guard(self,h):
        gravity=self.jaw_gravity_model.predict(h) if self.jaw_gravity_model else None
        return self.guards[h].update(*self.sensor(h),jaw_gravity=gravity)

    def sensor(self,h):
        robot=self.robots[h]
        wrench=self._array(robot.get_measured_joint_forces())[self.force_rows[h]]
        efforts=self._array(robot.get_measured_joint_efforts(joint_indices=self.jaw_indices[h]))
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
        velocity_fields={}
        if self._tissue_observer:
            state=self._tissue_observer.snapshot(omni.usd.get_context().get_stage_id())
            points=state['position_m']
            velocity_fields['velocity_m_s']=state['velocity_m_s']
            velocity_fields.update(self._tissue_observer.collision_snapshot(omni.usd.get_context().get_stage_id()))
        else:points=self.points().copy()
        rest=np.asarray(self.mesh.GetAttribute('omniphysics:restShapePoints').Get(),float)
        tets=np.asarray(self.mesh.GetAttribute('tetVertexIndices').Get(),int)
        np.savez_compressed(self.out/('tissue_'+label+'.npz'),points=points,rest=rest,tets=tets,
            pin=indices,target_world=world_targets,anchor_world=transform,**velocity_fields)
        summary=dict(label=label,nodes=len(points),tetrahedra=len(tets),
            observation_source='physx_tensor' if self._tissue_observer else 'usd',physics_dt_s=self.physics_dt,
            pin_target_error_m=float(np.linalg.norm(points[indices]-world_targets,axis=1).max()),
            points_sha256=hashlib.sha256(points.tobytes()).hexdigest(),
            tets_sha256=hashlib.sha256(tets.tobytes()).hexdigest())
        (self.out/('tissue_'+label+'.json')).write_text(json.dumps(summary,indent=2),encoding='utf-8')
        import shutil
        for suffix in ('.npz','.json'):
            shutil.copyfile(self.out/('tissue_'+label+suffix),
                self.out/(f'reset_{self.reset_sequence}_tissue_'+label+suffix))

    def points(self):
        if self._tissue_observer:
            import omni.usd
            return self._tissue_observer.snapshot(omni.usd.get_context().get_stage_id())['position_m']
        return np.asarray(self.mesh.GetAttribute('points').Get(),float)

    def shaft_transform(self,h):
        if self.gpu_tissue_observation:
            from scipy.spatial.transform import Rotation
            p,q=self.shaft_bodies[h].get_world_pose();q=self._array(q)
            M=np.eye(4);M[:3,3]=self._array(p);M[:3,:3]=Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
            return M
        from pxr import UsdGeom
        return np.asarray(UsdGeom.Xformable(self.stage.GetPrimAtPath(self.specs[h]['root']+'/shaft')).ComputeLocalToWorldTransform(0)).T
    def observe(self):
        poses=[];qs=[]
        for h in self.specs:
            q=self.q(h);p,R,_=self.controllers[h].pose(q);T=np.eye(4);T[:3,:3]=R;T[:3,3]=p
            poses.append(T);qs.append(q)
        return {'joint_position':np.asarray(qs,np.float32),'tool_pose':np.asarray(poses,np.float32),
                'contact_residual':np.asarray([g.filtered for g in self.guards.values()],np.float32),
                'jaw_position':np.asarray([self._array(self.robots[h].get_joint_positions(joint_indices=self.jaw_indices[h])) for h in self.specs],np.float32),
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
            M=self.shaft_transform(h)
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
            jaw_position_rad={h:self._array(self.robots[h].get_joint_positions(joint_indices=self.jaw_indices[h])).tolist() for h in self.specs},
            tissue_health=health,pin_drift_m=drift,controller='original Rosota TipController strict aperture',
            plant='PhysX dynamic articulations with measured joint feedback',
            calibrated_ports=False,right_tool=self.specs['right']['tool_kind'],cutting_implemented=False,scope_shaft_collision=False,
            commissioning_retreat_m=.015 if self.workspace=='legacy' else .018,workspace=self.workspace,
            grasp_assist=self.grasp_mode!='none' or bool(self.grasp.held),
            grasp_mode=self.grasp_mode,held_patches={h:v['cluster'] for h,v in self.grasp.held.items()},
            settlement_verified=self.settlement['measured_equilibrium'],
            diagnostic_no_tool_collision=self.diagnostic_no_tool_collision,
            disabled_robot_collider_count=len(self.disabled_robot_colliders),
            tissue_generation=self.tissue_generation,tissue_observation='physx_tensor' if self._tissue_observer else 'usd',
            jaw_gravity_compensation=self.jaw_gravity_compensation,
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
                            gripper_target_deg=self.controllers[h].row[6]*self.controllers[h].open_deg/100,rot_rejected=np.zeros(2))
                    else:result=self.controllers[h].tick(self.q(h))
                    rejected[h]+=result['rot_rejected']
                    self.robots[h].apply_action(ArticulationAction(joint_positions=result['joint_target'],
                        joint_velocities=result['joint_velocity_ff'],joint_indices=self.indices[h]))
                    jaws=self._array(self.robots[h].get_joint_positions(joint_indices=self.jaw_indices[h]))
                    measured=float(np.rad2deg(jaws[0]-jaws[1]))
                    if self.grasp_mode!='none':
                        from pxr import UsdGeom
                        s=self.specs[h]
                        M=self.shaft_transform(h)
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
                self._physics_step()
                for h in self.specs:self.update_guard(h)
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
        if self._tissue_observer:self._tissue_observer.close();self._tissue_observer=None
        if hasattr(self,'grasp'):
            for hand in list(self.grasp.held):self.grasp.release(hand)
        self._destroy_render()
        if self.sim:self.sim.stop()
        self.done=True
