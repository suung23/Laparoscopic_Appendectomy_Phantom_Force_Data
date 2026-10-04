"""Gymnasium benchmark for an already-attached appendix, NOT appendectomy.

Requires a running Isaac SimulationApp and an attached.usda from the probe.
Observation is privileged simulation state. No jaw-contact/grasp inference.
"""
from pathlib import Path
import gymnasium as gym
import numpy as np


class AppendixRetractionEnv(gym.Env):
    metadata={'render_modes':['rgb_array'],'render_fps':60}

    def __init__(self,app,stage_path,max_steps=160,render_mode=None):
        super().__init__()
        if render_mode not in (None,"rgb_array"): raise ValueError("Unsupported render mode")
        self.render_mode=render_mode
        self.rgb=None
        self.product=None
        self.app=app
        self.stage_path=Path(stage_path).resolve()
        if not self.stage_path.is_file(): raise FileNotFoundError(self.stage_path)
        self.max_steps=int(max_steps)
        if self.max_steps<1: raise ValueError('max_steps must be positive')
        self.action_space=gym.spaces.Box(-1,1,shape=(3,),dtype=np.float32)
        # Handle displacement xyz, mean grip displacement xyz, max pin drift, target lift.
        self.observation_space=gym.spaces.Box(-np.inf,np.inf,shape=(8,),dtype=np.float32)
        self.sim=None
        self.done=True

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        import omni.usd
        from pxr import UsdGeom,UsdPhysics,PhysxSchema
        from isaacsim.core.api import SimulationContext
        if self.rgb is not None:
            self.rgb.detach([self.product.path])
            self.product.destroy()
            self.rgb=None
            self.product=None
        if self.sim: self.sim.stop()
        SimulationContext.clear_instance()
        omni.usd.get_context().open_stage(str(self.stage_path))
        for _ in range(30): self.app.update()
        self.stage=omni.usd.get_context().get_stage()
        self.sim=SimulationContext(stage_units_in_meters=1.,physics_dt=1/120,rendering_dt=1/60)
        for prim in self.stage.Traverse():
            if prim.IsA(UsdPhysics.Scene):
                UsdPhysics.Scene(prim).CreateGravityMagnitudeAttr(0.)
                api=PhysxSchema.PhysxSceneAPI.Apply(prim)
                api.CreateEnableGPUDynamicsAttr(True)
                api.CreateBroadphaseTypeAttr('GPU')
        self.sim.initialize_physics()
        for _ in range(60): self.app.update()
        self.mesh=self.stage.GetPrimAtPath('/World/Atlas/appendix/sim')
        self.q0=np.asarray(self.mesh.GetAttribute('points').Get(),float).copy()
        if self.q0.ndim!=2 or not len(self.q0): raise RuntimeError('Cooked deformable nodes unavailable')
        self.pin=np.asarray(self.stage.GetPrimAtPath('/World/BaseAttachment').GetAttribute('omniphysics:vtxIndicesSrc0').Get(),int)
        self._bind_stage()
        self.steps=0
        self.previous_lift=0.
        self.done=False
        self.sim.play()
        for _ in range(getattr(self,'reset_settle_steps',0)):
            self.sim.step(render=True)
        if self.render_mode=='rgb_array':
            import omni.replicator.core as rep
            self.product=rep.create.render_product('/World/Endoscope',(640,360))
            self.rgb=rep.AnnotatorRegistry.get_annotator('rgb')
            self.rgb.attach([self.product])
        observation=self._observe()
        self.previous_lift=self.lift
        return observation,self._info(False)

    def _bind_stage(self):
        from pxr import UsdGeom
        self.grip=np.asarray(self.stage.GetPrimAtPath('/World/GripAttachment').GetAttribute('omniphysics:vtxIndicesSrc0').Get(),int)
        self.op=UsdGeom.Xformable(self.stage.GetPrimAtPath('/World/RetractionHandle')).GetOrderedXformOps()[0]
        self.origin=np.asarray(self.op.Get(),float).copy()
        self.command=np.zeros(3)

    def _observe(self):
        q=np.asarray(self.mesh.GetAttribute('points').Get(),float)
        if not np.isfinite(q).all(): raise RuntimeError('Non-finite PhysX tissue state')
        displacement=(q[self.grip]-self.q0[self.grip]).mean(axis=0)
        self.pin_drift=float(np.linalg.norm(q[self.pin]-self.q0[self.pin],axis=1).max())
        self.lift=float(displacement[2])
        return np.r_[self.command,displacement,self.pin_drift,.01].astype(np.float32)

    def _info(self,success):
        return {'task':'pre-attached appendix retraction','is_success':bool(success),
            'appendectomy_completed':False,'appendix_severed':False,
            'privileged_state_observation':True,'contact_based_grasp':False,
            'lift_m':self.lift,'pin_drift_m':self.pin_drift}

    def step(self,action):
        if self.done: raise RuntimeError('Call reset before stepping a finished episode')
        action=np.asarray(action,np.float32)
        if action.shape!=(3,) or not np.isfinite(action).all() or np.any(np.abs(action)>1):
            raise ValueError('Action must be three finite values in [-1,1]')
        from pxr import Gf
        self.command=np.clip(self.command+action*.001,-.025,.025)
        self.op.Set(Gf.Vec3d(*(self.origin+self.command)))
        for _ in range(2): self.sim.step(render=False)
        obs=self._observe()
        self.steps+=1
        failed=self.pin_drift>.001
        success=self.lift>=.01 and not failed
        reward=100.*(self.lift-self.previous_lift)-.001*float(np.dot(action,action))
        if failed: reward-=1.
        if success: reward+=1.
        self.previous_lift=self.lift
        terminated=bool(success or failed)
        truncated=bool(self.steps>=self.max_steps and not terminated)
        self.done=terminated or truncated
        return obs,float(reward),terminated,truncated,self._info(success)

    def render(self):
        if self.render_mode!='rgb_array': return None
        if self.rgb is None: raise RuntimeError('Reset before rendering')
        import omni.replicator.core as rep
        # Pause: image acquisition must not advance the controlled physical state.
        self.sim.pause()
        rep.orchestrator.step(rt_subframes=4,pause_timeline=True,delta_time=0.0)
        pixels=np.asarray(self.rgb.get_data())
        if not self.done: self.sim.play()
        if pixels.shape[:2]!=(360,640): raise RuntimeError('Missing endoscope image')
        return pixels[:,:,:3].astype(np.uint8).copy()

    def close(self):
        if self.sim: self.sim.stop()
        self.done=True
