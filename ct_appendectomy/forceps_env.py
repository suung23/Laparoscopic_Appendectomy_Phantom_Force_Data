"""Two-action jaw-contact lift fixture; not complete appendectomy or full RCM."""
import hashlib
import json
from pathlib import Path
import gymnasium as gym
import numpy as np
from retraction_env import AppendixRetractionEnv
from forceps import move_forceps
from tissue_metrics import tetra_health


class AppendixForcepsEnv(AppendixRetractionEnv):
    metadata={'render_modes':['rgb_array'],'render_fps':30}
    def __init__(self,app,stage_path,config_path,max_steps=220,render_mode=None):
        self.config=json.loads(Path(config_path).read_text(encoding='utf-8'))
        if self.config['stage_sha256']!=hashlib.sha256(Path(stage_path).read_bytes()).hexdigest():
            raise ValueError('Forceps configuration/stage mismatch')
        super().__init__(app,stage_path,max_steps,render_mode)
        # Reset returns the authored rest state. Settling introduces contact
        # integration before action zero and is not part of this task contract.
        self.reset_settle_steps=0
        self.action_space=gym.spaces.Box(-1,1,shape=(2,),dtype=np.float32)
        self.observation_space=gym.spaces.Box(-np.inf,np.inf,shape=(7,),dtype=np.float32)

    def _bind_stage(self):
        if self.stage.GetPrimAtPath('/World/GripAttachment').IsValid():
            raise RuntimeError('Contact benchmark must not have a grip attachment')
        self.forceps=dict(self.config['forceps'])
        for key in ('pin','grip','origin','axes'):
            self.forceps[key]=np.asarray(self.forceps[key],dtype=int if key in ('pin','grip') else float)
        self.grip=self.forceps['grip']
        self.command=np.zeros(2)
        self.held_steps=0
        self.tets=np.asarray(self.mesh.GetAttribute('tetVertexIndices').Get(),int)
        self.closed_reference=None
        self.reference_command_lift=0.
        move_forceps(self.stage,self.forceps,0.,0.)

    def _observe(self):
        q=np.asarray(self.mesh.GetAttribute('points').Get(),float)
        if not np.isfinite(q).all():raise RuntimeError('Non-finite tissue state')
        displacement=(q[self.grip]-self.q0[self.grip]).mean(axis=0)
        self.pin_drift=float(np.linalg.norm(q[self.pin]-self.q0[self.pin],axis=1).max())
        self.lift=float(displacement[2])
        self.health=tetra_health(self.q0,q,self.tets)
        self.maximum_displacement=float(np.linalg.norm(q-self.q0,axis=1).max())
        self.current_centroid=q[self.grip].mean(axis=0)
        if self.closed_reference is None:
            self.tracking_error=0.
        else:
            self.tracking_error=float(np.linalg.norm(self.current_centroid-self.closed_reference-np.array([0,0,self.command[1]-self.reference_command_lift])))
        return np.r_[self.command,displacement,self.pin_drift,self.tracking_error].astype(np.float32)

    def _info(self,success):
        return {'task':'hinged-jaw contact lift fixture','is_success':bool(success),
                'appendectomy_completed':False,'appendix_severed':False,
                'privileged_state_observation':True,'grip_attachment_present':False,
                'trocar_constraint_integrated':False,'lift_m':self.lift,
                'pin_drift_m':self.pin_drift,'tracking_error_m':self.tracking_error,
                'maximum_displacement_m':self.maximum_displacement,
                'tissue_health':self.health}

    def step(self,action):
        if self.done:raise RuntimeError('Reset before stepping a finished episode')
        action=np.asarray(action,np.float32)
        if action.shape!=(2,) or not np.isfinite(action).all() or np.any(np.abs(action)>1):
            raise ValueError('Two finite normalized actions required')
        previous_command=self.command.copy()
        self.command=np.clip(self.command+action*np.array([.025,.0002]),[0,0],[1,.016])
        if self.closed_reference is None and self.command[1]>previous_command[1] and previous_command[0]>=.99:
            self.closed_reference=self.current_centroid.copy()
            self.reference_command_lift=float(previous_command[1])
        if self.command[0]<.5:
            self.closed_reference=None
            self.reference_command_lift=0.
        move_forceps(self.stage,self.forceps,*self.command)
        # USD-authored kinematic jaw transforms are consumed through Kit updates.
        # Two 1/60 render ticks contain four 1/120 physics steps. A tensor-driven
        # headless fast path must prove equivalence before replacing this path.
        for _ in range(2):self.sim.step(render=True)
        obs=self._observe()
        self.steps+=1
        failed=self.pin_drift>.001 or self.maximum_displacement>.05 or self.health['inverted_tetrahedra']>0
        following=self.closed_reference is not None and self.command[0]>=.99 and self.command[1]-self.reference_command_lift>=.008 and self.tracking_error<.004 and self.lift>=.006
        self.held_steps=self.held_steps+1 if following and not failed else 0
        success=self.held_steps>=10
        reward=100*(self.lift-self.previous_lift)-.001*float(action@action)
        self.previous_lift=self.lift
        reward+=float(success)-float(failed)
        terminated=bool(success or failed)
        truncated=bool(self.steps>=self.max_steps and not terminated)
        self.done=terminated or truncated
        return obs,float(reward),terminated,truncated,self._info(success)
