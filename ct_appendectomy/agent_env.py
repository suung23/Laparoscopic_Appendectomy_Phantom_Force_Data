"""Versioned robot-agent interface over the contact-lift fixture.

Two actuator commands, not a robot arm or a complete appendectomy environment.
The compact observation is privileged and does not expose every FEM degree of freedom.
"""
import gymnasium as gym
import numpy as np

ENV_ID='RiiAppendixContactLift-v0'
_APP=None
STATE_NAMES=('jaw_closure','command_lift_10mm','tissue_dx_10mm','tissue_dy_10mm',
             'tissue_dz_10mm','pin_drift_mm','tracking_error_4mm','episode_fraction',
             'held_fraction','reference_dx_10mm','reference_dy_10mm','reference_dz_10mm',
             'reference_valid','reference_command_lift_10mm')


def scripted_action(step):
    """Verified engineering baseline; never labelled a learned policy."""
    return np.asarray([1/1.8,0] if step<72 else
                      ([0,1/1.8] if 108<=step<181 else [0,0]),np.float32)


class RobotAgentEnv(gym.Wrapper):
    def __init__(self,env,observation_mode='state'):
        super().__init__(env)
        if observation_mode not in ('state','rgb_state'):
            raise ValueError('Choose state or rgb_state observations')
        if observation_mode=='rgb_state' and env.render_mode!='rgb_array':
            raise ValueError('rgb_state requires render_mode=rgb_array')
        self.observation_mode=observation_mode
        state=gym.spaces.Box(-np.inf,np.inf,shape=(len(STATE_NAMES),),dtype=np.float32)
        self.observation_space=state if observation_mode=='state' else gym.spaces.Dict({
            'state':state,'rgb':gym.spaces.Box(0,255,(360,640,3),np.uint8)})
        self.potential=0.

    def _state(self,raw):
        core=self.env
        reference=np.zeros(3) if core.closed_reference is None else (
            core.closed_reference-core.q0[core.grip].mean(axis=0))/.01
        state=np.r_[raw/np.array([1,.01,.01,.01,.01,.001,.004]),
                    core.steps/core.max_steps,core.held_steps/10,reference,
                    float(core.closed_reference is not None),core.reference_command_lift/.01].astype(np.float32)
        if not np.isfinite(state).all():raise RuntimeError('Non-finite agent observation')
        return state

    def _observation(self,raw):
        state=self._state(raw)
        return state if self.observation_mode=='state' else {'state':state,'rgb':self.env.render()}

    def _potential(self):
        core=self.env
        held_lift=0. if core.closed_reference is None else float(
            core.current_centroid[2]-core.closed_reference[2])
        following=core.command[0]>=.99 and core.tracking_error<.004
        return .25*core.command[0]+(.75*np.clip(held_lift/.008,0,1) if following else 0.)

    def reset(self,*,seed=None,options=None):
        raw,info=self.env.reset(seed=seed,options=options)
        if seed is not None:self.action_space.seed(seed)
        self.potential=self._potential()
        return self._observation(raw),dict(info,env_id=ENV_ID,observation_mode=self.observation_mode)

    def step(self,action):
        raw,_,terminated,truncated,info=self.env.step(action)
        next_potential=0. if terminated else self._potential()
        # Potential shaping uses the learner's documented gamma=0.99. A time
        # limit bootstraps; an actual terminal success/failure does not.
        reward=.99*next_potential-self.potential-.002
        reward+=5. if info['is_success'] else (-5. if terminated else 0.)
        self.potential=next_potential
        info=dict(info,env_id=ENV_ID,observation_mode=self.observation_mode,
                  termination_reason=('success' if info['is_success'] else
                                      'tissue_stability' if terminated else
                                      'time_limit' if truncated else None))
        return self._observation(raw),float(reward),terminated,truncated,info


def bind_simulation_app(app):
    """Bind the one process-local Kit runtime, without copying it into Gym specs."""
    global _APP
    _APP=app


def make_env(stage_path,config_path,observation_mode='state',max_steps=220):
    if _APP is None:raise RuntimeError('Call bind_simulation_app(app) after initializing Isaac')
    # Import only after SimulationApp has initialized Isaac/PhysX extensions.
    from forceps_env import AppendixForcepsEnv
    return RobotAgentEnv(AppendixForcepsEnv(_APP,stage_path,config_path,max_steps,
                         render_mode='rgb_array'),observation_mode)


if ENV_ID not in gym.registry:
    gym.register(id=ENV_ID,entry_point='agent_env:make_env',nondeterministic=True)
