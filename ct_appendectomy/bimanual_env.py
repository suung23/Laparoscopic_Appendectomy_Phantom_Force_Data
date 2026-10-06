"""Goal-conditioned two-tool Gymnasium environment; explicit grasp assist.

One SimulationApp per process, supplied by the caller. State is privileged and
does not contain the complete FEM state. Tasks are manipulation, not surgery.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import gymnasium as gym
from gymnasium import spaces

TASKS = ('reach', 'dual_grasp', 'retract')
HANDS = ('Left', 'Right')


def save_config(path, stage_path, data):
    def encode(value):
        if isinstance(value, np.ndarray): return value.tolist()
        raise TypeError(type(value).__name__)
    payload = {'version': 1, 'stage_sha256': hashlib.sha256(Path(stage_path).read_bytes()).hexdigest(),
               'data': data, 'physics_dt': 1/480, 'control_dt': 1/60,
               'gravity': [0, -9.81, 0], 'grasp_assist': True}
    Path(path).write_text(json.dumps(payload, default=encode, indent=2), encoding='utf-8')


class BimanualEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 60}

    def __init__(self, app, assets, task='retract', max_steps=900, settle_steps=360):
        if task not in TASKS: raise ValueError('Unknown task')
        if max_steps < 1 or settle_steps < 1: raise ValueError('Invalid episode limits')
        self.app, self.assets, self.task = app, Path(assets), task
        self.max_steps, self.settle_steps = int(max_steps), int(settle_steps)
        self.config = json.loads((self.assets/'bimanual_config.json').read_text())
        self.stage_path = self.assets/'attached.usda'
        if self.config['version'] != 1: raise ValueError('Unsupported asset version')
        if hashlib.sha256(self.stage_path.read_bytes()).hexdigest() != self.config['stage_sha256']:
            raise ValueError('Stage/config mismatch')
        self.physics_dt=float(self.config['physics_dt'])
        if self.physics_dt not in (1/480,1/960,1/1920):
            raise ValueError('Unvalidated physics time-step profile')
        self.action_space = spaces.Box(-1., 1., (10,), np.float32)
        # joints/held 12, mouth-to-target 6, patch displacement 6, goal 1,
        # selected-patch ownership 2, task one-hot 3, remaining time 1.
        self.observation_space = spaces.Box(-np.inf, np.inf, (31,), np.float32)
        self.sim = None
        self.rgb = self.product = None
        self.semantic = None
        self.render_mode = 'rgb_array'
        self.done = True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if set(options)-{'task', 'lift_m', 'patches'}: raise ValueError('Unknown reset option')
        task = options.get('task', self.task)
        if task not in TASKS: raise ValueError('Unknown task')
        lift = float(options.get('lift_m', self.np_random.uniform(.004, .008)))
        patches = np.asarray(options.get('patches', [1, 5]))
        count = len(self.config['data']['clusters'])
        if not .004 <= lift <= .008: raise ValueError('Lift must be 4–8 mm')
        if patches.shape != (2,) or not np.issubdtype(patches.dtype, np.integer) or len(set(patches.tolist())) != 2 or np.any(patches < 0) or np.any(patches >= count):
            raise ValueError('Two distinct valid patch indices required')
        import omni.usd
        from isaacsim.core.api import SimulationContext
        from pxr import UsdPhysics, PhysxSchema, Gf
        from bimanual_tools import BimanualControl
        self.done = True
        self._destroy_render()
        if self.sim: self.sim.stop()
        SimulationContext.clear_instance()
        omni.usd.get_context().open_stage(str(self.stage_path.resolve()))
        for _ in range(30): self.app.update()
        stage = omni.usd.get_context().get_stage()
        self.sim = SimulationContext(stage_units_in_meters=1., physics_dt=self.physics_dt, rendering_dt=1/60)
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.Scene):
                physics = UsdPhysics.Scene(prim)
                physics.CreateGravityDirectionAttr(Gf.Vec3f(0, -1, 0))
                physics.CreateGravityMagnitudeAttr(9.81)
                api = PhysxSchema.PhysxSceneAPI.Apply(prim)
                api.CreateEnableGPUDynamicsAttr(True)
                api.CreateBroadphaseTypeAttr('GPU')
        self.sim.initialize_physics()
        data = json.loads(json.dumps(self.config['data']))
        data['pin'] = np.asarray(data['pin'], int)
        for cluster in data['clusters']:
            cluster['indices'] = np.asarray(cluster['indices'], int)
            cluster['origin'] = np.asarray(cluster['origin'], float)
        for arm in data['arms'].values():
            for key in arm: arm[key] = np.asarray(arm[key], float)
        data['axis'] = np.asarray(data['axis'], float)
        mesh = stage.GetPrimAtPath(data['sim_path'])
        self.rest = np.asarray(mesh.GetAttribute('points').Get(), float).copy()
        self.tets = np.asarray(mesh.GetAttribute('tetVertexIndices').Get(), int)
        self.control = BimanualControl(stage, data, self.rest)
        self.sim.play()
        for _ in range(self.settle_steps): self.sim.step(render=True)
        self.reference = self.control.points().copy()
        if np.max(np.linalg.norm(self.reference-self.rest, axis=1)) < .0001:
            raise RuntimeError('No gravity deformation: refuse inactive physics backend')
        self.patch_reference = self.control.centers()[patches].copy()
        self.patches, self.active_task, self.lift = patches, task, lift
        self.steps = self.hold = 0
        metrics = self._metrics()
        if metrics['failure']: self.done = True; raise RuntimeError('Invalid settled state: '+metrics['failure'])
        self.potential = self._potential(metrics)
        self.done = False
        return self._observe(metrics), self._info(metrics)

    def _metrics(self):
        from tissue_metrics import tetra_health
        q = self.control.points()
        if not np.isfinite(q).all(): raise FloatingPointError('Non-finite FEM state')
        centers = self.control.centers()[self.patches]
        mouths = np.array([self.control.arms[h].pose()['position_m'] for h in HANDS])
        targets = centers.copy()
        if self.active_task == 'reach':
            ports = np.array([self.control.arms[h].port for h in HANDS])
            direction = ports-centers
            targets += .018*direction/np.linalg.norm(direction, axis=1)[:, None]
        owned = np.array([self.control.held.get(h, {}).get('cluster') == int(i) for h, i in zip(HANDS, self.patches)])
        pin = self.control.data['pin']
        drift = float(np.linalg.norm(q[pin]-self.rest[pin], axis=1).max()) if len(pin) else 0.
        health = tetra_health(self.rest, q, self.tets)
        failure = 'tet_inversion' if health['inverted_tetrahedra'] else ('pin_drift' if drift > .001 else '')
        if np.linalg.norm(q-self.reference, axis=1).max() > .05: failure = 'excess_displacement'
        delta = centers-self.patch_reference
        distance = np.linalg.norm(mouths-targets, axis=1)
        reached = bool(np.all(distance < .002))
        lifted = bool(owned.all() and np.all(np.abs(delta[:, 1]-self.lift) < .0015) and np.all(np.linalg.norm(delta[:, [0, 2]], axis=1) < .003))
        achieved = {'reach': reached, 'dual_grasp': bool(owned.all()), 'retract': lifted}[self.active_task]
        trocar = {h: self.control.arms[h].trocar_metrics() for h in HANDS}
        if any(not x['aperture_clear'] for x in trocar.values()): failure = 'trocar_aperture_violation'
        return dict(error=targets-mouths, delta=delta, owned=owned, distance=distance,
                    achieved=achieved, failure=failure, pin_drift_m=drift, health=health, trocar=trocar)

    def _potential(self, m):
        reach = -float(np.minimum(m['distance']/.03, 2).mean())
        if self.active_task == 'reach': return reach
        value = reach+float(m['owned'].mean())
        if self.active_task == 'retract':
            value -= float(np.minimum(np.abs(m['delta'][:, 1]-self.lift)/self.lift, 2).mean())
        return value

    def _observe(self, m):
        return np.r_[self.control.observation(), m['error'].ravel()/.03,
                     m['delta'].ravel()/.01, self.lift/.01, m['owned'].astype(float),
                     [self.active_task == t for t in TASKS], 1-self.steps/self.max_steps].astype(np.float32)

    def _info(self, m):
        return dict(task=self.active_task, target_patches=self.patches.tolist(), target_lift_m=self.lift,
                    success=bool(self.hold >= 30 and not m['failure']), failure=m['failure'],
                    hold_steps=self.hold, pin_drift_m=m['pin_drift_m'], tissue_health=m['health'],
                    grasp_assist=True, appendectomy_completed=False,
                    trocar=m.get('trocar', {}), rosota_controller_conformant=False)

    def step(self, action):
        if self.done: raise RuntimeError('Call reset before stepping')
        action = np.asarray(action, dtype=float)
        if action.shape != (10,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1):
            raise ValueError('Expected 10 finite normalized actions')
        self.control.step_action(action, dt=1/60)
        self.sim.step(render=True)
        self.steps += 1
        try: m = self._metrics()
        except FloatingPointError:
            self.done = True
            raise  # Never return a fabricated healthy observation after solver failure.
        self.hold = self.hold+1 if m['achieved'] and not m['failure'] else 0
        success = self.hold >= 30
        terminated = bool(success or m['failure'])
        truncated = bool(self.steps >= self.max_steps and not terminated)
        potential = self._potential(m)
        reward = .99*(0. if terminated else potential)-self.potential-.002
        reward += 5. if success else (-5. if m['failure'] else 0.)
        self.potential, self.done = potential, terminated or truncated
        return self._observe(m), float(reward), terminated, truncated, self._info(m)

    def close(self):
        self._destroy_render()
        if self.sim: self.sim.stop()
        self.done = True

    def _destroy_render(self):
        if self.rgb is not None: self.rgb.detach([self.product.path])
        if getattr(self, 'semantic', None) is not None:
            self.semantic.detach([self.product.path])
            self.semantic = None
        if self.product is not None: self.product.destroy()
        self.rgb = self.product = None

    def _ensure_render_product(self):
        if not hasattr(self, 'control'): raise RuntimeError('Reset before rendering')
        import omni.replicator.core as rep
        if self.product is None:
            self.product = rep.create.render_product('/World/Endoscope', (640, 360))
            self.rgb = rep.AnnotatorRegistry.get_annotator('rgb')
            self.rgb.attach(self.product)
        return rep

    def render(self):
        rep = self._ensure_render_product()
        playing = self.sim.is_playing()
        self.sim.pause()
        try:
            rep.orchestrator.step(rt_subframes=4, delta_time=0.)
            pixels = np.asarray(self.rgb.get_data())[:, :, :3].copy()
            if pixels.shape != (360, 640, 3): raise RuntimeError('Missing endoscope RGB')
            return pixels
        finally:
            if playing: self.sim.play()

    def render_class_fraction(self, class_name):
        """Measure a class from the same endoscope product as the RGB frame."""
        rep = self._ensure_render_product()
        if self.semantic is None:
            self.semantic = rep.AnnotatorRegistry.get_annotator(
                'semantic_segmentation', init_params={'colorize': False})
            self.semantic.attach([self.product])
        from semantic_visibility import class_pixel_fraction
        playing = self.sim.is_playing()
        self.sim.pause()
        try:
            rep.orchestrator.step(rt_subframes=4, delta_time=0.)
            packet = self.semantic.get_data()
            if not isinstance(packet, dict) or not isinstance(packet.get('info'), dict):
                raise RuntimeError('Missing semantic segmentation metadata')
            return class_pixel_fraction(packet.get('data'),
                                       packet['info'].get('idToLabels'), class_name)
        finally:
            if playing: self.sim.play()

    def render_with_class_fraction(self, class_name):
        """Capture synchronized RGB and semantic visibility at one render tick."""
        rep = self._ensure_render_product()
        if self.semantic is None:
            self.semantic = rep.AnnotatorRegistry.get_annotator(
                'semantic_segmentation', init_params={'colorize': False})
            self.semantic.attach([self.product])
        playing = self.sim.is_playing()
        self.sim.pause()
        try:
            rep.orchestrator.step(rt_subframes=4, delta_time=0.)
            pixels = np.asarray(self.rgb.get_data())[:, :, :3].copy()
            packet = self.semantic.get_data()
            if pixels.shape != (360, 640, 3):
                raise RuntimeError('Missing endoscope RGB for semantic visibility capture')
            if not isinstance(packet, dict) or not isinstance(packet.get('info'), dict):
                raise RuntimeError('Missing semantic segmentation metadata')
            from semantic_visibility import class_pixel_fraction
            fraction = class_pixel_fraction(packet.get('data'),
                packet['info'].get('idToLabels'), class_name)
            semantic_shape = np.asarray(packet['data']).shape[:2]
            if semantic_shape != pixels.shape[:2]:
                raise RuntimeError('RGB and semantic image dimensions are not synchronized')
            return pixels, fraction
        finally:
            if playing: self.sim.play()
