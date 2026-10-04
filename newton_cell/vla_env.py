#!/usr/bin/env python3
"""`LaparoVLAEnv` on Newton -- the same interface, a different engine.

Drop-in for `isaac/vla_env.py`: same constructor keywords, same `(2, K, 7)`
action, same observation keys, same `info`. `isaac/vla_rollout.py` drives this
without modification when the backend is selected (see `newton_cell/rollout.py`).

    from vla_env import LaparoVLAEnv, Action
    env = LaparoVLAEnv(headless=True, control_hz=30, chunk=8)
    obs = env.reset()
    obs, info = env.step(action)

What is the same, by construction: the action semantics and every sign
(VLA_HANDOFF.md section 3), the kinematics (`laparo_ik`), the controller
(`laparo_control.BEST`), the trocar constraints and the leash, the tissue
material, and the scoring inputs.

What differs, deliberately and reported rather than hidden:

  * `track_lag_m` is ~0. The arm is a position-driven follower, which is the
    architecture SURGICAL_CELL.md specifies; the 0.38 mm p95 in the Isaac
    reference is that plant's servo error.
  * organ deformation will not match PhysX FEM, which the contract allows.
  * the force channels come from tissue contact rather than a joint reaction
    sensor, so there is no weight model to identify and no `noise_*` floor from
    a 54-pose least-squares fit -- see `forces.py`.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from arms import ARMS                                          # noqa: E402
from forces import ContactForces                               # noqa: E402
from grasp import GraspAssist                                  # noqa: E402
from scene import add_args                                     # noqa: E402
import sim as simmod                                           # noqa: E402
from view import CellRenderer                                  # noqa: E402

# Isaac's vla_env.Action, verbatim in behaviour
from importlib import util as _ilu                             # noqa: E402
_spec = _ilu.spec_from_file_location("_isaac_action",
                                     ROOT / "isaac" / "vla_env.py")

DEFAULTS = dict(
    physics_hz=480.0, control_hz=30.0, chunk=8, insert=0.10, min_depth=0.10,
    # rcm_abort_mm 3 -> 10 and the separate rcm_excursion_mm: see the long
    # note in isaac/vla_env.py DEFAULTS. Kept in step so the two backends
    # score the same episode the same way.
    cmd_leash_m=0.015, rcm_abort_mm=10.0, rcm_excursion_mm=3.0,
    rcm_abort_steps=5, manip_hz=60.0,
    roll_follow="absolute", gripper_rate_dps=250.0, grasp_force=0.03,
    jaw_length=0.015, grasp_radius=0.012, grasp_assist=True, stereo=False,
    endo_depth=False, d405_baseline=0.018, max_episode_steps=0, seed=0,
    # renderer
    samples=16, bounces=1, exposure=3.0, decimate_to=60000,
    device=None, iterations=10, radius_ratio=0.1, mass_floor=0.25,
    no_phantom=False, no_organ_pins=False, organ_support="none",
    organ_field_collision=False, phantom_collision=False,
    front=True, front_samples=6, front_bounces=0,
)

# Where to drop front-view PNGs, and how often. Set by `newton_cell/rollout.py`
# before it hands control to `isaac/vla_rollout.py`: that file constructs the
# env itself with a fixed keyword list, so an extra camera cannot be plumbed
# through its constructor without editing it -- and it is deliberately not
# edited. Module state is the seam that keeps `isaac/` untouched.
FRONT_OUT = None
FRONT_EVERY = 1

# Render-quality overrides, set by `eval.py` from its --preset before the env is
# built. Same seam and same reason as FRONT_OUT: `isaac/vla_rollout.py`
# constructs the env with a fixed keyword list and is deliberately not edited.
OVERRIDES = {}


class Action:
    """Helpers for the `(2, K, 7)` action array. Same as isaac/vla_env.Action."""

    LEFT, RIGHT = 0, 1
    DX, DY, DZ, RX, RY, RZ, GRIP = range(7)

    @staticmethod
    def zeros(chunk, gripper=(0.0, 0.0)):
        a = np.zeros((2, chunk, 7), np.float32)
        a[Action.LEFT, :, Action.GRIP] = gripper[0]
        a[Action.RIGHT, :, Action.GRIP] = gripper[1]
        return a

    @staticmethod
    def check(a, chunk):
        if isinstance(a, dict):
            per = {}
            for nm in ARMS:
                v = a.get(nm)
                if v is None:
                    continue
                v = np.atleast_2d(np.asarray(v, np.float32))
                if v.shape[1] != 7:
                    raise ValueError(f"action['{nm}'] must be (K,7), got {v.shape}")
                per[nm] = v
            if not per:
                raise ValueError("action dict has neither 'left' nor 'right'")
            k = max(len(v) for v in per.values())
            out = np.zeros((2, k, 7), np.float32)
            for i, nm in enumerate(ARMS):
                if nm in per:
                    out[i, :len(per[nm])] = per[nm]
            a = out
        a = np.asarray(a, np.float32)
        if a.ndim == 2:
            a = np.stack([a, a], 0)
        if a.ndim != 3 or a.shape[0] != 2 or a.shape[2] != 7:
            raise ValueError(f"action must be (2, K, 7), got {a.shape}")
        if a.shape[1] > chunk:
            raise ValueError(f"action chunk {a.shape[1]} exceeds env chunk={chunk}")
        if not np.all(np.isfinite(a)):
            raise ValueError("action contains NaN/Inf")
        return a


def _ns(**kw):
    d = dict(DEFAULTS)
    d.update(OVERRIDES)
    d.update({k: v for k, v in kw.items() if v is not None})
    return argparse.Namespace(**d)


class LaparoVLAEnv:
    """The Newton-backed evaluation environment."""

    def __init__(self, headless=True, **kw):
        if not headless:
            raise RuntimeError(
                "the Newton backend is headless-only: it exists because this "
                "host has no RT cores and no display. Use the Isaac backend on "
                "the RTX workstation for a GUI (VLA_HANDOFF.md 1.1).")
        unknown = set(kw) - set(DEFAULTS)
        if unknown:
            raise TypeError(f"unknown keyword(s): {sorted(unknown)}")
        self.cfg = _ns(**kw)
        self.chunk = int(self.cfg.chunk)
        wp.init()
        self.cell = simmod.Cell(self.cfg)
        self.arms = self.cell.arms
        self.L = self.cell.L
        self.forces = ContactForces(self.cell.model, self.arms)
        self.grasp = GraspAssist(self.cell.scene, self.arms,
                                 grasp_radius=self.cfg.grasp_radius) \
            if self.cfg.grasp_assist else None
        self.renderer = CellRenderer(self.cell, samples=self.cfg.samples,
                                     bounces=self.cfg.bounces,
                                     exposure=self.cfg.exposure,
                                     decimate_to=self.cfg.decimate_to,
                                     quiet=True)
        self._q_rest = self.cell.state_0.particle_q.numpy().copy()
        self._appendix_rest = self._appendix_centroid()
        self.n_steps = 0
        self._front_n = 0
        self._front_written = 0
        self._rcm_run = {k: 0 for k in ARMS}
        self._rcm_total = 0

    # ---- helpers ---------------------------------------------------------
    def _appendix_centroid(self):
        i, j = self.cell.scene.organ_span["appendix"]
        return self.cell.state_0.particle_q.numpy()[i:j].mean(axis=0)

    def _to_rig(self, p):
        return self.L.to_rig(p)

    # ---- API -------------------------------------------------------------
    def reset(self, seed=None):
        # A fresh episode re-settles the tissue and returns the arms to the
        # solved start pose. The scene is rebuilt only if it has to be; a reset
        # is a state reset, which is what makes it ~1 s rather than a rebuild.
        for arm in self.arms.values():
            arm.q = arm.q0.copy()
            arm.ctl.q = arm.q0.copy()
            arm.jaw_q[:] = 0.0
            arm.grip_cmd, arm.grip_meas = 100.0, 0.0
            p_tip, _ = arm.chain.fk(arm.q, arm.T_base)
            arm.p_cmd = p_tip.copy()
            arm.v_cmd = np.zeros(3)
            arm.roll_cmd = arm.roll_meas = 0.0
            arm.info = dict(depth_clamped=0, cone_clamped=0, leash_clamped=0)
        self.cell.state_0.particle_q.assign(self._q_rest)
        self.cell.state_0.particle_qd.zero_()
        self.cell._write_bodies(self.cell.state_0)
        if self.grasp:
            self.grasp.latched.clear()
        self.n_steps = 0
        self._rcm_run = {k: 0 for k in ARMS}
        self._rcm_total = 0
        self._appendix_rest = self._appendix_centroid()
        return self.observe()

    def step(self, action):
        a = Action.check(action, self.chunk)
        K = a.shape[1]
        for k in range(K):
            cmd = {}
            for i, nm in enumerate(ARMS):
                row = a[i, k]
                cmd[nm] = (row[0:3].astype(np.float64), float(row[5]),
                           float(row[6]))
            self.cell.control_substep(cmd)
            if self.grasp:
                self.grasp.update(self.cell.state_0)
                self.grasp.apply(self.cell.state_0)
        self.n_steps += 1

        info = self._info(a, K)
        return self.observe(), info

    def _info(self, a, K):
        info = {"sub_steps": int(K)}
        viol = []
        for nm, arm in self.arms.items():
            rcm_mm = arm.rcm_error() * 1000.0
            if rcm_mm > self.cfg.rcm_abort_mm:
                self._rcm_run[nm] += 1
            else:
                self._rcm_run[nm] = 0
            # counted on the PORT's level, not the abort's -- the abort ends
            # the run, this says the shaft left its 6 mm hole
            if rcm_mm > self.cfg.rcm_excursion_mm:
                self._rcm_total += 1
            if self._rcm_run[nm] >= self.cfg.rcm_abort_steps:
                viol.append(nm)
            rej = np.abs(a[ARMS.index(nm), :, 3:5]).sum(axis=0)
            info[nm] = dict(
                rot_rejected=np.array([rej[0], rej[1], 0.0], np.float32),
                depth_clamped=int(arm.info["depth_clamped"]),
                cone_clamped=int(arm.info["cone_clamped"]),
                leash_clamped=int(arm.info["leash_clamped"]),
                rcm_err_mm=float(rcm_mm),
                trocar_bite_mm=float(max(0.0, rcm_mm - 0.5)),
                track_lag_mm=0.0,          # position-driven follower, see docstring
                cmd_error_mm=float(np.linalg.norm(
                    arm.p_cmd - arm.link_frames()["tool_tip"][:3, 3]) * 1000.0),
                roll_backlog_deg=float(math.degrees(arm.roll_cmd - arm.roll_meas)),
            )
        info["rcm_violation"] = viol
        info["rcm_violation_steps"] = max(self._rcm_run.values())
        info["rcm_violation_total"] = self._rcm_total
        info["terminated"] = bool(viol)
        info["truncated"] = bool(self.cfg.max_episode_steps
                                 and self.n_steps >= self.cfg.max_episode_steps)
        return info

    # ---- observation -----------------------------------------------------
    def observe(self):
        self.renderer.sync()
        obs = {"t": float(self.cell.t), "T_world_rig": self.L.T_world_rig.astype(np.float32)}

        endo = self.renderer.render_endoscope()
        cam = self.renderer.cams.endoscope
        if self.cfg.front:
            obs["front"] = dict(
                rgb=self.renderer.render_front(samples=self.cfg.front_samples,
                                               bounces=self.cfg.front_bounces),
                T_world_cam=self.renderer.cams.front.T.astype(np.float32))
            self._dump_front(obs["front"]["rgb"])

        obs["endoscope"] = dict(
            rgb=endo, K=cam.K, T_rig_cam=self._T_rig_cam(cam),
            T_world_cam=cam.T.astype(np.float32),
            hfov_deg=cam.hfov_deg(), vfov_deg=cam.vfov_deg(),
            width=cam.width, height=cam.height)

        d_rgb, d_depth = self.renderer.render_d405()
        dcam = self.renderer.cams.d405
        obs["d405"] = dict(
            rgb=d_rgb, depth=d_depth, K=dcam.K,
            T_rig_cam=self._T_rig_cam(dcam),
            T_world_cam=dcam.T.astype(np.float32),
            hfov_deg=dcam.hfov_deg(), vfov_deg=dcam.vfov_deg(),
            width=dcam.width, height=dcam.height,
            baseline_m=float(self.cfg.d405_baseline))
        if self.cfg.stereo:
            obs["d405"]["rgb_right"] = self.renderer.render_d405_right(
                self.cfg.d405_baseline)

        for nm in ARMS:
            obs[nm] = self._arm_obs(nm)
        obs["left_force"] = self._force_obs("left")
        obs["task"] = self._task_obs()
        return obs

    def _dump_front(self, rgb):
        """Write the front frame where rollout.py will find it to encode."""
        if FRONT_OUT is None:
            return
        if self._front_n % max(1, FRONT_EVERY) == 0:
            import imageio.v3 as iio
            d = Path(FRONT_OUT)
            d.mkdir(parents=True, exist_ok=True)
            iio.imwrite(d / f"front_{self._front_written:05d}.png", rgb)
            self._front_written += 1
        self._front_n += 1

    def _T_rig_cam(self, cam):
        return (np.linalg.inv(self.L.T_world_rig) @ cam.T).astype(np.float32)

    def _arm_obs(self, nm):
        arm = self.arms[nm]
        F = arm.link_frames()
        T = F["tool_tip"]
        T_rig = np.linalg.inv(self.L.T_world_rig) @ T
        tip, axis = arm.shaft_line()
        n = arm.u_in
        tilt = float(math.acos(np.clip(np.dot(axis, n), -1.0, 1.0)))
        perp = axis - np.dot(axis, n) * n
        az = float(math.atan2(perp[1], perp[0])) if np.linalg.norm(perp) > 1e-9 else 0.0
        ins = arm.insertion_depth()
        roll = float(arm.roll_meas)
        return dict(
            tip_pose_rig=T_rig.astype(np.float32),
            tip_pos_rig=T_rig[:3, 3].astype(np.float32),
            tip_quat_rig=_R_to_quat(T_rig[:3, :3]).astype(np.float32),
            tip_pose_world=T.astype(np.float32),
            insertion_m=float(ins),
            rcm4=np.array([ins, tilt, az, roll], np.float32),
            shaft_dir_rig=(self.L.T_world_rig[:3, :3].T @ axis).astype(np.float32),
            port_id=int(arm.port_no),
            port_pos_rig=self._to_rig(arm.p_rcm).astype(np.float32),
            tip_cmd_pos_rig=self._to_rig(arm.p_cmd).astype(np.float32),
            cmd_error_m=float(np.linalg.norm(arm.p_cmd - T[:3, 3])),
            roll_cmd_rad=float(arm.roll_cmd),
            roll_meas_rad=float(arm.roll_meas),
            roll_backlog_rad=float(arm.roll_cmd - arm.roll_meas),
            gripper_cmd=float(arm.grip_cmd),
            gripper_meas=float(arm.grip_meas),
            jaw_rad=arm.jaw_q.astype(np.float32).copy(),
            joint_pos=arm.q.astype(np.float32).copy(),
            joint_vel=np.zeros(7, np.float32),
            grasped=bool(self.grasp.grasped(nm)) if self.grasp else False,
            rcm_err_m=float(arm.rcm_error()),
            trocar_bite_m=float(max(0.0, arm.rcm_error() - 0.0005)),
            track_lag_m=0.0,
        )

    def _force_obs(self, nm):
        w = self.forces.tool_wrench(self.cell.state_0, self.cell.contacts, nm)
        ch = ContactForces.channels(w)
        held = self.grasp.grasped(nm) if self.grasp else False
        gm = float(np.linalg.norm(w[:3]))
        ch.update(
            grip=float(min(gm, 0.075)),        # the model saturates, as in Isaac
            grip_raw=gm,
            contact=bool(gm > 1e-4),
            latched=bool(held),
            # No identified wrench model here, so no identification residual:
            # the gravity term the Isaac path has to cancel does not exist for a
            # kinematically driven instrument. The floor is the solver's own.
            noise_force_n=0.0,
            noise_torque_nm=0.0,
            noise_grip_n=0.0,
        )
        return ch

    def _task_obs(self):
        q = self.cell.state_0.particle_q.numpy()
        i, j = self.cell.scene.organ_span["appendix"]
        cen = q[i:j].mean(axis=0)
        handles = self.grasp.handle_positions(q) if self.grasp else np.zeros((0, 3))
        pin = getattr(self.cell.scene, "appendix_pin_world", cen)
        left_tip = self.arms["left"].link_frames()["tool_tip"][:3, 3]
        right_tip = self.arms["right"].link_frames()["tool_tip"][:3, 3]
        return dict(
            appendix_centroid_rig=self._to_rig(cen).astype(np.float32),
            appendix_lift_m=float(cen[2] - self._appendix_rest[2]),
            appendix_moved_m=float(np.linalg.norm(cen - self._appendix_rest)),
            handles_rig=np.array([self._to_rig(h) for h in handles], np.float32)
            if len(handles) else np.zeros((0, 3), np.float32),
            handle_free=self.grasp.free_mask() if self.grasp else np.zeros(0, bool),
            appendix_pin_rig=self._to_rig(pin).astype(np.float32),
            handle_to_pin_m=np.array([np.linalg.norm(h - pin) for h in handles],
                                     np.float32),
            left_tip_to_appendix_m=float(np.linalg.norm(q[i:j] - left_tip,
                                                        axis=1).min()),
            right_tip_to_appendix_m=float(np.linalg.norm(q[i:j] - right_tip,
                                                         axis=1).min()),
            grasped_any=bool(self.grasp.latched) if self.grasp else False,
        )

    def summary(self):
        return dict(backend="newton", particles=int(self.cell.model.particle_count),
                    tets=int(self.cell.model.tet_count),
                    physics_hz=float(self.cfg.physics_hz),
                    control_hz=float(self.cfg.control_hz))

    def close(self):
        pass


def _R_to_quat(R):
    t = float(np.trace(R))
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        return np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s,
                         (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(max(1e-12, 1.0 + R[i, i] - R[j, j] - R[k, k])) * 2.0
    q = np.zeros(4)
    q[0] = (R[k, j] - R[j, k]) / s
    q[1 + i] = 0.25 * s
    q[1 + j] = (R[j, i] + R[i, j]) / s
    q[1 + k] = (R[k, i] + R[i, k]) / s
    return q


class DemoPolicy:
    """Scripted baseline: approach the best handle, close on it, lift.

    State-driven rather than step-counted, so it behaves the same at any
    `--seconds`: approach, close once the tip is inside the latch gate, lift
    once the grasp has actually taken.

    The handle is chosen at least 25 mm from the appendix's world pin, for the
    reason VLA_HANDOFF.md section 4.4 gives: grasping next to the pin means
    pulling against a rigid anchor, which in the Isaac baseline saturated the
    command leash and pushed the shaft 16 mm out of a 6 mm port.
    """

    APPROACH_MM = 6.0        # tip-to-handle distance at which to start closing
    STEP_M = 0.0015          # per sub-step APPROACH motion, well under the leash
    # The lift is an order of magnitude slower than the approach, and that is
    # the whole difference between a scored run and an aborted one. The
    # appendix's far end is pinned to the world (VLA_HANDOFF.md section 4.4), so
    # lifting stretches the tissue between a rigid anchor and the jaws; pulling
    # hard on that drags the shaft out of a 6 mm port. Measured here at
    # 0.0015 m/sub-step (~45 mm/s): +19.1 mm of lift, then a sustained trocar
    # violation on the left arm at step 18 with RCM p95 4.54 mm. The Isaac
    # reference lifts 10.7 mm over a 20 s episode -- roughly 1 mm/s.
    LIFT_M = 0.00012         # per sub-step lift, ~3.6 mm/s at 30 Hz
    LIFT_TARGET_M = 0.015    # stop climbing once clear of the 10 mm success bar

    def __init__(self, env=None):
        self.env = env
        self.k = 0
        self.closed_for = 0

    def reset(self, obs):
        self.k = 0
        self.closed_for = 0

    def __call__(self, obs):
        chunk = self.env.chunk if self.env else 8
        task = obs["task"]
        tip = np.asarray(obs["left"]["tip_pos_rig"], np.float64)
        R = np.asarray(obs["left"]["tip_pose_rig"], np.float64)[:3, :3]
        grasped = bool(obs["left"]["grasped"])

        handles = np.asarray(task["handles_rig"], np.float64)
        if len(handles):
            far = np.asarray(task["handle_to_pin_m"], np.float64) > 0.025
            free = np.asarray(task["handle_free"], bool)
            ok = far & free
            if not ok.any():
                ok = free if free.any() else np.ones(len(handles), bool)
            cand = handles[ok]
            tgt = cand[int(np.argmin(np.linalg.norm(cand - tip, axis=1)))]
        else:
            tgt = np.asarray(task["appendix_centroid_rig"], np.float64)

        dist = float(np.linalg.norm(tgt - tip))
        if grasped:
            self.closed_for += 1
            lift = float(task["appendix_lift_m"])
            rate = self.LIFT_M if lift < self.LIFT_TARGET_M else 0.0
            d_rig = np.array([0.0, 0.0, rate])                 # lift, gently
            grip = 0.0
        elif dist * 1000.0 <= self.APPROACH_MM:
            d_rig = np.zeros(3)                                 # hold and close
            grip = 0.0
            self.closed_for += 1
        else:
            d_rig = tgt - tip                                   # approach
            grip = 100.0

        cap = self.LIFT_M if grasped else self.STEP_M
        n = np.linalg.norm(d_rig)
        if n > cap:
            d_rig = d_rig * (cap / n)

        a = Action.zeros(chunk, gripper=(grip, 100.0))
        a[Action.LEFT, :, 0:3] = (R.T @ d_rig).astype(np.float32)
        a[Action.LEFT, :, 6] = grip
        self.k += 1
        return a

    def label(self):
        return "newton-demo-grasp-lift"
