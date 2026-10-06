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
  * the force channels are post-step soft-contact estimates, not a joint
    reaction readback or calibrated tissue forces. The kinematic model has no
    payload weight to compensate; its `noise_*` values are placeholders, not
    measured noise floors -- see `forces.py`.
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
from task_metrics import (deformation_health, deformation_health_by_organ,
                          partition_tets_by_spans, resection_observations,
                          tissue_probe)  # noqa: E402
from view import CellRenderer                                  # noqa: E402

# Isaac's vla_env.Action, verbatim in behaviour
from importlib import util as _ilu                             # noqa: E402
_spec = _ilu.spec_from_file_location("_isaac_action",
                                     ROOT / "isaac" / "vla_env.py")

DEFAULTS = dict(
    physics_hz=480.0, control_hz=30.0, chunk=8, insert=0.10, min_depth=0.10,
    settle_seconds=2.0,
    # Scripted-policy diagnostic only; 12 mm/s preserves the prior 0.4 mm
    # action-substep displacement at 30 Hz. Learned policies set their own
    # actions and do not consume this parameter.
    demo_approach_mm_s=12.0,
    # Optional local Appendix300 appearance candidate. This is an RGB visual
    # transfer onto FEM rest nodes, not a patient-space video reconstruction.
    appendix_texture=None, appendix_texture_mix=1.0,
    # Diagnostic ablation only; normal environments retain tool/tissue contact.
    tool_tissue_collision=True,
    # Experimental swept-URDF clearance ranking for the scripted demo only.
    # It remains opt-in until it improves held-out tissue-contact outcomes.
    demo_swept_clearance=False,
    # Engineering starting point only; force limit/stiffness need physical
    # calibration before reporting tissue loads or grasp fidelity.
    grasp_stiffness_n_m=30.0, grasp_damping_ratio=0.8,
    # Rigid-tool/tissue and appendix-support contact material. ShapeConfig kf
    # is solver-specific; the VBD body-particle path uses the averaged mu.
    contact_ke=1.0e5, contact_kd=1.0, contact_kf=1.0e3,
    # Optional diagnostic override for both sides of VBD soft-rigid contact.
    contact_mu=None,
    # rcm_abort_mm 3 -> 10 and the separate rcm_excursion_mm: see the long
    # note in isaac/vla_env.py DEFAULTS. Kept in step so the two backends
    # score the same episode the same way.
    cmd_leash_m=0.015, rcm_abort_mm=10.0, rcm_excursion_mm=3.0,
    rcm_abort_steps=5, manip_hz=60.0,
    roll_follow="absolute", gripper_rate_dps=250.0, grasp_force=0.03,
    jaw_length=0.015, grasp_radius=0.012, grasp_assist=True, stereo=False,
    grasp_band=(0.30, 0.70), resect_offset_m=0.010,
    endo_depth=False, d405_baseline=0.018, max_episode_steps=0, seed=0,
    # renderer
    samples=16, bounces=1, exposure=3.0, decimate_to=60000,
    surface_subdivision_levels=1,
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
PHYSICS_ONLY = False

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
        self.forces = ContactForces(self.cell.model, self.arms,
                                    solver=self.cell.solver,
                                    dt=self.cell.physics_dt)
        self.grasp = GraspAssist(self.cell.scene, self.arms,
                                 self.cell.model, self.cell.physics_dt,
                                 grasp_radius=self.cfg.grasp_radius,
                                 force_limit_n=self.cfg.grasp_force,
                                 stiffness_n_m=self.cfg.grasp_stiffness_n_m,
                                 damping_ratio=self.cfg.grasp_damping_ratio) \
            if self.cfg.grasp_assist else None
        self.renderer = CellRenderer(self.cell, samples=self.cfg.samples,
                                     bounces=self.cfg.bounces,
                                     exposure=self.cfg.exposure,
                                     decimate_to=self.cfg.decimate_to,
                                     appendix_texture=self.cfg.appendix_texture,
                                     appendix_texture_mix=self.cfg.appendix_texture_mix,
                                     surface_subdivision_levels=self.cfg.surface_subdivision_levels,
                                     quiet=True)
        self._tet_indices = self.cell.model.tet_indices.numpy().reshape(-1, 4).copy()
        self._organ_tet_groups = partition_tets_by_spans(
            self._tet_indices, self.cell.scene.organ_span)
        # Captured CT meshes are not an equilibrium under gravity. Settle once
        # at the solved arm poses; otherwise every policy begins while the
        # appendix is still dropping onto its shelf and contacts are
        # contaminated by that startup transient.
        settle_ticks = max(0, int(round(
            float(self.cfg.settle_seconds) / self.cell.physics_dt)))
        health_period = max(1, self.cell.ticks_per_substep)
        for tick in range(settle_ticks):
            self.cell.physics_tick()
            if (tick + 1) % health_period == 0:
                q = self.cell.state_0.particle_q.numpy()
                qd = self.cell.state_0.particle_qd.numpy()
                health = deformation_health(
                    q, qd, self.cell.q_rest, self._tet_indices)
                if (not health["physics_state_finite"] or
                        health["tissue_inverted_tets"] != 0 or
                        health["tissue_nonfinite_tets"] != 0):
                    raise RuntimeError(
                        "passive tissue settling failed at "
                        f"{self.cell.t:.3f}s: {health}")
        self._q_rest = self.cell.state_0.particle_q.numpy().copy()
        self._qd_rest = self.cell.state_0.particle_qd.numpy().copy()
        self._episode_time0 = float(self.cell.t)
        health = deformation_health(
            self._q_rest, self._qd_rest, self.cell.q_rest, self._tet_indices)
        if (not health["physics_state_finite"] or
                health["tissue_inverted_tets"] != 0 or
                health["tissue_nonfinite_tets"] != 0):
            raise RuntimeError(f"passive tissue settling failed: {health}")
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
        for state in (self.cell.state_0, self.cell.state_1):
            state.particle_q.assign(self._q_rest)
            state.particle_qd.assign(self._qd_rest)
            if getattr(state, "body_qd", None) is not None:
                state.body_qd.assign(np.zeros_like(state.body_qd.numpy()))
        self.cell._write_bodies(self.cell.state_0)
        self.cell._write_bodies(self.cell.state_1)
        # `contacts` survives solver ticks and otherwise still describes the
        # previous episode's final pose. Refresh only after tissue and both
        # kinematic tool poses have been restored, before the initial obs reads
        # force channels from that contact set.
        self.cell.refresh_contacts()
        if self.grasp:
            self.grasp.reset()
        self._episode_time0 = float(self.cell.t)
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
            def apply_grasp_force(state):
                if self.grasp:
                    self.grasp.apply_forces(state)

            def update_grasp():
                if self.grasp:
                    self.grasp.update(self.cell.state_0)

            self.cell.control_substep(cmd, before_tick=apply_grasp_force,
                                      after_tick=update_grasp)
        self.n_steps += 1

        info = self._info(a, K)
        obs = self.observe()
        task = obs.get("task", {})
        if task:
            invalid_reasons = []
            if not bool(task.get("physics_state_finite", True)):
                invalid_reasons.append("non-finite FEM state")
            inverted = int(task.get("tissue_inverted_tets", 0))
            nonfinite = int(task.get("tissue_nonfinite_tets", 0))
            if inverted > 0:
                by_organ = task.get("tissue_health_by_organ", {})
                causes = [f"{name}:{int(h['tissue_inverted_tets'])}"
                          for name, h in by_organ.items()
                          if h.get("tissue_inverted_tets", 0) > 0]
                suffix = f" ({', '.join(causes)})" if causes else ""
                invalid_reasons.append(
                    f"{inverted} inverted tetrahedra{suffix}")
            if nonfinite > 0:
                by_organ = task.get("tissue_health_by_organ", {})
                causes = [f"{name}:{int(h['tissue_nonfinite_tets'])}"
                          for name, h in by_organ.items()
                          if h.get("tissue_nonfinite_tets", 0) > 0]
                suffix = f" ({', '.join(causes)})" if causes else ""
                invalid_reasons.append(
                    f"{nonfinite} non-finite tetrahedra{suffix}")
            if invalid_reasons:
                info["physics_invalid"] = "; ".join(invalid_reasons)
                # A failure-only trace links the inverted FEM element to the
                # instrument surfaces that contacted its nodes this tick.
                # Keep full contact traffic out of healthy training steps.
                worst_nodes = {}
                by_organ = task.get("tissue_health_by_organ", {})
                for organ, health in by_organ.items():
                    if (int(health.get("tissue_inverted_tets", 0)) <= 0 and
                            int(health.get("tissue_nonfinite_tets", 0)) <= 0):
                        continue
                    tet_i = int(health.get("tissue_worst_tet_index", -1))
                    group = self._organ_tet_groups.get(organ)
                    if group is None or tet_i < 0 or tet_i >= len(group[2]):
                        continue
                    start, _, local_tets = group
                    worst_nodes[organ] = (local_tets[tet_i] + start).tolist()
                info["physics_contact_diagnostics"] = (
                    self.forces.contact_diagnostics(
                        self.cell.contacts,
                        organ_spans=self.cell.scene.organ_span,
                        worst_tet_nodes=worst_nodes))
        return obs, info

    def _info(self, a, K):
        info = {"sub_steps": int(K)}
        if self.grasp:
            info["grasp"] = self.grasp.info()
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
        if PHYSICS_ONLY:
            # Keep the same physics-derived task, arm, trocar, and contact
            # channels while skipping the expensive path tracer. This mode is
            # for dynamics/contact evaluation only; it is not an image-policy
            # observation and must never be reported as a visual rollout.
            return {
                "t": float(self.cell.t - self._episode_time0),
                "T_world_rig": self.L.T_world_rig.astype(np.float32),
                "left": self._arm_obs("left"),
                "right": self._arm_obs("right"),
                "left_force": self._force_obs("left"),
                "grasp_assist": self._grasp_assist_obs(),
                "task": self._task_obs(),
            }
        self.renderer.sync()
        obs = {"t": float(self.cell.t - self._episode_time0),
               "T_world_rig": self.L.T_world_rig.astype(np.float32)}

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
        obs["grasp_assist"] = self._grasp_assist_obs()
        obs["task"] = self._task_obs()
        return obs

    def _grasp_assist_obs(self):
        """Expose controller-applied model forces separately from contact sensing."""
        force = self.grasp.last_force_n if self.grasp else {k: 0.0 for k in ARMS}
        deflection = (self.grasp.last_deflection_m if self.grasp
                      else {k: 0.0 for k in ARMS})
        return {
            "left_model_force_n": np.float32(force.get("left", 0.0)),
            "right_model_force_n": np.float32(force.get("right", 0.0)),
            "left_deflection_m": np.float32(deflection.get("left", 0.0)),
            "right_deflection_m": np.float32(deflection.get("right", 0.0)),
        }

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
        # `collide()` populated this contact set from the input state immediately
        # before the just-finished solver step. After the ping-pong swap that is
        # state_1; state_0 is already the post-solve configuration.
        w = self.forces.tool_wrench(self.cell.state_1, self.cell.contacts, nm)
        ch = ContactForces.channels(w)
        held = self.grasp.grasped(nm) if self.grasp else False
        gm = float(np.linalg.norm(w[:3]))
        ch.update(
            grip=float(min(gm, 0.075)),        # the model saturates, as in Isaac
            grip_raw=gm,
            contact=bool(gm > 1e-4),
            latched=bool(held),
            # The kinematic arm has no payload gravity to compensate, but the
            # post-step contact estimate is not an exact solver-force readback
            # and has no measured noise-floor or tissue-force calibration.
            noise_force_n=0.0,
            noise_torque_nm=0.0,
            noise_grip_n=0.0,
            force_estimator="post-step soft-contact reconstruction; static mixed stiffness",
            force_readback_available=False,
            force_calibrated=False,
            noise_floor_available=False,
        )
        return ch

    def _task_obs(self):
        q = self.cell.state_0.particle_q.numpy()
        qd = self.cell.state_0.particle_qd.numpy()
        health = deformation_health(q, qd, self._q_rest, self._tet_indices)
        health["tissue_health_by_organ"] = deformation_health_by_organ(
            q, qd, self._q_rest, self._organ_tet_groups)
        i, j = self.cell.scene.organ_span["appendix"]
        cen = q[i:j].mean(axis=0)
        handles = self.grasp.handle_positions(q) if self.grasp else np.zeros((0, 3))
        pin = getattr(self.cell.scene, "appendix_pin_world", cen)
        left_frames = self.arms["left"].link_frames()
        right_frames = self.arms["right"].link_frames()
        left_tip = left_frames["tool_tip"][:3, 3]
        right_tip = right_frames["tool_tip"][:3, 3]
        held = (self.grasp.latched.get("left", {}).get("handle", -1)
                if self.grasp else -1)
        metrics = resection_observations(
            q[i:j], handles, held, pin, right_tip,
            grasp_band=self.cfg.grasp_band,
            resect_offset_m=self.cfg.resect_offset_m)
        from laparo_ik import JAW_PIN_AXIS_F
        shaft_axis = -right_frames["tool_tip"][:3, 2]
        right_tissue, jaw_thickness, _ = tissue_probe(
            q[i:j], right_tip, shaft_axis,
            right_frames["tool_flange"][:3, :3],
            float(np.mean(np.abs(self.arms["right"].jaw_q))),
            JAW_PIN_AXIS_F)
        metrics["right_tissue_in_jaw"] = bool(right_tissue)
        metrics["right_jaw_tissue_m"] = float(jaw_thickness)
        metrics["resect_target_rig"] = (
            self._to_rig(metrics["resect_target_world"]).astype(np.float32)
            if np.isfinite(metrics["resect_target_world"]).all()
            else np.full(3, np.nan, dtype=np.float32))
        base = dict(
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
        base.update(health)
        base.update(metrics)
        return base

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
    STEP_M = 0.0004          # 12 mm/s, matching the Isaac demo approach rate
    STANDOFF_M = 0.025        # clearance above the target before descending
    STANDOFF_ENTER_M = 0.003
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
    # Clearance ranking is an offline privileged-policy heuristic. Keep its
    # spatial sampling close to the actual 0.4 mm default command increment;
    # the old 5 mm / 2048-point cap could skip thin jaw/cecum intersections.
    PATH_SAMPLE_M = 0.002
    COLLISION_POINTS_PER_LINK = 4096
    TISSUE_PARTICLE_RADIUS_M = 0.0002

    def __init__(self, env=None):
        self.env = env
        cfg = getattr(env, "cfg", None)
        control_hz = float(getattr(cfg, "control_hz", 30.0))
        speed_mm_s = float(getattr(
            cfg, "demo_approach_mm_s", self.STEP_M * 1000.0 * control_hz))
        if (not np.isfinite(control_hz) or control_hz <= 0.0 or
                not np.isfinite(speed_mm_s) or speed_mm_s <= 0.0):
            raise ValueError("demo approach speed and control rate must be positive and finite")
        self.step_m = speed_mm_s / (1000.0 * control_hz)
        self.k = 0
        self.closed_for = 0
        self.approach_phase = "standoff"
        self.target_handle_index = None
        self._tool_offsets = None
        self._tool_offsets_jaw_state = None

    @staticmethod
    def _rotation_between(a, b):
        """Minimal rotation carrying insertion axis a onto b."""
        a = np.asarray(a, np.float64).copy()
        b = np.asarray(b, np.float64).copy()
        a /= max(np.linalg.norm(a), 1e-12)
        b /= max(np.linalg.norm(b), 1e-12)
        cross = np.cross(a, b)
        s = float(np.linalg.norm(cross))
        c = float(np.clip(np.dot(a, b), -1.0, 1.0))
        if s < 1e-10:
            if c >= 0.0:
                return np.eye(3)
            basis = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 \
                else np.array([0.0, 1.0, 0.0])
            turn_axis = np.cross(a, basis)
            turn_axis /= np.linalg.norm(turn_axis)
            return 2.0 * np.outer(turn_axis, turn_axis) - np.eye(3)
        K = np.array([[0.0, -cross[2], cross[1]],
                      [cross[2], 0.0, -cross[0]],
                      [-cross[1], cross[0], 0.0]])
        return np.eye(3) + K + (K @ K) * ((1.0 - c) / (s * s))

    def _collision_offsets_in_tip_frame(self, arm):
        """Sample the real URDF collision meshes in the current tip frame."""
        jaw_state = tuple(np.round(np.asarray(arm.jaw_q), 5))
        if (self._tool_offsets is not None and
                self._tool_offsets_jaw_state == jaw_state):
            return self._tool_offsets
        tip_world = arm.link_frames()["tool_tip"]
        tip_world_inv = np.linalg.inv(tip_world)
        frames = arm.link_frames()
        geometry = arm.collision_geoms()
        offsets = []
        for link, geoms in geometry.items():
            points = []
            for geom in geoms:
                if len(geom.faces):
                    triangles = geom.verts[geom.faces]
                    edges = np.sort(np.vstack((geom.faces[:, [0, 1]],
                                               geom.faces[:, [1, 2]],
                                               geom.faces[:, [2, 0]])), axis=1)
                    edges = np.unique(edges, axis=0)
                    edge_midpoints = geom.verts[edges].mean(axis=1)
                    face_centres = triangles.mean(axis=1)
                    points.append(np.vstack([geom.verts, edge_midpoints,
                                             face_centres]))
                else:
                    points.append(geom.verts)
            if not points:
                continue
            cloud = np.vstack(points)
            if len(cloud) > self.COLLISION_POINTS_PER_LINK:
                selected = np.linspace(0, len(cloud) - 1,
                                       self.COLLISION_POINTS_PER_LINK,
                                       dtype=np.int64)
                cloud = cloud[selected]
            tip_from_link = tip_world_inv @ frames[link]
            offsets.append(cloud @ tip_from_link[:3, :3].T
                           + tip_from_link[:3, 3])
        if not offsets:
            return None
        self._tool_offsets = np.vstack(offsets)
        self._tool_offsets_jaw_state = jaw_state
        return self._tool_offsets

    def _cecum_surface_geometry_rig(self):
        """Return live cecum display vertices/faces in rig coordinates.

        Renderer faces index the display mesh, not the organ's FEM particle
        block. Resolve them through the renderer's FEM stencils so subdivision
        or a non-identity captured-surface binding cannot silently select the
        wrong tissue nodes for approach-clearance scoring.
        """
        cell = getattr(self.env, "cell", None)
        if cell is None or not hasattr(self.env, "_to_rig"):
            return None
        spans = getattr(cell.scene, "organ_span", {})
        if "cecum" not in spans:
            return None
        lo, hi = spans["cecum"]
        q = cell.state_0.particle_q.numpy()
        renderer = getattr(self.env, "renderer", None)
        faces = getattr(renderer, "_organ_faces", {}).get("cecum")
        faces = (None if faces is None else np.asarray(faces, np.int64))
        render_nodes = getattr(renderer, "_organ_render_nodes", {}).get("cecum")
        if faces is None or not len(faces):
            local = q[lo:hi]
            faces = None
        else:
            stencils = getattr(renderer, "_organ_render_stencils", {}).get("cecum")
            if stencils is not None:
                particle_ids, weights = stencils
                particle_ids = np.asarray(particle_ids, np.int64)
                weights = np.asarray(weights, np.float64)
                if (particle_ids.ndim != 2 or weights.shape != particle_ids.shape or
                        not len(particle_ids) or faces.min() < 0 or
                        faces.max() >= len(particle_ids) or
                        particle_ids.min() < 0 or particle_ids.max() >= hi - lo or
                        not np.isfinite(weights).all()):
                    raise RuntimeError("cecum render-to-FEM stencils are invalid")
                local = np.einsum(
                    "nk,nkj->nj", weights,
                    q[lo:hi][particle_ids], optimize=True)
            elif render_nodes is not None:
                render_nodes = np.asarray(render_nodes, np.int64)
                if (render_nodes.ndim != 1 or faces.max() >= len(render_nodes) or
                        render_nodes.min() < 0 or render_nodes.max() >= hi - lo):
                    raise RuntimeError("cecum render-node binding is invalid")
                local = q[lo:hi][render_nodes]
            else:
                # FEM-boundary fallback meshes use particle-local face indices.
                if faces.max() >= hi - lo:
                    raise RuntimeError("cecum fallback surface index exceeds FEM span")
        if not np.isfinite(local).all():
            raise RuntimeError("cecum live display surface contains non-finite positions")
        return self.env._to_rig(local), faces

    def _cecum_surface_rig(self):
        """Compatibility view of the live cecum display vertices."""
        geometry = self._cecum_surface_geometry_rig()
        return None if geometry is None else geometry[0]

    def _cecum_clearance_samples_rig(self):
        """Sample cecum triangle interiors as well as vertices for path ranking.

        Edge midpoints and face centroids reduce the large false clearances
        produced by a vertex-only nearest-neighbour query. This remains a
        sampled geometric heuristic; Newton's contact solver is authoritative.
        """
        geometry = self._cecum_surface_geometry_rig()
        if geometry is None:
            return None
        vertices, faces = geometry
        if faces is None or not len(faces):
            return vertices
        triangles = vertices[faces]
        edges = np.sort(np.vstack((faces[:, [0, 1]], faces[:, [1, 2]],
                                   faces[:, [2, 0]])), axis=1)
        edges = np.unique(edges, axis=0)
        edge_midpoints = vertices[edges].mean(axis=1)
        face_centres = triangles.mean(axis=1)
        return np.vstack((vertices, edge_midpoints, face_centres))

    def _swept_route_clearance(self, target, tip, R_tip, shaft_axis,
                               port, cecum, tool_offsets):
        """Minimum sampled clearance from swept URDF collision meshes to cecum.

        Candidate orientations follow the port-to-tip line. This geometric
        score is a planning heuristic; Newton contact remains the authority.
        """
        from scipy.spatial import cKDTree

        hover_z = max(float(tip[2]), float(target[2]) + self.STANDOFF_M)
        waypoints = [tip, np.array([target[0], target[1], hover_z]), target]
        tree = cKDTree(np.asarray(cecum, np.float64))
        best = float("inf")
        swept_batches = []
        for a, b in zip(waypoints[:-1], waypoints[1:]):
            length = float(np.linalg.norm(b - a))
            n = max(1, int(np.ceil(length / self.PATH_SAMPLE_M)))
            for alpha in np.linspace(0.0, 1.0, n + 1):
                p = a * (1.0 - alpha) + b * alpha
                axis = p - port
                norm = float(np.linalg.norm(axis))
                if norm < 1e-9:
                    continue
                R = self._rotation_between(shaft_axis, axis / norm) @ R_tip
                swept_batches.append(p + tool_offsets @ R.T)
        # Batch KD-tree lookups rather than making thousands of small Python
        # calls. This keeps the offline route-ranking heuristic affordable on
        # workstation CPUs while Newton remains authoritative for contact.
        for start in range(0, len(swept_batches), 8):
            swept = np.concatenate(swept_batches[start:start + 8], axis=0)
            d = tree.query(swept, k=1, workers=1)[0]
            best = min(best, float(d.min()) - self.TISSUE_PARTICLE_RADIUS_M)
        return best

    def reset(self, obs):
        self.k = 0
        self.closed_for = 0
        self.approach_phase = "standoff"
        self.target_handle_index = None

    def _locked_or_best_handle(self, candidate_ids, scores=None):
        """Keep one anatomical grasp point through the approach trajectory."""
        ids = np.asarray(candidate_ids, np.int64)
        if self.target_handle_index in ids:
            return int(self.target_handle_index)
        if not len(ids) or scores is None or len(scores) != len(ids):
            self.target_handle_index = None
            return None
        self.target_handle_index = int(ids[int(np.argmax(scores))])
        return self.target_handle_index

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
            n = len(handles)
            fractions = (np.arange(n, dtype=np.float64) + 0.5) / n
            cfg = getattr(self.env, "cfg", None)
            lo, hi = getattr(cfg, "grasp_band", (0.30, 0.70))
            in_band = (fractions >= lo) & (fractions <= hi)
            ok = far & free & in_band
            if not ok.any():
                ok = free & in_band
            if not ok.any():
                # Never turn a missing mid-body grasp target into a false
                # procedural success by silently picking an end handle.
                self.target_handle_index = None
                return Action.zeros(chunk, gripper=(100.0, 100.0))
            cand_ids = np.flatnonzero(ok)
            cand = handles[cand_ids]
            # This scripted policy has privileged anatomy access. Among the
            # allowed mid-body grasps, prefer the handle with the largest
            # clearance from cecum instead of the nearest point, which sent the
            # instrument through the adjacent bowel in the first Newton trials.
            cell = getattr(self.env, "cell", None)
            spans = getattr(getattr(cell, "scene", None), "organ_span", {})
            if self.target_handle_index in cand_ids:
                target_id = int(self.target_handle_index)
            elif (bool(getattr(getattr(self.env, "cfg", None),
                               "demo_swept_clearance", False)) and
                  cell is not None and "cecum" in spans and
                  hasattr(self.env, "_to_rig")):
                cecum_rig = self._cecum_clearance_samples_rig()
                arm = getattr(self.env, "arms", {}).get("left")
                offsets = (self._collision_offsets_in_tip_frame(arm)
                           if arm is not None else None)
                if cecum_rig is not None and offsets is not None:
                    port = np.asarray(obs["left"]["port_pos_rig"], np.float64)
                    shaft = np.asarray(obs["left"]["shaft_dir_rig"], np.float64)
                    route_clearance = [self._swept_route_clearance(
                        target, tip, R, shaft, port, cecum_rig, offsets)
                        for target in cand]
                    target_id = self._locked_or_best_handle(cand_ids,
                                                            route_clearance)
                else:
                    q = cell.state_0.particle_q.numpy()
                    lo, hi = spans["cecum"]
                    cecum_rig = self.env._to_rig(q[lo:hi])
                    clearance = np.linalg.norm(
                        cand[:, None, :] - cecum_rig[None, :, :], axis=2).min(axis=1)
                    target_id = self._locked_or_best_handle(cand_ids, clearance)
            elif cell is not None and "cecum" in spans and hasattr(self.env, "_to_rig"):
                q = cell.state_0.particle_q.numpy()
                lo, hi = spans["cecum"]
                cecum_rig = self.env._to_rig(q[lo:hi])
                clearance = np.linalg.norm(
                    cand[:, None, :] - cecum_rig[None, :, :], axis=2).min(axis=1)
                target_id = self._locked_or_best_handle(cand_ids, clearance)
            else:
                nearest = np.linalg.norm(cand - tip, axis=1)
                target_id = self._locked_or_best_handle(cand_ids, -nearest)
            tgt = handles[target_id]
        else:
            tgt = np.asarray(task["appendix_centroid_rig"], np.float64)

        dist = float(np.linalg.norm(tgt - tip))
        if grasped:
            self.approach_phase = "lift"
            self.closed_for += 1
            lift = float(task["appendix_lift_m"])
            rate = self.LIFT_M if lift < self.LIFT_TARGET_M else 0.0
            d_rig = np.array([0.0, 0.0, rate])                 # lift, gently
            grip = 0.0
        else:
            grip = 100.0
            if self.approach_phase == "standoff":
                hover_z = max(float(tip[2]), float(tgt[2]) + self.STANDOFF_M)
                waypoint = np.array([tgt[0], tgt[1], hover_z])
                d_rig = waypoint - tip
                if np.linalg.norm(d_rig) <= self.STANDOFF_ENTER_M:
                    self.approach_phase = "descend"
            elif self.approach_phase == "descend":
                if dist * 1000.0 <= self.APPROACH_MM:
                    self.approach_phase = "close"
                    d_rig = np.zeros(3)
                    grip = 0.0
                    self.closed_for += 1
                else:
                    d_rig = tgt - tip
            else:
                d_rig = np.zeros(3)                             # hold and close
                grip = 0.0
                self.closed_for += 1

        cap = self.LIFT_M if grasped else self.step_m
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
