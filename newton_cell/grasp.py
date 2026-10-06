#!/usr/bin/env python3
"""Force-limited compliant grasp assistance for the deformable appendix.

kPa-soft FEM tissue slips out of a friction pinch between two rigid jaws, so
the Isaac cell makes the grasp a CONSTRAINT instead: the appendix carries a row
of disjoint ~1 cm grip clusters, each welded to a kinematic handle, and closing
the jaws on tissue re-enables the nearest one (build_surgical_cell.author_pins,
`_author_grip_cluster`; VLA_HANDOFF.md section 7.2).

The Isaac notes are emphatic about one trap, and it is reproduced here:

    "attachment welds vertices at an offset from the handle origin, so the
    moment it is enabled the tissue TELEPORTS to where the handle is. The
    handle is authored at rest, and by the time the policy has approached and
    closed, the organ has been pushed somewhere else -- 22 mm in the scripted
    baseline. Enabled as-authored, that 22 mm snaps in one tick and the FEM
    kicks the instrument 13 mm out of a 6 mm port."

So the handle is moved to the tissue's CURRENT position before it is enabled,
and the latch then has nothing to snap.

At latch, a grip cluster's offsets from the jaw frame are captured. A damped
spring then applies a bounded external force to those FEM nodes. It does not
overwrite particle positions or velocities, so the tissue solver retains
control of deformation. The force cap and stiffness remain uncalibrated
engineering parameters and must be swept against physical measurements.
"""
from __future__ import annotations

import numpy as np


def compliant_grasp_forces(positions, velocities, targets, target_velocities,
                           stiffness_n_m, damping_n_s_m, force_limit_n):
    """Return per-node forces for a translational spring grasp.

    The total of nodal force magnitudes is capped, so the resultant load cannot
    exceed ``force_limit_n`` even when local spring directions differ.
    """
    q = np.asarray(positions, np.float64)
    qd = np.asarray(velocities, np.float64)
    target = np.asarray(targets, np.float64)
    target_v = np.asarray(target_velocities, np.float64)
    if q.ndim != 2 or q.shape[1] != 3 or len(q) == 0:
        raise ValueError("grasp positions must be a non-empty (N, 3) array")
    if qd.shape != q.shape or target.shape != q.shape or target_v.shape != q.shape:
        raise ValueError("grasp position, velocity, and target arrays must match")
    params = (stiffness_n_m, damping_n_s_m, force_limit_n)
    if not np.isfinite(q).all() or not np.isfinite(qd).all() or \
            not np.isfinite(target).all() or not np.isfinite(target_v).all() or \
            not np.isfinite(params).all() or stiffness_n_m < 0 or \
            damping_n_s_m < 0 or force_limit_n < 0:
        raise ValueError("grasp state and force parameters must be finite and non-negative")
    forces = (stiffness_n_m * (target - q)
              + damping_n_s_m * (target_v - qd)) / len(q)
    load = float(np.linalg.norm(forces, axis=1).sum())
    if load > force_limit_n and load > 0.0:
        forces *= force_limit_n / load
        load = float(force_limit_n)
    return forces, load


class GraspAssist:
    """Owns the latch state of the appendix's grip clusters."""

    def __init__(self, scene, arms, model, dt, grasp_radius=0.012,
                 close_frac=35.0, force_limit_n=0.03,
                 stiffness_n_m=30.0, damping_ratio=0.8):
        self.scene = scene
        self.arms = arms
        self.model = model
        self.dt = float(dt)
        self.grasp_radius = float(grasp_radius)
        self.close_frac = float(close_frac)
        self.force_limit_n = float(force_limit_n)
        self.stiffness_n_m = float(stiffness_n_m)
        self.damping_ratio = float(damping_ratio)
        if self.dt <= 0.0 or not np.isfinite(self.dt):
            raise ValueError("grasp physics dt must be positive and finite")
        if (not np.isfinite([self.force_limit_n, self.stiffness_n_m,
                             self.damping_ratio]).all() or
                min(self.force_limit_n, self.stiffness_n_m,
                    self.damping_ratio) < 0.0):
            raise ValueError("grasp force parameters must be finite and non-negative")
        self.handles = getattr(scene, "handles", []) or []
        self.latched = {}          # arm name -> handle, FEM nodes, jaw offsets
        self.events = {k: 0 for k in arms}
        self.last_force_n = {k: 0.0 for k in arms}
        self.last_deflection_m = {k: 0.0 for k in arms}

    # ------------------------------------------------------------------
    def handle_positions(self, q) -> np.ndarray:
        """Current centroid of every grip cluster, from live node positions."""
        if not self.handles:
            return np.zeros((0, 3))
        return np.array([q[h["idx"]].mean(axis=0) for h in self.handles])

    def free_mask(self) -> np.ndarray:
        held = {v["handle"] for v in self.latched.values()}
        return np.array([i not in held for i in range(len(self.handles))], bool)

    # ------------------------------------------------------------------
    def update(self, state, q=None) -> None:
        """Latch or release, once per control sub-step."""
        if not self.handles:
            return
        q = state.particle_q.numpy() if q is None else q
        cents = self.handle_positions(q)
        for name, arm in self.arms.items():
            closing = arm.grip_cmd <= self.close_frac
            cur = self.latched.get(name)
            if cur is not None and not closing:
                del self.latched[name]                 # opened: let go
                self.last_force_n[name] = 0.0
                self.last_deflection_m[name] = 0.0
                continue
            if cur is not None or not closing:
                continue
            # closing and not holding: latch the nearest FREE cluster whose
            # centroid is inside the jaw mouth
            tip = arm.link_frames()["tool_tip"][:3, 3]
            d = np.linalg.norm(cents - tip, axis=1)
            free = self.free_mask()
            d = np.where(free, d, np.inf)
            j = int(np.argmin(d))
            if not np.isfinite(d[j]) or d[j] > self.grasp_radius:
                continue
            h = self.handles[j]
            T = arm.link_frames()["tool_tip"]
            # Offsets captured NOW preserve the current tissue pose. The
            # compliant force ramps the cluster toward the jaw over subsequent
            # solver ticks rather than teleporting it to an authored rest pose.
            off = (q[h["idx"]] - T[:3, 3]) @ T[:3, :3]
            self.latched[name] = dict(handle=j, idx=h["idx"].copy(), off=off,
                                      last_target=q[h["idx"]].copy())
            self.events[name] += 1

    def apply_forces(self, state) -> None:
        """Accumulate bounded grasp forces in Newton's particle force array."""
        if not self.latched:
            return
        if state.particle_f is None:
            raise RuntimeError("Newton state does not expose particle_f for compliant grasp")
        q = state.particle_q.numpy()
        qd = state.particle_qd.numpy()
        particle_f = state.particle_f.numpy()
        inv_mass = self.model.particle_inv_mass.numpy()
        masses = np.divide(1.0, inv_mass, out=np.zeros_like(inv_mass),
                           where=inv_mass > 0.0)
        for name, lat in self.latched.items():
            T = self.arms[name].link_frames()["tool_tip"]
            idx = lat["idx"]
            target = lat["off"] @ T[:3, :3].T + T[:3, 3]
            target_v = (target - lat["last_target"]) / self.dt
            lat["last_target"] = target.copy()
            effective_mass = float(masses[idx].sum())
            damping = (2.0 * self.damping_ratio
                       * np.sqrt(self.stiffness_n_m * max(effective_mass, 1e-9)))
            force, load = compliant_grasp_forces(
                q[idx], qd[idx], target, target_v,
                self.stiffness_n_m, damping, self.force_limit_n)
            particle_f[idx] += force
            self.last_force_n[name] = load
            self.last_deflection_m[name] = float(
                np.linalg.norm(q[idx].mean(axis=0) - target.mean(axis=0)))
        state.particle_f.assign(particle_f)

    def reset(self) -> None:
        self.latched.clear()
        self.events = {k: 0 for k in self.arms}
        self.last_force_n = {k: 0.0 for k in self.arms}
        self.last_deflection_m = {k: 0.0 for k in self.arms}

    # ------------------------------------------------------------------
    def grasped(self, name) -> bool:
        return name in self.latched

    def info(self) -> dict:
        return dict(grasped={k: self.grasped(k) for k in self.arms},
                    events=dict(self.events),
                    force_n=dict(self.last_force_n),
                    deflection_mm={k: 1000.0 * v
                                   for k, v in self.last_deflection_m.items()},
                    handles=len(self.handles))
