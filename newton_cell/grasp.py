#!/usr/bin/env python3
"""Grasp assist: latch a grip cluster to the jaws, in place.

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

Latched nodes are carried rigidly by the jaw frame: their offsets are captured
at the moment of latching, and each step they are written to
`handle_transform @ offset`. Releasing hands them back to the solver.
"""
from __future__ import annotations

import numpy as np


class GraspAssist:
    """Owns the latch state of the appendix's grip clusters."""

    def __init__(self, scene, arms, grasp_radius=0.012, close_frac=35.0):
        self.scene = scene
        self.arms = arms
        self.grasp_radius = float(grasp_radius)
        self.close_frac = float(close_frac)
        self.handles = getattr(scene, "handles", []) or []
        self.latched = {}          # arm name -> dict(handle=i, idx=..., off=...)
        self.events = {k: 0 for k in arms}

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
            # Offsets captured NOW, against where the tissue actually is -- this
            # is the whole point: enabling the weld at its authored rest offset
            # teleports the organ to the handle and kicks the shaft out of the
            # port.
            off = (q[h["idx"]] - T[:3, 3]) @ T[:3, :3]
            self.latched[name] = dict(handle=j, idx=h["idx"].copy(), off=off)
            self.events[name] += 1

    def apply(self, state) -> None:
        """Carry every latched cluster rigidly with its jaw frame."""
        if not self.latched:
            return
        q = state.particle_q.numpy()
        qd = state.particle_qd.numpy()
        for name, lat in self.latched.items():
            T = self.arms[name].link_frames()["tool_tip"]
            q[lat["idx"]] = lat["off"] @ T[:3, :3].T + T[:3, 3]
            qd[lat["idx"]] = 0.0
        state.particle_q.assign(q)
        state.particle_qd.assign(qd)

    # ------------------------------------------------------------------
    def grasped(self, name) -> bool:
        return name in self.latched

    def info(self) -> dict:
        return dict(grasped={k: self.grasped(k) for k in self.arms},
                    events=dict(self.events),
                    handles=len(self.handles))
