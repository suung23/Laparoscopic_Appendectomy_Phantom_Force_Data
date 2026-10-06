#!/usr/bin/env python3
"""The cell + both arms + the step loop: the Newton twin of the Isaac plant.

`scene.py` builds the tissue, `arms.py` builds the instruments and owns the
controller; this composes them and runs time.

    .venv-newton/bin/python newton_cell/sim.py --seconds 2 --demo dive

Timing follows the Isaac env exactly: physics at `--physics-hz` (480, the rate
`laparo_control.BEST` was tuned and swept at), one action sub-step every
1/`--control-hz` (30), and `physics_hz/control_hz` = 16 physics ticks per
sub-step.

The arm is a POSITION-DRIVEN FOLLOWER, which is what SURGICAL_CELL.md specifies:

    same fr3.urdf model in both, same IK/controller mapping action -> joint
    targets, joints driven to those targets. Then the same action produces the
    same joint trajectory in both engines by construction.

So the controller's rate-limited joint target is taken as the achieved angle.
The one measurable consequence is that `track_lag` -- the PhysX servo's own
following error, 0.38 mm p95 in the Isaac reference run -- is ~0 here by
construction. That is a property of the plant, not of the controller, and it is
reported rather than hidden.
"""
from __future__ import annotations

import argparse
import math
import time

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverVBD

from arms import ARMS, NewtonArm          # also puts isaac/ on sys.path
from layout import CellLayout
from scene import CellScene, add_args, log

import laparo_control                                          # noqa: E402


class Cell:
    """Tissue + two instruments, steppable."""

    def __init__(self, args):
        self.args = args
        self.L = CellLayout()
        self.scene = CellScene(args)
        self.control_cfg = laparo_control.BEST.replace(manip_hz=args.manip_hz)

        self.arms = {}
        for name in ARMS:
            self.arms[name] = NewtonArm(name, self.L, self.control_cfg,
                                        insert=args.insert,
                                        min_depth=args.min_depth)
            a = self.arms[name]
            log(f"[{name:>5}] start pose: tip err "
                f"{a.start_info['tip_err']*1000:.2f} mm, RCM err "
                f"{a.start_info['rcm_err']*1000:.2f} mm, port {a.port_no}, "
                f"insertion {a.insertion_depth()*1000:.1f} mm")

        self.model = self._build()
        self.solver = SolverVBD(model=self.model, iterations=args.iterations,
                                particle_enable_self_contact=False)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        nb = self.model.body_count
        self._body_q = (self.state_0.body_q.numpy().copy() if nb
                        else np.zeros((0, 7), np.float32))
        self._write_bodies(self.state_0)
        self._write_bodies(self.state_1)

        self.physics_dt = 1.0 / args.physics_hz
        self.ticks_per_substep = int(round(args.physics_hz / args.control_hz))
        self.t = 0.0
        self.q_rest = self.state_0.particle_q.numpy().copy()

    # ------------------------------------------------------------------
    def _build(self):
        """Build the model with the arms' kinematic bodies included."""
        scene = self.scene
        a = self.args
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
        builder.particle_max_velocity = 20.0
        scene._prepare(builder)
        for name in ARMS:
            self.arms[name].register_bodies(builder, scene.tissue["friction"])
        return scene._finalize(builder)

    def _write_bodies(self, state):
        if not len(self._body_q):
            return
        for arm in self.arms.values():
            arm.fill_body_q(self._body_q)
        state.body_q.assign(self._body_q)

    # ------------------------------------------------------------------
    def physics_tick(self):
        s0, s1 = self.state_0, self.state_1
        self._write_bodies(s0)
        s0.clear_forces()
        self.model.collide(s0, self.contacts)
        self.solver.step(s0, s1, self.control, self.contacts, self.physics_dt)
        self.state_0, self.state_1 = s1, s0
        self.t += self.physics_dt

    def control_substep(self, cmd: dict, after_tick=None):
        """One action sub-step = 1/control_hz of simulated time.

        The control law runs at EVERY physics tick (480 Hz), not at control_hz
        -- that is what the Isaac env does (VLA_HANDOFF.md 3.6) and what
        `laparo_control.BEST` was swept at. The sub-step's delta is therefore
        spread linearly across the 16 ticks, so a chunk is a PATH, not a jump
        (3.2), and the tool frame it is expressed in is the one captured at the
        START of the sub-step.

        `cmd[arm]` is (d_xyz_tool, droll, grip_frac): one row of the (2, K, 7)
        action.
        """
        n = self.ticks_per_substep
        dt = self.physics_dt
        frames = {nm: arm.link_frames()["tool_tip"][:3, :3].copy()
                  for nm, arm in self.arms.items()}
        for nm, arm in self.arms.items():
            arm.info = dict(depth_clamped=0, cone_clamped=0, leash_clamped=0)
        for _ in range(n):
            for nm, arm in self.arms.items():
                d_xyz, droll, grip = cmd[nm]
                self._advance_arm(arm, np.asarray(d_xyz, np.float64) / n,
                                  float(droll) / n, grip, dt, frames[nm])
            self.physics_tick()
            if after_tick is not None:
                after_tick()

    def _advance_arm(self, arm, d_xyz_tool, droll, grip_frac, dt, R_tool):
        """Integrate the tool-frame delta onto the tip command and step the
        controller once. Mirrors vla_env.ArmRig.apply()'s command path: the
        insertion floor and the aperture cone clamp in place (never accumulate),
        and the leash provides the anti-windup a human operator provides by
        feeling the haptic wall and stopping (VLA_HANDOFF.md 3.5)."""
        F = arm.link_frames()
        p_tip = F["tool_tip"][:3, 3]
        p_cmd = arm.p_cmd + R_tool @ np.asarray(d_xyz_tool, np.float64)

        # --- hard constraints, clamped in place -------------------------
        info = arm.info
        v = p_cmd - arm.p_rcm
        depth = float(np.dot(v, arm.u_in))
        if depth < arm.min_depth:            # insertion floor
            p_cmd = p_cmd + (arm.min_depth - depth) * arm.u_in
            info["depth_clamped"] += 1
            v = p_cmd - arm.p_rcm
            depth = float(np.dot(v, arm.u_in))
        perp = v - depth * arm.u_in          # aperture cone
        tilt_max = arm.ctl.tilt_max
        if tilt_max > 0 and depth > 1e-6:
            rmax = depth * math.tan(tilt_max)
            rn = float(np.linalg.norm(perp))
            if rn > rmax > 0:
                p_cmd = arm.p_rcm + depth * arm.u_in + perp * (rmax / rn)
                info["cone_clamped"] += 1
        leash = self.args.cmd_leash_m       # anti-windup the operator provides
        d = p_cmd - p_tip
        dn = float(np.linalg.norm(d))
        if dn > leash > 0:
            p_cmd = p_tip + d * (leash / dn)
            info["leash_clamped"] += 1

        # v_cmd is low-passed at 8 Hz, exactly as vla_env.ArmRig.apply does --
        # the raw per-tick difference is far too noisy to feed forward.
        v_raw = (p_cmd - arm.p_cmd) / dt
        b = 1.0 - math.exp(-2.0 * math.pi * 8.0 * dt)
        arm.v_cmd = arm.v_cmd + (v_raw - arm.v_cmd) * b
        arm.p_cmd = p_cmd

        # roll accumulates as an absolute TARGET, so a rate limit is a delay
        # rather than a loss (VLA_HANDOFF.md 3.3). The accumulator counts what
        # the joint DELIVERED, not what was asked, or the residual never clears
        # and the target runs away from the shaft.
        arm.roll_cmd += float(droll)
        residual = arm.roll_cmd - arm.roll_meas

        rj = arm.ctl.ROLL_JOINT
        q7_pre = float(arm.ctl.q[rj])
        tip_meas, axis_meas = arm.shaft_line()
        q_t, qd_ff, p_shaped = arm.ctl.step(
            p_cmd, arm.v_cmd, p_tip, dt,
            r_cmd=arm.r_cmd0, line_meas=(tip_meas, axis_meas),
            q_meas=arm.q, droll=residual)
        # negated back into the ACTION's sign (tip +z), so roll_meas is
        # directly comparable to roll_cmd
        arm.roll_meas -= float(arm.ctl.q[rj] - q7_pre)
        arm.q = np.asarray(q_t, np.float64).copy()
        arm.p_shaped = p_shaped
        self._step_gripper(arm, grip_frac, dt)

    def _step_gripper(self, arm, grip_frac, dt):
        """0..100 -> per-jaw angle, slewed at gripper_rate_dps."""
        target = float(np.clip(grip_frac, 0.0, 100.0))
        rate = self.args.gripper_rate_dps * dt / 100.0 * \
            (100.0 / max(1e-9, 100.0))       # deg/s on the INCLUDED angle
        step = math.radians(self.args.gripper_rate_dps * dt) / 2.0
        want = target / 100.0 * arm.grip_half_max
        have = arm.jaw_q[0]
        move = float(np.clip(want - have, -step, step))
        arm.jaw_q[:] = have + move
        arm.grip_cmd = target
        arm.grip_meas = float(arm.jaw_q[0] / arm.grip_half_max * 100.0)


# ----------------------------------------------------------------------
def demo_dive(cell, k, n):
    """Drive both tips further in, then lift -- enough to load the tissue."""
    out = {}
    for name in ARMS:
        if k < n * 0.5:
            d = np.array([0.0, 0.0, -0.0006])      # tip -z = deeper (sec 3.2)
            g = 100.0
        else:
            d = np.array([0.0, 0.0, +0.0004])
            g = 0.0
        out[name] = (d, 0.0, g)
    return out


def demo_hold(cell, k, n):
    return {name: (np.zeros(3), 0.0, 100.0) for name in ARMS}


DEMOS = {"dive": demo_dive, "hold": demo_hold}


def main(argv=None) -> int:
    ap = add_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--physics-hz", type=float, default=480.0)
    ap.add_argument("--control-hz", type=float, default=30.0)
    ap.add_argument("--manip-hz", type=float, default=60.0)
    ap.add_argument("--insert", type=float, default=0.10)
    ap.add_argument("--min-depth", type=float, default=0.10)
    ap.add_argument("--cmd-leash-m", type=float, default=0.015)
    ap.add_argument("--gripper-rate-dps", type=float, default=250.0)
    ap.add_argument("--demo", choices=sorted(DEMOS), default="dive")
    a = ap.parse_args(argv)

    wp.init()
    log(f"device: {wp.get_device(a.device)}")
    cell = Cell(a)
    log(f"  physics {a.physics_hz:.0f} Hz, control {a.control_hz:.0f} Hz "
        f"({cell.ticks_per_substep} ticks/sub-step)")

    n = int(round(a.seconds * a.control_hz))
    fn = DEMOS[a.demo]
    t0 = time.time()
    for k in range(n):
        cell.control_substep(fn(cell, k, n))
        if k % max(1, n // 8) == 0 or k == n - 1:
            q = cell.state_0.particle_q.numpy()
            d = np.linalg.norm(q - cell.q_rest, axis=1)
            l, r = cell.arms["left"], cell.arms["right"]
            log(f"  t={cell.t:5.2f}s  tissue max {d.max()*1000:6.2f} mm  "
                f"| L depth {l.insertion_depth()*1000:6.1f} rcm "
                f"{l.rcm_error()*1000:5.2f} grip {l.grip_meas:5.1f}  "
                f"| R depth {r.insertion_depth()*1000:6.1f} rcm "
                f"{r.rcm_error()*1000:5.2f}")
    wall = time.time() - t0
    log(f"\n{n} sub-steps ({a.seconds:.2f} s sim) in {wall:.1f} s -> "
        f"rtf {a.seconds/wall:.3f}")
    q = cell.state_0.particle_q.numpy()
    for nm, (i, j) in cell.scene.organ_span.items():
        d = np.linalg.norm(q[i:j] - cell.q_rest[i:j], axis=1) * 1000
        log(f"  {nm:9s} mean {d.mean():6.2f} mm  max {d.max():6.2f} mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
