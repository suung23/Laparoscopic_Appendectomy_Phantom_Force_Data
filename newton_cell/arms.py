#!/usr/bin/env python3
"""One instrument arm, driven exactly as the Isaac backend drives it.

`laparo_ik` (kinematics) and `laparo_control.TipController` (the 182-config
grid-search winner `BEST`) are pure numpy and contain no engine calls at all --
`TipController`'s own docstring says "The caller owns the articulation: this
class never touches Isaac". So this module reuses BOTH verbatim and only
replaces what sits underneath them: instead of writing joint targets into a
PhysX articulation, it writes link transforms into Newton kinematic bodies.

That is the `SURGICAL_CELL.md` contract made concrete:

    Robot-arm movement must be identical across both engines.
    Organ deformation is allowed to differ.

Identical arm motion is not hoped for, it is structural -- the same action goes
through the same IK and the same controller to the same joint angles, and the
links are then placed at those angles rather than servoed toward them. What the
two engines can still disagree about is the tissue, which is allowed.

The one thing this loses relative to PhysX is joint-reaction force sensing: a
kinematically placed link reports no reaction. `forces.py` supplies a
post-step, model-based estimate from soft-contact state; it is not a sensor
readback, exact within-iteration solver reaction, or calibrated tissue force.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

from cell_spec import ROOT, constants, tools
from layout import CellLayout
from urdf_meshes import Geom, link_geometry, merged

sys.path.insert(0, str(ROOT / "isaac"))
import laparo_control                                        # noqa: E402
import trocar                                                # noqa: E402
from laparo_ik import Chain, port_inward_normal              # noqa: E402
from start_pose import SEED, solve_start_q                   # noqa: E402

ARMS = ("left", "right")

# Which arm uses which URDF and which endtip joints. Mirrors vla_env.ARM_CFG;
# `port_sign` is not needed here because layout.port_of_arm already resolves
# the port from world X (the specimen spin swaps the two ports' sides).
ARM_CFG = {
    "left":  dict(urdf="out/fr3_left_grasper.urdf",
                  links=("jaw_left", "jaw_right"),
                  joints=("gripper_jaw_left", "gripper_jaw_right"),
                  open_deg=37.0, instrumented=True),
    "right": dict(urdf="out/fr3_right_cutter.urdf",
                  links=("blade_left", "blade_right"),
                  joints=("cutter_blade_left", "cutter_blade_right"),
                  open_deg=25.0, instrumented=False),
}

# The chain's link names, in the order laparo_ik.Chain walks its steps. Step k
# lands in LINK_ORDER[k]; step 8 is the fixed fr3_link8_to_tool and step 9 the
# fixed tool_flange_to_tip, so the last two are frames, not bodies.
LINK_ORDER = ("fr3_link1", "fr3_link2", "fr3_link3", "fr3_link4", "fr3_link5",
              "fr3_link6", "fr3_link7", "fr3_link8", "tool_flange", "tool_tip")


def rot_axis(axis, ang) -> np.ndarray:
    a = np.asarray(axis, np.float64)
    n = np.linalg.norm(a)
    if n < 1e-15 or abs(ang) < 1e-18:
        return np.eye(3)
    k = a / n
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * (K @ K)


def T_of(R, p) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = p
    return M


class NewtonArm:
    """Kinematics + controller for one arm, plus its Newton kinematic bodies."""

    def __init__(self, name: str, layout: CellLayout, control_cfg,
                 insert: float = 0.10, min_depth: float = 0.10,
                 seed_q=None):
        self.name = name
        self.cfg = ARM_CFG[name]
        self.L = layout
        self.C = constants()
        self.tool = tools()[name]
        self.control_cfg = control_cfg

        self.chain = Chain(ROOT / self.cfg["urdf"])
        self.T_base = layout.arm_base(name)

        # port: resolved from world X by the layout, then 1-based for the
        # normal fit (which indexes TROCAR_PORTS_MM directly)
        self.port_idx = layout.port_of_arm(name)
        self.port_no = self.port_idx + 1
        self.p_rcm = layout.ports_world[self.port_idx].copy()
        # the trocarPorts group carries the phantom's spin, so the STL-frame
        # normal is rotated by it -- vla_env does `env.R_ports @ normal`
        from layout import Rz
        self.R_ports = Rz(layout.D["--rotate-phantom"])
        self.u_in = self.R_ports @ port_inward_normal(self.port_no)

        self.p_start = self.p_rcm + insert * self.u_in
        cfgc = control_cfg
        self.ik_kw = dict(k_manip=cfgc.k_manip, k_limit=cfgc.k_limit,
                          limit_margin=math.radians(cfgc.limit_margin_deg))
        q, sinfo = solve_start_q(
            self.chain, self.T_base, self.p_start, self.p_rcm,
            seed=SEED if seed_q is None else seed_q, roll_deg=0.0,
            elbow_lift=0.6, elbow_point=3, ik_kw=self.ik_kw)
        self.q0 = q.copy()
        self.q = q.copy()
        self.r_cmd0 = sinfo["r_cmd"]
        self.start_info = sinfo

        self.min_depth = min_depth
        self.ctl = laparo_control.TipController(
            cfgc, self.chain, self.T_base, self.p_rcm, self.q0,
            min_depth=min_depth, port_normal=self.u_in)

        # gripper: 0..100 maps onto 0..open_deg per jaw (included = 2x)
        self.grip_half_max = math.radians(self.cfg["open_deg"])
        self.jaw_q = np.zeros(2)              # per-jaw angle, rad
        self.grip_cmd = 100.0
        self.grip_meas = 0.0

        # tip command state -- the integral of the policy's deltas
        p_tip, _ = self.chain.fk(self.q, self.T_base)
        self.p_cmd = p_tip.copy()
        self.v_cmd = np.zeros(3)
        self.p_shaped = None
        self.info = dict(depth_clamped=0, cone_clamped=0, leash_clamped=0)
        self.roll_cmd = 0.0
        self.roll_meas = 0.0

        self.bodies = {}                      # link name -> newton body index
        self._geom_cache = {}

    # ---- kinematics ------------------------------------------------------
    def link_frames(self, q=None, jaw_q=None) -> dict:
        """World transform of every link, from the joint angles.

        Walks `laparo_ik.Chain.steps`, which is the URDF chain
        fr3_joint1..8 -> fr3_link8_to_tool -> tool_flange_to_tip, so the frames
        come out of exactly the description the controller was solved against.
        The two endtip links hang off tool_flange about the clevis pin.
        """
        q = self.q if q is None else q
        jaw_q = self.jaw_q if jaw_q is None else jaw_q
        out = {"fr3_link0": self.T_base.copy()}
        T = self.T_base.copy()
        qi = 0
        for k, (Tf, ax) in enumerate(self.chain.steps):
            T = T @ Tf
            if ax is not None:
                R = np.eye(4)
                R[:3, :3] = rot_axis(ax, q[qi])
                T = T @ R
                qi += 1
            out[LINK_ORDER[k]] = T.copy()

        flange = out["tool_flange"]
        pivot = np.asarray(self.tool.pivot_xyz, np.float64)
        pin = np.asarray(self.tool.pin_axis, np.float64)
        for i, (side, sign) in enumerate((("left", 1.0), ("right", -1.0))):
            link = f"{self.tool.link_prefix}_{side}"
            J = T_of(rot_axis(sign * pin, jaw_q[i]), pivot)
            out[link] = flange @ J
        return out

    def tip_pose(self, q=None) -> np.ndarray:
        return self.link_frames(q)["tool_tip"]

    def shaft_line(self, q=None):
        """(tip, insertion-direction) of the MEASURED shaft.

        `TipController.step` takes this as `line_meas` and closes the RCM loop
        with it; without it the controller only ever sees its own virtual arm,
        whose shaft passes through the port by construction.
        """
        F = self.link_frames(q)
        tip = F["tool_tip"][:3, 3]
        axis = -F["tool_tip"][:3, 2]          # tip -z = insertion (handoff sec 2)
        return tip, axis / np.linalg.norm(axis)

    def rcm_error(self, q=None) -> float:
        """Perpendicular distance from the port centre to the shaft line."""
        tip, axis = self.shaft_line(q)
        w = self.p_rcm - tip
        return float(np.linalg.norm(w - np.dot(w, axis) * axis))

    def insertion_depth(self, q=None) -> float:
        tip, axis = self.shaft_line(q)
        return float(np.dot(tip - self.p_rcm, axis))

    # ---- Newton bodies ---------------------------------------------------
    def collision_geoms(self) -> dict:
        """Which meshes actually touch tissue, per link.

        `LAPARO_COLLIDE_MESHES` also carries `main_housing` and `housing_cover`;
        those are the unit's casing, which lives outside the patient and can
        never reach an organ, so they are dropped. What is left is the 5 mm
        shaft and the distal clevis cluster, plus the two endtip links.
        """
        if self._geom_cache:
            return self._geom_cache
        C = self.C
        distal = {m for m in C["LAPARO_COLLIDE_MESHES"]
                  if not m.startswith(("main_housing", "housing_cover"))}
        col = link_geometry(ROOT / self.cfg["urdf"], "collision")
        keep = {}
        for link, items in col.items():
            if link == "tool_flange":
                sel = [g for g in items if Path(g.source).name in distal]
            elif link in self.cfg["links"]:
                sel = items
            else:
                continue                      # FR3 links stay out of the abdomen
            if sel:
                keep[link] = sel
        self._geom_cache = keep
        return keep

    def register_bodies(self, builder, tissue_mu: float, *,
                        contact_ke=1.0e5, contact_kd=1.0,
                        contact_kf=1.0e3,
                        tool_tissue_collision=True) -> None:
        """Add the instrument's colliding links as KINEMATIC Newton bodies.

        Kinematic, not dynamic: their transforms are written every substep from
        the controller's joint angles, so contact never moves them. That is what
        makes the arm trajectory identical to the Isaac run for a given action.
        """
        import newton
        frames = self.link_frames()
        for link, items in self.collision_geoms().items():
            v, f = merged(items)
            b = builder.add_body(xform=_wp_xform(frames[link]),
                                 mass=0.0, is_kinematic=True,
                                 label=f"{self.name}_{link}")
            # vertices are in the LINK frame; the body carries the placement
            builder.add_shape_mesh(
                b,
                mesh=newton.Mesh(np.ascontiguousarray(v, np.float32),
                                 np.ascontiguousarray(f, np.int32).flatten(),
                                 is_solid=True),
                cfg=builder.ShapeConfig(ke=float(contact_ke),
                                        kd=float(contact_kd),
                                        kf=float(contact_kf),
                                        mu=tissue_mu, is_solid=True,
                                        has_shape_collision=False,
                                        has_particle_collision=bool(
                                            tool_tissue_collision)),
                label=f"{self.name}_{link}")
            self.bodies[link] = b

    def fill_body_q(self, arr: np.ndarray) -> None:
        """Write this arm's kinematic body transforms into a host-side array.

        Host-side and not `state.body_q.assign()` per arm on purpose: this runs
        at 480 Hz, and a read-modify-write of the device array per arm per tick
        cost more than the solver did (rtf 0.096 -> 0.018 before the caller was
        changed to fill one array and upload it once).
        """
        if not self.bodies:
            return
        frames = self.link_frames()
        for link, b in self.bodies.items():
            T = frames[link]
            arr[b, 0:3] = T[:3, 3]
            arr[b, 3:7] = _quat_xyzw(T[:3, :3])


def _quat_xyzw(R) -> tuple:
    """Rotation matrix -> (x, y, z, w), the ordering warp transforms use."""
    t = float(np.trace(R))
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(max(1e-12, 1.0 + R[i, i] - R[j, j] - R[k, k])) * 2.0
        q = [0.0, 0.0, 0.0]
        q[i] = 0.25 * s
        q[j] = (R[j, i] + R[i, j]) / s
        q[k] = (R[k, i] + R[i, k]) / s
        w = (R[k, j] - R[j, k]) / s
        x, y, z = q
    n = math.sqrt(x * x + y * y + z * z + w * w)
    return (x / n, y / n, z / n, w / n)


def _wp_xform(T):
    import warp as wp
    return wp.transform(wp.vec3(*[float(v) for v in T[:3, 3]]),
                        wp.quat(*[float(v) for v in _quat_xyzw(T[:3, :3])]))


if __name__ == "__main__":
    L = CellLayout()
    cfg = laparo_control.BEST.replace(manip_hz=60.0)
    for name in ARMS:
        a = NewtonArm(name, L, cfg)
        F = a.link_frames()
        tip = F["tool_tip"]
        print(f"=== {name}  port {a.port_no}")
        print(f"  start pose: tip err {a.start_info['tip_err']*1000:6.2f} mm, "
              f"RCM err {a.start_info['rcm_err']*1000:5.2f} mm")
        print(f"  q0 (deg)  = {np.degrees(a.q0).round(1)}")
        print(f"  tip world = {tip[:3,3].round(4)}   rig = "
              f"{L.to_rig(tip[:3,3]).round(4)}")
        print(f"  insertion = {a.insertion_depth()*1000:6.2f} mm, "
              f"rcm err {a.rcm_error()*1000:.3f} mm")
        t, ax = a.shaft_line()
        print(f"  shaft dir = {ax.round(4)}  (u_in {a.u_in.round(4)}, "
              f"dot {float(np.dot(ax, a.u_in)):+.4f})")
        cg = a.collision_geoms()
        print(f"  colliders : {[(k, len(v)) for k, v in cg.items()]}")
