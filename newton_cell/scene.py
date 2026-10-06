#!/usr/bin/env python3
"""The surgical cell as a Newton model.

Builds the same cell `isaac/build_surgical_cell.py` writes to
`out/surgical_cell.usd`, but as a `newton.Model` -- so it runs on a GPU with no
RT cores. Geometry and material come from `cell_spec` / `layout`, which read
`build_surgical_cell.py` itself, so the two backends place and parametrise the
cell from one set of numbers.

    .venv-newton/bin/python newton_cell/scene.py --steps 60

What is a soft body and what is a collider follows the Isaac cell:

  organs (6)      FEM soft bodies, per-organ E (see cell_spec.tissue)
  appendix shelf  static collider -- the only thing under the appendix
  organ support   static plane, a safety net well below the organs
  field, phantom,
  endoscope       VISUAL ONLY (see below)
  instruments     KINEMATIC colliders driven from laparo_ik (see arms.py)

The instruments being kinematic rather than dynamic is the contract
`SURGICAL_CELL.md` sets out: arm motion has to be identical across the two
engines, which is achieved by driving joints to the same IK targets in both,
not by hoping two different solvers agree. It also means the tissue absorbs the
difference -- which is explicitly allowed -- and that contact force is read
straight off the soft-body contacts rather than out of a joint reaction sensor.

That in turn is why the field, phantom and endoscope carry no collider: with a
kinematic arm nothing responds to their contact forces, the trocar constraint is
enforced analytically by `trocar.py`, and the field in particular MUST NOT touch
the organs -- they are captured embedded in it, so its collider ejects them
(build_surgical_cell.py filters this pair for the same reason; reproduced here,
the stomach walked 24 mm in one second before the filter went in).

Organ pinning: the Isaac cell holds each organ's lower band of nodes at its
captured position (the mesentery) and pins the appendix at the end farthest
from the left robot, leaving the body free to grasp and lift. Here a pinned node
has its `ParticleFlags.ACTIVE` bit cleared and zero mass, set on the BUILDER
before `finalize()`. Zeroing `particle_inv_mass` on the finalized model does not
pin anything -- measured, such "pinned" nodes still drifted 38.6 mm in 1.5 s.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverVBD

from cell_spec import ORGANS, ROOT, cli_defaults, constants, lame, tissue
from layout import CellLayout


def log(msg: str) -> None:
    print(msg, flush=True)


def load_tet(path: Path):
    d = np.load(path)
    return d["vertices"].astype(np.float64), d["indices"].astype(np.int32)


def median_edge(verts: np.ndarray, tets: np.ndarray) -> float:
    """Median tet edge length -- sets the particle radius scale."""
    t = tets.reshape(-1, 4)
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    d = np.concatenate([np.linalg.norm(verts[t[:, i]] - verts[t[:, j]], axis=1)
                        for i, j in pairs])
    return float(np.median(d))


class CellScene:
    """The Newton model plus the bookkeeping the environment needs on top."""

    def __init__(self, args):
        self.args = args
        self.L = CellLayout()
        self.C = constants()
        self.D = cli_defaults()
        self.tissue = tissue()
        self.contact_mu = effective_contact_mu(
            getattr(args, "contact_mu", None), self.tissue["friction"])
        self.contact_material = np.asarray([
            getattr(args, "contact_ke", 1.0e5),
            getattr(args, "contact_kd", 1.0),
            getattr(args, "contact_kf", 1.0e3),
        ], dtype=np.float64)
        if (not np.isfinite(self.contact_material).all() or
                self.contact_material[0] <= 0.0 or
                (self.contact_material[1:] < 0.0).any()):
            raise ValueError("contact requires ke>0 and kd/kf>=0, all finite")
        self.organ_span = {}      # name -> (first_particle, end_particle)
        self.pinned = {}          # name -> index array of pinned particles
        self.static_shapes = {}   # label -> shape index
        self.visual_meshes = {}   # label -> (verts, faces), for the renderer
        self.model = None

    # ------------------------------------------------------------------
    def _add_static_mesh(self, builder, verts, faces, label, *,
                         ke, kd, kf, mu, is_solid, collide=True):
        """Register a static mesh: always for rendering, optionally as a collider.

        `collide=False` means the geometry is visual only. That is not a
        shortcut -- see `_add_field` for why the field in particular must not
        collide with the organs, and note that with kinematically driven
        instruments (the SURGICAL_CELL.md contract) a static collider has no
        one left to push: the arms do not respond to contact force, and the
        trocar constraint is enforced analytically by `trocar.py`, not by
        touching the dome.
        """
        self.visual_meshes[label] = (np.asarray(verts, np.float64),
                                     np.asarray(faces, np.int32))
        if not collide:
            return None
        m = newton.Mesh(np.ascontiguousarray(verts, dtype=np.float32),
                        np.ascontiguousarray(faces, dtype=np.int32).flatten(),
                        is_solid=is_solid)
        idx = builder.add_shape_mesh(
            body=-1, mesh=m,
            cfg=newton.ModelBuilder.ShapeConfig(ke=ke, kd=kd, kf=kf, mu=mu),
            label=label)
        self.static_shapes[label] = idx
        return idx

    def _add_phantom(self, builder):
        v = self.L.phantom_points_world()
        import trimesh
        ph = trimesh.load(ROOT / "phantom.stl", process=False)
        f = np.asarray(ph.faces, np.int32)
        self._add_static_mesh(builder, v, f, "phantom",
                              ke=1.0e5, kd=1.0, kf=1.0e3, mu=0.4, is_solid=False,
                              collide=self.args.phantom_collision)
        log(f"  phantom  {len(f):6d} faces, apex z {v[:,2].max()*1000:6.1f} mm")

    def _add_field(self, builder):
        p = ROOT / "mesh" / "field.ply"
        if not p.exists():
            log("  field    MISSING (mesh/field.ply) -- organs will have no bed")
            return
        import trimesh
        fl = trimesh.load(p, process=False)
        # the field is captured in the same frame as the organs and takes the
        # same bake (build_surgical_cell bakes both under transform_specimen)
        v = self.L.organs_to_world(np.asarray(fl.vertices, np.float64) * 0.001)
        f = np.asarray(fl.faces, np.int32)
        # Visual only by default, matching the Isaac cell's --organ-field-collision
        # (off by default): "the organs are captured EMBEDDED against the field,
        # so its collider overlaps them at spawn and ejects them -- measured up
        # to 100 mm. The anchors hold the organs instead." Reproduced here before
        # this flag existed: the stomach walked 24 mm in one second and particle
        # speeds sat at 1.3 m/s instead of settling.
        collide = self.args.organ_field_collision
        self._add_static_mesh(builder, v, f, "field",
                              ke=1.0e5, kd=1.0, kf=1.0e3, mu=0.5, is_solid=False,
                              collide=collide)
        log(f"  field    {len(f):6d} faces, extent "
            f"{((v.max(0)-v.min(0))*1000).round(1)} mm"
            f"{'' if collide else '  (visual only: organ<->field filtered)'}")

    def _add_endoscope(self, builder):
        p = ROOT / "mesh" / "endoscope.ply"
        if not p.exists():
            log("  endoscope MISSING (mesh/endoscope.ply)")
            return
        import trimesh
        en = trimesh.load(p, process=False)
        v = np.asarray(en.vertices, np.float64) * 0.001   # authored world mm, z up
        M = self.L.T_world_endoscope
        v = v @ M[:3, :3].T + M[:3, 3]
        f = np.asarray(en.faces, np.int32)
        self._add_static_mesh(builder, v, f, "endoscope",
                              ke=1.0e5, kd=1.0, kf=1.0e3, mu=0.3, is_solid=True,
                              collide=False)
        self.endoscope_points = v
        log(f"  endoscope {len(f):5d} faces, tip z {v[:,2].min()*1000:6.1f} mm")

    # ------------------------------------------------------------------
    def _pin_indices(self, name: str, world: np.ndarray) -> np.ndarray:
        """Which nodes of this organ are held at their captured position.

        Mirrors build_surgical_cell.author_pins: every organ gets its lower band
        (`--pin-band` of its height, from the bottom) held -- that is the
        mesentery -- and `PIN_BAND_CAP` widens it for organs whose free top
        oscillates. The appendix is handled separately by `_appendix_setup`.
        """
        if self.args.no_organ_pins or name == "appendix":
            return np.empty(0, np.int32)
        z = world[:, 2]
        band = self.C["PIN_BAND_CAP"].get(name, self.D["--pin-band"])
        cut = z.min() + band * (z.max() - z.min())
        return np.nonzero(z < cut)[0].astype(np.int32)

    def _appendix_setup(self, world: np.ndarray) -> np.ndarray:
        """Far-end world pin + the disjoint grip clusters, per author_pins.

        The appendectomy setup, verbatim in structure from the Isaac build:

        - the end FARTHEST from the left robot base is pinned to the world.
          VLA_HANDOFF.md section 4.4 exposes this as `appendix_pin_rig` and
          warns that grasping a handle next to it means pulling on a rigid
          anchor.
        - the rest of the body is partitioned into DISJOINT ~1 cm segments
          along its long axis, each welded to a kinematic handle at its
          centroid. Disjoint so no node is welded to two handles, which is what
          lets the near-end anchor and a mid-body grab both be active.
        - cluster 0 (the near end) stays anchored at rest -- "at rest the
          handle sits at the tip's captured position, so it doubles as the
          tip's anchor". The others are inactive until the jaws latch one.

        Returns the indices pinned at rest (far end + cluster 0).
        """
        if self.args.no_organ_pins:
            self.handles = []
            return np.empty(0, np.int32)
        base = self.L.arm_base("left")[:3, 3]
        d_base = np.linalg.norm(world - base, axis=1)
        far_pt, near_pt = world[d_base.argmax()], world[d_base.argmin()]

        df = np.linalg.norm(world - far_pt, axis=1)
        sel_far = np.nonzero(df < max(self.D["--pin-radius"],
                                      float(df.min()) + 0.004))[0]

        far_set = set(int(i) for i in sel_far)
        body = np.array([i for i in range(len(world)) if i not in far_set])
        axis = far_pt - near_pt
        Ln = float(np.linalg.norm(axis))
        axis = axis / Ln if Ln > 1e-9 else np.array([1.0, 0.0, 0.0])
        t = (world[body] - near_pt) @ axis
        span = float(t.max() - t.min()) if len(t) else 0.0
        K = int(max(3, min(7, round(span / 0.010))))       # ~1 cm segments
        edges = np.linspace(t.min() - 1e-6, t.max() + 1e-6, K + 1)
        bins = np.digitize(t, edges) - 1

        self.handles = []
        for b in range(K):
            sel = body[bins == b]
            if len(sel) < 3:
                continue
            c = world[sel].mean(axis=0)
            self.handles.append(dict(
                idx=sel.astype(np.int32),
                rest_centroid=c.copy(),
                # offsets so a latched handle can carry its nodes rigidly,
                # which is what the USD vertex attachment does
                offsets=(world[sel] - c).copy(),
                anchor=(len(self.handles) == 0),
            ))
        self.appendix_pin_world = far_pt.copy()
        log(f"    appendix far-end pin {len(sel_far)} nodes at "
            f"{(far_pt*1000).round(1)} mm; {len(self.handles)} grip cluster(s) "
            f"over {span*1000:.0f} mm "
            f"({', '.join(str(len(h['idx'])) for h in self.handles)} nodes)")
        anchored = [h["idx"] for h in self.handles if h["anchor"]]
        return np.concatenate([sel_far] + anchored).astype(np.int32)

    def _organ_support_plane(self, builder) -> None:
        """The lifted ground plane the organs rest on.

        `--organ-support plane` (the default) keeps the deformable scene's own
        infinite ground plane and lets `transform_specimen` lift it with the
        specimen, so it ends up ~52 mm up, right under the organs, holding them
        against gravity. Its visual grid is hidden; only the collider matters.

        Without it every organ's FREE part -- the fraction above `--pin-band` --
        has nothing beneath it and sags: measured here at 26 mm for the stomach
        and 17 mm for the cecum over one second, still going.
        """
        z = float(self.L.organ_offset[2])
        r = 1.0
        v = np.array([[-r, -r, z], [r, -r, z], [r, r, z], [-r, r, z]])
        f = np.array([[0, 1, 2], [0, 2, 3]], np.int32)
        self._add_static_mesh(builder, v, f, "organ_support",
                              ke=2.0e5, kd=1.0, kf=1.0e3, mu=0.6, is_solid=False)
        log(f"  support  organ plane at z {z*1000:.1f} mm "
            f"(the deformable scene's ground plane, lifted with the specimen)")

    def _appendix_shelf(self, builder) -> None:
        """An invisible shelf conforming to the appendix's underside.

        Verbatim in intent from build_surgical_cell.add_appendix_support: the
        field was reconstructed from a D405 looking straight DOWN, so there is
        no geometry UNDER the appendix -- through the surgical opening it hangs
        over a hole and kPa-soft tissue drapes straight down. Without this the
        body simply falls (measured here before it was added: 148 mm in one
        second of sim, still accelerating).

        A grid over the footprint, each vertex 2.5 mm below the local minimum
        of the render verts above it, so the body sags half a millimetre and
        then rests -- and still lifts freely off it.
        """
        p = ROOT / "mesh" / "organ_appendix.ply"
        if not p.exists():
            log("  ! mesh/organ_appendix.ply missing -- no appendix shelf")
            return
        import trimesh
        v = self.L.organs_to_world(
            np.asarray(trimesh.load(p, process=False).vertices, np.float64) * 0.001)
        lo, hi = v.min(axis=0), v.max(axis=0)
        pad, cell, drop = 0.006, 0.004, 0.0025
        xs = np.arange(lo[0] - pad, hi[0] + pad + cell, cell)
        ys = np.arange(lo[1] - pad, hi[1] + pad + cell, cell)
        zfloor = float(lo[2]) - drop
        pts = np.empty((len(ys), len(xs), 3))
        for j, y in enumerate(ys):
            for i, x in enumerate(xs):
                near = v[(np.abs(v[:, 0] - x) < 1.5 * cell)
                         & (np.abs(v[:, 1] - y) < 1.5 * cell)]
                z = (float(near[:, 2].min()) - drop) if len(near) else zfloor
                pts[j, i] = (x, y, z)
        faces = []
        for j in range(len(ys) - 1):
            for i in range(len(xs) - 1):
                a = j * len(xs) + i
                b, c, d = a + 1, a + len(xs), a + len(xs) + 1
                faces += [[a, c, b], [b, c, d]]
        self._add_static_mesh(builder, pts.reshape(-1, 3), np.array(faces, np.int32),
                              "appendix_shelf",
                              ke=float(self.contact_material[0]),
                              kd=float(self.contact_material[1]),
                              kf=float(self.contact_material[2]),
                              mu=0.5, is_solid=False)
        log(f"  shelf    {len(faces):6d} faces under the appendix "
            f"({len(xs)}x{len(ys)} grid, {drop*1000:.1f} mm below its underside)")

    def _add_organs(self, builder):
        present = [(n, ROOT / "tet" / f"{n}.npz") for n in ORGANS]
        present = [(n, p) for n, p in present if p.exists()]
        for n in ORGANS:
            if not (ROOT / "tet" / f"{n}.npz").exists():
                log(f"  skipping {n}: tet/{n}.npz missing")

        # captured -> world, using the same bake the Isaac cell applies
        self.organ_world = {}
        for name, p in present:
            verts, idx = load_tet(p)
            self.organ_world[name] = self.L.organs_to_world(verts)

        T = self.tissue
        nu = T["poisson"]
        log(f"  material nu={nu} rho={T['density']:.0f} damping={T['damping']} "
            f"(from build_surgical_cell.py, NOT VLA_HANDOFF section 7.4)")

        for name, p in present:
            verts, idx = load_tet(p)
            world = self.organ_world[name]
            E = T["youngs"][name]
            mu_i, lam_i = lame(E, nu)
            first = builder.particle_count
            builder.add_soft_mesh(
                pos=wp.vec3(0.0, 0.0, 0.0),
                rot=wp.quat_identity(),
                scale=1.0,
                vel=wp.vec3(0.0, 0.0, 0.0),
                vertices=[wp.vec3(*q) for q in world],
                indices=idx.tolist(),
                density=T["density"],
                k_mu=mu_i,
                k_lambda=lam_i,
                k_damp=T["damping"],
            )
            self.organ_span[name] = (first, builder.particle_count)
            if name == "appendix":
                pin = self._appendix_setup(world)
                for h in self.handles:       # handles index into the global array
                    h["idx"] = h["idx"] + first
            else:
                pin = self._pin_indices(name, world)
            self.pinned[name] = pin + first
            log(f"    {name:9s} {len(world):5d} nodes {len(idx)//4:6d} tets  "
                f"E={E/1e3:4.1f} kPa  mu={mu_i:8.1f} lam={lam_i:9.1f}  "
                f"pinned {len(pin):4d} ({100*len(pin)/max(1,len(world)):4.1f}%)")

    # ------------------------------------------------------------------
    def build(self) -> newton.Model:
        """Tissue only. `sim.Cell` uses _prepare/_finalize so it can slot the
        arms' kinematic bodies in between."""
        a = self.args
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
        builder.particle_max_velocity = 20.0
        self._prepare(builder)
        return self._finalize(builder)

    def _prepare(self, builder) -> None:
        a = self.args
        # Particle radius must stay well under the tet edge length; at radius ~=
        # edge length every particle overlaps its neighbours and self-contact
        # tears the body apart on the first step (sim/scene.py, measured).
        edges = [median_edge(*load_tet(ROOT / "tet" / f"{n}.npz"))
                 for n in ORGANS if (ROOT / "tet" / f"{n}.npz").exists()]
        edge = min(edges) if edges else 0.002
        builder.default_particle_radius = a.radius_ratio * edge
        log(f"cell: tet edge (min median) {edge*1000:.2f} mm -> particle radius "
            f"{builder.default_particle_radius*1000:.3f} mm")

        log("  colliders:")
        if not a.no_phantom:
            self._add_phantom(builder)
            self._add_field(builder)
            self._add_endoscope(builder)
        if a.organ_support == "plane":
            self._organ_support_plane(builder)
        if not a.no_organ_pins:
            self._appendix_shelf(builder)

        log("  organs:")
        self._add_organs(builder)

    def _finalize(self, builder) -> newton.Model:
        a = self.args
        # Pin the mesentery band. A pinned node is one with the ACTIVE flag
        # cleared and zero mass, set on the BUILDER: clearing ACTIVE is what
        # tells the solver not to integrate it, and doing it before finalize()
        # means the colouring and the solver's precomputed adjacency both see
        # it. Zeroing `particle_inv_mass` on the finalized model instead does
        # NOT pin anything -- measured, pinned nodes still drifted 38.6 mm in
        # 1.5 s of sim.
        pin_all = ([] if not any(len(v) for v in self.pinned.values())
                   else np.concatenate([v for v in self.pinned.values() if len(v)]))
        pin_all = np.asarray(pin_all, np.int32)
        for i in pin_all:
            builder.particle_flags[int(i)] = (
                builder.particle_flags[int(i)] & ~int(newton.ParticleFlags.ACTIVE))
            builder.particle_mass[int(i)] = 0.0
        if len(pin_all):
            log(f"  pinned {len(pin_all)} of {builder.particle_count} nodes "
                f"({100*len(pin_all)/builder.particle_count:.1f}%) at capture "
                f"pose (ACTIVE cleared, mass 0)")

        builder.color()                      # VBD needs a colouring; after geometry
        model = builder.finalize(device=a.device)

        # tetgen leaves sliver tets whatever the quality knobs; a node belonging
        # only to slivers ends up ~1e-9 kg against a 4e-4 kg median and ordinary
        # elastic forces launch it on the first step (sim/scene.py, measured).
        if a.mass_floor > 0:
            mass = model.particle_mass.numpy().copy()
            free = mass > 0                  # never resurrect a pinned node
            floor = a.mass_floor * float(np.median(mass[free]))
            raised = int((mass[free] < floor).sum())
            mass[free] = np.maximum(mass[free], floor)
            model.particle_mass.assign(mass.astype(np.float32))
            inv = np.zeros_like(mass)
            np.divide(1.0, mass, out=inv, where=mass > 0)   # pinned nodes stay 0
            model.particle_inv_mass.assign(inv.astype(np.float32))
            log(f"  mass floor {a.mass_floor:.2f}x median ({floor:.2e} kg): "
                f"raised {raised}/{len(mass)} ({100*raised/len(mass):.1f}%)")

        # Keep Newton's global soft-contact defaults for particle contacts;
        # instrument and shelf ShapeConfig values are set explicitly from the
        # measured-evaluation knobs above.
        model.soft_contact_ke = 1.0e4
        model.soft_contact_kd = 1.0
        model.soft_contact_kf = 1.0e3
        model.soft_contact_mu = self.contact_mu
        self.model = model
        log(f"  model: {model.particle_count} particles, {model.tet_count} tets, "
            f"{model.shape_count} shapes")
        return model


def add_args(ap):
    ap.add_argument("--device", default=None, help="warp device, e.g. cuda:0")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--substeps", type=int, default=8)
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--contact-mu", type=float, default=None,
                    help="diagnostic override applied to Newton soft-contact and instrument ShapeConfig mu")
    ap.add_argument("--radius-ratio", type=float, default=0.1)
    ap.add_argument("--mass-floor", type=float, default=0.25)
    ap.add_argument("--no-phantom", action="store_true")
    ap.add_argument("--no-organ-pins", action="store_true")
    ap.add_argument("--organ-support", choices=("plane", "none"), default="none",
                    help="'plane' keeps the deformable scene's ground plane, "
                         "lifted with the specimen. Default 'none': measured, "
                         "it lands at z=48 mm while the organs sit at 53-78 mm, "
                         "so it never touches them -- and being 2x2 m it filled "
                         "36 %% of the endoscope view")
    ap.add_argument("--organ-field-collision", action="store_true",
                    help="let the organs collide with the field. Off by "
                         "default, as in build_surgical_cell.py: they are "
                         "captured embedded in it and get ejected at spawn")
    ap.add_argument("--phantom-collision", action="store_true",
                    help="let the organs collide with the phantom dome. Off by "
                         "default: they sit well clear of its walls and the "
                         "instruments are kinematic, so it has nothing to push")
    return ap


def effective_contact_mu(override, tissue_mu):
    """Resolve one coefficient for both sides of Newton VBD tool/FEM contact.

    SolverVBD's body-particle contact formulation uses the particle-side and
    shape-side ``mu`` values; ShapeConfig.kf is not consumed by that path.
    Setting both sides through this override is therefore necessary for a
    true zero-friction tool/FEM diagnostic.
    """
    value = float(tissue_mu if override is None else override)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("contact mu must be finite and nonnegative")
    return value


def main(argv=None) -> int:
    ap = add_args(argparse.ArgumentParser(description=__doc__))
    a = ap.parse_args(argv)

    wp.init()
    log(f"device: {wp.get_device(a.device)}")
    scene = CellScene(a)
    model = scene.build()

    solver = SolverVBD(model=model, iterations=a.iterations,
                       particle_enable_self_contact=False)
    log(f"solver: {type(solver).__name__}  iterations={a.iterations}")

    state_0, state_1 = model.state(), model.state()
    control, contacts = model.control(), model.contacts()
    frame_dt = 1.0 / a.fps
    dt = frame_dt / a.substeps

    q0 = state_0.particle_q.numpy().copy()
    t0 = time.time()
    for frame in range(a.steps):
        for _ in range(a.substeps):
            state_0.clear_forces()
            model.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, dt)
            state_0, state_1 = state_1, state_0
        q = state_0.particle_q.numpy()
        if not np.isfinite(q).all():
            log(f"\ndiverged at frame {frame}: non-finite particle positions")
            return 1
        if frame % max(1, a.steps // 6) == 0 or frame == a.steps - 1:
            spd = float(np.abs(state_0.particle_qd.numpy()).max())
            log(f"  frame {frame:4d} t={frame*frame_dt:5.2f}s  "
                f"mean |dx| {np.linalg.norm(q-q0,axis=1).mean()*1000:6.3f} mm  "
                f"max speed {spd:7.4f} m/s")
    wall = time.time() - t0
    sim_t = a.steps * frame_dt
    log(f"\n{a.steps} frames ({sim_t:.2f} s sim) in {wall:.1f} s wall  "
        f"-> rtf {sim_t/wall:.3f}")

    qf = state_0.particle_q.numpy()
    log("per-organ displacement from capture pose:")
    for name, (i, j) in scene.organ_span.items():
        d = np.linalg.norm(qf[i:j] - q0[i:j], axis=1) * 1000
        log(f"  {name:9s} mean {d.mean():6.2f} mm   max {d.max():6.2f} mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
