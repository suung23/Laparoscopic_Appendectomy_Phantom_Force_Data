#!/usr/bin/env python3
"""Newton scene: laparoscopic phantom + reconstructed rabbit organs.

    .venv-sam2/bin/python sim/scene.py --steps 120 --record out/rollout.npz

Everything is SI: metres, kilograms, seconds. The reconstruction pipeline works
in millimetres (uint16 depth PNGs hold metric depth exactly that way), so tet
meshes are converted once in tools/tetrahedralize.py and arrive here in metres.

Frames
------
The reconstruction lives in the D405 camera frame: +x right, +y DOWN the image,
+z away from the lens. Newton is z-up. CAM_TO_WORLD maps one to the other, so a
"lower in the image" point ends up lower in the world rather than further away.

Solver
------
SolverVBD. The organs tetrahedralise to ~1.6 mm elements, and explicit
integration at that element size needs dt ~1e-5 s to stay stable -- both
SolverSemiImplicit and SolverXPBD blow up to five digits of m/s within a few
frames at 60 fps, whatever the substep count or particle mass distribution.
VBD is implicit and holds at dt = 1/(60*8); the free-fall check reproduces
g*t to four digits.

Material
--------
k_mu / k_lambda are Lame parameters in pascals. The defaults below are literature
values for bowel-like soft tissue (E ~ 20 kPa, nu ~ 0.45), NOT identified from
this recording. Fitting them against the instrument-contact frames is a separate
step; until that runs, treat the dynamics as plausible rather than calibrated.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverSemiImplicit, SolverVBD, SolverXPBD

ROOT = Path(__file__).resolve().parent.parent

# The D405 looked straight DOWN at the animal on the table, so the camera's
# depth axis is the world's down axis -- not a horizontal one. Mapping depth to
# a horizontal axis instead stands the rabbit on end inside the phantom.
#   cam +x (image right) -> world +x
#   cam +y (image down)  -> world -y   (horizontal, along the table)
#   cam +z (deeper)      -> world -z   (further from the lens = lower)
CAM_TO_WORLD = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)

TISSUE_DENSITY = 1050.0  # kg/m^3, soft tissue
YOUNGS_MODULUS = 20.0e3  # Pa
POISSON_RATIO = 0.45
DAMPING = 1.0

# Every organ is a soft body. They share one material; the appendix gets its
# own, because it is the structure the instruments actually manipulate and the
# one whose response has to be right.
ORGANS = [
    ("cecum", "tet/cecum.npz"),
    ("bowel1", "tet/bowel1.npz"),
    ("bowel2", "tet/bowel2.npz"),
    ("bowel3", "tet/bowel3.npz"),
    ("stomach", "tet/stomach.npz"),
    ("appendix", "tet/appendix.npz"),
]

# Pa. Eyeballed, not identified from the recording -- see HANDOFF.md.
ORGAN_YOUNGS = {"appendix": 12.0e3}


def lame(E: float, nu: float) -> tuple[float, float]:
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return mu, lam


def load_tet(path: Path):
    d = np.load(path)
    return d["vertices"].astype(np.float64), d["indices"].astype(np.int32)


def median_edge(verts: np.ndarray, tets: np.ndarray) -> float:
    """Median tet edge length -- sets the particle radius scale."""
    t = tets.reshape(-1, 4)
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    d = np.concatenate([np.linalg.norm(verts[t[:, i]] - verts[t[:, j]], axis=1) for i, j in pairs])
    return float(np.median(d))


def placement_offset(all_verts_m: list[np.ndarray], drop_height: float) -> np.ndarray:
    """One shared translation for every organ.

    Centring each organ on the phantom axis independently would throw away the
    relative anatomy the reconstruction captured -- the appendix ends up inside
    the cecum. The offset is computed once over the union and applied to all.
    """
    stacked = np.vstack([v @ CAM_TO_WORLD.T for v in all_verts_m])
    return np.array([
        -stacked[:, 0].mean(),
        -stacked[:, 1].mean(),
        -stacked[:, 2].min() + drop_height,
    ])


def place(verts_m: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """Camera frame -> world, using the shared offset."""
    return verts_m @ CAM_TO_WORLD.T + offset


def build(args) -> tuple[newton.Model, dict]:
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
    builder.particle_max_velocity = 20.0

    # Particle radius must stay well under the tet edge length. At radius ~= edge
    # length every particle overlaps its own neighbours, and self-contact tears
    # the body apart on the first step (max speed pinned to the clamp from frame 0).
    edges = [median_edge(load_tet(ROOT / rel)[0], load_tet(ROOT / rel)[1])
             for _, rel in ORGANS if (ROOT / rel).exists()]
    edge = min(edges) if edges else 0.002
    builder.default_particle_radius = args.radius_ratio * edge
    print(f"tet edge (min median): {edge*1000:.2f} mm  ->  particle radius "
          f"{builder.default_particle_radius*1000:.3f} mm")

    ground_cfg = newton.ModelBuilder.ShapeConfig(ke=2.0e5, kd=1.0, kf=1.0e3, mu=0.6)
    builder.add_ground_plane(cfg=ground_cfg)

    # --- phantom: static collider -------------------------------------
    import trimesh

    if args.no_phantom:
        print("phantom collider disabled")
    else:
        # the reconstructed surgical field: one textured solid, static collider
        field_path = ROOT / "mesh" / "field.ply"
        if field_path.exists():
            fl = trimesh.load(field_path, process=False)
            fv = np.asarray(fl.vertices, np.float64) * 0.001 @ CAM_TO_WORLD.T
            fv += placement_offset([np.asarray(fl.vertices, np.float64) * 0.001],
                                   args.drop_height)
            builder.add_shape_mesh(
                body=-1,
                mesh=newton.Mesh(fv, np.asarray(fl.faces, np.int32).flatten(), is_solid=False),
                cfg=newton.ModelBuilder.ShapeConfig(ke=1.0e5, kd=1.0, kf=1.0e3, mu=0.5),
                label="field",
            )
            print(f"field collider: {len(fl.faces)} faces, "
                  f"extent {((fv.max(0)-fv.min(0))*1000).round(1)} mm")

        # endoscope: mesh only, static. Enters through the 92 mm centre port in
        # the dome at (-9.5, 50.2, 186) mm and reaches down into the abdomen.
        # Barrel diameter 8.8 mm was measured off the 2026-07-12 recording.
        endo_path = ROOT / "mesh" / "endoscope.ply"
        if endo_path.exists():
            en = trimesh.load(endo_path, process=False)
            ev = np.asarray(en.vertices, np.float64) * 0.001   # authored in world mm, z up
            builder.add_shape_mesh(
                body=-1,
                mesh=newton.Mesh(ev, np.asarray(en.faces, np.int32).flatten(), is_solid=True),
                cfg=newton.ModelBuilder.ShapeConfig(ke=1.0e5, kd=1.0, kf=1.0e3, mu=0.3),
                label="endoscope",
            )
            print(f"endoscope: {len(en.faces)} faces, tip z={ev[:,2].min()*1000:.0f} mm")

        ph = trimesh.load(ROOT / "phantom.stl", process=False)
        ph_v = np.asarray(ph.vertices, np.float64) * 0.001  # mm -> m
        ph_v[:, 2] -= ph_v[:, 2].min()  # rest the dome on the floor
        ph_v[:, 0] -= ph_v[:, 0].mean()
        ph_v[:, 1] -= ph_v[:, 1].mean()
        phantom_mesh = newton.Mesh(ph_v, np.asarray(ph.faces, np.int32).flatten(), is_solid=False)
        builder.add_shape_mesh(
            body=-1,  # -1 = static world
            mesh=phantom_mesh,
            cfg=newton.ModelBuilder.ShapeConfig(ke=1.0e5, kd=1.0, kf=1.0e3, mu=0.4),
            label="phantom",
        )

    mu, lam = lame(args.youngs, args.poisson)
    print(f"material: E={args.youngs/1e3:.1f} kPa nu={args.poisson}  ->  mu={mu:.0f} Pa lambda={lam:.0f} Pa")

    # --- organs: FEM soft bodies --------------------------------------
    present = [(n, r) for n, r in ORGANS if (ROOT / r).exists()]
    for n, r in ORGANS:
        if (ROOT / r) not in [ROOT / x for _, x in present]:
            print(f"skipping {n}: {r} missing")
    offset = placement_offset([load_tet(ROOT / r)[0] for _, r in present], args.drop_height)
    print(f"shared placement offset: {(offset*1000).round(1)} mm")

    info = {}
    for name, rel in present:
        verts, idx = load_tet(ROOT / rel)
        world = place(verts, offset)
        first = builder.particle_count
        E = ORGAN_YOUNGS.get(name, args.youngs)
        mu_i, lam_i = lame(E, args.poisson)
        builder.add_soft_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=[wp.vec3(*p) for p in world],
            indices=idx.tolist(),
            density=args.density,
            k_mu=mu_i,
            k_lambda=lam_i,
            k_damp=args.damping,
        )
        info[name] = (first, builder.particle_count)
        ext = (world.max(0) - world.min(0)) * 1000
        print(f"  {name:9s} {len(world):5d} nodes  {len(idx)//4:6d} tets  E={E/1e3:.0f} kPa  "
              f"particles [{first}:{builder.particle_count})")

    if args.solver == "vbd":
        builder.color()  # VBD needs a graph colouring; must follow all geometry

    model = builder.finalize(device=args.device)

    # tetgen's Delaunay refinement leaves sliver tets no matter how the quality
    # knobs are set (isotropic remeshing of the surface first does not remove
    # them either). A node that belongs only to slivers ends up ~1e-9 kg against
    # a 4e-4 kg median, and ordinary elastic forces then accelerate it to five
    # digits of m/s on the first step. Lifting the lightest particles to a floor
    # costs a little mass accuracy and buys a sim that does not explode.
    if args.mass_floor > 0:
        mass = model.particle_mass.numpy().copy()
        floor = args.mass_floor * float(np.median(mass))
        raised = int((mass < floor).sum())
        mass = np.maximum(mass, floor)
        model.particle_mass.assign(mass.astype(np.float32))
        model.particle_inv_mass.assign(np.where(mass > 0, 1.0 / mass, 0.0).astype(np.float32))
        print(f"mass floor {args.mass_floor:.2f}x median ({floor:.2e} kg): raised {raised} of "
              f"{len(mass)} particles ({100*raised/len(mass):.1f}%)")

    model.soft_contact_ke = 1.0e4
    model.soft_contact_kd = 1.0
    model.soft_contact_kf = 1.0e3
    model.soft_contact_mu = 0.5
    return model, info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60, help="frames to simulate")
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--substeps", type=int, default=8)
    ap.add_argument("--solver", choices=["vbd", "semi_implicit", "xpbd"], default="vbd",
                    help="vbd is the only one stable here; see the note in the module docstring")
    ap.add_argument("--device", default=None, help="warp device, e.g. cuda:0 (default: auto)")
    ap.add_argument("--youngs", type=float, default=YOUNGS_MODULUS)
    ap.add_argument("--poisson", type=float, default=POISSON_RATIO)
    ap.add_argument("--density", type=float, default=TISSUE_DENSITY)
    ap.add_argument("--damping", type=float, default=DAMPING)
    ap.add_argument("--drop-height", type=float, default=0.005, help="metres above the floor")
    ap.add_argument("--radius-ratio", type=float, default=0.1,
                    help="particle radius as a fraction of median tet edge length")
    ap.add_argument("--no-phantom", action="store_true", help="ground plane only, for isolating issues")
    ap.add_argument("--iterations", type=int, default=10, help="solver iterations (vbd/xpbd)")
    ap.add_argument("--mass-floor", type=float, default=0.25,
                    help="raise particle mass to this fraction of the median (0 = off)")
    ap.add_argument("--record", help="npz to write particle trajectories to")
    ap.add_argument("--viewer", choices=["none", "gl", "viser", "usd"], default="none",
                    help="gl = native window you can orbit; viser = same thing in a browser; "
                         "usd = write a .usd you can open in Blender / Isaac Sim / Omniverse")
    ap.add_argument("--viewer-out", default="out/scene.usd", help="path for --viewer usd")
    args = ap.parse_args()

    wp.init()
    print(f"device: {wp.get_device(args.device)}")

    model, info = build(args)
    if args.solver == "vbd":
        solver = SolverVBD(model=model, iterations=args.iterations,
                           particle_enable_self_contact=False)
    elif args.solver == "semi_implicit":
        solver = SolverSemiImplicit(model=model)
    else:
        solver = SolverXPBD(model=model, iterations=args.iterations)
    print(f"solver: {type(solver).__name__}  particles={model.particle_count}  "
          f"tets={model.tet_count}  shapes={model.shape_count}")

    viewer = None
    if args.viewer != "none":
        from newton import viewer as nv
        if args.viewer == "gl":
            viewer = nv.ViewerGL()
        elif args.viewer == "viser":
            viewer = nv.ViewerViser()
        else:
            Path(args.viewer_out).parent.mkdir(parents=True, exist_ok=True)
            viewer = nv.ViewerUSD(output_path=args.viewer_out, fps=int(args.fps),
                                  num_frames=args.steps, up_axis="Z")
        viewer.set_model(model)
        print(f"viewer: {type(viewer).__name__}")

    state_0, state_1 = model.state(), model.state()
    control, contacts = model.control(), model.contacts()
    frame_dt = 1.0 / args.fps
    dt = frame_dt / args.substeps

    traj = []
    q0 = state_0.particle_q.numpy().copy()
    for frame in range(args.steps):
        if viewer is not None and not viewer.is_running():
            print(f"viewer closed at frame {frame}")
            break
        for _ in range(args.substeps):
            state_0.clear_forces()
            model.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, dt)
            state_0, state_1 = state_1, state_0
        if viewer is not None:
            viewer.begin_frame(frame * frame_dt)
            viewer.log_state(state_0)
            viewer.log_contacts(contacts, state_0)
            viewer.end_frame()

        q = state_0.particle_q.numpy()
        if not np.isfinite(q).all():
            print(f"\ndiverged at frame {frame}: non-finite particle positions")
            return 1
        traj.append(q.copy())
        if frame % max(1, args.steps // 6) == 0 or frame == args.steps - 1:
            drop = (q0[:, 2].mean() - q[:, 2].mean()) * 1000
            print(f"  frame {frame:4d}  t={frame*frame_dt:5.2f}s  "
                  f"mean z drop {drop:6.2f} mm  max speed "
                  f"{np.abs(state_0.particle_qd.numpy()).max():6.3f} m/s")

    print("\nfinal per-organ displacement:")
    qf = traj[-1]
    for name, (a, b) in info.items():
        d = np.linalg.norm(qf[a:b] - q0[a:b], axis=1) * 1000
        print(f"  {name:9s} mean {d.mean():6.2f} mm   max {d.max():6.2f} mm")

    if args.record:
        out = Path(args.record)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out,
            trajectory=np.asarray(traj, np.float32),
            rest=q0.astype(np.float32),
            organs=np.array([(n, a, b) for n, (a, b) in info.items()], dtype=object),
            fps=args.fps,
        )
        print(f"\nwrote {out} ({out.stat().st_size/1e6:.1f} MB)")
    if viewer is not None:
        viewer.close()
        if args.viewer == "usd":
            print(f"wrote {args.viewer_out} - open it in Blender, Isaac Sim or Omniverse")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
