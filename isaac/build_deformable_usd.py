#!/usr/bin/env python3
"""Rabbit surgical field -> Isaac Sim PhysX deformable bodies.

The Newton pipeline (sim/scene.py) bakes physics into an animation. This builds
the same scene as *live* PhysX FEM soft bodies instead, so the organs can be
poked, grabbed and dragged.

Placement is not recomputed here -- scene.py's own CAM_TO_WORLD / placement_offset
are imported and reused, so organs, field and phantom land exactly where the
Newton scene puts them.

    ~/isaacsim-5.1/python.sh isaac/build_deformable_usd.py              # GUI, interactive
    ~/isaacsim-5.1/python.sh isaac/build_deformable_usd.py --headless \
        --bench 120                                                     # measure step cost

pxr and omni only exist once the Kit runtime is up, so SimulationApp has to be
constructed before any of them are imported. That is why the imports below are
not all at the top of the file.
"""
from __future__ import annotations

import argparse
import sys
import time
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="out/scene_physx.usd")
    ap.add_argument("--headless", action="store_true", help="no window; for benchmarking")
    ap.add_argument("--bench", type=int, default=0,
                    help="step physics this many frames and report the rate, then exit")
    ap.add_argument("--drop-height", type=float, default=0.005,
                    help="metres above the floor, matching scene.py")
    ap.add_argument("--youngs", type=float, default=5.0e3,
                    help="Pa, Young's modulus of the general tissue. scene.py's "
                         "Newton constant is 20 kPa, which is firm muscle -- at "
                         "that stiffness the organs barely dent under an "
                         "instrument. Laparoscopic bowel/mesentery behaves like "
                         "single-digit kPa, so the Isaac cell defaults to 5 kPa; "
                         "pass the scene.py value explicitly to compare")
    ap.add_argument("--appendix-youngs", type=float, default=1.5e3,
                    help="Pa, Young's modulus of the appendix specifically. It "
                         "is the structure the instruments manipulate in an "
                         "appendectomy: it has to be limp enough to be grasped "
                         "and lifted clear of the caecum, so it is the softest "
                         "tissue in the scene")
    ap.add_argument("--hex-resolution", type=int, default=16,
                    help="hexes along the longest edge of each simulation mesh "
                         "(0 = PhysX picks). The auto pick gave the appendix "
                         "only 154 nodes -- a slender tube spanned by a handful "
                         "of big voxels is stiff in BENDING no matter how low E "
                         "is, so it could never drape over the jaws")
    ap.add_argument("--poisson", type=float, default=None)
    ap.add_argument("--no-field", action="store_true",
                    help="skip the 457k-face field collider (much cheaper)")
    ap.add_argument("--substeps", type=int, default=8,
                    help="physics substeps per 60 Hz frame; scene.py uses 8")
    ap.add_argument("--iterations", type=int, default=32,
                    help="solver position iterations per substep")
    ap.add_argument("--contact-offset", type=float, default=0.003,
                    help="metres at which organs start feeling each other")
    ap.add_argument("--rest-offset", type=float, default=0.0008,
                    help="metres the surfaces hold apart at rest")
    ap.add_argument("--self-collision", action="store_true",
                    help="stop a folded organ passing through itself; slow to cook")
    ap.add_argument("--static", action="store_true",
                    help="author and display the scene without stepping physics")
    ap.add_argument("--hex-mesh", action="store_true", default=False,
                    help="cook a hexahedral simulation mesh instead of a "
                         "tetrahedral one. Hexes never sliver, but the runtime "
                         "cooking has not simulated cleanly here")
    ap.add_argument("--keep-tets", action="store_true",
                    help="ignored: the hierarchy cooker always regenerates the "
                         "sim mesh from the surface, so the handoff tets cannot "
                         "be fed straight in through this path")
    ap.add_argument("--show-shell", action="store_true",
                    help="draw the phantom dome and field. They enclose the "
                         "organs, so by default they collide but stay hidden")
    return ap.parse_args()


ARGS = parse_args()

# --- boot Kit before touching pxr/omni ------------------------------------
from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": ARGS.headless})

import omni.usd  # noqa: E402
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402

# --- scene.py owns the geometry conventions; reuse rather than restate ----
sys.path.insert(0, str(ROOT / "sim"))

# scene.py imports newton and warp at module scope and Isaac Sim's interpreter
# has neither. The helpers we want need only numpy, so stand in empty modules
# long enough to get the import through.
for _mod in ("newton", "newton.solvers", "warp"):
    sys.modules.setdefault(_mod, types.ModuleType(_mod))
sys.modules["newton"].solvers = sys.modules["newton.solvers"]
for _name in ("SolverSemiImplicit", "SolverVBD", "SolverXPBD"):
    setattr(sys.modules["newton.solvers"], _name, None)

from scene import (  # noqa: E402
    CAM_TO_WORLD,
    ORGANS,
    ORGAN_YOUNGS,
    POISSON_RATIO,
    TISSUE_DENSITY,
    YOUNGS_MODULUS,
    load_tet,
    place,
    placement_offset,
)


def log(msg: str) -> None:
    print(msg, flush=True)


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    """0-255 sRGB -> linear float, which is what displayColor is read as."""
    c = np.asarray(c, np.float64)[:, :3] / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def ply_vertex_colors(path: Path):
    """Per-vertex linear colour from a .ply, or None if it carries none."""
    import trimesh

    if not path.exists():
        return None
    m = trimesh.load(path, process=False)
    vc = getattr(m.visual, "vertex_colors", None)
    if vc is None or len(vc) == 0:
        return None
    return srgb_to_linear(np.asarray(vc))


def set_display_color(mesh, colors: np.ndarray | None, n_points: int) -> bool:
    """Author displayColor. Falls back to a flat tissue tone when unavailable.

    tetgen keeps the input surface vertices as the first nodes of the volume
    mesh, so the surface colours line up with the leading points and the
    interior nodes -- never rendered -- take the mean.
    """
    if colors is None:
        return False
    if len(colors) < n_points:
        fill = np.tile(colors.mean(axis=0), (n_points - len(colors), 1))
        colors = np.vstack([colors, fill])
    pv = mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex)
    pv.Set([Gf.Vec3f(*c) for c in colors[:n_points].tolist()])
    return True


def set_extent(prim_or_schema, verts: np.ndarray) -> None:
    """Author the bounding box.

    Without `extent` Hydra treats the prim's bounds as empty and frustum-culls
    it, so the geometry is on the stage, lit, and never drawn.
    """
    lo, hi = verts.min(axis=0).tolist(), verts.max(axis=0).tolist()
    boundable = UsdGeom.Boundable(prim_or_schema)
    boundable.CreateExtentAttr().Set([Gf.Vec3f(*lo), Gf.Vec3f(*hi)])


# --------------------------------------------------------------------------
# geometry, placed exactly as the Newton scene places it
# --------------------------------------------------------------------------

def organ_geometry(drop_height: float):
    present = [(n, r) for n, r in ORGANS if (ROOT / r).exists()]
    for n, r in ORGANS:
        if not (ROOT / r).exists():
            log(f"  skipping {n}: {r} missing")
    offset = placement_offset([load_tet(ROOT / r)[0] for _, r in present], drop_height)
    log(f"  shared placement offset: {(offset * 1000).round(1)} mm")
    out = []
    for name, rel in present:
        verts, idx = load_tet(ROOT / rel)
        out.append((name, place(verts, offset), idx.reshape(-1, 4)))
    return out


def collider_geometry(drop_height: float, include_field: bool, organ_offset=None):
    import trimesh

    out = []
    field_path = ROOT / "mesh" / "field.ply"
    if include_field and field_path.exists():
        fl = trimesh.load(field_path, process=False)
        fv_m = np.asarray(fl.vertices, np.float64) * 0.001
        # Use the SAME offset the organs got, not the field's own. The field and
        # organs were photographed together in one camera frame; giving each its
        # own placement_offset floats them apart -- the field ends up above the
        # organs with nothing under them. The shared offset restores the real
        # registration, so the field sits under the organs as their body wall.
        off = organ_offset if organ_offset is not None else placement_offset([fv_m], drop_height)
        fv = fv_m @ CAM_TO_WORLD.T + off
        out.append(("field", fv, np.asarray(fl.faces, np.int32)))

    endo_path = ROOT / "mesh" / "endoscope.ply"
    if endo_path.exists():
        en = trimesh.load(endo_path, process=False)
        out.append(("endoscope", np.asarray(en.vertices, np.float64) * 0.001,
                    np.asarray(en.faces, np.int32)))

    ph_path = ROOT / "phantom.stl"
    if ph_path.exists():
        ph = trimesh.load(ph_path, process=False)
        pv = np.asarray(ph.vertices, np.float64) * 0.001
        pv[:, 2] -= pv[:, 2].min()
        pv[:, 0] -= pv[:, 0].mean()
        pv[:, 1] -= pv[:, 1].mean()
        out.append(("phantom", pv, np.asarray(ph.faces, np.int32)))
    return out


# --------------------------------------------------------------------------
# deformable authoring -- Isaac Sim has shipped two schema generations, so
# probe for both rather than assuming which one this build carries.
# --------------------------------------------------------------------------

def deformable_backend() -> str | None:
    try:
        from omni.physx.scripts import deformableUtils
        if hasattr(deformableUtils, "set_physics_volume_deformable_body"):
            return "omniphysics"
    except ImportError:
        pass
    if hasattr(PhysxSchema, "PhysxDeformableBodyAPI"):
        return "physx_legacy"
    return None


def make_material(stage, path, youngs, poisson, density):
    """omni.physx owns the material schema; let it author the attributes."""
    from omni.physx.scripts import deformableUtils as du

    du.add_deformable_material(
        stage, path,
        density=density,
        youngs_modulus=youngs,
        poissons_ratio=poisson,
        dynamic_friction=0.5,
        static_friction=0.5,
    )
    return UsdShade.Material.Get(stage, path)


def author_organ(stage, name, verts, tets, material):
    """The organ as a volume deformable cooked from its surface.

    A bare TetMesh does not render in Hydra, so a sibling render Mesh is bound to
    the body's bind pose and PhysX skins it -- no per-frame Python, which is what
    segfaulted Kit before.

      /World/organs/<name>        Xform, DeformableBodyAPI
        /sim                      TetMesh, cooked HEX simulation mesh (guide)
        /coll                     TetMesh, cooked collision mesh (guide)
        /render                   Mesh, skinned visible surface

    **The sim and collision meshes must be DIFFERENT prims.** A hexahedral
    simulation mesh cannot double as the collision mesh: PhysX then fails the
    whole body with "Failed to cook PxDeformableVolumeMesh!" (the underlying
    complaint is "collision mesh UsdGeomTetMesh needs to have
    surfaceFaceVertexIndices set") and the organ silently never simulates --
    it renders, it is skinned, it just never moves, not even under gravity.
    Measured on a plain cube in isolation: sim==coll + hex = 0.00 mm over 1 s,
    every other combination free-falls ~4.9 m. That single shared path is why
    the organs behaved like rigid scenery.
    """
    from omni.physx.scripts import deformableUtils as du

    root = f"/World/organs/{name}"
    sim_path = f"{root}/sim"
    coll_path = f"{root}/coll"
    render_path = f"{root}/render"
    surf = surface_faces(name)
    UsdGeom.Xform.Define(stage, root)

    # render mesh = cooking source + visible surface; PhysX skins it
    render = UsdGeom.Mesh.Define(stage, render_path)
    render.CreatePointsAttr().Set([Gf.Vec3f(*p) for p in verts.tolist()])
    render.CreateFaceVertexIndicesAttr().Set(surf.flatten().tolist())
    render.CreateFaceVertexCountsAttr().Set([3] * len(surf))
    render.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    set_extent(render.GetPrim(), verts)
    if not set_display_color(render, ply_vertex_colors(ROOT / "mesh" / f"organ_{name}.ply"),
                             len(verts)):
        log(f"  ! {name}: no vertex colour on the source ply")

    # A hexahedral sim mesh voxelises the interior, so it cannot contain the
    # sliver tets that make every tetrahedral path blow up on this data. The
    # tets shipped with the handoff and PhysX's own tet cook both sliver; the
    # voxel hex does not. --hex-resolution sets how many hexes span the longest
    # edge: a slender organ voxelised at the auto resolution is far stiffer in
    # bending than its material E implies, so the appendix needs a real count.
    ok = du.create_auto_volume_deformable_hierarchy(
        stage, root, sim_path, coll_path, render_path,
        simulation_hex_mesh_enabled=True,
        cooking_src_simplification_enabled=False,
        set_visibility_with_guide_purpose=True,
    )
    if not ok:
        log(f"  ! {name}: create_auto_volume_deformable_hierarchy failed")

    root_prim = stage.GetPrimAtPath(root)
    if ARGS.hex_resolution > 0:
        a = root_prim.GetAttribute("physxDeformableBody:resolution")
        if not a:
            a = root_prim.CreateAttribute("physxDeformableBody:resolution",
                                          Sdf.ValueTypeNames.UInt)
        a.Set(int(ARGS.hex_resolution))

    # With sim != coll the helper only gives the SIM mesh guide purpose; the
    # collision tet mesh would otherwise draw as grey tets over the organ. Hide
    # it with VISIBILITY rather than purpose -- purpose is what physics parsing
    # keys off, and the helper pointedly leaves the collision mesh's alone.
    coll_prim_vis = stage.GetPrimAtPath(coll_path)
    if coll_prim_vis:
        UsdGeom.Imageable(coll_prim_vis).CreateVisibilityAttr().Set(
            UsdGeom.Tokens.invisible)

    # By default the collision offset is ~0, so two organs only register contact
    # once their surfaces already overlap -- and since they are packed together
    # anatomically, they simply pass through each other. A few mm of contact/rest
    # offset makes them feel each other before interpenetrating. Self-collision
    # keeps a folded bowel loop from passing through itself.
    coll_prim = stage.GetPrimAtPath(coll_path)
    if coll_prim:
        PhysxSchema.PhysxCollisionAPI.Apply(coll_prim)
        pc = PhysxSchema.PhysxCollisionAPI(coll_prim)
        pc.CreateContactOffsetAttr().Set(ARGS.contact_offset)
        pc.CreateRestOffsetAttr().Set(ARGS.rest_offset)
    # self-collision cooking on a hex deformable is very slow (minutes per body)
    # and is not needed for organ-vs-organ contact, so leave it off by default
    attrs = [("physxDeformable:solverPositionIterationCount", ARGS.iterations)]
    if ARGS.self_collision:
        attrs += [("physxDeformable:selfCollision", True),
                  ("physxDeformable:selfCollisionFilterDistance",
                   float(ARGS.contact_offset))]
    for attr, val in attrs:
        a = root_prim.GetAttribute(attr)
        if not a:
            t = (Sdf.ValueTypeNames.Bool if isinstance(val, bool)
                 else Sdf.ValueTypeNames.Int if isinstance(val, int)
                 else Sdf.ValueTypeNames.Float)
            a = root_prim.CreateAttribute(attr, t)
        a.Set(val)

    if material:
        UsdShade.MaterialBindingAPI.Apply(root_prim).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics")
    return root_prim


def surface_faces(name: str) -> np.ndarray:
    """The tet files carry the surface triangles alongside the volume."""
    rel = dict(ORGANS)[name]
    return np.load(ROOT / rel)["surface_faces"].astype(np.int32)


def author_collider(stage, name, verts, faces, collider_material=None):
    path = f"/World/colliders/{name}"
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr().Set([Gf.Vec3f(*p) for p in verts.tolist()])
    m.CreateFaceVertexIndicesAttr().Set(faces.flatten().tolist())
    m.CreateFaceVertexCountsAttr().Set([3] * len(faces))
    m.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    set_extent(m.GetPrim(), verts)

    # the field carries the photographed colour off the RealSense frames; the
    # phantom is a moulded plastic trainer and has none
    src = {"field": ROOT / "mesh" / "field.ply",
           "endoscope": ROOT / "mesh" / "endoscope.ply"}.get(name)
    if not set_display_color(m, ply_vertex_colors(src) if src else None, len(verts)):
        m.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set(
            [Gf.Vec3f(0.34, 0.32, 0.30)])

    prim = m.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    # the field and dome are open shells; a convex hull would seal the cavity
    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set(
        UsdPhysics.Tokens.none)

    # The organs rest embedded in the field volume (they are inside the abdomen),
    # so on the first step PhysX has to resolve that overlap. With no material it
    # does so violently and flings them out. A firm-but-compliant contact -- zero
    # restitution so nothing bounces, real friction so nothing slides off, and a
    # few mm of contact/rest offset so the overlap is pushed out gently rather
    # than as a hard spike -- lets them settle instead.
    PhysxSchema.PhysxCollisionAPI.Apply(prim)
    pc = PhysxSchema.PhysxCollisionAPI(prim)
    pc.CreateContactOffsetAttr().Set(0.004)   # start resolving 4 mm out
    pc.CreateRestOffsetAttr().Set(0.001)       # hold 1 mm apart at rest

    if collider_material:
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            collider_material, UsdShade.Tokens.weakerThanDescendants, "physics")
    return prim


def make_collider_material(stage, path):
    """Firm-but-soft contact for the static shell: no bounce, real friction."""
    mat = UsdShade.Material.Define(stage, path)
    prim = mat.GetPrim()
    UsdPhysics.MaterialAPI.Apply(prim)
    api = UsdPhysics.MaterialAPI(prim)
    api.CreateStaticFrictionAttr().Set(0.6)
    api.CreateDynamicFrictionAttr().Set(0.5)
    api.CreateRestitutionAttr().Set(0.0)       # organs must not bounce off
    # compliant contact so the initial overlap unwinds softly, not as a spike
    PhysxSchema.PhysxMaterialAPI.Apply(prim)
    PhysxSchema.PhysxMaterialAPI(prim).CreateCompliantContactStiffnessAttr().Set(1.0e4)
    PhysxSchema.PhysxMaterialAPI(prim).CreateCompliantContactDampingAttr().Set(10.0)
    return mat


def force_gpu_dynamics(stage):
    """PhysX simulates deformables on the GPU only -- every scene needs the flag."""
    found = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Scene):
            continue
        px = PhysxSchema.PhysxSceneAPI.Apply(prim)
        px.CreateEnableGPUDynamicsAttr().Set(True)
        px.CreateBroadphaseTypeAttr().Set("GPU")
        px.CreateSolverTypeAttr().Set("TGS")
        found.append((prim.GetPath(), px.GetEnableGPUDynamicsAttr().Get()))
    for path, on in found:
        log(f"  physics scene {path}: GPU dynamics = {on}")
    if not found:
        log("  ! no UsdPhysics.Scene on the stage")
    return found


def build_stage(stage, backend):
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    UsdGeom.Xform.Define(stage, "/World/organs")
    UsdGeom.Xform.Define(stage, "/World/colliders")

    # Nothing here authors a light, and unlit geometry renders black against a
    # black viewport -- indistinguishable from an empty stage.
    from pxr import UsdLux
    dome = UsdLux.DomeLight.Define(stage, "/World/domeLight")
    dome.CreateIntensityAttr().Set(1000.0)
    key = UsdLux.DistantLight.Define(stage, "/World/keyLight")
    key.CreateIntensityAttr().Set(3000.0)
    key.CreateAngleAttr().Set(1.0)
    UsdGeom.Xformable(key.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 0.0, 45.0))
    log("  dome + key light")

    scene = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(9.81)

    # PhysX runs deformables on the GPU only -- without these the bodies are
    # authored correctly but never simulate.
    px = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    px.CreateEnableGPUDynamicsAttr().Set(True)
    px.CreateBroadphaseTypeAttr().Set("GPU")
    px.CreateSolverTypeAttr().Set("TGS")

    youngs = ARGS.youngs if ARGS.youngs is not None else YOUNGS_MODULUS
    poisson = ARGS.poisson if ARGS.poisson is not None else POISSON_RATIO

    log("organs:")
    total = 0
    organ_lo = np.full(3, np.inf)
    organ_hi = np.full(3, -np.inf)
    for name, verts, tets in organ_geometry(ARGS.drop_height):
        organ_lo = np.minimum(organ_lo, verts.min(axis=0))
        organ_hi = np.maximum(organ_hi, verts.max(axis=0))
        # scene.py's ORGAN_YOUNGS is the Newton pipeline's table; the Isaac cell
        # sets its own appendectomy-tuned values (see --youngs/--appendix-youngs)
        E = ARGS.appendix_youngs if name == "appendix" else youngs
        mat = make_material(stage, f"/World/materials/{name}", E, poisson,
                            TISSUE_DENSITY)
        author_organ(stage, name, verts, tets, mat)
        total += len(tets)
        log(f"  {name:9s} {len(verts):5d} nodes  {len(tets):6d} tets  E={E/1e3:.0f} kPa")

    # Solid support right under the organs. The field is a thin shell with the
    # surgical opening cut through it, so the organs -- which sit below the field
    # surface -- fall straight through the hole to the floor with nothing to rest
    # on. A plane just beneath the lowest organ vertex is the no-hole support the
    # cavity floor would provide: they settle on it instead of dropping away.
    from omni.physx.scripts import physicsUtils
    support_z = float(organ_lo[2]) - 0.002
    physicsUtils.add_ground_plane(stage, "/World/groundPlane", "Z", 0.3,
                                  Gf.Vec3f(0.0, 0.0, support_z), Gf.Vec3f(0.4))
    log(f"  support plane at z={support_z*1000:.1f} mm "
        f"(organs bottom {organ_lo[2]*1000:.1f} mm)")
    # the shell material -- firm but soft -- goes on the ground too
    collider_mat = make_collider_material(stage, "/World/materials/shell")
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath("/World/groundPlane")).Bind(
        collider_mat, UsdShade.Tokens.weakerThanDescendants, "physics")
    log("  ground plane at z=0 (0.3 m)")

    # the offset organ_geometry used, so the field can share it
    organ_offset = placement_offset(
        [load_tet(ROOT / r)[0] for _, r in ORGANS if (ROOT / r).exists()],
        ARGS.drop_height)

    log("colliders:")
    for name, verts, faces in collider_geometry(ARGS.drop_height, not ARGS.no_field,
                                                organ_offset):
        prim = author_collider(stage, name, verts, faces, collider_mat)
        # the shell encloses the organs; drawn, it is all you see from outside.
        # Hide it by default (collision is unaffected) so the organs are visible.
        if name in ("phantom", "field") and not ARGS.show_shell:
            UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
        log(f"  {name:9s} {len(verts):7d} verts  {len(faces):7d} faces")

    # Frame the organs. When the shell is hidden the camera can sit close; when
    # shown, pull back to clear the 355 mm dome.
    if ARGS.show_shell:
        lo, hi = organ_lo - 0.18, organ_hi + 0.18
    else:
        lo, hi = organ_lo.copy(), organ_hi.copy()
    centre = (lo + hi) * 0.5
    dist = max(float(np.linalg.norm(hi - lo)) * 1.4, 0.12)
    cam = UsdGeom.Camera.Define(stage, "/World/organCam")
    cam.CreateFocalLengthAttr().Set(24.0)
    cam.CreateClippingRangeAttr().Set(Gf.Vec2f(0.001, 1000.0))
    xf = UsdGeom.Xformable(cam.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(float(centre[0]),
                                     float(centre[1] - dist),
                                     float(centre[2] + dist * 0.6)))
    xf.AddRotateXYZOp().Set(Gf.Vec3f(60.0, 0.0, 0.0))
    log(f"  organCam {dist*1000:.0f} mm back from "
        f"{[round(float(v)*1000, 1) for v in centre]} mm")
    return total


def main() -> int:
    backend = deformable_backend()
    if backend is None:
        log("no deformable schema in this build -- cannot continue")
        return 1
    log(f"deformable backend: {backend}")

    out_path = ROOT / ARGS.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    total = build_stage(stage, backend)

    stage.GetRootLayer().Export(str(out_path))
    log(f"\nwrote {out_path}  ({total} tets)")

    from omni.isaac.core import SimulationContext  # noqa: E402

    # The tets are ~1.6 mm and that element size, not the mass ratio, sets the
    # timestep -- the same finding that forced Newton onto an implicit solver at
    # dt = 1/(60*8). At the SimulationContext default of 1/60 PhysX inverts
    # thousands of tets within a few frames and throws the organs 100 mm+ out of
    # place, even though every tet is well conditioned at rest.
    sim = SimulationContext(stage_units_in_meters=1.0,
                            physics_dt=1.0 / (60.0 * ARGS.substeps),
                            rendering_dt=1.0 / 60.0)

    # SimulationContext can introduce its own physics scene, which would not
    # carry the GPU flags set during authoring. Deformables are GPU-only, so
    # assert them on every scene that exists and verify it took.
    force_gpu_dynamics(stage)
    sim.initialize_physics()
    force_gpu_dynamics(stage)

    if not ARGS.static:
        sim.play()
        # Deformable cooking is asynchronous and driven by app updates, not by
        # sim.step(). Without pumping the app first the bodies never get their
        # simulation mesh and silently sit still. Let it finish before stepping.
        for _ in range(60):
            simulation_app.update()
        log("pumped app updates for async deformable cooking")
    else:
        log("static: physics authored but not stepped")

    if ARGS.bench:
        def organ_points():
            """Current vertex positions, per organ."""
            snap = {}
            for prim in stage.GetPrimAtPath("/World/organs").GetChildren():
                # measure the render mesh: with a hexahedral sim mesh the tet
                # prim is an empty placeholder and the deformation shows up here
                # measure the render mesh -- with a hex sim mesh the tet prim is
                # an empty placeholder and the deformation shows on the skin
                child = prim.GetChild("render")
                attr = child.GetAttribute("points") if child else None
                if attr and attr.Get():
                    snap[prim.GetName()] = np.array(attr.Get(), dtype=np.float64)
            return snap

        log("\ncolliders on stage:")
        for prim in stage.GetPrimAtPath("/World/colliders").GetChildren():
            has = prim.HasAPI(UsdPhysics.CollisionAPI)
            enabled = prim.GetAttribute("physics:collisionEnabled")
            approx = prim.GetAttribute("physics:approximation")
            log(f"  {prim.GetName():9s} CollisionAPI={has} "
                f"enabled={enabled.Get() if enabled else None} "
                f"approx={approx.Get() if approx else None}")

        before = organ_points()
        log("\nfall trace (lowest organ vertex, mm):")
        for i in range(6):
            for _ in range(10):
                sim.step(render=False)
            zmin = min(v[:, 2].min() for v in organ_points().values())
            log(f"  step {(i+1)*10:3d}   z_min {zmin*1000:9.1f}")
        for _ in range(10):          # warm up: cooking and first-step allocations
            sim.step(render=False)
        t0 = time.perf_counter()
        for _ in range(ARGS.bench):
            sim.step(render=False)
        dt = time.perf_counter() - t0
        after = organ_points()

        log(f"\nphysics only: {ARGS.bench} steps in {dt:.2f}s"
            f"  ->  {ARGS.bench/dt:.1f} steps/s  ({1000*dt/ARGS.bench:.1f} ms/step)")

        # A step rate this high on 44k tets usually means nothing moved. Prove
        # the bodies actually simulate rather than trusting the timing alone.
        log("\nvertex displacement after stepping:")
        moved = False
        for name in sorted(before):
            if name not in after:
                continue
            d = np.linalg.norm(after[name] - before[name], axis=1)
            log(f"  {name:9s} mean {d.mean()*1000:7.3f} mm   max {d.max()*1000:7.3f} mm")
            if d.max() > 1e-6:
                moved = True
        if not moved:
            log("\nnothing moved -- the deformables are authored but not simulating")
        simulation_app.close()
        return 0 if moved else 2

    if not ARGS.headless:
        # Point the viewport at the organs; the default camera sits metres away
        # and frames empty space at this scale.
        try:
            from omni.kit.viewport.utility import get_active_viewport
            vp = get_active_viewport()
            vp.set_active_camera("/World/organCam")
            log("viewport camera -> /World/organCam")
        except Exception as exc:
            log(f"could not set viewport camera ({exc}); "
                "select an organ in Stage and press F")

    log("\nrunning -- close the window to quit")
    while simulation_app.is_running():
        if ARGS.static:
            simulation_app.update()
        else:
            sim.step(render=True)
    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
