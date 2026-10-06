#!/usr/bin/env python3
"""Bind the live Newton cell to the path tracer and render frames.

    .venv-newton/bin/python newton_cell/view.py --out out/newton_probe

Owns the mapping from simulation state to render geometry: which mesh belongs
to which material, where the deforming organ surfaces come from, and how the
instruments' moving links get their vertices refreshed each frame.

Organ surfaces use captured CT-derived PLY topology with a render-only Loop
subdivision. Every smoothed vertex is a sparse interpolation stencil over
source render vertices, which bind to FEM nodes within a strict rest tolerance;
the same stencil follows live simulated positions. This adds no FEM degrees of
freedom and does not change contact geometry. If source correspondence fails,
the renderer falls back to the FEM boundary mesh. Smoothing is a geometric
approximation and must be assessed against held-out endoscopic views.
"""
from __future__ import annotations

import json
import math

from pathlib import Path

import numpy as np

from cameras import CellCameras
from render import (MAT_APPENDIX, MAT_FIELD, MAT_FLOOR, MAT_ORGAN, MAT_PHANTOM,
                    MAT_PLATE, MAT_ROBOT, MAT_SCOPE, MAT_SHELF, MAT_TABLE,
                    MAT_TOOL_DARK, MAT_TOOL_STEEL, PathTracer)
from appearance import (appendix300_texture_provenance,
                        appendix_texture_uv,
                        bind_surface_vertices_to_fem_nodes,
                        loop_subdivide_surface)
from cell_spec import ROOT
from urdf_meshes import link_geometry, merged


def srgb_to_linear(c):
    """0-255 sRGB -> linear, exactly build_deformable_usd.srgb_to_linear.

    The reconstruction's .ply colours are photographic sRGB bytes; the Isaac
    cell converts them once and authors the result as `displayColor`. Matching
    that conversion is what makes the two backends draw the same specimen.
    """
    c = np.asarray(c, np.float64)[:, :3] / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def ply_vertex_colors(path):
    """Per-vertex LINEAR colour from a .ply, or None if it carries none."""
    import trimesh
    if not path.exists():
        return None
    m = trimesh.load(path, process=False)
    vc = getattr(m.visual, "vertex_colors", None)
    if vc is None or len(vc) == 0:
        return None
    return srgb_to_linear(np.asarray(vc)), np.asarray(m.vertices, np.float64)


def transfer_colors(src_pts, src_col, dst_pts):
    """Nearest-neighbour colour transfer fallback between surface samplings."""
    from scipy.spatial import cKDTree
    tree = cKDTree(np.asarray(src_pts, np.float64))
    _, idx = tree.query(np.asarray(dst_pts, np.float64), k=1)
    return np.asarray(src_col)[idx]


def tet_surface(tets: np.ndarray) -> np.ndarray:
    """Boundary triangles of a tetrahedral mesh.

    A face shared by two tets is interior; one that appears once is on the
    surface. Winding is taken from the owning tet so normals point outward.
    """
    t = np.asarray(tets, np.int64).reshape(-1, 4)
    faces = np.concatenate([t[:, [0, 2, 1]], t[:, [0, 1, 3]],
                            t[:, [1, 2, 3]], t[:, [0, 3, 2]]], axis=0)
    key = np.sort(faces, axis=1)
    _, idx, counts = np.unique(key, axis=0, return_index=True, return_counts=True)
    return faces[idx[counts == 1]].astype(np.int32)


ORGAN_MATERIAL = {"appendix": MAT_APPENDIX}

# The only instrument geometry the ENDOSCOPE can see: the 5 mm shaft and the
# distal clevis cluster (the jaws are separate links). These stay at full
# resolution because quadric decimation eats a long thin tube alive -- at 6k
# faces the shafts broke into floating fragments in frame. Everything else on
# the instrument is casing that lives outside the patient and is only ever a few
# hundred pixels of the front view, so it is decimated hard: keeping it at CAD
# resolution put 658k triangles in a software-traversed BVH and took the
# endoscope to 247 s/frame.
DISTAL_MESHES = {
    "laparo_shaft.stl", "part_13.stl", "part_15.stl",
    "gripper_shaft_inner.stl", "gripper_shaft_outer.stl",
    "cutter_shaft_inner.stl", "cutter_shaft_outer.stl",
}


class CellRenderer:
    """A PathTracer kept in sync with a `sim.Cell`."""

    def __init__(self, cell, samples=48, bounces=1, exposure=1.0, device=None,
                 decimate_to=60000, robot_faces=6000, hole_radius=0.035,
                 hole_z=0.12, quiet=False, appendix_texture=None,
                 appendix_texture_mix=1.0, surface_subdivision_levels=1):
        self.cell = cell
        self.scene = cell.scene
        self.decimate_to = int(decimate_to)
        # The robot and instrument shells are SMOOTH and small in frame; their
        # CAD tessellation is not. Undecimated they add ~700k triangles (FR3
        # links ~160k/arm, the tool_flange visual 193k/arm) to a BVH traversed
        # in software, and that slowed EVERY camera -- one control step went
        # 35 s -> 260 s when they were added for the front view.
        self.robot_faces = int(robot_faces)
        # Decimation is per BODY and proportional: keep 1/body_ratio of a body's
        # faces, never below min_body_faces. One flat budget over a merged
        # multi-body mesh shatters the small parts.
        self.body_ratio = 6
        self.min_body_faces = 1200
        self.hole_radius = float(hole_radius)
        self.hole_z = float(hole_z)
        self._quiet = quiet
        self.appendix_texture = (Path(appendix_texture).expanduser()
                                 if appendix_texture else None)
        self.appendix_texture_mix = float(appendix_texture_mix)
        if not 0.0 <= self.appendix_texture_mix <= 1.0:
            raise ValueError("appendix_texture_mix must be in [0, 1]")
        if self.appendix_texture is not None and not self.appendix_texture.is_file():
            raise FileNotFoundError(f"Appendix texture not found: {self.appendix_texture}")
        if (isinstance(surface_subdivision_levels, bool) or
                not np.isfinite(surface_subdivision_levels) or
                int(surface_subdivision_levels) != surface_subdivision_levels or
                not 0 <= surface_subdivision_levels <= 3):
            raise ValueError("surface_subdivision_levels must be an integer in [0, 3]")
        self.surface_subdivision_levels = int(surface_subdivision_levels)
        self.appendix_texture_source = (
            appendix300_texture_provenance(self.appendix_texture)
            if self.appendix_texture is not None else None)
        self.tracer = PathTracer(device=device, samples=samples,
                                 bounces=bounces, exposure=exposure)
        # TWO scenes, because the two clinical cameras sit INSIDE the phantom
        # dome and physically cannot see the robot, the table or the room. Left
        # in one BVH, the FR3 shells and the instrument casing put ~700k extra
        # triangles in front of every endoscope ray for nothing, and decimating
        # them instead wrecked them (trimesh's quadric decimation shatters this
        # CAD geometry however the budget is split). Splitting is both faster
        # and lossless.
        self.tracer_room = PathTracer(device=device, samples=samples,
                                      bounces=bounces, exposure=exposure)
        self._organ_faces = {}
        self._organ_render_nodes = {}
        self._organ_render_stencils = {}
        self._organ_binding_reports = {}
        self._arm_parts = self._build(self.tracer, room=False)
        self._room_parts = self._build(self.tracer_room, room=True)
        q = cell.state_0.particle_q.numpy()
        centroid = self._organ_centroid(q)
        self.cams = CellCameras(cell.L, centroid)

    def _log(self, msg):
        if not self._quiet:
            print(msg, flush=True)

    # ------------------------------------------------------------------
    def _organ_centroid(self, q):
        pts = [q[i:j] for (i, j) in self.scene.organ_span.values()]
        allp = np.vstack(pts)
        return 0.5 * (allp.min(axis=0) + allp.max(axis=0))

    def _cut_view_hole(self, v, f):
        """Open the phantom shell over the D405, as the Isaac cell does.

        `cut_phantom_view_hole` exists there because the scanned dome is much
        thicker than the real ~3 cm hole at its top, so it occludes the overhead
        camera. Without it the D405 renders the inside of the shell 20 mm from
        the lens instead of the organs 120 mm below (measured: depth came back
        16-23 mm).
        """
        d = self.cell.L.d405_position
        c = v[f].mean(axis=1)
        drop = (np.linalg.norm(c[:, :2] - d[:2], axis=1) < self.hole_radius) \
            & (c[:, 2] > self.hole_z)
        return f[~drop], int(drop.sum())

    @staticmethod
    def _decimate(v, f, target):
        """Reduce a scanned collider to a render-sane triangle count.

        The BVH is traversed in software (no RT cores), so triangle count is
        the renderer's main cost. `field.ply` ships 458k faces of scanned
        surface; at 60k it is visually identical at these distances and the
        whole trace gets several times faster.
        """
        if len(f) <= target:
            return v, f
        try:
            import trimesh
            m = trimesh.Trimesh(vertices=v, faces=f, process=False)
            m = m.simplify_quadric_decimation(face_count=target)
            return (np.asarray(m.vertices, np.float64),
                    np.asarray(m.faces, np.int32))
        except Exception:
            return v, f                       # decimation is an optimisation only

    def _build(self, t, room):
        vm = self.scene.visual_meshes
        static_mat = {"field": MAT_FIELD, "phantom": MAT_PHANTOM,
                      "endoscope": MAT_SCOPE, "appendix_shelf": MAT_SHELF,
                      "organ_support": MAT_SHELF}
        for name, mat in static_mat.items():
            if name not in vm:
                continue
            v, f = vm[name]
            col = None
            if name == "field":
                # only the field: at 458k faces it dominates the trace. The
                # phantom is left alone -- decimating its shell welded the inner
                # and outer walls together in places and the dome came back
                # speckled with z-fighting facets.
                v0 = v.copy()
                v, f = self._decimate(v, f, self.decimate_to)
                got = ply_vertex_colors(ROOT / "mesh" / "field.ply")
                if got is not None:
                    col = transfer_colors(v0, got[0], v)
            if name == "phantom":
                f, n = self._cut_view_hole(v, f)
                self._log(f"  phantom view hole: dropped {n} face(s) over the D405")
            t.add(name, v, f, mat, colors=col)

        # Draw each captured surface mesh while binding its vertices directly
        # to the matching FEM rest nodes. This preserves the source render
        # topology and colours without adding a second deforming-body model.
        # If a source mesh cannot be bound uniquely within tolerance, fall back
        # to the tet boundary rather than letting appearance geometry drift.
        from scene import load_tet
        q = self.cell.state_0.particle_q.numpy()
        for name, (i, j) in self.scene.organ_span.items():
            _, idx = load_tet(ROOT / "tet" / f"{name}.npz")
            faces = tet_surface(idx)
            render_nodes = np.arange(j - i, dtype=np.int32)
            render_vertices = q[i:j]
            col = None
            import trimesh
            mesh_path = ROOT / "mesh" / f"organ_{name}.ply"
            if mesh_path.is_file():
                captured = trimesh.load(mesh_path, process=False)
                cap_pts = np.asarray(captured.vertices, np.float64)
                cap_faces = np.asarray(captured.faces, np.int32)
                if (captured.is_watertight and captured.is_winding_consistent and
                        cap_faces.ndim == 2 and cap_faces.shape[1] == 3):
                    cap_world = self.cell.L.organs_to_world(cap_pts * 0.001)
                    try:
                        render_nodes, bind_report = bind_surface_vertices_to_fem_nodes(
                            cap_world, self.cell.q_rest[i:j])
                    except ValueError as exc:
                        self._log(f"  {name} captured surface binding rejected: {exc}; "
                                  "using FEM boundary render")
                        got = ply_vertex_colors(mesh_path)
                        if got is not None:
                            col = transfer_colors(cap_world, got[0], q[i:j])
                    else:
                        faces = cap_faces
                        render_vertices = q[i:j][render_nodes]
                        self._organ_binding_reports[name] = bind_report
                        self._log(f"  {name} render surface: "
                                  f"{bind_report['render_vertices']} captured vertices "
                                  f"bound to FEM nodes; max rest error "
                                  f"{bind_report['max_rest_error_m']*1e6:.3f} um")
                        colors = getattr(captured.visual, "vertex_colors", None)
                        if colors is not None and len(colors) == len(cap_pts):
                            col = srgb_to_linear(np.asarray(colors))
                else:
                    self._log(f"  {name} captured surface rejected: mesh is not a "
                              "closed, consistently wound triangle surface")
                    got = ply_vertex_colors(mesh_path)
                    if got is not None:
                        cap_col, cap_pts = got
                        cap_world = self.cell.L.organs_to_world(cap_pts * 0.001)
                        col = transfer_colors(cap_world, cap_col, q[i:j])
            self._organ_render_nodes[name] = render_nodes
            rest_vertices = self.cell.q_rest[i:j][render_nodes]
            base_colors = col
            if self.surface_subdivision_levels:
                (rest_vertices, faces, stencil_indices, stencil_weights,
                 subdivision_report) = loop_subdivide_surface(
                    rest_vertices, faces, self.surface_subdivision_levels)
                if base_colors is not None:
                    base_colors = np.einsum(
                        "nk,nkj->nj", stencil_weights,
                        np.asarray(base_colors, np.float32)[stencil_indices],
                        optimize=True).astype(np.float32)
                if name in self._organ_binding_reports:
                    self._organ_binding_reports[name]["surface_subdivision"] = subdivision_report
            else:
                stencil_indices = render_nodes[:, None].astype(np.int32)
                stencil_weights = np.ones((len(render_nodes), 1), np.float32)
            particle_stencil_indices = render_nodes[stencil_indices]
            self._organ_render_stencils[name] = (particle_stencil_indices,
                                                  stencil_weights)
            self._organ_faces[name] = faces
            live_base = q[i:j]
            render_vertices = np.einsum(
                "nk,nkj->nj", stencil_weights,
                live_base[particle_stencil_indices],
                optimize=True).astype(np.float32)
            col = base_colors
            if name == "appendix" and self.appendix_texture is not None:
                from PIL import Image
                image = np.asarray(Image.open(self.appendix_texture).convert("RGB"),
                                   np.float32) / 255.0
                image = np.where(image <= 0.04045, image / 12.92,
                                 ((image + 0.055) / 1.055) ** 2.4).astype(np.float32)
                uv = appendix_texture_uv(rest_vertices)
                self._log("  appendix appearance: Appendix300 RGB candidate "
                          f"({self.appendix_texture.name}), high-resolution wrapped UV sampling "
                          "from rest coordinates; not video-to-3D registration")
            else:
                image, uv = None, None
            t.add(f"organ_{name}", render_vertices, faces,
                  ORGAN_MATERIAL.get(name, MAT_ORGAN), colors=col,
                  texcoords=uv, texture_data=image,
                  texture_mix=self.appendix_texture_mix if image is not None else 0.0)

        self._add_furniture(t, room)

        # instruments: every VISUAL mesh of the links that move, split by the
        # material the URDF gives them (gloss black housing, steel endtip)
        parts = []
        for aname, arm in self.cell.arms.items():
            vis = link_geometry(ROOT / arm.cfg["urdf"], "visual")
            for link, items in vis.items():
                if link.startswith("fr3_link"):
                    if not room:
                        continue          # not visible from inside the dome
                    v, f = merged(items)
                    key = f"{aname}_{link}"
                    t.add(key, v, f, MAT_ROBOT)
                    parts.append((key, aname, link, v.copy()))
                    continue
                steel = [g for g in items if g.material and g.material[0] > 0.5]
                dark = [g for g in items if not (g.material and g.material[0] > 0.5)]
                for tag, sel, mat in (("steel", steel, MAT_TOOL_STEEL),
                                      ("dark", dark, MAT_TOOL_DARK)):
                    if not sel:
                        continue
                    use = sel
                    if not room and tag == "dark" and link == "tool_flange":
                        # The casing lives outside the patient; from inside the
                        # dome only the shaft and the distal cluster are ever in
                        # frame, and dropping the rest takes 193k triangles per
                        # arm out of the clinical BVH.
                        use = [g for g in sel
                               if Path(g.source).name in DISTAL_MESHES]
                        if not use:
                            continue
                    v, f = merged(use)
                    key = f"{aname}_{link}_{tag}"
                    t.add(key, v, f, mat)
                    parts.append((key, aname, link, v.copy()))
        t.build()
        return parts

    @staticmethod
    def _box(cx, cy, cz, sx, sy, sz):
        h = np.array([sx, sy, sz]) / 2.0
        c = np.array([cx, cy, cz])
        v = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                      [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]]) * h + c
        f = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                      [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
                      [1, 2, 6], [1, 6, 5], [0, 4, 7], [0, 7, 3]], np.int32)
        return v, f

    @staticmethod
    def _stack(parts):
        vs, fs, n = [], [], 0
        for v, f in parts:
            vs.append(v)
            fs.append(f + n)
            n += len(v)
        return np.vstack(vs), np.vstack(fs).astype(np.int32)

    def _decimate_each(self, geoms):
        """Decimate every body SEPARATELY, in proportion to its own size.

        Merging first and then decimating to one budget destroys the small
        bodies: at a 6k budget over the 193k merged instrument the servo housing
        and its linkages came back as floating shards, and the FR3 shells went
        visibly faceted. Per-body with a floor keeps every part recognisable at
        a fraction of the CAD tessellation.
        """
        return [self._decimate(g.verts, g.faces,
                               max(self.min_body_faces,
                                   len(g.faces) // self.body_ratio))
                for g in geoms]

    def _decimate_housing(self, items, sel):
        """Full resolution for the distal geometry, decimate all the casing."""
        keep = [g for g in sel if Path(g.source).name in DISTAL_MESHES]
        drop = [g for g in sel if Path(g.source).name not in DISTAL_MESHES]
        return self._stack([(g.verts, g.faces) for g in keep]
                           + self._decimate_each(drop))

    def _add_furniture(self, t, room):
        """Specimen plate always; table and floor only for the overview.

        The PLATE is the blue surgical drape the specimen rests on, and it is
        very much visible from inside the dome -- it is the blue surround in
        every reference endoscope frame. It is one box, so it costs the clinical
        BVH nothing. Leaving it out (which the first version of the two-scene
        split did) turns that surround black and the view stops matching the
        Isaac render.

        The table and the floor genuinely are invisible from in there, and they
        are large, so they stay in the overview scene only.
        """
        C, D, L = self.scene.C, self.scene.D, self.cell.L
        px, py = C["PLATE_SIZE"][0], C["PLATE_SIZE"][1]
        lift = D["--lift"]
        v, f = self._box(0.0, 0.0, lift / 2.0, px, py, max(lift, 1e-3))
        t.add("plate", v, f, MAT_PLATE,
              colors=np.tile(np.asarray(C["PLATE_COLOR"], np.float32), (len(v), 1)))
        if not room:
            return
        tx, _, tz = C["TABLE_SIZE"]
        depth = L.table_depth
        cy = L.base_y + D["--table-shift"]
        v, f = self._box(0.0, cy, -tz / 2.0, tx, depth, tz)
        t.add("table", v, f, MAT_TABLE,
              colors=np.tile(np.asarray(C["ALU"], np.float32), (len(v), 1)))
        v, f = self._box(0.0, cy, -tz - 0.005, 3.0, 2.4, 0.01)
        t.add("floor", v, f, MAT_FLOOR,
              colors=np.tile(np.asarray(C["FLOOR_COLOR"], np.float32), (len(v), 1)))

    # ------------------------------------------------------------------
    def sync(self):
        """Refresh every moving surface in both scenes."""
        q = self.cell.state_0.particle_q.numpy()
        frames = {a: arm.link_frames() for a, arm in self.cell.arms.items()}
        for tracer, parts in ((self.tracer, self._arm_parts),
                              (self.tracer_room, self._room_parts)):
            for name, (i, j) in self.scene.organ_span.items():
                stencil_indices, stencil_weights = self._organ_render_stencils[name]
                live = np.einsum("nk,nkj->nj", stencil_weights,
                                 q[i:j][stencil_indices], optimize=True)
                tracer.update(f"organ_{name}", live)
            for key, aname, link, v0 in parts:
                T = frames[aname][link]
                tracer.update(key, v0 @ T[:3, :3].T + T[:3, 3])
            tracer.commit()

    def render_endoscope(self, samples=None):
        # MAT_SCOPE hidden: the objective sits on the tip's bevelled end face,
        # so the barrel is behind it and a real laparoscope never images itself.
        # Measured with it visible: 27 % of the frame, at zero distance from the
        # tip light, blown to white.
        return self.tracer.render(self.cams.endoscope,
                                  self.cams.endoscope_light,
                                  samples=samples, want_depth=False,
                                  hide=(MAT_SCOPE,))[0]

    def render_d405(self, samples=None, want_depth=True):
        return self.tracer.render(self.cams.d405, self.cams.endoscope_light,
                                  samples=samples, want_depth=want_depth)

    def render_front(self, samples=None, bounces=None):
        """The overview: both arms, the phantom, the table.

        Lit by a broad key light at the camera plus a strong ambient, not by the
        scope tip -- the tip light is inside the dome and would leave the room
        black. This is the cell-level view the reference render pairs with the
        endoscope feed.
        """
        cam = self.cams.front
        key = dict(pos=cam.T[:3, 3] + np.array([0.0, 0.0, 0.8]),
                   dir=np.array([0.0, 0.35, -0.94]),
                   color=np.array([1.0, 0.98, 0.95]),
                   intensity=6.0, radius=0.25,
                   cos_outer=math.cos(math.radians(85.0)), softness=0.6)
        # The overview is ambient-dominated and almost noise-free, so it needs
        # far fewer samples than the endoscope -- which matters, because it also
        # has the whole robot in frame. Rendered at the scope's 48 spp with a
        # bounce it cost 365 s/frame; 8 spp direct-only is visually the same
        # here and ~40x cheaper.
        t = self.tracer_room
        keep = t.bounces
        if bounces is not None:
            t.bounces = int(bounces)
        try:
            return t.render(cam, key, ambient=(0.35, 0.35, 0.37),
                            samples=samples, want_depth=False,
                            exposure=0.9)[0]
        finally:
            t.bounces = keep

    def render_d405_right(self, baseline, samples=None):
        return self.tracer.render(self.cams.d405_right(baseline),
                                  self.cams.endoscope_light,
                                  samples=samples, want_depth=False)[0]


def main(argv=None) -> int:
    import argparse
    import time
    from pathlib import Path

    import warp as wp
    import imageio.v3 as iio

    from scene import add_args, log
    import sim as simmod

    ap = add_args(argparse.ArgumentParser(description=__doc__))
    for name, kw in (("--seconds", dict(type=float, default=0.0)),
                     ("--physics-hz", dict(type=float, default=480.0)),
                     ("--control-hz", dict(type=float, default=30.0)),
                     ("--manip-hz", dict(type=float, default=60.0)),
                     ("--insert", dict(type=float, default=0.10)),
                     ("--min-depth", dict(type=float, default=0.10)),
                     ("--cmd-leash-m", dict(type=float, default=0.015)),
                     ("--gripper-rate-dps", dict(type=float, default=250.0))):
        ap.add_argument(name, **kw)
    ap.add_argument("--out", default="out/newton_probe")
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--decimate-to", type=int, default=60000)
    ap.add_argument("--bounces", type=int, default=1)
    ap.add_argument("--exposure", type=float, default=6.0,
                    help="tone-map scale. 6.0 puts the tissue at luma "
                         "p50 165 / p90 212, in the range the Isaac cell "
                         "documents for the same view (field 161-164, dome "
                         "143-164). Calibrated AFTER the scope barrel and the "
                         "stray support plane stopped dominating the frame")
    ap.add_argument("--appendix300-texture", default=None,
                    help="optional local Appendix300 RGB candidate; visual-only "
                         "cylindrical UV projection onto appendix FEM rest nodes")
    ap.add_argument("--appendix300-texture-mix", type=float, default=1.0,
                    help="blend ray-sampled video RGB with CT colours: 0=CT, 1=video")
    ap.add_argument("--surface-subdivision-levels", type=int, default=1,
                    help="render-only Loop smoothing levels (0 disables; FEM/contact mesh is unchanged)")
    a = ap.parse_args(argv)

    wp.init()
    log(f"device: {wp.get_device(a.device)}")
    cell = simmod.Cell(a)
    if a.seconds > 0:
        n = int(round(a.seconds * a.control_hz))
        for k in range(n):
            cell.control_substep(simmod.demo_dive(cell, k, n))
        log(f"stepped {n} sub-steps ({a.seconds:.2f} s)")

    r = CellRenderer(cell, samples=a.samples, bounces=a.bounces,
                     exposure=a.exposure, decimate_to=a.decimate_to,
                     appendix_texture=a.appendix300_texture,
                     appendix_texture_mix=a.appendix300_texture_mix,
                     surface_subdivision_levels=a.surface_subdivision_levels)
    log(r.cams.describe())
    r.sync()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    q_live = cell.state_0.particle_q.numpy()
    live_bindings = {}
    for name, (i, j) in cell.scene.organ_span.items():
        stencil_indices, stencil_weights = r._organ_render_stencils[name]
        live_surface = np.einsum("nk,nkj->nj", stencil_weights,
                                 q_live[i:j][stencil_indices], optimize=True)
        rest_surface = np.einsum("nk,nkj->nj", stencil_weights,
                                 cell.q_rest[i:j][stencil_indices], optimize=True)
        displacement = np.linalg.norm(live_surface - rest_surface, axis=1) * 1000.0
        record = dict(r._organ_binding_reports.get(name, {
            "render_vertices": int(len(live_surface)), "unique_binding": False,
            "binding": "FEM boundary fallback",
        }))
        record.update({
            "live_render_vertex_displacement_p95_mm": float(
                np.quantile(displacement, 0.95)),
            "live_render_vertex_displacement_max_mm": float(displacement.max()),
        })
        live_bindings[name] = record
    (out / "render-surface-bindings.json").write_text(json.dumps({
        "status": "runtime_render_surface_binding",
        "simulation_geometry_unchanged": True,
        "binding": "render vertices use fixed Loop stencils over source vertices mapped to live FEM particles",
        "surface_subdivision_levels": r.surface_subdivision_levels,
        "contact_geometry_changed": False,
        "organs": live_bindings,
        "fallback_organs": sorted(set(r.scene.organ_span) -
                                   set(r._organ_binding_reports)),
        "privacy": "Local aggregate binding metadata only; no raw images or absolute paths.",
    }, indent=2) + "\n", encoding="utf-8")
    t0 = time.time()
    endo = r.render_endoscope()
    t1 = time.time()
    d405, depth = r.render_d405()
    t2 = time.time()
    front = r.render_front()
    t3 = time.time()
    iio.imwrite(out / "endoscope.png", endo)
    iio.imwrite(out / "d405.png", d405)
    iio.imwrite(out / "front.png", front)
    np.save(out / "d405_depth.npy", depth)
    if r.appendix_texture is not None:
        (out / "appendix-appearance.json").write_text(json.dumps({
            **r.appendix_texture_source,
            "source_texture": r.appendix_texture.name,
            "mapping": "cylindrical UVs on FEM rest nodes; full-resolution RGB sampled per ray hit",
            "texture_mix_with_captured_vertex_colours": r.appendix_texture_mix,
            "deforms_with_fem_nodes": True,
            "source_roi_anatomy_verified": False,
            "video_to_3d_registration": False,
            "measured_albedo_or_brdf": False,
            "render_surface_binding": {
                "source_node_binding": r._organ_binding_reports.get("appendix"),
                "surface_subdivision_levels": r.surface_subdivision_levels,
                "simulation_geometry_changed": False,
                "contact_geometry_changed": False,
            },
            "clinical_visual_validation": False,
            "privacy": "Local texture and rendered frames remain in the local ignored output directory.",
        }, indent=2) + "\n", encoding="utf-8")
    log(f"front     {front.shape} in {t3-t2:.2f} s (mean luma {front.mean():.1f})")
    log(f"\nendoscope {endo.shape} in {t1-t0:.2f} s  "
        f"(mean luma {endo.mean():.1f})")
    log(f"d405      {d405.shape} in {t2-t1:.2f} s  "
        f"(mean luma {d405.mean():.1f}, depth "
        f"{np.nanmin(depth[depth>0])*1000:.0f}..{depth.max()*1000:.0f} mm)")
    log(f"wrote {out}/endoscope.png, d405.png, d405_depth.npy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
