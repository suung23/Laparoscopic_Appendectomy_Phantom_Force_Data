#!/usr/bin/env python3
"""Headless path-traced rendering, on a GPU with no RT cores.

Isaac renders this cell with RTX, which needs hardware ray-tracing cores. An
H100 has none, which is the whole reason this backend exists (VLA_HANDOFF.md
1.1). Newton's own viewers do not fill the gap either: `ViewerRTX` is Omniverse
OVRTX and needs the same RT cores, and `ViewerGL`/`ViewerUSD` are rasterisers,
which is a visible step down in exactly the thing that matters here -- how light
behaves inside an abdomen lit by a single fibre at the end of a scope.

So this is a path tracer written in Warp kernels. It needs no RT cores, no
Vulkan, no EGL, no X server and no root -- just CUDA, which is all this host
offers. `wp.mesh_query_ray` gives BVH traversal; the light transport is here.

Why a path tracer is the right shape for THIS scene, not just a nicer one:

  * A laparoscope carries its own light. The dominant source is a small emitter
    at the scope tip, a few centimetres from wet tissue, so the image is made
    almost entirely of inverse-square falloff, soft shadows from a finite-size
    source, and specular sheen. A rasteriser with a point light and a shadow map
    gets none of those right, and they are most of what makes an endoscopic
    frame look like one.
  * The organs deform every frame. Re-fitting a BVH over moving vertices is
    cheap (`wp.Mesh.refit`), so deformation costs nothing structural.

The camera model is the CALIBRATED pinhole, not a convenient one: the endoscope
uses the ChArUco intrinsics `build_surgical_cell.py` authors
(fx=684.128, ppx=611.381, ppy=297.075 at 1280x720), with square pixels, because
that is what the Isaac renderer actually draws with (see its --endo-intrinsics
note). `K` is reported in the observation so a policy sees the same projection
on both machines.
"""
from __future__ import annotations

import math

import numpy as np
import warp as wp

# ---------------------------------------------------------------------------
# material table: albedo (linear), roughness, specular, and a subsurface-ish
# wrap term that keeps kPa-soft tissue from reading like painted plastic
# ---------------------------------------------------------------------------
MAT_FIELD, MAT_ORGAN, MAT_APPENDIX, MAT_PHANTOM, MAT_TOOL_DARK, \
    MAT_TOOL_STEEL, MAT_SCOPE, MAT_SHELF, \
    MAT_TABLE, MAT_PLATE, MAT_FLOOR, MAT_ROBOT = range(12)

MATERIALS = np.array([
    # albedo r, g, b            rough  spec  wrap
    [0.45, 0.13, 0.11, 0.55, 0.25, 0.40],   # field: surgical bed, deep red
    [0.55, 0.17, 0.15, 0.30, 0.45, 0.45],   # organ: wet serosa, glossy
    [0.62, 0.23, 0.21, 0.28, 0.50, 0.50],   # appendix: the target, a touch paler
    [0.95, 0.95, 0.95, 0.70, 0.10, 0.10],   # phantom: white moulded plastic
    [0.045, 0.045, 0.050, 0.35, 0.10, 0.0],  # tool housing / shaft: gloss black
    [0.780, 0.790, 0.820, 0.18, 0.45, 0.0],  # endtip: stainless steel
    [0.10, 0.10, 0.11, 0.30, 0.55, 0.0],    # endoscope barrel
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],         # shelf: invisible helper
    [0.90, 0.91, 0.93, 0.25, 0.35, 0.0],    # table: brushed aluminium
    [0.21, 0.45, 0.66, 0.60, 0.10, 0.0],    # specimen plate: surgical drape
    [0.50, 0.50, 0.50, 0.85, 0.03, 0.0],    # room floor
    [0.88, 0.88, 0.89, 0.35, 0.25, 0.0],    # FR3 shells: white
], dtype=np.float32)

# The albedo columns above are only a FALLBACK: any part added with captured
# per-vertex colours overrides them (see PathTracer.add). They still matter for
# the geometry this repo authors rather than reconstructs -- table, plate, tools.
VISIBLE = np.array([1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1], dtype=np.int32)


@wp.func
def _srgb(x: float) -> float:
    if x <= 0.0031308:
        return 12.92 * x
    return 1.055 * wp.pow(x, 1.0 / 2.4) - 0.055


@wp.func
def _onb(n: wp.vec3):
    """Build a tangent frame around n (Duff et al., branchless)."""
    s = 1.0
    if n[2] < 0.0:
        s = -1.0
    a = -1.0 / (s + n[2])
    b = n[0] * n[1] * a
    t = wp.vec3(1.0 + s * n[0] * n[0] * a, s * b, -s * n[0])
    bt = wp.vec3(b, s + n[1] * n[1] * a, -n[1])
    return t, bt


@wp.func
def _cos_hemi(n: wp.vec3, r1: float, r2: float) -> wp.vec3:
    t, b = _onb(n)
    r = wp.sqrt(r1)
    phi = 2.0 * 3.14159265 * r2
    x = r * wp.cos(phi)
    y = r * wp.sin(phi)
    z = wp.sqrt(wp.max(0.0, 1.0 - r1))
    return wp.normalize(t * x + b * y + n * z)


@wp.kernel
def _trace(
    mesh: wp.uint64,
    tri_mat: wp.array(dtype=wp.int32),
    materials: wp.array2d(dtype=wp.float32),
    visible: wp.array(dtype=wp.int32),
    vcol: wp.array(dtype=wp.vec3),
    indices: wp.array(dtype=wp.int32),
    # camera
    cam_pos: wp.vec3, cam_r: wp.vec3, cam_u: wp.vec3, cam_f: wp.vec3,
    fx: float, fy: float, cx: float, cy: float,
    width: int, height: int,
    # tip light
    l_pos: wp.vec3, l_dir: wp.vec3, l_color: wp.vec3,
    l_intensity: float, l_radius: float, l_cos_outer: float, l_softness: float,
    # ambient fill
    amb: wp.vec3,
    samples: int, bounces: int, seed: int, exposure: float, max_t: float,
    out_rgb: wp.array3d(dtype=wp.float32),
    out_depth: wp.array2d(dtype=wp.float32),
):
    px, py = wp.tid()
    if px >= width or py >= height:
        return
    state = wp.rand_init(seed, px * 7919 + py * 104729)
    acc = wp.vec3(0.0, 0.0, 0.0)
    depth_out = float(0.0)

    for s in range(samples):
        # jittered pinhole ray, OpenCV convention (+x right, +y down, +z fwd)
        jx = wp.randf(state) - 0.5
        jy = wp.randf(state) - 0.5
        if samples == 1:
            jx = 0.0
            jy = 0.0
        sx = (float(px) + 0.5 + jx - cx) / fx
        sy = (float(py) + 0.5 + jy - cy) / fy
        d = wp.normalize(cam_f + cam_r * sx + cam_u * sy)
        o = cam_pos
        throughput = wp.vec3(1.0, 1.0, 1.0)
        radiance = wp.vec3(0.0, 0.0, 0.0)

        for b in range(bounces + 1):
            # Advance to the first VISIBLE surface. Invisible geometry (the
            # appendix shelf, and the scope barrel when rendering from the
            # scope's own camera) is passed straight through, and doing that in
            # an inner loop matters: skipping it with `continue` on the bounce
            # loop spends a bounce on a surface that was never drawn, which with
            # bounces=0 returned nothing at all.
            t = float(0.0)
            face = int(-1)
            hit_u = float(0.0)
            hit_v = float(0.0)
            n = wp.vec3(0.0, 0.0, 0.0)
            travelled = float(0.0)
            found = int(0)
            for _ in range(8):
                query = wp.mesh_query_ray(mesh, o, d, max_t)
                if not query.result:
                    break
                if visible[tri_mat[query.face]] != 0:
                    t = query.t
                    face = query.face
                    n = wp.normalize(query.normal)
                    hit_u = query.u
                    hit_v = query.v
                    travelled = travelled + query.t
                    found = 1
                    break
                o = o + d * (query.t + 1.0e-4)
                travelled = travelled + query.t + 1.0e-4
            if found == 0:
                radiance = radiance + wp.cw_mul(throughput, amb)
                break
            if wp.dot(n, d) > 0.0:
                n = -n
            p = o + d * t + n * 1.0e-5

            if b == 0:
                depth_out = depth_out + travelled

            m = tri_mat[face]
            # Albedo comes from the CAPTURED per-vertex colour, interpolated
            # over the hit triangle; the material row supplies only the finish
            # (roughness / specular / wrap). That is what the Isaac cell draws
            # with too -- build_deformable_usd.py authors the reconstruction's
            # .ply colours as `displayColor`, sRGB->linear. Using a flat
            # per-material albedo instead turned the olive/tan specimen a
            # uniform pink.
            i0 = indices[face * 3 + 0]
            i1 = indices[face * 3 + 1]
            i2 = indices[face * 3 + 2]
            bu = hit_u
            bv = hit_v
            alb = (vcol[i0] * (1.0 - bu - bv) + vcol[i1] * bu + vcol[i2] * bv)
            rough = materials[m, 3]
            spec = materials[m, 4]
            wrap = materials[m, 5]

            # --- direct lighting from the scope tip -------------------------
            # sampled over the emitter's finite radius, which is what makes the
            # shadows soft rather than stencil-hard
            lp = l_pos + wp.vec3(wp.randf(state) - 0.5, wp.randf(state) - 0.5,
                                 wp.randf(state) - 0.5) * (2.0 * l_radius)
            to_l = lp - p
            dist = wp.length(to_l)
            wi = to_l / wp.max(dist, 1.0e-6)

            # cone: the fibre bundle emits into a wide but finite cone
            ca = wp.dot(-wi, l_dir)
            cone = wp.clamp((ca - l_cos_outer) / wp.max(1.0e-4, l_softness *
                                                        (1.0 - l_cos_outer)),
                            0.0, 1.0)
            if cone > 0.0:
                lit = float(1.0)
                so = wp.vec3(p[0], p[1], p[2])
                remain = float(dist - 1.0e-4)
                for _ in range(4):            # skip invisible occluders
                    shadow = wp.mesh_query_ray(mesh, so, wi, remain)
                    if not shadow.result:
                        break
                    if visible[tri_mat[shadow.face]] != 0:
                        lit = 0.0
                        break
                    so = so + wi * (shadow.t + 1.0e-4)
                    remain = remain - shadow.t - 1.0e-4
                    if remain <= 0.0:
                        break
                if lit > 0.0:
                    ndl = wp.dot(n, wi)
                    # wrapped diffuse: a cheap stand-in for the light that
                    # scatters a millimetre inside wet tissue and comes back out
                    diff = wp.clamp((ndl + wrap) / (1.0 + wrap), 0.0, 1.0)
                    fall = l_intensity / wp.max(1.0e-6, dist * dist)
                    e = wp.cw_mul(l_color, wp.vec3(fall, fall, fall)) * cone
                    radiance = radiance + wp.cw_mul(
                        wp.cw_mul(throughput, alb), e) * diff
                    # Blinn-Phong lobe: the wet sheen the scope light throws back
                    h = wp.normalize(wi - d)
                    nh = wp.max(0.0, wp.dot(n, h))
                    shin = 2.0 / wp.max(1.0e-4, rough * rough * rough) - 2.0
                    sv = spec * wp.pow(nh, shin) * (shin + 8.0) / 25.13274
                    sv = sv * wp.max(0.0, ndl)
                    radiance = radiance + wp.cw_mul(throughput, e) * sv

            radiance = radiance + wp.cw_mul(wp.cw_mul(throughput, alb), amb)

            if b >= bounces:
                break
            # --- one cosine-weighted diffuse bounce -------------------------
            d = _cos_hemi(n, wp.randf(state), wp.randf(state))
            o = p
            throughput = wp.cw_mul(throughput, alb)

        acc = acc + radiance

    inv = 1.0 / float(samples)
    c = acc * inv * exposure
    # Tone-map LUMINANCE, not each channel. A per-channel Reinhard compresses
    # the strongest channel hardest, which desaturates exactly the reds that
    # carry this image -- 0.55/0.17/0.15 serosa came back pale pink. Scaling
    # all three by one luminance ratio keeps the hue and only pulls the range in.
    lum = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    k = 1.0
    if lum > 1.0e-6:
        k = (lum / (1.0 + lum)) / lum
    out_rgb[py, px, 0] = _srgb(wp.clamp(c[0] * k, 0.0, 1.0))
    out_rgb[py, px, 1] = _srgb(wp.clamp(c[1] * k, 0.0, 1.0))
    out_rgb[py, px, 2] = _srgb(wp.clamp(c[2] * k, 0.0, 1.0))
    out_depth[py, px] = depth_out * inv


class Camera:
    """A calibrated pinhole, OpenCV convention: +x right, +y down, +z forward."""

    def __init__(self, width, height, fx, fy, cx, cy, T_world_cam):
        self.width, self.height = int(width), int(height)
        self.fx, self.fy, self.cx, self.cy = float(fx), float(fy), float(cx), float(cy)
        self.T = np.asarray(T_world_cam, np.float64)

    @staticmethod
    def from_hfov(width, height, hfov_deg, T_world_cam):
        f = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        return Camera(width, height, f, f, width / 2.0, height / 2.0, T_world_cam)

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]], np.float32)

    def basis(self):
        R = self.T[:3, :3]
        return R[:, 0], R[:, 1], R[:, 2], self.T[:3, 3]

    def hfov_deg(self) -> float:
        return math.degrees(2.0 * math.atan(self.width / (2.0 * self.fx)))

    def vfov_deg(self) -> float:
        return math.degrees(2.0 * math.atan(self.height / (2.0 * self.fy)))


class PathTracer:
    """One combined mesh + BVH over the whole cell, re-fitted as tissue moves."""

    def __init__(self, device=None, samples=48, bounces=1, exposure=6.0,
                 max_t=8.0):
        self.device = device
        self.samples = samples
        self.bounces = bounces
        self.exposure = exposure
        # Ray length cap. The cell is ~0.4 m and the room geometry ~4 m, so an
        # effectively infinite max_t only makes the BVH walk further for rays
        # that hit nothing.
        self.max_t = float(max_t)
        self._parts = []          # (name, n_verts, slice, material)
        self.points = None
        self.mesh = None
        self._frame = 0

    # ---- construction ----------------------------------------------------
    def add(self, name, verts, faces, material, colors=None):
        """`colors` is per-vertex LINEAR rgb. Without it the material's own
        albedo is replicated onto every vertex, so the kernel has one path."""
        v = np.asarray(verts, np.float32)
        if colors is None:
            c = np.tile(MATERIALS[int(material), :3], (len(v), 1))
        else:
            c = np.asarray(colors, np.float32).reshape(-1, 3)
            if len(c) != len(v):
                raise ValueError(f"{name}: {len(c)} colours for {len(v)} verts")
        self._parts.append([name, v, np.asarray(faces, np.int32),
                            int(material), None, c.astype(np.float32)])

    def build(self):
        vs, fs, mats, cols, n = [], [], [], [], 0
        for part in self._parts:
            name, v, f, m, _, c = part
            part[4] = slice(n, n + len(v))
            vs.append(v)
            fs.append(f + n)
            mats.append(np.full(len(f), m, np.int32))
            cols.append(c)
            n += len(v)
        self._pts_np = np.vstack(vs).astype(np.float32)
        idx = np.vstack(fs).astype(np.int32).flatten()
        self.tri_mat = wp.array(np.concatenate(mats), dtype=wp.int32,
                                device=self.device)
        self.points = wp.array(self._pts_np, dtype=wp.vec3, device=self.device)
        self.indices = wp.array(idx, dtype=wp.int32, device=self.device)
        self.mesh = wp.Mesh(points=self.points, indices=self.indices,
                            support_winding_number=False)
        self.vcol = wp.array(np.vstack(cols).astype(np.float32), dtype=wp.vec3,
                             device=self.device)
        self.materials = wp.array(MATERIALS, dtype=wp.float32, device=self.device)
        self.visible = wp.array(VISIBLE, dtype=wp.int32, device=self.device)
        self._out = {}
        self._vis_override = None
        return self

    def slice_of(self, name):
        for p in self._parts:
            if p[0] == name:
                return p[4]
        raise KeyError(name)

    def update(self, name, verts):
        """Move one part's vertices (deforming tissue, or a moving instrument)."""
        sl = self.slice_of(name)
        self._pts_np[sl] = np.asarray(verts, np.float32)

    def commit(self):
        """Upload the moved vertices and re-fit the BVH."""
        self.points.assign(self._pts_np)
        self.mesh.refit()

    # ---- rendering -------------------------------------------------------
    def _buffers(self, cam):
        key = (cam.width, cam.height)
        if key not in self._out:
            self._out[key] = (
                wp.zeros((cam.height, cam.width, 3), dtype=wp.float32,
                         device=self.device),
                wp.zeros((cam.height, cam.width), dtype=wp.float32,
                         device=self.device))
        return self._out[key]

    def render(self, cam: Camera, light, ambient=(0.008, 0.006, 0.006),
               samples=None, want_depth=True, hide=(), exposure=None):
        """`hide` lists material ids invisible to THIS camera.

        The scope camera hides the scope: its objective sits on the tip's end
        face, so the barrel is entirely behind it and a real laparoscope never
        images itself. Left in, the barrel filled 27 % of the frame at zero
        distance and the tip light blew it out.
        """
        vis = self.visible
        if hide:
            v = VISIBLE.copy()
            for m in hide:
                v[int(m)] = 0
            if self._vis_override is None:
                self._vis_override = wp.zeros(len(VISIBLE), dtype=wp.int32,
                                              device=self.device)
            self._vis_override.assign(v)
            vis = self._vis_override
        rgb_buf, d_buf = self._buffers(cam)
        r, u, f, pos = cam.basis()
        ns = int(samples or self.samples)
        self._frame += 1
        wp.launch(
            _trace, dim=(cam.width, cam.height),
            inputs=[
                self.mesh.id, self.tri_mat, self.materials, vis,
                self.vcol, self.indices,
                wp.vec3(*pos), wp.vec3(*r), wp.vec3(*u), wp.vec3(*f),
                cam.fx, cam.fy, cam.cx, cam.cy, cam.width, cam.height,
                wp.vec3(*light["pos"]), wp.vec3(*light["dir"]),
                wp.vec3(*light["color"]), float(light["intensity"]),
                float(light["radius"]), float(light["cos_outer"]),
                float(light["softness"]),
                wp.vec3(*ambient), ns, int(self.bounces),
                int(self._frame),
                float(self.exposure if exposure is None else exposure),
                float(self.max_t),
            ],
            outputs=[rgb_buf, d_buf], device=self.device)
        rgb = (np.clip(rgb_buf.numpy(), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        if not want_depth:
            return rgb, None
        return rgb, d_buf.numpy().astype(np.float32)
