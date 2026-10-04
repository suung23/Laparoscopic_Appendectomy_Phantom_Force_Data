#!/usr/bin/env python3
"""Read link geometry out of a URDF, as world-placeable triangle meshes.

Both the physics (which instrument parts collide with tissue) and the renderer
(which parts are drawn, and in what colour) need the same thing: for every link,
the meshes hanging off it with their link-frame origins and materials. This
parses that once.

Isaac gets this from its URDF importer. There is no importer here, so the URDF
is read directly -- which is also why `tool_urdf.py` bothers to keep the file
byte-identical to the Isaac one: it is the shared description, and both engines
now genuinely read the same bytes.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

_MESH_CACHE: dict = {}


def rpy_to_R(r, p, y):
    """URDF fixed-axis roll-pitch-yaw. Same convention as laparo_ik.rpy_to_R."""
    import math
    cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p),
                              math.sin(p), math.cos(y), math.sin(y))
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr]])


def _origin_of(node) -> np.ndarray:
    o = node.find("origin")
    xyz = [0.0, 0.0, 0.0]
    rpy = [0.0, 0.0, 0.0]
    if o is not None:
        xyz = [float(v) for v in o.get("xyz", "0 0 0").split()]
        rpy = [float(v) for v in o.get("rpy", "0 0 0").split()]
    T = np.eye(4)
    T[:3, :3] = rpy_to_R(*rpy)
    T[:3, 3] = xyz
    return T


def load_mesh(path: str):
    """(vertices, faces) for one mesh file, cached. Vertices unscaled."""
    if path in _MESH_CACHE:
        return _MESH_CACHE[path]
    import trimesh
    m = trimesh.load(path, process=False, force="mesh")
    v = np.asarray(m.vertices, np.float64)
    f = np.asarray(m.faces, np.int32)
    _MESH_CACHE[path] = (v, f)
    return v, f


class Geom:
    """One mesh on one link, already in that link's frame."""

    __slots__ = ("link", "verts", "faces", "material", "source", "kind")

    def __init__(self, link, verts, faces, material, source, kind):
        self.link, self.verts, self.faces = link, verts, faces
        self.material, self.source, self.kind = material, source, kind

    def __repr__(self):
        return (f"<Geom {self.link}/{Path(self.source).name} "
                f"{len(self.verts)}v {self.kind} {self.material}>")


def link_geometry(urdf: Path, kind: str = "visual") -> dict:
    """{link_name: [Geom, ...]} with every mesh baked into the LINK frame.

    `kind` is "visual" or "collision". The origin transform and the mesh scale
    (the tool meshes are authored in millimetres) are applied to the vertices
    here, so a consumer only has to apply the link's world transform.
    """
    root = ET.parse(urdf).getroot()
    materials = {}
    for m in root.findall("material"):
        c = m.find("color")
        if c is not None:
            materials[m.get("name")] = tuple(
                float(v) for v in c.get("rgba", "1 1 1 1").split())[:3]

    out = {}
    for link in root.findall("link"):
        name = link.get("name")
        items = []
        for node in link.findall(kind):
            geo = node.find("geometry")
            mesh = geo.find("mesh") if geo is not None else None
            if mesh is None:
                continue                      # primitives are unused here
            fn = mesh.get("filename", "").replace("file://", "")
            if not Path(fn).is_file():
                continue
            scale = mesh.get("scale")
            s = np.array([float(v) for v in scale.split()]) if scale \
                else np.ones(3)
            v, f = load_mesh(fn)
            T = _origin_of(node)
            vv = (v * s) @ T[:3, :3].T + T[:3, 3]
            mat = node.find("material")
            mname = mat.get("name") if mat is not None else None
            items.append(Geom(name, vv, f, materials.get(mname, None), fn, kind))
        if items:
            out[name] = items
    return out


def merged(geoms: list):
    """Concatenate a list of Geom into one (verts, faces) pair."""
    if not geoms:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int32)
    vs, fs, n = [], [], 0
    for g in geoms:
        vs.append(g.verts)
        fs.append(g.faces + n)
        n += len(g.verts)
    return np.vstack(vs), np.vstack(fs).astype(np.int32)


if __name__ == "__main__":
    import sys
    ROOT = Path(__file__).resolve().parent.parent
    for arm, u in (("left", "out/fr3_left_grasper.urdf"),
                   ("right", "out/fr3_right_cutter.urdf")):
        print(f"=== {arm}: {u}")
        for kind in ("visual", "collision"):
            g = link_geometry(ROOT / u, kind)
            tot = sum(len(x) for x in g.values())
            print(f"  {kind:9s} {len(g)} link(s), {tot} mesh(es)")
            for ln, items in g.items():
                v, f = merged(items)
                print(f"    {ln:20s} {len(items):2d} mesh  {len(v):7d}v "
                      f"{len(f):7d}f  mats "
                      f"{sorted({str(i.material) for i in items})[:2]}")
