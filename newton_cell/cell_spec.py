#!/usr/bin/env python3
"""The cell's authored constants, read out of `isaac/build_surgical_cell.py`.

`build_surgical_cell.py` is the single source of truth for where everything in
the surgical cell lives -- port positions measured on the phantom, the CAD
frames of the two motorised instruments, the jaw pivot and pin axis, the table.
Every one of those numbers is documented in place next to the measurement that
produced it, and several carry hard-won sign corrections.

That module cannot be imported here: it calls `parse_args()` and then
`SimulationApp({...})` at module scope, so importing it starts Isaac Sim (and
eats argv). Copying the constants across would create exactly the two-copies
drift the file itself warns about ("Keeping one builder is deliberate -- two
copies of this would drift").

So they are EXTRACTED instead. `constants()` parses the module with `ast` and
evaluates its top-level literal assignments. Nothing is imported and no Isaac
code runs. Every name this package needs is listed in `REQUIRED`, and a missing
one is a hard failure -- if a future edit to build_surgical_cell.py renames a
constant, the Newton build stops instead of quietly using a stale value.
"""
from __future__ import annotations

import ast
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "isaac" / "build_surgical_cell.py"

# Names this package reads. Extraction fails loudly if any is absent.
REQUIRED = (
    # table / room
    "TABLE_SIZE", "PLATE_SIZE", "ALU", "PLATE_COLOR", "FLOOR_COLOR",
    "PHANTOM_COLOR",
    # trocar ports
    "TROCAR_PORTS_MM", "USED_PORT_IDX", "LEFT_COLOR", "RIGHT_COLOR", "AMBER",
    # left grasper
    "MOTOR_TOOL_DIR", "LAPARO_MESH_DIR", "LAPARO_MESH_SCALE",
    "LAPARO_EXTRA_MESHES", "LAPARO_COLLIDE_MESHES",
    "LAPARO_VIS_XYZ", "LAPARO_VIS_RPY", "LAPARO_TIP_XYZ", "LAPARO_TIP_RPY",
    "LAPARO_MASS", "LAPARO_COM", "LAPARO_INERTIA",
    # jaws
    "JAW_MESHES", "JAW_PIVOT_XYZ", "JAW_PIN_AXIS", "JAW_OPEN_TOTAL_DEG",
    "JAW_MASS", "JAW_COM", "JAW_INERTIA",
    # right cutter
    "CUTTER_TOOL_DIR", "CUTTER_MESH_DIR",
    "CUTTER_VIS_XYZ", "CUTTER_VIS_RPY", "CUTTER_TIP_XYZ", "CUTTER_TIP_RPY",
    "CUTTER_MASS", "CUTTER_COM", "CUTTER_INERTIA",
    "BLADE_MESHES", "BLADE_PIVOT_XYZ", "BLADE_PIN_AXIS", "BLADE_OPEN_TOTAL_DEG",
    # materials / naming
    "TOOL_MATERIALS", "SILVER_TOOL_STEMS", "BOLT_TOOL_STEMS_PREFIX",
    "MESH_NUM_PREFIX",
    # hand removal
    "HAND_LINKS", "HAND_JOINTS",
    # specimen / endoscope placement
    "ORGAN_EXTRA_TRANSLATE", "ENDO_POSE_TRANSLATE", "ENDO_POSE_ROTXYZ",
    # cameras
    "D405_RES", "D405_HFOV_DEG", "D405_FOCAL",
    "ENDO_CAM_RES", "ENDO_CAM_HFOV_DEG", "ENDO_CAM_FOCAL",
    "ENDO_TIP_ANGLE_DEG", "ENDO_BEVEL_CLOCK_DEG",
    # endoscope tip light -- the dominant light source inside the abdomen
    "ENDO_LIGHT_COLOR", "ENDO_LIGHT_INTENSITY", "ENDO_LIGHT_EXPOSURE",
    "ENDO_LIGHT_RADIUS", "ENDO_LIGHT_DIFFUSE", "ENDO_LIGHT_SPECULAR",
    "ENDO_CONE_DEG", "ENDO_CONE_SOFTNESS",
    # organ stiffness caps (two organs go numerically unstable at the full E)
    "YOUNGS_CAP", "PIN_BAND_CAP",
)

# Tissue properties are argparse DEFAULTS in build_surgical_cell.py, not module
# constants, so they are read separately by `cli_defaults()`. They matter more
# than anything else in this file: see `tissue()` for why the VLA handoff's
# quoted numbers are not the ones the cell is actually built with.
REQUIRED_CLI = (
    "--appendix-youngs", "--organ-youngs", "--organ-density",
    "--organ-damping", "--organ-friction",
    "--lift", "--rotate-phantom", "--rotate-organs", "--port-offset",
    "--spacing", "--camera-drop", "--table-shift", "--table-depth",
    "--pin-radius", "--pin-band",
    "--organ-iterations", "--organ-contact-offset", "--organ-rest-offset",
    "--appendix-raise", "--endoscope-intensity", "--scene-light-scale",
)


class _Evaluator(ast.NodeVisitor):
    """Evaluate the literal subset of build_surgical_cell.py's module scope.

    Handles the four shapes the constants actually use: literals (numbers,
    strings, tuples, lists, dicts, sets), `dict(k=...)` calls, `ROOT / "a" / "b"`
    path expressions, and references to names defined earlier in the file.
    Anything else is skipped rather than guessed at.
    """

    def __init__(self):
        self.env = {"ROOT": ROOT, "math": math}

    def _eval(self, node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Tuple):
            return tuple(self._eval(e) for e in node.elts)
        if isinstance(node, ast.List):
            return [self._eval(e) for e in node.elts]
        if isinstance(node, ast.Set):
            return {self._eval(e) for e in node.elts}
        if isinstance(node, ast.Dict):
            return {self._eval(k): self._eval(v)
                    for k, v in zip(node.keys, node.values)}
        if isinstance(node, ast.Name):
            if node.id not in self.env:
                raise KeyError(node.id)
            return self.env[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -self._eval(node.operand)
        if isinstance(node, ast.BinOp):
            left, right = self._eval(node.left), self._eval(node.right)
            if isinstance(node.op, ast.Div):
                return left / right          # covers Path / "str" too
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            raise TypeError(type(node.op))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == "dict":
            return {kw.arg: self._eval(kw.value) for kw in node.keywords}
        raise TypeError(ast.dump(node)[:80])

    def visit_Assign(self, node):
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            return
        try:
            self.env[node.targets[0].id] = self._eval(node.value)
        except Exception:
            pass                             # not a literal; nothing needs it


def _cli_defaults_from(tree: ast.AST, ev: "_Evaluator") -> dict:
    """Every `ap.add_argument("--x", ..., default=V)` default in the module.

    The tissue properties live here rather than in module constants, so they
    have to be read out of the parser definition. Only literal defaults are
    taken; a computed one would be skipped and then caught by the REQUIRED_CLI
    check.
    """
    out = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        flags = [a.value for a in node.args
                 if isinstance(a, ast.Constant) and isinstance(a.value, str)]
        opt = next((f for f in flags if f.startswith("--")), None)
        if opt is None:
            continue
        for kw in node.keywords:
            if kw.arg == "default":
                try:
                    out[opt] = ev._eval(kw.value)
                except Exception:
                    pass
    return out


_CACHE = None
_CLI_CACHE = None


def cli_defaults() -> dict:
    """build_surgical_cell.py's argparse defaults -- the values the cell is
    actually built with, since nobody passes these flags."""
    constants()                              # populates _CLI_CACHE
    return _CLI_CACHE


def constants() -> dict:
    """Top-level literal constants of build_surgical_cell.py."""
    global _CACHE, _CLI_CACHE
    if _CACHE is not None:
        return _CACHE
    if not SOURCE.is_file():
        raise SystemExit(f"cannot find the cell spec: {SOURCE}")
    tree = ast.parse(SOURCE.read_text(), str(SOURCE))
    ev = _Evaluator()
    for node in tree.body:                   # module scope only
        if isinstance(node, ast.Assign):
            ev.visit_Assign(node)
    cli = _cli_defaults_from(tree, ev)

    missing = [n for n in REQUIRED if n not in ev.env]
    missing += [n for n in REQUIRED_CLI if n not in cli]
    if missing:
        raise SystemExit(
            f"{SOURCE.name} no longer defines: {', '.join(missing)}.\n"
            f"newton_cell reads the cell layout from that file so the two "
            f"engines cannot drift apart. Update newton_cell/cell_spec.py "
            f"REQUIRED/REQUIRED_CLI (and whatever uses it) to match the rename.")
    _CACHE = {n: ev.env[n] for n in REQUIRED}
    _CLI_CACHE = {n: cli[n] for n in REQUIRED_CLI}
    return _CACHE


# --------------------------------------------------------------------------
# derived: the two tool specs, as build_arm_tool_urdf() consumes them
# --------------------------------------------------------------------------
class ToolSpec:
    """One arm's instrument. Mirrors build_surgical_cell.ToolSpec."""

    def __init__(self, name, mesh_dir, vis_xyz, vis_rpy, tip_xyz, tip_rpy,
                 mass, com, inertia, jaw_meshes, pivot_xyz, pin_axis,
                 open_total_deg, link_prefix, joint_prefix, urdf):
        self.name, self.mesh_dir = name, mesh_dir
        self.vis_xyz, self.vis_rpy = vis_xyz, vis_rpy
        self.tip_xyz, self.tip_rpy = tip_xyz, tip_rpy
        self.mass, self.com, self.inertia = mass, com, inertia
        self.jaw_meshes, self.pivot_xyz = jaw_meshes, pivot_xyz
        self.pin_axis, self.open_total_deg = pin_axis, open_total_deg
        self.link_prefix, self.joint_prefix = link_prefix, joint_prefix
        self.urdf = urdf

    @property
    def joints(self):
        return (f"{self.joint_prefix}_left", f"{self.joint_prefix}_right")


def tools() -> dict:
    C = constants()
    return {
        "left": ToolSpec(
            "left grasper", C["LAPARO_MESH_DIR"],
            C["LAPARO_VIS_XYZ"], C["LAPARO_VIS_RPY"],
            C["LAPARO_TIP_XYZ"], C["LAPARO_TIP_RPY"],
            C["LAPARO_MASS"], C["LAPARO_COM"], C["LAPARO_INERTIA"],
            C["JAW_MESHES"], C["JAW_PIVOT_XYZ"], C["JAW_PIN_AXIS"],
            C["JAW_OPEN_TOTAL_DEG"], "jaw", "gripper_jaw",
            "out/fr3_left_grasper.urdf"),
        "right": ToolSpec(
            "right cutter", C["CUTTER_MESH_DIR"],
            C["CUTTER_VIS_XYZ"], C["CUTTER_VIS_RPY"],
            C["CUTTER_TIP_XYZ"], C["CUTTER_TIP_RPY"],
            C["CUTTER_MASS"], C["CUTTER_COM"], C["CUTTER_INERTIA"],
            C["BLADE_MESHES"], C["BLADE_PIVOT_XYZ"], C["BLADE_PIN_AXIS"],
            C["BLADE_OPEN_TOTAL_DEG"], "blade", "cutter_blade",
            "out/fr3_right_cutter.urdf"),
    }


# --------------------------------------------------------------------------
# tissue
# --------------------------------------------------------------------------
# Poisson's ratio is the one organ property build_surgical_cell.py does NOT
# touch: build_deformable_usd.py authors it from sim/scene.py's POISSON_RATIO
# and the cell build overrides only youngsModulus and density afterwards. So it
# is read from sim/scene.py, the module that actually sets it.
SCENE_PY = ROOT / "sim" / "scene.py"

ORGANS = ("cecum", "bowel1", "bowel2", "bowel3", "stomach", "appendix")


def poisson_ratio() -> float:
    """POISSON_RATIO out of sim/scene.py (build_deformable_usd.py imports it)."""
    tree = ast.parse(SCENE_PY.read_text(), str(SCENE_PY))
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id == "POISSON_RATIO":
            return float(ast.literal_eval(node.value))
    raise SystemExit(f"no POISSON_RATIO in {SCENE_PY}")


def tissue() -> dict:
    """The organ material the Isaac cell is ACTUALLY built with.

    Read this before trusting VLA_HANDOFF.md section 7.4, which says "every
    organ is E = 20 kPa, nu = 0.45 (appendix 12 kPa)". The 20 kPa is stale: it
    is `sim/scene.py`'s YOUNGS_MODULUS, which is the standalone Newton organ
    scene's default and the value build_deformable_usd.py cooks with. The cell
    build then OVERWRITES youngsModulus and density on every organ material
    (build_surgical_cell.py, "organ material tuning"), and those overrides are
    what `out/surgical_cell.usd` carries:

        appendix          12 kPa   --appendix-youngs   ("holds its shape yet
                                                        the grasp lifts it")
        bowel2, bowel3    15 kPa   YOUNGS_CAP          (numerically unstable at
                                                        30 kPa: sliver elements
                                                        buzz 1-2 mm/s at rest)
        cecum, bowel1,
        stomach           30 kPa   --organ-youngs      ("5 kPa was too soft --
                                                        a firm push sank the
                                                        tool tip inside")
        density         1200       --organ-density     (1050 was "so light the
                                                        tissue flew off")
        damping         0.05       --organ-damping
        friction        0.6        --organ-friction
        nu              0.45       sim/scene.py        (never overridden)

    Building the Newton cell from the handoff's numbers instead would make five
    of six organs 1.5x too soft and everything 12 % too light -- which is
    exactly the sim-to-sim gap this port is supposed to avoid.
    """
    C, D = constants(), cli_defaults()
    nu = poisson_ratio()
    per_organ = {}
    for name in ORGANS:
        if name == "appendix":
            E = D["--appendix-youngs"]
        else:
            E = C["YOUNGS_CAP"].get(name, D["--organ-youngs"])
        per_organ[name] = float(E)
    return dict(
        youngs=per_organ,
        poisson=nu,
        density=float(D["--organ-density"]),
        damping=float(D["--organ-damping"]),
        friction=float(D["--organ-friction"]),
        contact_offset=float(D["--organ-contact-offset"]),
        rest_offset=float(D["--organ-rest-offset"]),
        iterations=int(D["--organ-iterations"]),
    )


def lame(E: float, nu: float) -> tuple:
    """Young's/Poisson -> the Lame pair Newton's add_soft_mesh wants.

    Same conversion as sim/scene.py:lame(). PhysX takes (E, nu) directly and
    Newton takes (k_mu, k_lambda), so this is where the two engines' material
    inputs are made to describe the same solid.
    """
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return mu, lam


def mesh_material_name(stem: str) -> str:
    """Verbatim build_surgical_cell.mesh_material_name."""
    C = constants()
    if stem in C["SILVER_TOOL_STEMS"]:
        return "stainless_steel"
    if stem.startswith(tuple(C["BOLT_TOOL_STEMS_PREFIX"])):
        return "bolt_mat"
    return "gloss_black"


def staged_mesh_path(mesh: Path) -> Path:
    """Verbatim build_surgical_cell.staged_mesh_path.

    Newton loads meshes by path and does not care about USD prim naming, but
    the staging is kept so the URDF this package writes is byte-identical to
    the one the Isaac path writes -- that URDF is the interface the two engines
    share, and a diff between the two machines should come back empty.
    """
    if not mesh.stem[:1].isdigit():
        return mesh
    dest = ROOT / "out" / "laparo_meshes" / f"{constants()['MESH_NUM_PREFIX']}{mesh.name}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink() or dest.exists():
        dest.unlink()
    dest.symlink_to(mesh.resolve())
    return dest


if __name__ == "__main__":
    C = constants()
    print(f"read {len(C)} constants from {SOURCE.relative_to(ROOT)}")
    for k in REQUIRED:
        s = repr(C[k])
        print(f"  {k:24s} {s[:92]}{'...' if len(s) > 92 else ''}")
    print(f"\ncli defaults ({len(cli_defaults())}):")
    for k, v in cli_defaults().items():
        print(f"  {k:26s} {v}")
    T = tissue()
    print(f"\ntissue as the cell is actually built "
          f"(nu={T['poisson']}, rho={T['density']}, damping={T['damping']}, "
          f"friction={T['friction']}):")
    for name, E in T["youngs"].items():
        mu, lam = lame(E, T["poisson"])
        print(f"  {name:9s} E={E/1e3:5.1f} kPa  ->  mu={mu:9.1f} Pa  "
              f"lambda={lam:10.1f} Pa")
