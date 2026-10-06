# `newton_cell` — the H100 / headless backend

A **second** backend for the surgical cell, alongside the existing Isaac Sim one.
Nothing in [`isaac/`](../isaac/) is modified, moved or replaced: the RTX
workstation keeps running exactly as
[VLA_HANDOFF.md](../VLA_HANDOFF.md) describes, and this directory adds a path
that runs on a GPU without RT cores.

| | RTX workstation | this backend |
|---|---|---|
| entry point | `~/isaacsim-5.1/python.sh isaac/vla_rollout.py` | `.venv-newton/bin/python …` |
| physics | Isaac Sim 5.1 + PhysX | Newton 1.4 + Warp (CUDA, sm_90) |
| rendering | Isaac RTX (hardware ray tracing) | Warp path tracer (CUDA, no RT cores) |
| display | GUI available | headless only |
| status of `isaac/` | authoritative | untouched |

Both are driven by the *same* `laparo_ik`, the same `laparo_control.BEST`, the
same action interface and the same URDFs. Only the engine underneath differs.

## Why a second backend at all

`VLA_HANDOFF.md` §1.1 states the constraint: Isaac Sim's renderer is ray-traced
and NVIDIA does not support it on GPUs without RT cores, which is what an H100
is. `SURGICAL_CELL.md` "Target architecture" already anticipated this split and
specified the contract it has to honour:

> **Robot-arm movement must be identical** across both engines.
> **Organ deformation is allowed to differ** (PhysX FEM vs Newton).

That is achievable because the arm is a position-driven follower of a shared
action interface, not a dynamics simulation whose solver has to match.

## What is here so far

| file | role |
|---|---|
| `fr3_urdf.py` | FR3 URDF from `franka_description` instead of the Isaac extension |
| `cell_spec.py` | reads the cell's authored constants **out of** `isaac/build_surgical_cell.py` |
| `tool_urdf.py` | FR3 + motorised instrument → `out/fr3_{left_grasper,right_cutter}.urdf` |
| `layout.py` | world poses: phantom, trocar ports, robot bases, rig frame, D405 |
| `scene.py` | the cell as a `newton.Model` — organs as FEM soft bodies, pins, shelf |
| `urdf_meshes.py` | link geometry out of the URDFs (no Isaac importer here) |
| `arms.py` | one arm: `Chain` + `TipController` + kinematic Newton bodies |
| `sim.py` | cell + both arms + the 480 Hz step loop |
| `render.py` | the Warp path tracer (no RT cores, no Vulkan, no X) |
| `cameras.py` | endoscope (calibrated pinhole, 30° oblique) + D405 + tip light |
| `view.py` | binds the live sim to the tracer and renders frames |
| `forces.py` | contact-force channels on the `tool_tip` axes |
| `grasp.py` | force-limited compliant grasp assistance on FEM nodes |
| `vla_env.py` | `LaparoVLAEnv` / `Action` / `DemoPolicy` |
| `rollout.py` | shim that runs `isaac/vla_rollout.py` against this backend |
| `eval.py` | entry point with render presets (`--preset fast` for training loops) |

### One source of truth, not two copies

`build_surgical_cell.py` cannot be imported without Isaac (it calls
`parse_args()` and `SimulationApp()` at module scope). Copying its constants
here would create exactly the drift that file warns about, so `cell_spec.py`
**parses** them with `ast` instead — no import, no Isaac, no second copy. Every
name it needs is listed in `REQUIRED` / `REQUIRED_CLI`; a rename on the Isaac
side stops this build with a message rather than silently using a stale number.

### The shared URDFs are validated, not overwritten

`out/fr3_left_grasper.urdf` and `out/fr3_right_cutter.urdf` are the one
interface both backends read. On the RTX machine `build_surgical_cell.py`
writes them. Running `tool_urdf.py` there does **not** clobber them: it builds
into a temporary directory, compares the parsed kinematic chain, keeps the
existing file if they agree, and fails loudly (naming `--force`) if they do not.

## Verified so far

```
$ .venv-newton/bin/python newton_cell/tool_urdf.py
  verified 8 joint origins against fr3/kinematics.yaml (exact)
  verified 16 mesh reference(s) resolve on disk
  verified left  tool_tip frame vs VLA_HANDOFF.md section 2 (insertion = -z, det=+1)
  verified right tool_tip frame vs VLA_HANDOFF.md section 2 (insertion = -z, det=+1)
```

- FR3 joint origins match the manufacturer's `kinematics.yaml` **exactly**, so
  `laparo_control.BEST` (a 182-configuration grid-search winner tuned against
  these origins) still describes this robot.
- `laparo_ik.Chain` parses both URDFs; on both arms
  `insertion · shaft = +1.000000` and `tip +y · clevis pin = +1.000000`, and the
  tip stands off the flange by exactly the authored `tip_xyz` (515.9 mm grasper,
  494.2 mm cutter).
- `layout.py` derives left = port 3 / right = port 2 purely from world X, which
  is what `VLA_HANDOFF.md` §4.3 documents as `port_id`.

### Tissue settles (`scene.py`, H100, 2 s of sim)

```
model: 11962 particles, 43956 tets
  frame   0  max speed 0.1722 m/s
  frame  60  max speed 0.0011 m/s
  frame 119  max speed 0.0010 m/s      -> settled, not buzzing
120 frames (2.00 s sim) in 20.8 s wall -> rtf 0.096
  cecum    mean 0.03 mm   stomach mean 0.01 mm
  appendix mean 2.02 mm   max 4.47 mm  (drapes onto its shelf and rests)
```

`rtf 0.096` against the Isaac path's documented 0.11 — the same order, before
the arms and rendering are added. At-rest speed settles to ~0.4–1.0 mm/s, which
is at or below the "1–2 mm/s at rest" buzz `build_surgical_cell.py` documents
for `bowel2` under PhysX.

`LaparoVLAEnv` now performs the same 2 s passive settling at construction and
uses that finite, non-inverted state for later resets. Tissue health is checked
at each control interval during settling; episode time starts at zero after the
settle, so warm-up is not counted as policy performance. This closes a reset
bug where the policy previously began moving while the appendix was still
falling. The Newton demo now moves to a 25 mm overhead waypoint at 12 mm/s,
then descends toward the mid-body handle with the most cecum clearance. This
matches the Isaac baseline's standoff sequence and adds a clearance rule for
Newton's anatomy. These changes address a measured source of numerical failure;
they do not establish calibrated tissue mechanics or a completed appendectomy.

### Three things that had to be right, and were not at first

Each was caught by measurement, not by reading:

1. **Pinning.** Zeroing `particle_inv_mass` on the finalized model pins nothing
   in Newton — "pinned" nodes still drifted **38.6 mm**. Pinning is
   `ParticleFlags.ACTIVE` cleared plus zero mass, on the *builder*, before
   `finalize()`.
2. **The field must not collide with the organs.** They are captured embedded
   in it. With the collider on, the stomach walked **24 mm in one second** and
   speeds sat at 1.3 m/s. `build_surgical_cell.py` filters this pair and says
   why; this backend now defaults the same way (`--organ-field-collision`).
3. **The appendix needs its shelf.** The field was reconstructed from a
   top-down D405, so there is no geometry *under* the appendix. Without the
   shelf it simply falls — **148 mm in one second and still accelerating**.
   It also pins at the end farthest from the **left robot base**, not at the
   cecum.

## Tissue properties — read this before trusting the handoff

`VLA_HANDOFF.md` §7.4 says *"every organ is E = 20 kPa, ν = 0.45 (appendix
12 kPa)"*. **The 20 kPa is stale.** It is `sim/scene.py`'s `YOUNGS_MODULUS`,
which is the standalone Newton organ scene's default and the value
`build_deformable_usd.py` cooks with. `build_surgical_cell.py` then *overwrites*
`youngsModulus` and `density` on every organ material, and those overrides are
what `out/surgical_cell.usd` actually carries:

| organ | E | source |
|---|---|---|
| appendix | **12 kPa** | `--appendix-youngs` |
| bowel2, bowel3 | **15 kPa** | `YOUNGS_CAP` (unstable at 30 kPa: sliver elements buzz at rest) |
| cecum, bowel1, stomach | **30 kPa** | `--organ-youngs` ("5 kPa was too soft — a firm push sank the tool tip inside") |
| all | ρ = **1200** kg/m³ | `--organ-density` (1050 was "so light the tissue flew off") |
| all | ν = **0.45**, damping 0.05, friction 0.6 | `sim/scene.py` / `--organ-damping` / `--organ-friction` |

`cell_spec.tissue()` returns these, read from source. Building the Newton cell
from the handoff's numbers instead would make five of six organs 1.5× too soft
and everything 12 % too light.

## Environment

Created with `uv`, outside the repo's tracked files (`.venv-newton/`, `assets/`
are gitignored):

```bash
uv venv --python 3.11 .venv-newton
uv pip install --python .venv-newton/bin/python \
    numpy trimesh imageio imageio-ffmpeg matplotlib scipy pillow xacro "newton[importers]"
```

> The NVIDIA engine is **`newton`** on PyPI. `newton-physics` is an unrelated
> package that installs a module called `newton_physics` and no Warp — installing
> it silently gives you the wrong thing.

Verified on this host: Newton 1.4.0 / Warp 1.15.0, `NVIDIA H100 NVL` `sm_90`,
driver 550.144.03, CUDA 12.4, Ubuntu 22.04. No `sudo`, no X, no Vulkan needed.

---

## Running it

```bash
# once: FR3 + tool URDFs (validates rather than overwrites an Isaac-built pair)
.venv-newton/bin/python newton_cell/tool_urdf.py

# a rollout, scored, with mp4s
.venv-newton/bin/python newton_cell/rollout.py \
    --episodes 1 --seconds 20 --video --out out/newton_eval

# your own policy -- identical interface to the Isaac backend
.venv-newton/bin/python newton_cell/rollout.py \
    --policy mypolicy.py:Policy --episodes 20 --seconds 25 \
    --out out/eval_run1 --video

# one probe frame from each camera, to eyeball the scene
.venv-newton/bin/python newton_cell/view.py --out out/newton_probe --samples 96
```

`rollout.py` is a **shim**, not a fork: it imports `isaac/vla_rollout.py`
unmodified and puts `newton_cell/` ahead of `isaac/` on `sys.path` so that
file's late `from vla_env import LaparoVLAEnv` binds here. The episode recorder,
the scorer, the npz layout and the ffmpeg step are therefore literally the same
code on both machines. Running `isaac/vla_rollout.py` directly still gets Isaac,
unchanged — the backend is chosen by which entry point you launch, so the RTX
workstation's behaviour is untouched.

## Rendering

`ViewerRTX` is Omniverse OVRTX and needs the same RT cores an H100 lacks;
`ViewerGL`/`ViewerUSD` are rasterisers. So `render.py` is a **path tracer in
Warp kernels** — pure CUDA, no RT cores, no Vulkan, no EGL, no X, no root, which
is exactly what this host offers.

That shape suits the scene rather than just being prettier. A laparoscope
carries its own light: a small emitter at the scope tip, centimetres from wet
tissue, so the image is almost entirely inverse-square falloff, soft shadows
from a finite-size source, and specular sheen — none of which a rasteriser with
a shadow map reproduces. Deforming organs cost nothing structural, since a
`wp.Mesh.refit` per frame is cheap.

### Colour comes from the reconstruction, not from a palette

The organs and the surgical field were photographed, not authored: every
`mesh/organ_*.ply` and `mesh/field.ply` carries per-vertex colour from the
capture (appendix mean RGB 146/113/44, field 165/129/93 — olive bowel on a tan
bed). `build_deformable_usd.py` converts those sRGB bytes to linear once and
authors them as `displayColor`, and that is what the Isaac renderer draws.

So the tracer interpolates **per-vertex captured colour** across each hit
triangle and uses it as albedo; the material table supplies only the finish
(roughness, specular, subsurface wrap). Organs are simulated as tets but
captured as surface meshes, so each tet-boundary vertex takes the colour of its
nearest captured vertex (`view.transfer_colors`).

Using flat per-material albedo instead — which is what this backend did at
first — turns the whole specimen a uniform pink and loses the olive/tan/blue
that makes the reference view readable. The blue is the specimen plate
(`PLATE_COLOR`, the surgical drape), which is authored rather than captured.

#### Optional Appendix300 appearance candidate

The Newton endoscope can optionally render a provenance-checked Appendix300
surface-color candidate on the simulated appendix:

```bash
.venv-newton/bin/python newton_cell/eval.py --preset balanced \
  --appendix300-texture out/ct_appendectomy/appendix300_sam2_texture_v21/appendix_appendix300_temporal_rgb.png \
  --out out/appendix300_fem_texture_preview
```

The candidate's sibling `appearance.json` is mandatory; the Appendix300 DOI,
CC BY 4.0 license, and image hash are checked before rendering. This candidate
uses a local SAM2 visible-object track (mean adjacent-mask IoU 0.72 over 0.5 s),
but the mask has not been verified by an expert as appendix tissue. A
deterministic cylindrical UV parameterisation is bound to appendix FEM rest
nodes. The path tracer samples the full-resolution RGB image at each ray hit,
so appearance detail is no longer limited to the sparse vertex colours and
travels with the deforming mesh. This remains a visual appearance experiment,
not camera-to-mesh registration,
patient-specific reconstruction, or measured albedo/BRDF. The video crop can
carry illumination and specular glare, so the scope light will not be
physically correct where those remain baked into the texture. The option is
off by default; ordinary runs retain captured CT colors. Render outputs and
source imagery remain in the ignored local `out/` tree. Validation evidence is
recorded in `validation/appendix300-uv-texture-2026-10-06.json`; the FEM surface
is still visibly coarse and this does not pass the visual realism gate.

The Newton renderer now uses the captured organ PLY surface topology and binds
its vertices directly to unique FEM rest nodes, with a 10 µm maximum match
guard and a FEM-boundary fallback. All six current assets pass the guard; their
measured maximum rest-position mismatch is below 0.011 µm. This is a clean
render/simulation correspondence, not a higher-resolution reconstruction: a
paired endoscope render showed no meaningful detail gain, and the surfaces
remain visibly faceted. A render-only Loop subdivision candidate now smooths
the display surface with sparse stencils over those mapped FEM nodes; the
simulation and contact meshes stay unchanged. One level multiplies triangle
count by four, while its measured rest-surface smoothing offset is p95
0.78–0.82 mm and maximum 0.99–1.09 mm across these assets. This intentionally
changes the visible outline slightly and needs same-camera RTX comparison and
held-out visual review before promotion. Use
`--surface-subdivision-levels 0` to render the unsmoothed control. See
`validation/newton-captured-render-surface-binding-2026-10-06.json`.

### Three cameras

| output | what it is |
|---|---|
| `episode_NNN_endoscope.mp4` | the 30° forward-oblique scope, 1280×720 — the policy's view |
| `episode_NNN_d405.mp4` (+ `_depth`) | overhead RealSense D405, 848×480 RGB-D |
| `episode_NNN_front.mp4` | **cell overview** — both arms, phantom, table |

The front view is not a clinical camera; nothing on the rig sees it. It exists
so an eval video is readable without knowing the port layout, and it is lit by
a broad key light rather than the scope tip (which is inside the dome and would
leave the room black). Being ambient-dominated it is nearly noise-free, so it
renders at 8 spp direct-only — at the endoscope's 48 spp with a bounce it cost
365 s/frame, which is 40× more than it needs.

It is plumbed through module state (`vla_env.FRONT_OUT`), set by `rollout.py`
before it hands over. `isaac/vla_rollout.py` builds the env with a fixed keyword
list, so a third camera cannot go through its constructor — and that file is
deliberately not edited.

### The endoscope camera

The endoscope uses the **calibrated** pinhole `build_surgical_cell.py` authors
(ChArUco, fx = 684.128, principal point 611.4/297.1 at 1280×720, square pixels
because the RTX viewport ignores `verticalAperture`), with the 30° forward-oblique
bevel on both the view axis and the illumination axis. `K` is reported in the
observation, so a policy sees the same projection on both machines.

## Status

### Current blocking evidence

- The 2026-10-06 physics fail-fast smoke test stopped after 1.07 simulated
  seconds with 16 inverted tetrahedra. The guard worked, but tissue mechanics
  failed; see `validation/newton-physics-failfast-2026-10-06.json`.
- A longer 13.6 s scripted attempt completed only 2/5 stages and failed at
  `at_site`, with up to 87 inverted elements and 3.69 m/s nodal speed; see
  `validation/newton-bimanual-contact-health-2026-10-06.json`.
- The force-capped compliant grasp has synthetic unit coverage and was
  exercised in a local RTX 4090 Newton rollout. The cecum inversion gate stopped
  that scripted episode before a grasp latch; its stiffness and force cap remain
  uncalibrated. The newer high-resolution Appendix300 UV render is a provisional
  appearance transfer, and the coarse FEM surface plus unregistered source ROI
  do not satisfy the endoscope realism gate.
- A swept-URDF clearance ranking was tested against the cecum contact failure.
  Dynamic target reselection worsened the failure; locking the target produced
  exactly the baseline actions and outcome for the tested seed. It remains an
  opt-in diagnostic (`--demo-swept-clearance`), not a validated safety feature;
  see `validation/newton-swept-approach-clearance-2026-10-06.json`.
- A reset now clears restored rigid-body velocities and refreshes Newton's
  contact manifold before the first observation. Paired seed 1–3 contact
  ablations reproduce cecum inversion only with instrument/tissue contact;
  all runs still fail before grasp. A scripted 3/6/12 mm/s approach probe
  changes failure severity but none prevents inversion. `--demo-approach-mm-s`
  is diagnostic only; see
  `validation/newton-tool-tissue-collision-ablation-2026-10-06.json` and
  `validation/newton-demo-approach-speed-probe-2026-10-06.json`.

The H100 validation process is still live but has produced no new output; the
Windows runtime does not have Newton/Warp, and the SSH configuration remains
inaccessible to the current sandbox. Do not treat older episode scores below
as evidence that the current branch passes mechanics or task gates.


### Historical commissioning checks

| | evidence |
|---|---|
| Newton + Warp on H100 | Newton 1.4.0 / Warp 1.15.0, `sm_90` |
| FR3 without Isaac | 8 joint origins **exact** vs `kinematics.yaml` |
| Arm + tool URDFs | tip frame matches `VLA_HANDOFF.md` §2 on both arms |
| Existing IK + `BEST` controller | reused **unmodified**; start pose solves to tip err 0.00 mm, RCM err 0.00 mm, insertion exactly 100.00 mm, `shaft · port normal = +1.000000` |
| Tissue FEM, correct material | settles to ~0.4–1.0 mm/s; organs hold <0.03 mm; appendix drapes 2 mm onto its shelf |
| Arms driving tissue | 480 Hz control, 16 ticks/sub-step, delta interpolated in the sub-step's tool frame; depth 100→117→106 mm over a dive, **RCM error 0.01–0.02 mm throughout** |
| Path tracer on H100 | runs with no RT cores / Vulkan / EGL / X / root |
| D405 camera | depth 96–218 mm after the phantom view hole is cut |
| Exposure calibration | dome at luma p50 157 / p90 206, vs the 143–164 the Isaac cell documents |

### Historical 20 s scripted episode (pre-force-capped grasp revision)

```
$ .venv-newton/bin/python newton_cell/rollout.py --episodes 1 --seconds 20 --video
backend: newton (.../newton_cell/vla_env.py)
1 episode(s) x 75 step(s) of 8 sub-steps = 20.0 s of simulated time each
episode 0: SUCCESS lift +12.6 mm, grasp 1x (92% of steps), closest 0.8 mm,
           RCM p95 0.02 mm -> episode_000.npz [2609 s wall]
    newton_eval/episode_000_endoscope.mp4     1280x720 h264
    newton_eval/episode_000_d405.mp4
    newton_eval/episode_000_d405_depth.mp4
success rate 100% over 1 episode(s)
```

Scored by `Episode.score()` in `isaac/vla_rollout.py` — the same code, not a
reimplementation:

| | Newton (this run) | Isaac reference (VLA_HANDOFF §5.7) |
|---|---|---|
| `success` | **true** (12.6 mm lift) | true (10.7 mm lift) |
| `rcm_err_p95_mm` / `max` | **0.016 / 0.32** | 0.32 / 2.57 |
| `rcm_excursion_frac` | **0.0** | 0.0 |
| `trocar_bite_p95_mm` | **0.0** | 0.0002 |
| `grasp_events` / held | 1 / **92 %** of steps | 1 |
| `tissue_disturbance_mm` | 12.9 | 9.3 |

That archived run predates the compliant-grasp change. Current Newton rollout
NPZ files record `left/right_grasp_assist_model_force_n` and
`left/right_grasp_assist_deflection_m` separately from the tool-contact wrench;
the Isaac runner's metric schema remains unchanged. These model-force channels
must not be interpreted as measured grasper or tissue forces.

The rig-health numbers are the ones that matter, and the handoff is blunt about
why: *"a success obtained by tearing the trocar is not a success."* The shaft
stays inside its 6 mm port throughout.

**Lift rate is the whole game, and it is worth knowing why.** The appendix's far
end is pinned to the world (`VLA_HANDOFF.md` §4.4), so lifting stretches tissue
between a rigid anchor and the jaws. Measured, same policy, only the rate changed:

| demo lift rate | achieved lift | RCM p95 | outcome |
|---|---|---|---|
| 45 mm/s | +19.1 mm | **4.54 mm** | **ABORTED** — sustained trocar violation, step 18 |
| 1.5 mm/s | +6.5 mm | 0.02 mm | clean, but under the 10 mm bar |
| **3.6 mm/s** | **+12.6 mm** | **0.02 mm** | **SUCCESS** |

That is the same trade the handoff warns about, reproduced independently on a
different engine.

### Two scenes, not one

The endoscope and the D405 sit **inside** the phantom dome. They cannot see the
robot, the table or the room, so those are not in their BVH:

| | triangles | endoscope frame @16 spp |
|---|---|---|
| one shared scene | 1,039,382 | 153 s |
| clinical scene only | **197,082** | **6.3 s** |

A 24x speedup for geometry that was never visible. The front view keeps its own
full scene (`tracer_room`) at native CAD resolution.

**One exception: the specimen plate is in both.** It is the blue surgical drape
the specimen rests on, and it is the blue surround in every reference endoscope
frame — very much visible from inside the dome. It is one box: adding it back to
the clinical scene cost **12 triangles** (197,082 -> 197,094) and no measurable
time. The first version of the split dropped it along with the table and the
floor, and the endoscope surround went black.

Decimating the robot instead was tried first and abandoned: trimesh's quadric
decimation shatters this CAD geometry however the budget is split — a flat 6k
budget over the merged 193k instrument turned the servo housings into floating
shards and the FR3 shells visibly faceted, and per-body proportional budgets
were no better. Splitting the scenes is both faster and lossless.

### Performance, and where it goes

| | |
|---|---|
| tissue only | `rtf 0.096` |
| + both arms | `rtf 0.048` |
| + all three cameras every control step (a scored episode) | ~`rtf 0.02` |
| Isaac reference, everything on | `rtf 0.11` |

Per control step, at the shipped defaults (16 spp / 1 bounce clinical,
8 spp / direct-only front):

| camera | cost |
|---|---|
| endoscope 1280x720 | 6.3 s |
| D405 848x480 RGB-D | 2.9 s |
| front 960x720 | ~70 s |

The front view now dominates, because its scene is 5x the triangles. Drop
`front=False` (or raise `--image-every`) if you only need the clinical feeds.

Physics is competitive; **rendering dominates a scored run**. A 20 s episode is
75 control steps and each one path-traces a 1280×720 endoscope frame plus an
848×480 D405 frame, so a 43-minute episode is ~35 minutes of rendering. Levers,
in order of value:

- `--samples` (env `samples=`) is linear. 24 spp is the default; 8 spp is
  usable for training-time monitoring, 96+ for a presentation video.
- `bounces=0` drops indirect light and is ~5× faster (14.5 s → 2.8 s per
  1280×720 frame at 16 spp).
- `--image-every N` renders every Nth step — `isaac/vla_rollout.py` already
  supports it and it applies unchanged here.
- One process per GPU. This host has 8; `CUDA_VISIBLE_DEVICES=<n>` picks one,
  and episodes parallelise across them with no shared state.

### Implemented components (validation is uneven)

These components are wired in the backend; this table does not imply that
physics, contact, appearance, or task-performance gates have passed. See the
timestamped `validation/` reports for failures and the scope of each check.

| stage | file |
|---|---|
| FR3 + tool URDFs, no Isaac | `fr3_urdf.py`, `tool_urdf.py` |
| cell layout + tissue spec, from source | `cell_spec.py`, `layout.py` |
| tissue FEM, pins, appendix shelf | `scene.py` |
| instruments driven by the shipped IK + controller | `arms.py`, `sim.py` |
| compliant grasp assist (bounded spring-damper loads on FEM nodes) | `grasp.py` |
| post-step model-based contact-load estimates on the tool_tip axes | `forces.py` |
| path-traced endoscope + D405 (+ stereo, + depth) | `render.py`, `cameras.py`, `view.py` |
| `LaparoVLAEnv` / `Action` / `DemoPolicy` | `vla_env.py` |
| scored rollout → npz + mp4, via the shipped runner | `rollout.py` |

### Deliberate differences from the Isaac backend

- **`track_lag` is ~0 by construction.** The arm is a position-driven follower,
  which is what `SURGICAL_CELL.md` specifies; the 0.38 mm p95 in the Isaac
  reference run is the PhysX servo's own following error, a property of that
  plant. Reported rather than hidden.
- **Organ deformation will not match PhysX FEM.** Explicitly allowed by the
  contract: *"Robot-arm movement must be identical… Organ deformation is
  allowed to differ."*
- **The force channels come from tissue contact, not a joint reaction sensor.**
  A kinematically driven link reports no reaction, so `forces.py` estimates
  load from the post-step soft-contact manifold. It follows Newton's penalty,
  damping and friction law form, but uses static mixed material stiffness
  rather than the per-contact AVBD penalty that changes during solver
  iterations. It is not a readback of the exact force used in a solver
  iteration or a calibrated tissue-force measurement. The kinematic model has
  no robot payload weight to compensate, but that does not calibrate tissue
  load. The grasp assist adds a damped, force-capped
  spring to selected FEM nodes instead of overwriting their positions. Its
  reported spring load is a model input, not a measured tissue-contact force;
  neither the 30 N/m stiffness nor the 0.03 N cap is physically calibrated.
  The force-feedback and tissue-deformation response still require runtime
  tests and calibration against measured tissue data.
- **The field, phantom and endoscope carry no collider.** With a kinematic arm
  nothing responds to them, the trocar constraint is enforced analytically by
  `trocar.py`, and the field in particular must not touch the organs.
- **No GUI.** `--gui` is refused with a message rather than silently ignored.

### Contact-material diagnostics

Newton VBD's FEM-particle to rigid-instrument contact averages the particle
`soft_contact_mu` and instrument shape `mu` values. Its body-particle path does
not consume `ShapeConfig.kf`; setting `--contact-kf 0` is therefore not a valid
friction ablation for this contact pair. `--contact-mu X` is an explicit
diagnostic override applied to both sides. It is not a calibrated material
setting and is not a training default. In the 2026-10-06 one-scene, three-seed
probe, `mu=0` removed cecum element inversion but produced 85–136 mm/s peak
tissue-node speed and completed no appendectomy stages. The authored setting
remains unchanged pending measured contact/deformation validation; see
`validation/newton-tool-friction-ablation-2026-10-06.json` and
`validation/newton-tool-mu-ablation-2026-10-06.json`.
