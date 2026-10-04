# Bimanual exploration and physics candidate checks

`ExplorationEnv` extends the existing CT/atlas scene with two independently
actuated instruments and an actuated endoscope. It supports Gymnasium reset,
step, RGB render and close. Pass the existing SimulationApp and the prepared
`multitask_assets` folder. Only one environment may own the Isaac process.

The 16 normalized velocity actions are left tool (yaw, pitch, insertion, roll,
jaw), right tool (same five), then scope pan, tilt, roll, right, up, insertion.
The 42 privileged observations are tool joints/held flags (12), paired goal
errors / 30 mm (6), normalized scope commands (6), visited mask (7), target
one-hot (7), remaining time (1), and three geometric tissue diagnostics.
RGB is available through `render()`; the bundled reference planner does not
consume pixels and must not be described as a vision policy.

The current benchmark covers seven paired targets around the initial instrument
poses. The next target minimizes travel among unvisited targets using the current
tool state. Both tips must remain within 2 mm for 15 consecutive steps. Revisits
do not earn coverage again. The scope follows the next tool-target midpoint.
Coverage describes tool workspace visitation, not visible organ surface area.
Camera translation is bounded to a 4 mm ball around this scene's planned eye;
the camera is a point model, not a collision-validated endoscope shaft.

This is local exploration around the appendix, not whole-abdomen route planning.
The other CT organs remain static visuals; the environment does not certify
shaft-organ or trocar clearance. It checks appendix FEM validity on every
control step, rejects invalid actions before mutation, and terminates on the
same pin drift/inversion/displacement limits as the bimanual environment.

## Reproduce

Close other Isaac processes, then from repository root:

```powershell
$env:OMNI_KIT_ACCEPT_EULA='YES'
& '..\..\work\isaac-venv\Scripts\python.exe' ct_appendectomy/run_exploration.py --headless --out out/ct_appendectomy/exploration_new
& '..\..\work\venv\Scripts\python.exe' ct_appendectomy/encode_exploration.py out/ct_appendectomy/exploration_new
& '..\..\work\isaac-venv\Scripts\python.exe' ct_appendectomy/run_exploration.py --headless --policy checkpoint --train-steps 1024 --out out/ct_appendectomy/exploration_ppo_new
```

Use a fresh output folder to avoid stale recording frames. Captures are actual
Isaac RGB images and do not advance physics. MP4/GIF captions label the planner
as geometric, not learned RL. Nominal video sampling is 5 Hz; the final sample
may be less than a full sample period after its predecessor. The recording is
not a measurement of wall-clock real-time performance.

PPO uses the 42-state/16-action contract, with a distinct environment ID and
checksums for both the exploration and base environment code. A failed learned
evaluation is not a successful demo; its recording can be encoded explicitly
with `encode_exploration.py --allow-failed` and retains the evaluation status.

## Physics candidate, not an engine replacement claim

`newton_contact_probe.py` runs Newton 1.4 / Warp CUDA VBD on a conforming 2 mm
voxel approximation of the atlas appendix, with two kinematic spherical tools,
posterior gravity, fixed base and ground support. A matched run disables tool
particle contact. Both runs check tetrahedral inversion every control frame,
base drift and the difference in tissue displacement under enabled contact.

```powershell
& '..\..\work\venv\Scripts\python.exe' ct_appendectomy/newton_contact_probe.py
& '..\..\work\venv\Scripts\python.exe' ct_appendectomy/newton_contact_probe.py --substeps 16 --iterations 32 --out out/ct_appendectomy/newton_contact_fine
& '..\..\work\venv\Scripts\python.exe' ct_appendectomy/newton_contact_probe.py --substeps 16 --iterations 32 --cuda-graph --out out/ct_appendectomy/newton_contact_graph
```

This coupon has different meshes and tools from the PhysX CAD scene, so it
cannot rank engines fairly. It does not have calibrated material parameters,
CAD jaw contact, grasp, cutting, fluids, organ coupling or Isaac integration.
An engine candidate must pass before replacing the working scene backend.

Measured on this machine: the 960 Hz / 32-iteration eager VBD coupon passed
with zero inverted tetrahedra in sampled control frames for both contact modes;
the maximum contact-induced displacement difference was 6.35 mm. The 240 Hz /
12-iteration run failed. CUDA Graph replay failed the repeated contact test
with four inverted tetrahedra and remains an experimental negative control,
not an accepted acceleration. These are numerical tests, not clinical validation.

`prepare_physics_profile.py` creates a separate hashed PhysX asset profile at
480/960/1920 Hz with a specified solver iteration count. It never overwrites the
source folder. Raising precision alone did not fix the CAD contact baseline at
960 Hz/128 iterations; this remains a failed engineering gate. The base Gym
environment now reads the physics interval from its hashed asset configuration.
Existing PPO checkpoints pin the base environment source hash and must be
retrained or explicitly migrated after this contract implementation change.

Recorded exploration outcome: the geometric planner visited 7/7 paired targets
in 259 steps, with both tips moving 9.07 mm from their start and the scope also
moving. The actual 256-step PPO update and checkpoint load passed, but evaluation
visited only the initial target (1/7) and reached the 600-step limit. It did not
learn successful exploration. Videos label the two policies separately.

## Primary references checked 2026-10-04

- [Rosota's contact-stable tissue study](https://arxiv.org/abs/2609.25642):
  implicit integration, separate simulation/contact meshes and live-pose grasp
  constraints are relevant to this implementation. The paper explicitly limits
  its fidelity claims.
- [LapGym](https://arxiv.org/abs/2302.09606): SOFA-based laparoscopic RL tasks
  provide a surgical task benchmark reference.
- [ORBIT-Surgical](https://github.com/orbit-surgical/orbit-surgical): an Isaac
  Sim-based surgical robot learning framework.
- [NVIDIA Medical Physics Simulation](https://isaac-for-healthcare.github.io/medical-physics-simulation/):
  the public documentation lists the soft-tissue surgical component as not yet
  ported. Available generative video and endoluminal components do not establish
  a ready-to-use, validated appendectomy physics backend.
- [Newton](https://github.com/newton-physics/newton): the installed CUDA VBD
  solver is tested locally; using this engine alone does not establish SOTA.

Remaining gates are whole-tool collision-aware routes, calibrated organ contact,
visual coverage measurement, learned exploration on held-out cases, and a matched
cross-engine contact benchmark. Surgical quality is not certified by waypoint
coverage or a successful video recording.
