# Bimanual RL environment v1

This is an Isaac Sim 5.1 / PhysX manipulation environment in the hybrid human
CT + atlas scene. It is not a validated surgical trainer or a SOTA claim.
The appendix is atlas-derived, not patient-specific. Other CT organs are static
visuals. Grasp assistance uses geometry-gated FEM attachments, not measured
friction, force sensing or injury thresholds. Cutting, ligation, bleeding,
mesenteric mechanics and full appendectomy are not implemented here.

## Interface

`BimanualEnv(app, assets, task='retract')` receives an existing SimulationApp;
only one live environment is supported per process. The caller owns app.close().
`reset(seed=..., options=...)`, `step(action)`, `render()`, `close()` follow the
Gymnasium contract. No external trackers or network calls are used.

Actions: 10 normalized velocities, Left then Right, each yaw, pitch, insertion,
roll, jaw opening. Physics 480 Hz, control 60 Hz of simulated time. Wall-clock
throughput is measured separately. Both instruments rotate about fixed virtual
ports. Anatomical port placement and shaft-organ collisions remain unvalidated.

Observation: float32 vector of 31 privileged values: joints/held flags (12),
mouth-to-task-target vectors divided by 30 mm (6), patch displacements divided
by 10 mm (6), requested lift divided by 10 mm (1), selected-patch ownership (2),
task one-hot (3), remaining episode fraction (1). This is not the complete FEM
state. `render()` returns the actual endoscope RGB uint8 image, 640 x 360;
visual-policy learning is not implemented by the bundled state-policy trainer.
The RL camera is fixed; interactive camera controls are provided separately by
the bimanual desktop scene. Camera actions are not part of this 10-action contract.

Tasks require 30 consecutive control steps of achievement:

- `reach`: both instrument mouths within 2 mm of pregrasp targets, 18 mm
  toward each port from its selected patch. This task stops before tissue contact.
- `dual_grasp`: each hand owns its designated, distinct patch.
- `retract`: both designated patches held at requested height within 1.5 mm,
  with transverse displacement below 3 mm.

Default patches [1, 5]; explicit `patches` accepts any two distinct valid indices,
but other pairs require separate reachability tests. `lift_m` is 4–8 mm, drawn
using the environment RNG unless explicitly supplied. This variation changes
task goals, not anatomy. There is only one CT case in this benchmark.

Reward uses gamma=.99 potential shaping, step cost -.002, success +5, failure
-5. Terminal shaping potential is zero; time-limit truncation still bootstraps.
Failure overrides simultaneous success. Pin drift >1 mm, inverted tetrahedra,
or displacement >50 mm terminate. These are engineering validity limits, not
clinically calibrated damage metrics. FEM health is sampled every control step,
not every physics substep. Solver non-finite output aborts rather than fabricating
a valid observation. Reset rejects a non-deforming/inactive physics backend.

## Run locally from repository root

Use the existing Isaac Python environment, after closing the other Isaac process.

```powershell
$env:OMNI_KIT_ACCEPT_EULA='YES'
& '..\..\work\isaac-venv\Scripts\python.exe' ct_appendectomy/run_scene.py --scene out/ct_appendectomy/bimanual_scene/ct_appendectomy.usda --bimanual --headless --frames 360 --out out/ct_appendectomy/multitask_assets
& '..\..\work\isaac-venv\Scripts\python.exe' ct_appendectomy/bimanual_benchmark.py --headless --seeds 42 43
& '..\..\work\isaac-venv\Scripts\python.exe' ct_appendectomy/bimanual_benchmark.py --headless --tasks reach --train-steps 2048 --policy checkpoint --out out/ct_appendectomy/bimanual_ppo
```

Scripted and random baselines are explicitly labelled. PPO uses the shared local
implementation with a new 31-input/10-output model; legacy 14/2 checkpoints are
incompatible. Checkpoints verify task, stage and configuration hashes before
loading. Training seed starts at 10000, default evaluation seeds are 42 and 43.
This split varies goals only; it is not held-out anatomical generalization.
The sampled lift goal affects `retract`; different seeds do not yet create
different reach geometry. Reach evaluations are repeatability checks.
An update smoke test does not establish convergence. Read the episode success
rate and physical failure reasons before assessing policy competence.

The benchmark writes reset time, control-step p50/p95, per-episode outcomes and
tissue health locally. Rendering readback and startup costs are not included in
control-step latency. Benchmark completion does not imply all episodes succeeded.

`run-multitask.ps1 -Headless -Task reach` runs the scripted approach check;
add `-Prepare` to rebuild physics assets, or `-Train -Steps 2048` to train and
evaluate PPO. `--compare-scripted` in the Python runner evaluates both policies
and labels every episode independently.

The experimental `run_scene.py --jaw-collision convexDecomposition` preserves
more of the CAD shape than a single convex hull. It did not resolve the observed
approach inversions and is not the default or an accepted stability fix.

## Remaining asset-quality gates

1. Validated CT/appendix registration, peritoneum/insufflation geometry and ports.
2. Coupled surrounding-organ and mesentery mechanics, calibrated contact/grasp.
3. Validated live cutting, ligation, injury and bleeding models.
4. Multiple cases, expert annotations and task completion review.
5. Learned-policy held-out success, baseline comparisons and sim-to-real study.

These gates must be evidenced before calling the asset a high-fidelity surgery
simulator or claiming state-of-the-art performance.
