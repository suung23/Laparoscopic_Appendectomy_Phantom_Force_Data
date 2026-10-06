# Robot-agent RL interface

The current task is **contact-based appendix grasp and lift**, with two actuator commands. It does not yet control a robot arm, trocar-constrained shaft, scissors, ligature or a full appendectomy. PhysX FEM and RTX run on the local GPU; the small PPO network runs on CPU to avoid GPU contention. There is no network server or external experiment tracker.

## Run on this workstation

Use the repository PowerShell launchers with the existing Isaac Python 3.11 installation. Close another Isaac simulator instance before starting to avoid sharing GPU memory between independent physics processes. EULA acceptance is inherited from the already configured installation.

```powershell
# Visible scripted agent: close, hold, lift, finish, reset, repeat.
./run-agent.ps1 -Policy scripted -Episodes 0

# Repeatable finite baselines.
./run-agent.ps1 -Headless -Policy scripted -Episodes 3
./run-agent.ps1 -Headless -Policy random -Episodes 3

# PPO from random initialization; convergence is not guaranteed by this command.
./run-agent.ps1 -Headless -TrainSteps 4096 -Episodes 3

# Optional, explicitly labelled demonstration initialization, then real PPO updates.
./run-agent.ps1 -Headless -TrainSteps 2048 -DemoWarmStart -Episodes 3

# Deterministic evaluation of the saved learned network, with the Isaac window.
./run-agent.ps1 -Policy checkpoint -Checkpoint out/ct_appendectomy/agent/policy.pt -Episodes 0

# Evaluate that checkpoint after validating it against its training scene,
# then switch Kit to a separate material-sensitivity asset for evaluation.
./run-agent.ps1 -Headless -Policy checkpoint `
  -Checkpoint out/ct_appendectomy/agent/policy.pt -Episodes 10 `
  -EvaluationAssets out/ct_appendectomy/material_sensitivity_control `
  -EvaluationDomain control_mean_shear_equivalent

# Resume network + Adam optimizer for additional PPO interactions.
./run-agent.ps1 -Headless -Checkpoint out/ct_appendectomy/agent/policy.pt -TrainSteps 4096

# Validate an RGB + state observation path with the baseline controller.
./run-agent.ps1 -Headless -Policy scripted -Episodes 1 -RgbObservation
```

`-Prepare` rebuilds the current CT/atlas scene and prepares a fresh cooked forceps stage. Otherwise the saved stage/config are reused and their SHA-256 contract is checked. Changing assets invalidates incompatible checkpoints. The default prepared stage is `out/ct_appendectomy/suite/forceps/attached.usda`. The launcher also prepares assets if its configuration file is missing. Public CT/atlas data and the installed Isaac environment are prerequisites, not bundled with this repository.

Use `-OutputDirectory out/ct_appendectomy/my_experiment` to keep separate run reports and checkpoints. Reusing the default output directory replaces its summary and latest trajectory; JSONL journals retain run IDs.

`./build-launcher.ps1` also creates `RobotAgent.exe`, a terminal-free visible agent launcher. It replays `out/ct_appendectomy/agent/policy.pt` when present, otherwise uses the scripted baseline. Its panel shows policy, episode, step, return, tissue lift and base drift, and has a pause/resume button. A scripted demonstration is labelled `scripted`; it is not a learned network. An existing incompatible checkpoint causes an error rather than silent fallback.

The panel's Scripted / Random / Learned buttons select the controller for the next episode. The current episode finishes normally before switching; its log retains the actual controller name. Closing the Isaac window stops the continuous demo.

Local verification on 2026-10-04: the new executable compiled, but the Windows UI launcher returned `ShellExecuteW returned 5` twice. The new panel's visible interaction is therefore unverified. Headless Python execution, Gym registration, real PPO updates, checkpoint resume and RGB-observation rollouts were verified. This launch failure is not evidence that a GUI window opened.

## Environment contract

Import `agent_env` **after** initializing Isaac SimulationApp. This registers `RiiAppendixContactLift-v0` with Gymnasium:

```python
import gymnasium as gym
import agent_env

agent_env.bind_simulation_app(app)  # Keep Kit out of Gym's copied spec kwargs.
env = gym.make(
    agent_env.ENV_ID,
    stage_path="out/ct_appendectomy/suite/forceps/attached.usda",
    config_path="out/ct_appendectomy/suite/forceps/forceps_config.json",
    observation_mode="state",  # or "rgb_state"
)
obs, info = env.reset(seed=42)
while True:
    action = policy(obs)  # float32 [jaw_increment, vertical_increment], each [-1, 1]
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
```

The two normalized actions increment jaw closure by 0.025 and tool height by 0.2 mm. Closure is bounded to [0,1], height to [0,16 mm]. One action advances 1/30 simulated second through four 1/120 physics steps. This is not a free six-degree-of-freedom manipulation task.

State observations contain 14 float32 values, named by `STATE_NAMES`: two actuator commands, tissue-centroid displacement xyz, pin drift, tracking error, elapsed fraction, held fraction, reference displacement xyz, reference validity and reference command height. Positions are normalized by documented 1/4/10 mm scales in `agent_env.py`. This is a compact privileged observation, not the complete FEM state and not an image policy. `rgb_state` adds uint8 360×640×3 RGB as a dictionary entry; the bundled MLP trainer accepts state observations only. Rendering does not advance physics.

Success requires ten held steps after closing and commanding at least 8 mm lift relative to the grasp reference, with tissue lift at least 6 mm and tracking error below 4 mm. The first upward action after closing records the tissue/command reference at the current height; opening below 0.5 clears it, allowing regrasp after exploration. Base drift over 1 mm, node displacement over 50 mm or an inverted tetrahedron ends the episode as `tissue_stability`. These are engineering gates, not clinical injury thresholds. The external 220-action budget produces `truncated=True`; success/failure produces `terminated=True`. Invalid actions are rejected, and a finished episode must reset before further actions.

Reward is +5 on success, -5 on physical failure, -0.002 per action, plus potential shaping `0.99 * Phi(next) - Phi(current)`. Phi includes closure progress (0.25) and retained lift progress (0.75). Terminal Phi is zero; a time limit retains its bootstrap value. Use gamma 0.99 with this reward version. `info.is_success` and `termination_reason` determine outcomes; return alone is not a success test.

## Rosota FR3 waypoint PPO pretraining

`rosota_rl_adapter.py` wraps the actual `RosotaCTEnv` controller path in a 14-D
normalized action and 53-D privileged state. Each arm controls local tip xyz,
shaft-axis roll and jaw aperture (-1 closed, +1 fully open); two tilt channels
are fixed at zero under the strict aperture profile. Translation is capped at
0.12 mm per 30 Hz action and roll at 0.4 rad/s. Four additional actions move
the endoscope through its geometric trocar pivot, but scope movement has no
reward term in this waypoint task. The state contains measured joint/tool/goal/scope
poses, jaw state, locally tared mount reactions and contact/hold flags. The
reaction and jaw-effort features are simulator sensor signals, not calibrated
tissue force. This is not an image policy.

`run_rosota_rl.py` trains the local PPO implementation and evaluates matched
seed sets against a geometric waypoint controller and a random policy. Its
current objective is only two-arm trocar-constrained waypoint reach; it does
not reward scope exploration, train a grasp policy, dissect, or complete appendectomy.
The current CT articulation has no shaft-organ collision geometry. A ten-step,
0.5 mm waypoint hold is an engineering benchmark threshold, not a surgical
success criterion. `--eval-assets` records a different scene hash but
does not establish patient-level held-out generalization. Reset-state hashes
are compared because reset equilibrium is not yet repeatable.

Example after preparing `multitask_assets`:

```powershell
./../../work/isaac-venv/Scripts/python.exe ct_appendectomy/run_rosota_rl.py `
  --headless --train-steps 4096 --eval-episodes 10 `
  --assets out/ct_appendectomy/multitask_assets `
  --out out/ct_appendectomy/rosota_rl_waypoint_v1
```

The adapter, PPO updates and deterministic policy contracts pass CPU/mock tests
only. The local Isaac Python executable is blocked by Windows Application
Control, so no Rosota PPO or evaluation episode is claimed.

To select the unassisted patch-traction benchmark, add
`--task unassisted-traction` to that command and use a separate output directory. It
sets `grasp_mode="none"` and requires the simulator's jaw/tissue contact path.
Its 65-value privileged observation adds each target patch's position in the
local shaft frame and its displacement from reset. The dense reward encourages
both jaws to approach their respective target patches, register the contact
latch, and lift each patch 5 mm. Success requires both patches within 5 mm of
their tips and lifted at least 5 mm for ten policy steps. This contact latch is
an uncalibrated simulator proxy, not force in validated tissue units. The task
does not cut tissue or complete an appendectomy. Adapter tests run without
Isaac; live contact, physics-substep health, repeatability, held-out anatomy,
RGB-policy learning and expert review remain unverified.

For an image-input variant, select `--task vision-traction` with its own
checkpoint and output directory. For example:

```powershell
./../../work/isaac-venv/Scripts/python.exe ct_appendectomy/run_rosota_rl.py `
  --headless --task vision-traction --train-steps 4096 --eval-episodes 10 `
  --assets out/ct_appendectomy/multitask_assets `
  --out out/ct_appendectomy/rosota_vision_traction_visibility_v2
```

Use `--visibility-shaping-weight 0` for the matched reward ablation. The weight
is included in the checkpoint contract so a checkpoint cannot silently move
between shaping conditions.

The adapter captures the actual `/World/Endoscope`
RGB render at 640×360 and resizes it to 160×90, paired with 45 robot/jaw/scope
proprioceptive values. It deliberately omits the target-patch positions and FEM
displacements from policy input. Those values still drive reward and success
measurement, so this is a visual actor with simulator-privileged reward
shaping, not a fully deployable end-to-end controller. Isaac semantic
segmentation also reports the appendix's visible pixel fraction, and a small
potential-based reward term encourages better appendix framing. Segmentation is
simulator-only ground truth and is excluded from the policy observation; the
visibility term measures visible image area, not exploration coverage or
anatomical completeness. RGB and semantic masks are captured from one render
tick. CPU/mock tests verify the fraction calculation and reward channel, but the
live Replicator path, shaping scale and any learned improvement remain
unverified. The checkpoint contract
separates this CNN policy from the 53- and 65-value MLP policies. A mock image
rollout verifies PPO updates, but Isaac rendering, GPU/physics throughput,
contact safety, held-out anatomy, visual policy performance and surgery remain
unverified.

## Learning and evidence

The local PPO implementation uses tanh Gaussian actions, clipped policy updates, a value baseline, GAE, gradient clipping and separate masks for termination and truncation. See the [PPO paper](https://arxiv.org/abs/1707.06347) and [Gymnasium time-limit semantics](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/). Optional behavior cloning collects one successful real simulator demonstration; its steps and loss are reported separately from PPO interaction counts.

Outputs stay under ignored `out/ct_appendectomy/agent/`: `report.json`, episode/training JSONL, the latest action/observation trajectory, `latest_endoscope.png`, and `policy.pt`. Checkpoints contain network parameters, Adam state, environment contract, asset hashes and cumulative training steps. Loading uses `torch.load(weights_only=True)` and rejects a different environment or scene. Resume is a new episode with restored model/optimizer, not a bit-exact mid-episode continuation.

The bimanual `run_rosota_rl.py` path now reads FEM health after every underlying `sim.step()` during each 1/480 s controller interval. A 30 Hz action therefore records 16 solver ticks at `physics_substeps=1`, 32 at 2, or 64 at 4. Any sampled inversion stops the remaining substeps in that interval and marks the episode failed. PPO training/evaluation episode records aggregate sample counts, inversion counts, minimum signed volume ratio, and maximum p99 volume change; missing interval samples fail the sampling-complete flag. CPU/mock tests cover aggregation and early stop, but no Isaac runtime execution has verified this instrumentation yet.

The paired bimanual reaction-transfer probe now slices only the trace rows produced by its current action, then requires complete solver-substep sampling and the expected tick count before using a trial in its paired comparison. It no longer falls back to the last 16 global rows when an action terminates early.

The separate 2-actuator `agent_runner.py` contact-lift fixture still records tissue metrics once per action rather than every underlying physics tick. It has no force sensor/readback; reports explicitly mark `force_readback_available=false`. Do not infer force from deformation, jaw command or reward.

A successful runner status means execution and contracts passed. It does **not** mean the policy succeeded: inspect evaluation successes and per-episode outcomes. Evaluation uses separate seeds, but anatomy is fixed and those seeds do not create distinct patients; PhysX may also have small numerical nondeterminism. Do not report a few episodes here as generalization, clinical skill or full-surgery competence.

One SimulationApp owns one global USD stage. Multiple Gym instances in the same process are unsupported. Actual multi-environment throughput, image-policy learning, articulated robot control and anatomical domain randomization remain later work.

## Asset-held-out evaluation

`ct_appendectomy/build_material_holdout.py` authors local material sensitivity assets from the forceps training scene. It maps the published human appendix storage-modulus group mean to an isotropic small-strain equivalent Young's modulus using the scene's Poisson ratio. These conditions are literature-scaled probes, not measured Young's moduli, fitted FEM parameters, patient variants or validated tissue states. Generate the available conditions with:

```powershell
./../../work/venv/Scripts/python.exe ct_appendectomy/build_material_holdout.py `
  --source-assets out/ct_appendectomy/suite/forceps `
  --out out/ct_appendectomy/material_sensitivity_control --group control
./../../work/venv/Scripts/python.exe ct_appendectomy/build_material_holdout.py `
  --source-assets out/ct_appendectomy/suite/forceps `
  --out out/ct_appendectomy/material_sensitivity_purulent --group purulent
./../../work/venv/Scripts/python.exe ct_appendectomy/build_material_holdout.py `
  --source-assets out/ct_appendectomy/suite/forceps `
  --out out/ct_appendectomy/material_sensitivity_gangrenous --group gangrenous
```

With `-EvaluationAssets`, the runner first loads a checkpoint only if its saved scene/config hashes match `-Assets` (the training asset directory), then closes that environment and opens the evaluation asset directory. The episode report records both asset hashes, whether they differ, and the human-readable condition label. The asset builder also verifies the USD physics-purpose binding from `/World/Atlas/appendix/sim` resolves through `/World/Atlas/appendix.material:binding:physics` to `/World/TissueMaterial`. This proves scene composition, not that the Isaac runtime consumes the override. The runner explicitly sets `patient_level_holdout_verified=false`; changing one material parameter is not anatomical generalization. Each directory is evaluated in a separate run and output directory. The installed Isaac Python runtime remains subject to the recorded Windows Application Control launch restriction, so authored assets and CLI wiring are preflighted but no new RL result is claimed until the actual episodes run.

## Recorded local results

- Initial scripted interface: 3/3 successful episodes (the earlier 13-value observation contract).
- Current 14-value registered interface: exact initial-state reset and render-state invariance, invalid-action rejection without mutation, and nonzero-height grasp-reference checks passed. Its 189-step demonstration succeeded.
- Final PowerShell launcher regression on the 14-value interface: 1/1 scripted success, fresh successful report, and state/action/next-state trajectory export passed (`ct-agent-scripted-current.json`).
- Demonstration-initialized PPO: 2,048 environment steps, eight updates, nonzero parameter change, saved checkpoint reload passed. Deterministic evaluation: **0/3 successes**, all time limits. Do not label this a competent learned controller.
- RGB+state random rollout: 220 steps, correct truncation, no inverted tetrahedra.
- Checkpoint resume: loaded the 2,048-step model/Adam state, performed another 256 steps and saved/reloaded the 2,304-step checkpoint. Evaluation: 0/1 successes.

Evidence is in `validation/ct-agent-*.json`. The primary 2,048-step policy remains at `out/ct_appendectomy/agent/policy.pt`; the resume test writes a separate checkpoint under `out/ct_appendectomy/agent_resume` and does not overwrite it.
