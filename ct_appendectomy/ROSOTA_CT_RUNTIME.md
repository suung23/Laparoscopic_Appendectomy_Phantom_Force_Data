# CT + Rosota articulation commissioning

This is a separate entry point from the legacy 16-action demonstration.
`run_rosota_ct.py` loads the existing CT/atlas tissue, authors two dynamic
PhysX FR3 articulations, and uses `rosota_controller.RosotaArmController` to
call the unchanged upstream `laparo_control.TipController` at 480 Hz.
Targets go through articulation drives; policy steps never teleport joints.
Joint positions are explicitly initialized only at reset.

The contact-feedback extension is documented in `CONTACT_CONTROL.md`:
measured jaw-effort closure inhibition, torque caps, reopening, and separate
mount-reaction arm stops. The historical free-space evidence below does not
by itself validate tissue contact.

## What the new path actually connects

- FR3 joint origins, limits, masses, centres of mass and inertia tensors come
  from the installed Isaac FR3 URDF. The Rosota tool-tail transform comes from
  the original builder constants through `cell_spec`.
- Seven arm joints and two jaw joints are simulated per arm. The distal CAD
  rigid bodies are fixed/revolute children of the articulation, not independent
  kinematic targets. The existing tissue contact geometry remains enabled.
- Every physics tick reads the reached joint state, calls the original
  controller, applies joint position/velocity targets, steps physics, and
  checks the actual shaft body against the trocar plane. FK from measured
  joints is independently compared with the rigid-body pose readback.
- The scope optical centre moves on the same fixed pivot as its orientation.
  It can pivot, insert/withdraw and roll, but cannot independently translate
  sideways while leaving the shaft fixed.
- RGB capture pauses the timeline and asserts no tissue motion during capture.
  Captions identify a controller commissioning run, not learned RL.

## Gym interface

`RosotaCTEnv(app, assets, fr3_urdf, out, max_steps=300)` is a Gymnasium env.
`reset(seed=...)` returns `(observation, info)` and `step(action)` returns
`(observation, reward, terminated, truncated, info)`.

The action is a dictionary:

- `tools`: `(2,1,7)`, left/right, one Rosota sample per step, with metres/radians
  and gripper opening in `[0,100]`. This version deliberately supports K=1;
  the reference Rosota environment also supports longer chunks.
- `scope`: four normalized velocities in `[-1,1]`: pivot yaw, pivot pitch,
  insertion, roll. This is an explicitly separate extension, not part of the
  upstream seven-component tool action.

The tool frame is captured once per action sample and held through its 16
physics ticks. Tool `-z` inserts and `rx/ry` are reported as rejected. Invalid
second-arm or scope input is rejected before either controller is changed.

Observations contain measured joint positions `(2,7)`, measured-joint FK tool
poses `(2,4,4)`, scope pose `(4,4)` and goal positions `(2,3)`, all float32.
The contact extension adds float32 `contact_residual` `(2,6)`, `jaw_position`
and `jaw_effort` `(2,2)`, plus int8 `contact_stop` and `jaw_contact_hold` `(2,)`.
These force channels are measured joint reactions, not calibrated tissue
forces. Translation requests are capped at 3.6 mm/s for contact commissioning.
The reward is negative mean tip-to-goal distance / 10 mm, with an additional
failure penalty. Physical invalidity terminates; the episode limit truncates.
This is a privileged-state reaching environment. It is not a pixel-trained
exploration policy or an appendectomy completion task. No policy was trained
on this new action contract during commissioning.

## Explicit engineering assumptions

The bases and ports are constructed around the prior CT tool start frames.
The tip-to-port start distance is 120 mm; the source controller's 100 mm
minimum insertion is enforced. This is not measured robot-to-patient or CT
abdominal-wall registration. Exported `scene.usda` records the authored scene.
The original scene/config files are hash-checked and never overwritten.

The original nearby tool placement produced inverted tissue elements after
adding dynamic jaws. Commissioning now starts 15 mm further withdrawn, with
collision still enabled, to test the controller in free space. This does not
solve the failed tissue-contact case and is not a successful grasp test.
The failed initial run is retained in local output/validation records.

Both current distal CAD assets are graspers. The right-hand controller is
therefore configured with the **grasper** tail and 74-degree included opening;
it is not falsely labelled as the reference right cutter (50 degrees).
Full robot CAD and arm collision geometry are not included. Do not use this
scene to evaluate arm-arm, arm-patient or housing collisions.

Scope geometry uses a 5 mm shaft / 6 mm aperture engineering profile and a
100–160 mm insertion range. It has no scope-shaft collision body or calibrated
camera-to-shaft extrinsics. RCM geometry is enforced; tissue clearance and
abdominal-wall reaction forces are not validated.

The robot drive gains come from the simulator controller profile and are not
hardware FR3 gains. A measured-effort jaw closure latch is implemented, but
sustained tissue grasp/traction, cutting, haemostasis, the full Rosota
force/camera observation contract and platform submission remain unvalidated.
`rosota_full_contract=False` remains explicit in the results.

## Local run

From repository root, with other Isaac GPU processes closed:

`./run-rosota-ct.ps1` runs and encodes into a fresh folder. Add `-Visible` to
show the Isaac window while recording. The script checks the actual report,
because the Isaac process exit code alone is insufficient evidence of success.
For the underlying commands:

```powershell
$env:OMNI_KIT_ACCEPT_EULA='YES'
& '..\..\work\isaac-venv\Scripts\python.exe' ct_appendectomy/run_rosota_ct.py --headless --fr3-urdf '..\..\work\isaac-venv\Lib\site-packages\isaacsim\exts\isaacsim.robot_motion.motion_generation\motion_policy_configs\FR3\fr3.urdf' --out out/ct_appendectomy/rosota_new
& '..\..\work\venv\Scripts\python.exe' ct_appendectomy/encode_exploration.py out/ct_appendectomy/rosota_new
```

Choose a fresh output folder. `report.json`, `reset_diagnostics.json`,
`physics_trace.json`, `scene.usda`, chain fixtures and actual RGB frames stay
local under `out/`. `physics_trace.json` is sampled every physical tick after
reset, not just at camera frame rate. The startup settling interval is separate.
Video encoding is nominal sampling playback, not wall-clock performance.
