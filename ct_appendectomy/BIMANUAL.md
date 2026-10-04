# Bimanual appendix manipulation

`IsaacSimulator.exe` now opens the prepared bimanual scene. `./run-bimanual.ps1`
does the same, preparing missing assets if necessary. Use `-Prepare` to rebuild
and find a camera view of both settled grasp sites; `-Headless -Demo` runs the
scripted acceptance test. The camera is fixed throughout the recorded test.

Each hand has its own yaw, pitch, insertion, shaft roll and jaw-opening control.
The shaft pivots mathematically through its fixed port. Two copies of the
Rosota clean-export grasper CAD are used; no full robot-arm dynamics or clinical
trocar calibration is implied. The UI button **Both hands: approach / grasp /
lift / release** executes a scripted manipulation cycle. **Release both** opens
the jaws and cancels it. Camera controls are available through a separate button.

The controller exposes `step_action(action, dt)` with ten normalized velocity
actions: left yaw/pitch/insertion/roll/opening, followed by the same five for the
right. Invalid actions are rejected before either hand changes. `observation()`
returns each hand's five joint coordinates and its grasp flag (12 values).
This is a control interface, not a newly trained policy. The existing
`RiiAppendixContactLift-v0` Gym benchmark and its checkpoints remain separate.

## Grasp model and validation

The private upstream implements a grasp assist (`newton_cell/grasp.py`,
`isaac/teleop_left_touch.py`, and `isaac/build_surgical_cell.py` at commit
`9135fe9eb8549d011124c8c57b7c86edeaaba735`). This implementation follows that
approach: small disjoint FEM patches have registered attachments that are
disabled until a closing jaw mouth contains appendix tissue. Other hands
cannot own the same patch. Attachment offsets are updated to the live tissue
position before activation. Opening releases the attachment. The captured
patch follows both translation and rotation of the tool.
The executed jaw coordinate stops at the grasp angle rather than integrating
a hidden closure command; releasing therefore opens from its visible position.
The bimanual mode uses hex resolution 64 and a 1/480 s physics timestep with
1/60 s control. Grasp patches have a 2.5 mm selection radius, with at least
eight nodes, to keep their physical footprint across mesh resolutions.

This is an explicit geometry-gated constraint, **not a measured contact-force
sensor or a validated friction-only grasp**. Jaw visuals use CAD triangles;
their collision geometry uses convex hulls. The gate tests appendix nodes in
the mouth, not a clinical organ-recognition model. Other CT organs still have
no coupled deformable/contact response. Instruments may intersect those static
visual surfaces; surgery-quality access/exposure and collision avoidance remain
unfinished. Mesentery and artery also remain uncoupled static atlas surfaces.

The acceptance sequence closes in empty space, opens, approaches two patches,
closes, lifts 8 mm, holds, opens and withdraws. It checks distinct ownership,
4 mm minimum tissue lift, 3 mm maximum tracking error, release, unchanged camera
and tetrahedron orientation at the lift checkpoint and final state. Activation
motion is checked on the actual attached patch (1 mm maximum); the largest
movement of any other node is reported separately. This is not a continuous
strain, force, injury, or transient inversion certification.

## Gravity and the rabbit reference

The human scene applies a viewer-axis swap after `ct.json`'s RAS-to-viewer map.
Their linear product is identity. Gravity `(0,-9.81,0)` therefore points
posteriorly in anatomical RAS, under an explicit **supine posture assumption**.
USD's Z-up metadata does not determine this explicitly authored gravity vector.

Rosota's rabbit meshes come from a top-down D405 capture. Upstream uses
`diag(1,-1,-1)` to map camera coordinates to its Z-up world, and gravity is
`(0,0,-9.81)`. The meshes are transformed together, never centred individually.
`inspect_rosota_anatomy.py` verifies their original hashes and exports a
separate reference USD and two-view figure in `out/rosota_anatomy`.

The captured cecum has a 28.0 × 58.7 × 24.3 mm bounding box; the appendix has
17.9 × 65.3 × 16.7 mm. Their nearest sampled mesh vertices are 0.566 mm apart.
These are reconstruction measurements, not complete-organ lengths or reviewed
anatomical landmarks. The rabbit assets are **not substituted into human CT**.
No new clinical anatomy or material-calibration claim follows from inspecting them.
