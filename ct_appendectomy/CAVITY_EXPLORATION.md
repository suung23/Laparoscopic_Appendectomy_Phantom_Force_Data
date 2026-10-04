# Supine cavity exploration

Run `./run-cavity.ps1` from the repository for this single-instrument scene.
It prepares the CT/atlas scene and opens Isaac with endoscope and Rosota gripper
controls. `-Headless -RecordScan` generates 90 RTX camera frames.
`IsaacSimulator.exe` now opens the newer two-hand scene described in `BIMANUAL.md`.

The endoscope panel provides pan, tilt, roll, lateral motion and insertion.
The camera light follows its optical centre. Translation is bounded to an 8 mm
ball around a position selected outside all source organ surfaces and inside
the body envelope. The sphere is checked against source geometry, not a moving
instrument, the full scope shaft, or a clinically calibrated trocar path.

Gravity points posteriorly (-Y in this scene's anatomical coordinates) at
9.81 m/s². The appendix has a base attachment and settles against a local
posterior support collider. This is an engineering support approximation,
not segmented peritoneum. CT organs remain static in their reconstructed poses;
this is not a fully dynamic abdominal cavity. Mesentery and artery are still
static atlas surfaces and not mechanically coupled to the appendix.

Seven private Rosota clean-export STL assets are imported at millimetre-to-metre
scale. The shaft, clevis components and independent left/right jaws retain
their measured pivot frames. Positive opposing hinge rotation opens the jaws
up to 74 degrees total. Visuals use the original triangles; contact uses convex
hulls, which do not resolve every serration. Jaw closure and anterior/posterior
motion are available in the tool panel. Tool forces and tissue properties are
not experimentally calibrated; no grasp-success claim is made for this scene.

`rosota_distal_manifest.json` pins the source commit, paths and Git blob hashes.
Raw private CAD lives only in ignored `assets/rosota_distal`; generated USD,
images and videos live in ignored `out`. No upstream redistribution license is
assumed. The original 42-file snapshot is preserved.

This scene changes gravity, contact geometry and viewing coordinates. The
old `RiiAppendixContactLift-v0` checkpoints remain bound to their legacy scene.
`RobotAgent.exe` still runs that legacy benchmark. Do not describe an endoscope
scan of the new scene as learned-agent performance or an appendectomy success.

Validation rejects inverted tetrahedra, base drift above 1 mm and collision
surface penetration above 1 mm. Simulation cage nodes are reported separately
because a coarse FEM cage can extend outside the collision surface. A run with
`--disable-support` is the contact negative control.
