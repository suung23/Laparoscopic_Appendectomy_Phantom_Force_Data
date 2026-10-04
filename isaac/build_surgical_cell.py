#!/usr/bin/env python3
"""Bimanual surgical cell in Isaac Sim: reconstructed organs + phantom plus two
Franka FR3 arms, on a 140 x (auto) x 70 cm aluminium table (the Y-depth grows
with --port-offset so the bases stay on it; 80 cm at 0.27 m, 100 cm at 0.46 m).

The organ/phantom half is reused from out/scene_soft.usd (built by
isaac/build_deformable_usd.py), where the phantom is baked at the world origin
with its base on z=0. Here the whole specimen (organs + phantom + colliders) is
posed for the real rig, then the arms, table and trocar markers are added.

Layout (metres, Z-up; table top = robot-base plane = z=0):
  - the specimen is spun 180 deg about Z (--rotate-phantom) so the used trocar
    ports (2,3) face the arms, and raised --lift (0.01 m) onto its plate
  - two FR3 bases --spacing apart along X, symmetric about X=0
  - the base line sits --port-offset (0.23 m, measured in the XY plane) in -Y
    from the midpoint of the two used ports, so the arms stand off the phantom
  - the aluminium table hangs below z=0; a room floor sits at its base
  - a RealSense D405 sits in the phantom's top hole (middle of the four highest
    ports), looking straight down (-Z), X/Y world-aligned -- the capture camera

    ~/isaacsim-5.1/python.sh isaac/build_surgical_cell.py                 # build + verify PNG
    ~/isaacsim-5.1/python.sh isaac/build_surgical_cell.py --view --reuse-robot   # interactive

FR3 kinematics come from the local motion-generation fr3.urdf. Its meshes are not
bundled next to it, so the mesh paths are retargeted onto the franka_description
meshes that ship with the URDF importer -- the link names are identical.

pxr/omni only exist once Kit is up, so SimulationApp is constructed before they are
imported -- hence the mid-file imports.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="out/scene_soft.usd",
                    help="the organs+phantom deformable scene to build on")
    ap.add_argument("--out", default="out/surgical_cell.usd")
    ap.add_argument("--robot-usd", default="out/fr3.usd",
                    help="where the imported FR3 is cached / referenced from")
    ap.add_argument("--spacing", type=float, default=0.90,
                    help="metres between the two FR3 bases")
    ap.add_argument("--port-offset", type=float, default=0.55,
                    help="metres (XY plane) from the used-ports (2,3) midpoint "
                         "to the line joining the two bases. 0.55 m is both "
                         "what PORT_OFFSET_STUDY.md recommends (2026-07-30: "
                         "below it J4 comes within 9 deg of its stop in the "
                         "corners of the scope view, above it the drive's "
                         "deflection under tissue load exceeds the 6 mm hole's "
                         "0.32 mm lateral budget) and WHAT THE REAL RIG NOW "
                         "STANDS AT (user). It was 0.27, the older measured "
                         "value, which is why the LEFT arm reached its working "
                         "pose with 1.4 deg of J4 left and went uncontrollable "
                         "there: J4 pinned at its LOWER stop is a fully folded "
                         "elbow, i.e. the target is too close to the base, and "
                         "no posture fixes it -- swept over 6 elbow lifts x 24 "
                         "shaft rolls the margin never moved. Standing the "
                         "bases back is the whole fix (0.27 -> 1.4 deg, "
                         "0.43 -> 5.7, 0.51 -> 11.9, 0.59 -> 26.9)")
    ap.add_argument("--rotate-phantom", type=float, default=180.0,
                    help="degrees to spin the phantom (with its trocar ports) about "
                         "Z, so the used ports face the arms")
    ap.add_argument("--rotate-organs", type=float, default=0.0,
                    help="net Z-rotation of the organs+field relative to their "
                         "captured pose. 0 keeps them exactly as the D405 filmed "
                         "them -- the live D405 view then matches the real camera "
                         "feed. (An earlier 180 made the sim view 180 deg off the "
                         "real feed.)")
    ap.add_argument("--lift", type=float, default=0.01,
                    help="metres the specimen plate raises phantom+organs above the "
                         "robot-base plane (table top)")
    ap.add_argument("--reuse-robot", action="store_true",
                    help="skip the URDF import if the robot USD already exists")
    ap.add_argument("--plain-right-arm", action="store_true",
                    help="give the right arm the stock Franka hand instead of "
                         "the motorised cutter (the pre-2026-07-28 behaviour)")
    ap.add_argument("--no-table", action="store_true",
                    help="skip the aluminium table + specimen plate")
    ap.add_argument("--organ-support", choices=("plane", "slab", "none"),
                    default="plane",
                    help="what holds the organs up. The field is an open shell "
                         "with the surgical opening cut through it, so once the "
                         "organs actually simulate they fall straight through it "
                         "unless something catches them. 'plane' keeps the "
                         "deformable scene's own infinite ground plane (lifted "
                         "onto the organs, grid hidden) -- the robust default, "
                         "verified to settle all six organs and leave the arms "
                         "untouched. 'slab' uses a finite hidden box instead "
                         "(can drop a small organ that a neighbour ejects). "
                         "'none' leaves them unsupported (they fall away)")
    ap.add_argument("--no-camera", action="store_true",
                    help="skip the RealSense D405 at the top of the phantom")
    ap.add_argument("--d405-intrinsics",
                    default="429.048,427.996,420.269,236.313",
                    metavar="FX,FY,PPX,PPY",
                    help="calibrated pinhole intrinsics of the real D405 at "
                         "848x480. Default = the capture unit (S/N 409122273236) "
                         "read live via rs-enumerate-devices -c, then undistorted "
                         "to a pinhole exactly like tools/extract_endonerf.py "
                         "(cv2.getOptimalNewCameraMatrix, alpha=0). Pass 'nominal' "
                         "for the ideal 87-deg datasheet model")
    ap.add_argument("--endo-intrinsics",
                    default="684.128,697.759,611.381,297.075",
                    metavar="FX,FY,PPX,PPY",
                    help="calibrated pinhole intrinsics of the real laparoscope "
                         "camera at 1280x720. Default = ChArUco calibration "
                         "(endoscope_interinsic/, 30 frames, RMS 0.58 px), "
                         "undistorted to a pinhole like the D405 "
                         "(cv2.getOptimalNewCameraMatrix, alpha=0). Pass 'nominal' "
                         "for the generic 70-deg model")
    ap.add_argument("--camera-drop", type=float, default=0.005,
                    help="metres to lower the D405 optical centre below the phantom "
                         "apex (into the ~3 cm top hole)")
    ap.add_argument("--view-hole-radius", type=float, default=0.016,
                    help="metres: radius of the hole cut through the phantom shell "
                         "around the D405 axis (the scanned mesh came out too thick "
                         "and blocks the view; the real dome has a ~3 cm opening)")
    ap.add_argument("--endo-pitch", type=float, default=1.267,
                    help="degrees added to the scope's authored X euler, i.e. "
                         "nod the whole scope. The fitted pose looks 40 deg "
                         "below horizontal and puts a wide band of empty dome "
                         "across the top of the frame; nodding down aims the "
                         "view into the organ mass instead. The start-pose aim "
                         "re-solves against the moved camera, so both tools "
                         "follow the new frame centre by themselves. Pivots "
                         "about the scope prim's own origin, not about its port. "
                         "Default 1.267 lands the authored X euler on -15.5, the "
                         "pose tuned by hand in the GUI on 2026-08-15 (it was "
                         "15.0 -> -1.767, a much deeper nod, until then)")
    ap.add_argument("--endo-offset", default="0.00989,-0.06833,0",
                    help="X,Y,Z metres added to the scope position in WORLD "
                         "axes, after the advance. The advance runs along the "
                         "shaft, which is the only motion a real scope has "
                         "through its port; this is the free translation for "
                         "framing a different part of the specimen. The "
                         "specimen's long axis is world Y (the scope looks "
                         "+Y at it), so Y slides along the animal. The X,Y here "
                         "are what carry the GUI-tuned pose of 2026-08-15 (was "
                         "0,-0.085,0); its Z lives in --endo-drop")
    ap.add_argument("--endo-drop", type=float, default=0.02038,
                    help="metres to LOWER the scope in world Z, after the "
                         "advance. Rotation alone cannot give a near-horizontal "
                         "view of tissue: the fitted tip sits at z~165 mm and "
                         "the tissue surface is at z~10-80 mm, so levelling the "
                         "boresight from up there just looks OVER the field and "
                         "fills the frame with dome. A real scope goes INTO the "
                         "cavity and looks across at tissue height -- this is "
                         "that height. Default 0.02038 is the GUI-tuned pose of "
                         "2026-08-15 (was -0.010, i.e. 10 mm the other way)")
    ap.add_argument("--endo-yaw", type=float, default=-2.346,
                    help="degrees added to the scope's authored Y euler. Swings "
                         "the view sideways, to bring the target's flank and the "
                         "tissue to its right into frame. Default -2.346 lands "
                         "the authored Y euler on 0.2 -- the GUI-tuned pose of "
                         "2026-08-15 squares the scope up, undoing the fit's own "
                         "2.546 deg of yaw")
    ap.add_argument("--endo-roll", type=float, default=0.0,
                    help="degrees added to the scope's authored Z euler. Rolls "
                         "the picture, to stand the target's long axis upright")
    ap.add_argument("--endo-advance", type=float, default=0.135,
                    help="metres to push the whole scope IN along its own shaft "
                         "axis (negative withdraws). The fitted pose "
                         "(ENDO_POSE_TRANSLATE) is where the scope sat in the "
                         "recording used for the view match; in the real "
                         "sessions the operator advances and withdraws it, so "
                         "the working distance is a setting, not a constant. "
                         "The camera and tip light are parented under the scope "
                         "and ride along, so illumination stays physical -- "
                         "moving only the camera would leave the light behind")
    ap.add_argument("--endo-k1", type=float, default=0.0,
                    help="OpenCV radial distortion k1 on the scope camera. "
                         "NEGATIVE is barrel -- centre magnified, scale falling "
                         "off toward the edge, which is what makes a shaft look "
                         "fat where it passes the tip and thin at its distal "
                         "end. Rendered by RTX itself (omni:lensdistortion), not "
                         "warped in post. 0 restores the calibrated pinhole")
    ap.add_argument("--endo-k2", type=float, default=0.0,
                    help="OpenCV radial distortion k2 (second-order term)")
    ap.add_argument("--endo-cone-deg", type=float, default=80.0,
                    help="tip-light cone half-angle (default ENDO_CONE_DEG, 80). "
                         "80 was chosen to LIGHT the dome walls so the white "
                         "phantom read white. A real laparoscopic view wants the "
                         "opposite -- the periphery falling dark -- so narrow it "
                         "toward 50-60 for that look")
    ap.add_argument("--no-endoscope-light", action="store_true",
                    help="skip the laparoscope tip light inside the phantom")
    ap.add_argument("--endoscope-intensity", type=float, default=50.0,
                    help="laparoscope light intensity (absolute UsdLux units; "
                         "500 as tuned in the GUI on the rig)")
    ap.add_argument("--physics-hz", type=float, default=480.0,
                    help="physics steps per second authored on the scene. The "
                         "organs' ~1.6 mm tets need 1/480 (the deformable scene's "
                         "own rate); at the 60 Hz default the FEM elements invert "
                         "on tool contact and the GPU solver crashes")
    ap.add_argument("--shell-stiffness", type=float, default=3.0e5,
                    help="N/m, compliant-contact stiffness of the shell/field "
                         "material. scene_soft.usd ships 1e4 (tuned for organs "
                         "settling), which against the position-driven tool is "
                         "a ~100 N drive vs a 10 N/mm spring: ~10+ mm of "
                         "penetration, and the thin field shell pops straight "
                         "through. 3e5 keeps the same saturated drive under "
                         "0.5 mm")
    ap.add_argument("--shell-damping", type=float, default=300.0,
                    help="N s/m, compliant-contact damping to go with "
                         "--shell-stiffness (scaled up from 10 with the "
                         "stiffness so contact stays overdamped)")
    ap.add_argument("--jaw-friction", type=float, default=0.6,
                    help="static/dynamic friction on the grasper endtip "
                         "colliders. The appendix hold is a CONSTRAINT, not "
                         "friction, so this need not be high -- 2.0 made a "
                         "pushed organ STICK to the tip and ride up with it "
                         "(user). 0.6 lets the tip slide off a poked organ")
    ap.add_argument("--blade-friction", type=float, default=0.02,
                    help="static/dynamic friction on the CUTTER's two blade "
                         "colliders, with combine mode 'min' so it sets the "
                         "pair regardless of the organ's own. Near zero on "
                         "purpose: scissors must not hold tissue. At the "
                         "grasper's 0.6 the closed blades pinched the appendix "
                         "and carried it exactly like the gripper (user). A "
                         "slick blade still PUSHES -- normal force is "
                         "unaffected -- so retracting and holding tissue aside "
                         "keep working; what it loses is the ability to grip")
    ap.add_argument("--tip-rest-offset", type=float, default=0.0015,
                    help="metres, collision rest offset (skin) on the jaw "
                         "endtip colliders. Gives the sharp mesh tip a rounded "
                         "shell it holds OUTSIDE the tissue, so a hard push "
                         "dents the organ instead of the tip sinking inside it")
    ap.add_argument("--tip-contact-offset", type=float, default=0.005,
                    help="metres, collision contact offset on the jaw endtip "
                         "colliders -- start resolving tool-vs-organ contact "
                         "this far out so a fast push is caught before it "
                         "tunnels")
    ap.add_argument("--organ-iterations", type=int, default=48,
                    help="deformable solverPositionIterationCount re-asserted "
                         "on every organ in the cell. Higher = stiffer contact "
                         "resolution, so the tool tip cannot punch as deep into "
                         "a pushed organ (build_deformable ships 32)")
    ap.add_argument("--organ-friction", type=float, default=0.6,
                    help="static/dynamic friction re-asserted on every organ "
                         "material (scene_soft ships 0.5). Kept modest: high "
                         "tissue friction made a poked organ ride up with the "
                         "tip instead of sliding off")
    ap.add_argument("--scene-light-scale", type=float, default=0.3,
                    help="multiplier on the inherited dome/key scene lights -- the "
                         "room outside the phantom was far too bright at 1.0")
    ap.add_argument("--tissue-desat", type=float, default=0.70,
                    help="0..1: how far to pull the organs' and field's baked "
                         "capture colour toward --tissue-tint. The plys carry "
                         "the real D405 colour of that rabbit, which came out "
                         "olive-green under the capture lighting; a laparoscope "
                         "sees the same tissue as pale cream. 0 = leave the "
                         "capture colour alone, 1 = keep only its per-vertex "
                         "BRIGHTNESS and replace the hue entirely")
    ap.add_argument("--tissue-tint", default="0.90,1.0,0.86",
                    help="R,G,B linear, the tone --tissue-desat pulls toward. "
                         "The default is the pale KHAKI of a real laparoscopic "
                         "frame -- G above R above B. Read off the reference "
                         "frame after a first pass at warm cream (1.0, 0.905, "
                         "0.754) came out too yellow: rabbit bowel under a "
                         "scope is green-grey, not ivory")
    ap.add_argument("--tissue-gain", type=float, default=2.1,
                    help="multiplier on tissue displayColor after the tint. "
                         "Albedo, not exposure -- >1 clips to white")
    ap.add_argument("--no-tint-field", action="store_true",
                    help="retint the organs only, leaving the field (the body "
                         "wall the organs sit in) at its capture colour")
    ap.add_argument("--plate-color", default=None,
                    help="R,G,B linear for the specimen plate (default "
                         "PLATE_COLOR, a bright sky-blue drape). The plate is "
                         "the single biggest non-tissue surface in the scope "
                         "view and no lighting change darkens it -- it sits "
                         "right under the organs, so it is close to the tip "
                         "light. Darken it here for a closed-cavity look")
    ap.add_argument("--phantom-color", default=None,
                    help="R,G,B linear for the phantom dome (default "
                         "PHANTOM_COLOR, near-white -- the real trainer is "
                         "white moulded plastic). Darken it to stop the dome "
                         "walls reading as a bright ceiling in the scope view")
    ap.add_argument("--table-shift", type=float, default=0.05,
                    help="metres to slide the table toward the phantom (+Y) from the "
                         "base-line/specimen midpoint")
    ap.add_argument("--table-depth", type=float, default=0.0,
                    help="metres, table Y-depth. 0 = AUTO: grow it 1:1 with the "
                         "stand-off so the robot bases keep a constant back "
                         "margin as --port-offset increases (the 80 cm depth is "
                         "sized for 0.27 m; at the RCM-optimal 0.46 m the bases "
                         "would otherwise overhang the back edge). "
                         "depth = round_5cm(0.80 + max(0, port_offset - 0.27)): "
                         "0.27->80, 0.46->100, 0.50->105 cm")
    ap.add_argument("--no-trocars", action="store_true",
                    help="skip drawing the phantom trocar port markers")
    ap.add_argument("--hide-phantom", action="store_true",
                    help="keep the phantom shell hidden (shown by default so the "
                         "trocar port markers can be checked against the dome)")
    ap.add_argument("--no-organ-pins", action="store_true",
                    help="skip pinning the organs in place. By default every "
                         "organ's lower band of SIM-mesh nodes is held at its "
                         "captured position by a directly-authored "
                         "OmniPhysicsVtxXformAttachment (explicit vertex "
                         "indices; the mesentery). The auto sphere-overlap "
                         "attachment path silently bonded nothing for 3 of the "
                         "6 organs, so it is not used. The appendix is pinned "
                         "only around its base at the cecum -- its body must "
                         "stay free to grasp and lift")
    ap.add_argument("--pin-radius", type=float, default=0.008,
                    help="metres: radius around the appendix's base point (its "
                         "closest point to the cecum) whose sim nodes get "
                         "pinned; the rest of the appendix stays free")
    ap.add_argument("--pin-band", type=float, default=0.65,
                    help="fraction of each organ's height (from the bottom) "
                         "whose sim nodes are pinned; the exposed top stays "
                         "free to deform under the instruments. 0.45 left the "
                         "free upper half visibly creeping at rest "
                         "(user-observed); 0.65 keeps only the exposed crust "
                         "soft")
    ap.add_argument("--grip-radius", type=float, default=0.012,
                    help="metres: radius of the appendix tip cluster bonded "
                         "to the grasp handle. 8 mm welded so little tissue "
                         "that the neighbourhood stretched thin at the jaws "
                         "(the 'gap'); 12 mm moves the local tissue as one "
                         "piece with the tool")
    ap.add_argument("--appendix-youngs", type=float, default=12000.0,
                    help="Pa: appendix Young's modulus override. 1.5 kPa "
                         "stretched like jelly, 4 kPa was still too soft "
                         "(user); 12 kPa holds its shape yet the constraint "
                         "grasp still lifts it limply")
    ap.add_argument("--organ-youngs", type=float, default=30000.0,
                    help="Pa: Young's modulus authored over EVERY non-appendix "
                         "organ (scene_soft ships 5 kPa). 5 kPa was too soft -- "
                         "a firm push sank the tool tip inside (user); 30 kPa "
                         "resists the ~100 N drive so the tip dents instead")
    ap.add_argument("--organ-density", type=float, default=1200.0,
                    help="kg/m^3 authored on every organ material (scene_soft "
                         "ships 1050). Raised so the tissue is not so light it "
                         "flies off the instrument (user: density too low)")
    ap.add_argument("--organ-damping", type=float, default=0.05,
                    help="elasticityDamping authored on every organ material "
                         "(PhysxDeformableBodyMaterialAPI; physx default "
                         "0.005). Damps at-rest wobble of the unpinned tissue. "
                         "(NB it did NOT touch the bowel2 buzz -- that was a "
                         "numerical instability, capped via YOUNGS_CAP.)")
    ap.add_argument("--appendix-raise", type=float, default=0.008,
                    help="metres to raise the appendix above its captured "
                         "pose (baked into its points; pins/shelf/handle all "
                         "follow). The top-down capture leaves it buried "
                         "under its neighbours, too deep for the jaws to "
                         "reach cleanly")
    ap.add_argument("--organ-contact-offset", type=float, default=0.0015,
                    help="metres, organ collision contactOffset. scene_soft "
                         "ships 3 mm, but the organs are packed tighter than "
                         "that, so at rest every pair is inside the offset and "
                         "the solver keeps pushing them apart -- the observed "
                         "permanent trembling. 1.5 mm still cushions tool "
                         "contact at 480 Hz")
    ap.add_argument("--organ-rest-offset", type=float, default=0.0,
                    help="metres, organ collision restOffset (scene_soft ships "
                         "0.8 mm -- same story: captured surfaces touch, so a "
                         "positive rest gap is a permanent extrusion force)")
    ap.add_argument("--organ-field-collision", action="store_true",
                    help="keep organ<->field collision. Off by default: the "
                         "organs are captured EMBEDDED against the field, so "
                         "its (tool-proof, 3e5 N/m) collider overlaps them at "
                         "spawn and ejects them -- measured up to 100 mm. The "
                         "anchors hold the organs instead; the field still "
                         "collides with the instruments")
    ap.add_argument("--filter-organ-organ", action="store_true",
                    help="filter organ<->organ contact (default: keep it, so "
                         "pushing one organ nudges its neighbours). The bowel2 "
                         "at-rest buzz was NOT contact (filtering it changed "
                         "nothing) -- it was a numerical instability fixed by "
                         "the per-organ stiffness cap (YOUNGS_CAP) instead")
    ap.add_argument("--hide-field", action="store_true",
                    help="keep the field (rabbit body wall around the organs) "
                         "hidden. scene_soft.usd authors it invisible so the "
                         "organs show from outside; in the cell the cameras sit "
                         "INSIDE the phantom looking at the surgical site, where "
                         "the field is the abdomen the organs live in -- shown "
                         "by default (it always collides either way)")
    # ---- bimanual start pose (both instruments through their own port) ----
    ap.add_argument("--no-start-pose", action="store_true",
                    help="leave both arms in the URDF zero pose instead of "
                         "baking the inserted surgical start pose")
    ap.add_argument("--start-insert", type=float, default=0.10,
                    help="m past its own trocar port each tool SPAWNS at (hard "
                         "constraint; the aim only chooses the DIRECTION). This "
                         "is a parked pose, deliberately short of the organ "
                         "bed: at the 0.145 it briefly carried, the solver had "
                         "to place the tips among the organs (the depth is "
                         "pinned, so the clearance term can only pick the "
                         "least-buried direction, not back off), and the cell "
                         "opened with tissue already deformed around a tool and "
                         "organs ejected out of the overlap. The tips reach the "
                         "working pose by FLYING there instead -- see "
                         "start_pose.APPROACH_TIP_WORLD and --approach-speed on "
                         "the teleop/env. Keep teleop's --insert and vla_env's "
                         "cfg['insert'] on the same value -- both RE-SOLVE the "
                         "start pose rather than trusting the stage. It no "
                         "longer has to track --endo-advance: the scope only "
                         "has to frame the tips where the approach PARKS them, "
                         "and that is now a fixed point in the world")
    ap.add_argument("--start-aim", choices=("approach", "endoscope"),
                    default="approach",
                    help="which DIRECTION each tool spawns pointing. approach: "
                         "along the ray from its port to its working tip "
                         "(start_pose.APPROACH_TIP_WORLD), so the spawn is just "
                         "the working pose withdrawn and the approach is a pure "
                         "slide in. endoscope: the old aim, at the middle of the "
                         "scope image -- which only works while the tools are "
                         "DEEPER than the camera, and at the 10 cm spawn puts "
                         "the left tip 1.5 mm from the camera origin with the IK "
                         "20.7 mm short against its joint limits")
    ap.add_argument("--start-min-sep", type=float, default=0.045,
                    help="m: minimum start tip-to-tip separation (--start-aim "
                         "endoscope only; the approach aim inherits whatever "
                         "separation the working tips have)")
    ap.add_argument("--start-max-tilt", type=float, default=60.0,
                    help="deg: how far a shaft may lean off its port's surface "
                         "normal at the SPAWN pose. Matches the operating cap "
                         "the controller enforces (trocar.TILT_CAP_DEG), so the "
                         "arms are not parked outside the cone they are then "
                         "allowed to move in")
    ap.add_argument("--start-v-frac", type=float, default=0.5,
                    help="target image height for the start tips (0 top, 1 bottom)")
    ap.add_argument("--start-min-clear", type=float, default=0.010,
                    help="m: keep the start tips this far off the organ surfaces")
    ap.add_argument("--start-roll-left", type=float, default=0.0,
                    help="deg: extra roll of the LEFT instrument about its own "
                         "shaft at the start pose, on top of the 'tip +y up' "
                         "convention. The shaft IS flange +z, so this is a "
                         "rotation of the end effector about flange z")
    ap.add_argument("--start-roll-right", type=float, default=0.0,
                    help="deg: same for the RIGHT instrument. 0 after measuring "
                         "it: rendered side by side (isaac/try_right_roll.py, "
                         "offsets 0/-60/-120/180/120/60 from the rig-photo "
                         "viewpoint), the SAME '+y up' convention on both arms "
                         "is what makes the two bodies mirror-symmetric, long "
                         "sides outward and tools facing each other. An offset "
                         "breaks that symmetry AND costs joint headroom -- -60 "
                         "drops the right arm's limit margin 38.1 -> 13.2 deg. "
                         "What made the cutter look wrong before was that the "
                         "bake did not converge the roll AT ALL, not the frame "
                         "convention")
    ap.add_argument("--start-elbow-lift", type=float, default=0.6,
                    help="null-space gain lifting the ELBOW in world z at the "
                         "start pose. The posture level alone (manipulability + "
                         "joint limits) settles it low over the table; the real "
                         "rig carries it high in an arch. 0 disables")
    ap.add_argument("--start-elbow-point", type=int, default=3,
                    help="which chain point to lift (3 = joint-4 origin, the "
                         "FR3's visual elbow)")
    ap.add_argument("--start-seed-from", choices=("left", "right", "none"),
                    default="right",
                    help="which arm is solved first from the neutral seed; the "
                         "other is then seeded from its MIRROR, so the two "
                         "elbows settle into matching postures instead of each "
                         "finding its own manipulability optimum. 'none' "
                         "solves both from the neutral seed (the old behaviour)")
    ap.add_argument("--view", action="store_true",
                    help="open a window and hold it (no window = build + verify PNG)")
    ap.add_argument("--verify-png", default="out/surgical_cell.png",
                    help="headless: render one frame here to check placement")
    return ap.parse_args()


ARGS = parse_args()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": not ARGS.view,
                                "renderer": "RaytracedLighting"})

import omni.kit.commands  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.utils.extensions import get_extension_path_from_name  # noqa: E402
from pxr import (Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux,  # noqa: E402
                 UsdPhysics, UsdShade)


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------
# Table / specimen constants
# --------------------------------------------------------------------------
# The aluminium table everything sits on: 140 x 80 cm top, 70 cm tall. The top
# is the robot-base plane (world z=0); the box hangs below to z=-0.70.
TABLE_SIZE = (1.40, 0.80, 0.70)     # x, y, z (height), metres
PLATE_SIZE = (0.46, 0.52, None)     # x, y footprint; height = ARGS.lift
ALU = (0.90, 0.91, 0.93)            # aluminium base colour (metallic material)
PLATE_COLOR = (0.2122, 0.4508, 0.6649)  # dim cavity-wall brown. It WAS a bright
                                    # surgical-drape sky-blue (linear 0.2122,
                                    # 0.4508, 0.6649 = sRGB #7FB3D5), which is
                                    # what the specimen really rests on -- but
                                    # the plate is the largest surface in the
                                    # SCOPE view and a bright blue field is the
                                    # loudest difference from a real
                                    # laparoscopic frame, where the periphery is
                                    # dim tissue. No lighting change reaches it:
                                    # it sits directly under the organs, so it
                                    # is close to the tip light, and killing the
                                    # dome and key lights outright (measured)
                                    # leaves it exactly as bright. Albedo is the
                                    # only lever. --plate-color restores the
                                    # blue for rig-verification renders.
FLOOR_COLOR = (0.50, 0.50, 0.50)    # neutral grey room floor
# The phantom dome -- the "abdominal wall" that walls in the scope view. Its
# collider is authored in build_deformable_usd.py, which gives every colour-less
# collider a dark neutral grey (0.34, 0.32, 0.30), and that made the scope view's
# surround read as dim brown shadow. The real trainer is WHITE moulded plastic
# (rig photo, 2026-07-29), so the cell re-authors it here instead of forcing a
# deformable re-cook. displayColor is read as LINEAR by RTX.
#
# Albedo alone was not enough: the dome walls sit far off the tip light's axis,
# so white paint still shaded to 50 % grey. Widening ENDO_CONE_DEG (below) is the
# other half of the fix -- together they take the wall from luma 58-103/255 to
# 143-164/255 with the field's exposure unchanged (161 -> 164).
#
# Superseded 2026-08-12, for the same reason as PLATE_COLOR. The white dome is
# what the rig really looks like from outside, but inside the scope it becomes a
# bright ceiling over the whole upper frame, and a real laparoscopic view has dim
# tissue there. This is now a dim warm wall; --phantom-color 0.95,0.95,0.95
# restores the white trainer for rig-verification renders.
PHANTOM_COLOR = (0.95, 0.95, 0.95)

# --------------------------------------------------------------------------
# Laparoscopic trocar ports
# --------------------------------------------------------------------------
# Hole positions measured on the phantom, in the phantom's own STL frame
# (millimetres). The bimanual instruments use ports 2 and 3 (1-based). NB the
# whole specimen is spun 180 deg about Z (--rotate-phantom) so those ports face
# the arms -- which swaps their physical side, so left/right is decided from the
# final world X below, not from the port number.
TROCAR_PORTS_MM = [
    (-95.171, 125.068, 133.195),   # 1
    (-53.106,  77.202, 175.961),   # 2  used
    ( 53.810,  79.279, 177.066),   # 3  used
    ( 98.208, 126.619, 134.031),   # 4
    (116.733,  16.543, 153.019),   # 5
    (-110.588, 10.071, 152.489),   # 6
    ( 56.075, -48.227, 180.953),   # 7
    (-50.545, -52.212, 179.154),   # 8
    (105.819,-109.028, 149.272),   # 9
    (-91.281,-113.699, 148.559),   # 10
    (  8.161,-121.251, 164.292),   # 11
]
USED_PORT_IDX = (1, 2)              # 0-based: ports 2 and 3
LEFT_COLOR = (0.10, 0.90, 0.25)     # green, physical left  (-X, by the left arm)
RIGHT_COLOR = (0.20, 0.55, 1.00)    # blue,  physical right (+X, by the right arm)
AMBER = (1.00, 0.75, 0.0)           # every other, unused port

# --------------------------------------------------------------------------
# Left laparoscopic grasper -- MOTORISED end-effector (2026-07-28 rebuild)
# --------------------------------------------------------------------------
# The franka-hand-to-laparoscope mount is GONE, and so is the Franka hand: the
# instrument is now a purpose-built motorised unit (laparo_endeffector/
# motorized_laparo_gripper_urdf) that bolts straight onto the bare FR3 flange.
# A 190 x 60 x 182 mm housing carries an STS3215 servo driving the jaw linkage;
# the 5 mm laparoscopic shaft is COAXIAL with the flange Z axis and the jaws
# open 516 mm out along it. That is the big change from the old mount, whose
# tube stuck out sideways along flange -Y with the tip 460 mm off-axis.
#
#   T_flange->tip = [[-0.707107, 0.707107, 0,  0.000093],
#                    [ 0.707107, 0.707107, 0, -0.000085],
#                    [ 0,        0,       -1,  0.515949],
#                    [ 0,        0,        0,  1       ]]
#
# Meshes come from the CLEAN re-export (2026-07-28 evening), made by the fixed
# `laparo_endeffector/Flange URDF.py` v3 after the first export turned out to
# be broken in three ways. That first attempt is kept next door under
# motorized_laparo_gripper_urdf/ for reference only -- do not build from it:
#   * its STL units were MIXED, 19 bodies in mm and 8 in cm, because the
#     exporter had two writers with different units;
#   * it drew only 18 of the 25 bodies, silently dropping the whole outer
#     casing including main_housing.stl;
#   * two duplicated components (vis, connector_F) were written byte-identical,
#     i.e. one instance of each landed on top of its twin.
# The re-export is all-millimetre with ONE scale, carries every body, and puts
# the duplicates 68.7 mm and 10.1 mm apart where they belong. Verified body by
# body: shaft 114.2 -> 491.2 mm, clevis pin at 494.9 mm (0.055 mm from the
# JAW_PIVOT_XYZ below, its axis within 0.05 deg of JAW_PIN_AXIS), jaw tips at
# 516.0 mm = exactly the urdf's tip joint.
#
# One body is knowingly imperfect: `vis_part_1_1`, an M3 screw head Fusion
# reports as 45 deg rotated but hands out unrotated (see TOLERATE_BODIES in
# the exporter). It is 3 mm, buried inside the housing and visual-only.
MOTOR_TOOL_DIR = (ROOT / "laparo_endeffector"
                  / "motorized_laparo_gripper_urdf_new"
                  / "motorized_laparo_gripper_urdf")
LAPARO_MESH_DIR = MOTOR_TOOL_DIR / "meshes"
LAPARO_MESH_SCALE = 0.001                          # STL mm -> m, for every body
# The casing bodies the FIRST export dropped. Kept as an explicit completeness
# check: if a future re-export loses them again, the build fails loudly rather
# than quietly putting an invisible box on the arm.
LAPARO_EXTRA_MESHES = (
    "main_housing.stl", "housing_cover.stl", "motor_cover.stl",
    "motor_linkage.stl", "motor_handle_linkage.stl",
    "handle_cover.stl", "handle_linkage.stl",
)
# Only these get a collider. The importer convex-hulls every collision mesh,
# so the set is chosen to make each hull a sane shape: laparo_shaft is a clean
# 5 mm x 377 mm tube (its hull IS the tube -- no synthetic cylinder needed any
# more), the distal cluster is small, and the housing hull is just its box.
# Everything else lives inside the housing hull and would only add contacts.
LAPARO_COLLIDE_MESHES = {
    "laparo_shaft.stl", "part_13.stl", "part_15.stl",
    "gripper_shaft_inner.stl", "gripper_shaft_outer.stl",
    "main_housing.stl", "housing_cover.stl",
}
LAPARO_VIS_XYZ = (0.0, 0.0, 0.162154)              # mesh placement in the flange frame
LAPARO_VIS_RPY = (3.141593, 0.0, 2.356194)         # flip about X, +135 deg yaw
LAPARO_TIP_XYZ = (0.000093, -0.000085, 0.515949)   # T_flange->tip translation (m)
LAPARO_TIP_RPY = (3.141593, 0.0, 2.356194)         # tip frame: +z back up the shaft
LAPARO_MASS = 0.473                                # CAD total, housing + servo + tool
LAPARO_COM = (0.008485, -0.008485, 0.096000)       # 12 mm off-axis, mid-housing
LAPARO_INERTIA = dict(ixx="0.001457750", ixy="0.000644583", ixz="0",
                      iyy="0.001457750", iyz="0", izz="0.001574767")
# --------------------------------------------------------------------------
# Articulated gripper jaws
# --------------------------------------------------------------------------
# The jaws rotate about the clevis pin. Both pivot and axis are cross-checked
# against the pin meshes themselves rather than taken on trust: the PCA long
# axis of gripper_shaft_inner/outer (a 5.05 mm x r0.74 mm cylinder) comes out
# at (0.7071, 0.7071, 0) in the flange frame, and its centre at
# (0.005, -0.117, 494.911) mm -- 0.04 mm from the urdf's joint origin below.
#
# SIGNS: the shipped tool.urdf has them BACKWARDS. Its limits (left negative,
# right positive) sweep each blade THROUGH the other one -- the blade-to-blade
# gap goes 0.03 mm closed -> 0.04 mm at 17 deg, i.e. they grind past each
# other, and only re-separate once they have swapped sides. The opposite
# pairing opens cleanly and monotonically (0.03 -> 2.90 -> 6.15 mm), each
# blade moving further onto the side it already rests on at q=0. That is the
# pairing used here, so that as before q=0 is CLOSED (tips 0.09 mm apart) and
# POSITIVE q OPENS both jaws -- 25 mm of tip separation at the 74 deg full
# opening. Self-collision is off, so this cannot be left to the solver.
JAW_MESHES = {"left": "gripper_tip_left.stl", "right": "gripper_tip_right.stl"}
JAW_PIVOT_XYZ = (0.000093, -0.000085, 0.494949)    # clevis pin centre, flange frame
JAW_PIN_AXIS = (0.707107, 0.707106, 0.0)           # pin axis, flange frame, unit
JAW_OPEN_TOTAL_DEG = 74.0                          # fully open included angle (urdf)
JAW_MASS = 0.004                                   # a 21 mm steel jaw, ~4 g
JAW_COM = (0.0, 0.0, 0.009473)                     # pivot->blade centroid, link frame
JAW_INERTIA = dict(ixx="1.5e-07", ixy="0", ixz="0",
                   iyy="1.5e-07", iyz="0", izz="1.5e-07")  # ~rod mL^2/12
# GUI-play defaults authored on the jaw drives in the left robot USD (teleop
# overrides via the articulation controller). USD angular drives take
# torque-per-DEGREE units; 10 Nm/deg (~573 Nm/rad) pins a 4 g jaw rigidly.
JAW_DRIVE_STIFFNESS = 10.0
JAW_DRIVE_DAMPING = 0.5
JAW_DRIVE_MAX_FORCE = 50.0
# The Franka hand + fingers are removed on BOTH arms -- the units replace them.
HAND_LINKS = ("fr3_hand", "fr3_hand_tcp", "fr3_leftfinger", "fr3_rightfinger",
              "gripper_center")
HAND_JOINTS = ("fr3_hand_joint", "fr3_hand_tcp_joint", "fr3_finger_joint1",
               "fr3_finger_joint2", "gripper_center_joint")
LEFT_ROBOT_USD = "out/fr3_left_grasper.usd"        # FR3 (no hand) + grasper
RIGHT_ROBOT_USD = "out/fr3_right_cutter.usd"       # FR3 (no hand) + cutter

# --------------------------------------------------------------------------
# Right arm: motorised laparoscopic CUTTER
# --------------------------------------------------------------------------
# Same instrument family as the left grasper -- same housing, same servo, same
# 5 mm shaft -- with scissor blades instead of grasper jaws, and built for the
# other arm: the CAD flange and tip frames are turned 180 deg about Z. That
# shows up as the mesh placement yaw going 135 deg -> -45 deg, and the COM x/y
# flipping sign. The tip frame's ORIENTATION is unchanged (the exporter's
# rpy roll flips to -pi, which is the same rotation), so every tool-frame
# quantity the teleop uses -- SHAFT_AXIS_F, the pin axis -- is identical to the
# grasper's. Only the numbers below differ.
#
# Verified from the STL bytes exactly as the grasper was:
#   * mesh placement IS the grasper's rule turned 180 deg about Z (6.9e-7)
#   * following the urdf joint chain to a blade agrees with that one rigid
#     placement rule to 0.0004 mm
#   * shaft 114.2 -> 469.4 mm, clevis pin at 473.15 mm (0.10 mm from the pivot
#     below, axis within 0.04 deg), blade tips at 494.2 mm = the tip joint
#   * the tool is 21.8 mm SHORTER than the grasper (494.2 vs 516.0 mm)
#
# NO sign trap here, unlike the grasper: the two blades are mirror images, so
# both pairings give identical opening (gap and tip separation match to the
# digit). The blades are offset along the pin and shear past each other like
# scissors -- the blade-to-blade gap stays ~1.5 mm across the whole sweep
# instead of closing. q=0 is closed (tips 1.30 mm apart); at the 50 deg full
# opening the tips are 15.95 mm apart.
CUTTER_TOOL_DIR = (ROOT / "laparo_endeffector" / "motorized_laparo_cutter_urdf"
                   / "motorized_laparo_cutter_urdf")
CUTTER_MESH_DIR = CUTTER_TOOL_DIR / "meshes"
CUTTER_VIS_XYZ = (0.0, 0.0, 0.162154)              # same stand-off up the flange
CUTTER_VIS_RPY = (3.141593, 0.0, -0.785398)        # -45 deg: the grasper's +135, turned 180
CUTTER_TIP_XYZ = (-0.000093, 0.000085, 0.494154)   # 21.8 mm shorter than the grasper
CUTTER_TIP_RPY = (-3.141593, 0.0, 2.356194)        # same ORIENTATION as the grasper's tip
CUTTER_MASS = 0.473
CUTTER_COM = (-0.008485, 0.008485, 0.096000)       # grasper's COM with x,y negated
CUTTER_INERTIA = dict(ixx="0.001457750", ixy="0.000644583", ixz="0",
                      iyy="0.001457750", iyz="0", izz="0.001574767")
BLADE_MESHES = {"left": "cutter_tip_left.stl", "right": "cutter_tip_right.stl"}
BLADE_PIVOT_XYZ = (-0.000093, 0.000085, 0.473154)  # clevis pin centre, flange frame
BLADE_PIN_AXIS = (0.707107, 0.707106, 0.0)         # same pin line as the grasper
BLADE_OPEN_TOTAL_DEG = 50.0                        # fully open included angle (urdf)


class ToolSpec:
    """Everything build_arm_tool_urdf() needs to emit one arm's tool."""

    def __init__(self, name, mesh_dir, vis_xyz, vis_rpy, tip_xyz, tip_rpy,
                 mass, com, inertia, jaw_meshes, pivot_xyz, pin_axis,
                 open_total_deg, link_prefix, joint_prefix, urdf, usd,
                 body_mat="gloss_black"):
        # body_mat: the housing+shaft colour, which is the one thing that is
        # NOT shared between the two instruments (see TOOL_MATERIALS)
        self.body_mat = body_mat
        self.name, self.mesh_dir = name, mesh_dir
        self.vis_xyz, self.vis_rpy = vis_xyz, vis_rpy
        self.tip_xyz, self.tip_rpy = tip_xyz, tip_rpy
        self.mass, self.com, self.inertia = mass, com, inertia
        self.jaw_meshes, self.pivot_xyz = jaw_meshes, pivot_xyz
        self.pin_axis, self.open_total_deg = pin_axis, open_total_deg
        self.link_prefix, self.joint_prefix = link_prefix, joint_prefix
        self.urdf, self.usd = urdf, usd

    @property
    def joints(self):
        return (f"{self.joint_prefix}_left", f"{self.joint_prefix}_right")


TOOLS = {
    "left": ToolSpec(
        "left grasper", LAPARO_MESH_DIR, LAPARO_VIS_XYZ, LAPARO_VIS_RPY,
        LAPARO_TIP_XYZ, LAPARO_TIP_RPY, LAPARO_MASS, LAPARO_COM,
        LAPARO_INERTIA, JAW_MESHES, JAW_PIVOT_XYZ, JAW_PIN_AXIS,
        JAW_OPEN_TOTAL_DEG, "jaw", "gripper_jaw",
        "out/fr3_left_grasper.urdf", LEFT_ROBOT_USD,
        body_mat="gloss_mid"),
    "right": ToolSpec(
        "right cutter", CUTTER_MESH_DIR, CUTTER_VIS_XYZ, CUTTER_VIS_RPY,
        CUTTER_TIP_XYZ, CUTTER_TIP_RPY, CUTTER_MASS, CUTTER_COM,
        CUTTER_INERTIA, BLADE_MESHES, BLADE_PIVOT_XYZ, BLADE_PIN_AXIS,
        BLADE_OPEN_TOTAL_DEG, "blade", "cutter_blade",
        "out/fr3_right_cutter.urdf", RIGHT_ROBOT_USD,
        body_mat="gloss_black"),
}
# every articulated endtip joint name, for the drive/contact authoring passes
ENDTIP_JOINTS = tuple(j for t in TOOLS.values() for j in t.joints)

# Organs whose cooked mesh goes numerically unstable at the firm --organ-youngs
# (fine/sliver elements buzz at rest, ~1-2 mm/s, at 480 Hz): capped to a lower,
# stable modulus. bowel2 was the clear offender; bowel3 twitched mildly. The
# stable organs keep the full --organ-youngs.
YOUNGS_CAP = {"bowel2": 15000.0, "bowel3": 15000.0}

# Organs pinned MORE than the global --pin-band (fraction of height pinned from
# the bottom): bowel2's large free top oscillates at rest, so most of it is
# fixed, leaving only a thin exposed crust deformable.
PIN_BAND_CAP = {"bowel2": 0.90}

# Per-part tool colours matching the scope footage of the real instruments: a
# silver metal endtip (jaws/blades + clevis cluster) on both, dark steel
# fasteners, and a body+shaft whose colour is PER ARM -- the left grasper is a
# light grey/white, the right cutter is black, and both are glossy, not matte.
# `gloss_black` and `stainless_steel` are the names the shipped tool.urdf uses;
# the rest are ours.
TOOL_MATERIALS = {                       # urdf material name -> diffuse rgb
    "gloss_black":      (0.045, 0.045, 0.050),
    "gloss_mid":        (0.188, 0.190, 0.196),
    "stainless_steel":  (0.780, 0.790, 0.820),
    "bolt_mat":         (0.220, 0.220, 0.240),
}
# How each of those reads once RTX has it. A urdf <material> reaches the stage
# as an OmniPBR with ONLY diffuse_color_constant set, i.e. metallic 0 and the
# default rough dielectric -- so the "stainless steel" endtip renders as a flat
# light grey that photographs as WHITE plastic next to the real instrument
# (compare laparo_endeffector/20260728_140327.jpg: the jaws are a polished
# mirror that takes its brightness from what it reflects, not from its albedo).
# `author_tool_metal_look` writes these back over the imported materials.
#
# The diffuse values above stay as the URDF authored them -- they are what a
# non-RTX/preview path falls back to. Under a metal shader the albedo is the
# REFLECTANCE tint, so the steel is dropped to a neutral, slightly darker grey:
# a 0.78 metal reflects almost everything and blows out under the scope light.
#
# The two body colours are DIELECTRIC, not metal: a white instrument body is
# moulded polymer and a black one is anodised or painted, and both take a sharp
# white highlight off the scope light rather than mirroring the field the way
# the steel endtip does. Metal 0 with a low roughness is exactly that -- "glossy
# but not chrome". Roughness 0.10 is what puts the highlight on them; the 0.35
# the black body shipped with read as matte.
TOOL_PBR = {                             # urdf material -> (metal, rough, rgb)
    "gloss_black":     (0.0, 0.10, (0.045, 0.045, 0.050)),
    # The left arm's body is a DARK grey -- halfway between the two instruments
    # (user), i.e. (0.330 + 0.045) / 2. It is not the near-white the reference
    # photo suggests, because the scope light sits ~30 mm off the shaft and
    # falls off as 1/r^2: the part of the tool nearest the lens is lit several
    # stops harder than the tissue the exposure is set for, so albedo here buys
    # clipping, not brightness. It went 0.82 (a flat white slab with no visible
    # form) -> 0.48 -> 0.33 -> 0.19, and the shaft still blows out near the
    # lens while the shadowed housing reads correctly. If the lit shaft needs
    # to come down further, the knob is --endoscope-intensity, not this.
    "gloss_mid":       (0.0, 0.10, (0.188, 0.190, 0.196)),
    "stainless_steel": (1.0, 0.12, (0.560, 0.570, 0.590)),
    "bolt_mat":        (1.0, 0.45, (0.220, 0.220, 0.240)),
}
# the distal cluster on the shaft axis: clevis body, pushrod and pin (the jaws
# are their own links). These are the only bare-metal parts of the instrument.
SILVER_TOOL_STEMS = {"part_13", "part_15",
                     "gripper_shaft_inner", "gripper_shaft_outer"}
BOLT_TOOL_STEMS_PREFIX = ("vis_", "Vis_", "Composant", "connector_F")
# A USD prim name may not start with a digit. The exporter now applies this
# prefix itself (bodies 10/11/13/15 ship as part_10...part_15), so
# staged_mesh_path has nothing left to rename -- it stays as a safety net in
# case a future export regresses.
MESH_NUM_PREFIX = "part_"
# the same distal cluster, as it appears in the imported collider prim paths.
DISTAL_COLLIDER_STEMS = ("part_13", "part_15",
                         "gripper_shaft_inner", "gripper_shaft_outer")


def mesh_material_name(stem: str, body_mat: str = "gloss_black") -> str:
    """Pick the urdf material for a laparoscope mesh from its file stem.

    `body_mat` is the arm's own body colour (ToolSpec.body_mat) -- the endtip
    and the fasteners are the same on both instruments, only the housing and
    shaft differ."""
    if stem in SILVER_TOOL_STEMS:
        return "stainless_steel"
    if stem.startswith(BOLT_TOOL_STEMS_PREFIX):
        return "bolt_mat"
    return body_mat                      # housing, covers, shaft, linkage, servo


def staged_mesh_path(mesh: Path) -> Path:
    """A URDF-importer-safe path for one tool mesh.

    Four of the CAD bodies are named 10/11/13/15.stl. The importer turns a
    mesh filename straight into a USD prim name, and a USD prim name may not
    start with a digit -- it emits `Ill-formed SdfPath </visuals/tool_flange/10>`
    and then fails the whole import with "Used null prim". So those meshes are
    symlinked into out/ under a `part_` prefix and referenced from there; the
    rest are used in place.
    """
    if not mesh.stem[:1].isdigit():
        return mesh
    dest = ROOT / "out" / "laparo_meshes" / f"{MESH_NUM_PREFIX}{mesh.name}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink() or dest.exists():
        dest.unlink()
    dest.symlink_to(mesh.resolve())
    return dest

# --------------------------------------------------------------------------
# RealSense D405 overhead camera (the capture camera; sim/scene.py notes it
# "looked straight DOWN at the animal", 848x480). It sits in the ~3 cm hole at
# the top of the phantom, in the middle of the top four ports, looking -Z.
# --------------------------------------------------------------------------
D405_RES = (848, 480)               # capture resolution, rs_bag.py
D405_HFOV_DEG = 87.0                # nominal D405 horizontal FOV
D405_FOCAL = 18.0                   # arbitrary; only focal:aperture ratio sets FOV
D405_BODY = (0.042, 0.042, 0.023)   # approx module size (X, Y, Z), metres
D405_COLOR = (0.05, 0.05, 0.06)     # dark camera housing

# --------------------------------------------------------------------------
# Laparoscope tip light. A surgical endoscope pipes ~300 W xenon/LED down a fibre
# and emits a bright, slightly warm, wide illumination cone from the scope tip.
# Modelled as a small SphereLight + ShapingAPI cone at the endoscope tooltip,
# aimed down the same bevelled view axis as the objective. Brightness comes from
# --endoscope-intensity; the real light source was set to 50 percent.
# --------------------------------------------------------------------------
# Endoscope pose (world; replaces the default spin+lift), authored as translate
# + ONE rotateXYZ op so the transform widget stays happy.
#
# FITTED TO THE REAL RECORDING, 2026-07-29, by isaac/match_endoscope_view.py:
# the D405 bag of the same session is turned into a coloured point cloud in
# world coordinates and rendered from candidate scope poses until the picture
# agrees with data/episode_cam0_20260705_175255.mp4 (95.3 % per-pixel
# drape/tissue agreement). The tip camera lands at world
# (+3.3, -101.8, +164.9) mm with its boresight 40 deg below horizontal --
# 15 mm and 4 deg from where the OLD pose put it, because the old pose was
# hand-set in the GUI to make a 0-degree camera look at the field.
#
# **The scope is SQUARE to the rig, by construction, not by fit.** The recording
# was hand-held, so a free fit reproduces its ~4 deg of bank and ~6 deg of yaw --
# invisible in the video, glaring in the render, where the dome's mouth and the
# rectangular plate are big clean symmetric shapes. Roll and yaw are therefore
# pinned to 0 (`--roll`/`--yaw` to override): the boresight lies in the world YZ
# plane and so does the shaft, which runs straight down the robots' forward
# direction (+Y toward the phantom), as the rig photo shows it does. Pinning them
# cost nothing -- agreement went UP, 0.944 -> 0.953.
#
# What changed most is the SHAFT. With ENDO_TIP_ANGLE_DEG = 30 the view leaves
# the tip 30 deg below the shaft, so holding the view at 40 deg down puts the
# shaft at 10 deg down -- nearly horizontal, entering from -Y. That is the
# geometry the 30-degree scope forces; the previous 40-deg shaft only worked
# because the camera was (wrongly) modelled as looking straight along it.
# Clocking the bevel the other way (ENDO_BEVEL_CLOCK_DEG = 180) would instead
# give a 70-deg shaft for the same view.
#
# Retired pose: translate (0, -0.09615, 0.02735), rotateXYZ (14, 0, 180)
# -- the GUI stack [translate, rotY 0, rotX -14, rotZ 180].
ENDO_POSE_TRANSLATE = (0.01477, -0.04363, 0.05360)
ENDO_POSE_ROTXYZ = (-16.767, 2.546, 177.134)

# Extra organs+field offset measured in the GUI (user typed it on the two prims
# with the +1 cm plate lift already baked into the points, so the net capture ->
# world translation is this PLUS the lift). Baked into the points with it.
ORGAN_EXTRA_TRANSLATE = (0.0, -0.01077, 0.03823)
# A further organs-only nudge measured in the GUI afterwards (the field stays
# put) -- again typed on top of the already-baked points, so it adds on.
ORGANS_ONLY_TRANSLATE = (0.0, -0.00111, 0.00274)

# Endoscope light, colour/intensity as tuned in the GUI on the rig (2026-07-23).
# The GUI session had accidentally dragged radius/diffuse/specular to negative
# values; those are restored to physical ones -- a 1.5 mm fibre-bundle emitter
# with standard multipliers.
#
# Its POSE is no longer a constant: on a 30-degree scope the fibre exits the same
# bevelled face as the objective, so add_endoscope_light puts the emitter at the
# tooltip and aims it down the VIEW axis. The retired GUI pose was
# translate (0, 0, 0.1), orientXYZ (-38, 2, -9) in the scope's local frame --
# hand-aimed at the organs from the scope's pre-2026-07-29 pose, which no longer
# exists.
ENDO_LIGHT_COLOR = (1.0, 0.91, 0.9)
ENDO_LIGHT_INTENSITY = 500.0
ENDO_LIGHT_EXPOSURE = 0.0
ENDO_LIGHT_RADIUS = 0.0015          # 1.5 mm emissive fibre bundle
ENDO_LIGHT_DIFFUSE = 1.0
ENDO_LIGHT_SPECULAR = 1.0
ENDO_CONE_DEG = 80.0                # illumination cone half-angle. Was 55: the
                                    # dome walls fell outside it and the white
                                    # PHANTOM_COLOR still shaded to grey. 80 deg
                                    # lights the walls (corner luma 114 -> 159)
                                    # and leaves the field's exposure alone
                                    # (161 -> 163) -- a cone mask admits more
                                    # directions, it does not spread the centre.
ENDO_CONE_SOFTNESS = 0.5            # soft cone edge

# The scope camera at the same tip. The real tower runs 1280x720; its measured
# intrinsics arrive via --endo-intrinsics (default = calibrated).
ENDO_CAM_RES = (1280, 720)
ENDO_CAM_HFOV_DEG = 70.0            # 'nominal' fallback field of view
ENDO_CAM_FOCAL = 18.0               # arbitrary; only focal:aperture ratio sets FOV

# TIP ANGULATION. The scope on the rig is a 30-degree forward-oblique
# laparoscope, not a 0-degree one (photo of the distal end, 2026-07-29): the end
# face is ground at 30 deg to the plane perpendicular to the shaft, and BOTH the
# objective and the light fibre sit on that face. So the direction of view -- and
# the illumination axis -- leave the tip 30 deg off the shaft axis, not along it.
# The cell was built with a 0-degree camera until now, which is exactly 30 deg of
# view direction unaccounted for.
#
# ENDO_BEVEL_CLOCK_DEG rolls the bevel about the shaft: 0 deflects the view
# toward image DOWN (the scope is clocked so the bevel looks down into the
# abdomen), +90 toward image right.
ENDO_TIP_ANGLE_DEG = 30.0
ENDO_BEVEL_CLOCK_DEG = 0.0


def phantom_placement_offset():
    """The exact scale+centre the phantom collider got in build_deformable_usd.py.

    phantom.stl vertices are metres = mm*0.001, then z shifted so the base sits on
    z=0 and X/Y shifted to their mean, so the dome is centred over the origin.
    Measured trocar holes live in that same STL frame, so they take the same
    transform to land on the placed dome.
    """
    import numpy as np
    import trimesh

    ph = trimesh.load(ROOT / "phantom.stl", process=False)
    v = np.asarray(ph.vertices, np.float64) * 0.001
    return np.array([v[:, 0].mean(), v[:, 1].mean(), v[:, 2].min()])


def rot_z(pts, deg):
    """Rotate an (N,3) array about the world Z axis through the origin."""
    import numpy as np
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return np.asarray(pts, np.float64) @ R.T


def port_positions_local():
    """Trocar holes in the phantom's placed (pre-rotation, unlifted) frame."""
    import numpy as np
    off = phantom_placement_offset()
    return np.asarray(TROCAR_PORTS_MM, np.float64) * 0.001 - off


def port_positions_world():
    """Final world positions of the trocar holes: phantom spin + plate lift.

    The ports are drilled through the phantom, so they carry its 180 deg Z-spin
    and the lift. This is what the robot placement and the D405 are measured
    against (the D405 translates with the ports but keeps identity orientation).
    """
    p = rot_z(port_positions_local(), ARGS.rotate_phantom)
    p[:, 2] += ARGS.lift
    return p


def _lift_existing(prim) -> None:
    """Shift a prim by the full organs offset (plate lift + GUI-measured extra),
    composing with its existing translate op if it has one. Used for the organ
    support plane, which must ride with the organs or they settle back down."""
    d = (ORGAN_EXTRA_TRANSLATE[0], ORGAN_EXTRA_TRANSLATE[1],
         ORGAN_EXTRA_TRANSLATE[2] + ARGS.lift)
    xf = UsdGeom.Xformable(prim)
    for op in xf.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            v = op.Get()
            op.Set(Gf.Vec3d(v[0] + d[0], v[1] + d[1], v[2] + d[2]))
            return
    xf.AddTranslateOp().Set(Gf.Vec3d(*d))


def bake_rot_lift_points(stage, root_path) -> int:
    """Bake the specimen spin+lift into a subtree's point data (not xformOps).

    PhysX deformables step in world space from their cooked point data: an
    ancestor RotateZ authored after cooking is honoured by Hydra while physics is
    stopped, but the moment the sim initialises the bodies snap back to their
    authored points -- the organs visibly "un-rotate". So for the organs (and the
    field they rest in) the transform must live in the points themselves. A 180 deg
    spin about Z through the origin is just (x,y) -> (-x,-y), applied uniformly to
    every point-array attribute (points, physxDeformable* rest/sim points, ...),
    plus the +z lift; extents are re-derived. Physics stays valid: gravity is -Z
    and every cooked array gets the same rigid map.
    """
    import numpy as np

    a = np.radians(ARGS.rotate_organs)
    R = np.array([[np.cos(a), -np.sin(a), 0.0],
                  [np.sin(a), np.cos(a), 0.0],
                  [0.0, 0.0, 1.0]])
    lift = np.array([0.0, 0.0, ARGS.lift]) + np.array(ORGAN_EXTRA_TRANSLATE)
    if root_path == "/World/organs":
        # the organs got one further GUI nudge after the shared offset; the
        # field stays put
        lift = lift + np.array(ORGANS_ONLY_TRANSLATE)
    root = stage.GetPrimAtPath(root_path)
    if not (root and root.IsValid()):
        log(f"  ! missing: {root_path}")
        return 0
    if UsdGeom.Xformable(root).GetOrderedXformOps():
        log(f"  ! {root_path} already carries xformOps -- not baking twice")
        return 0
    n_attrs = 0
    for prim in Usd.PrimRange(root):
        # the appendix alone gets an extra raise: the top-down capture leaves
        # it buried under/between its neighbours, too deep for the jaws to
        # reach cleanly (user-observed in the endoscope view)
        off = lift
        if root_path == "/World/organs" and \
                "/appendix" in prim.GetPath().pathString:
            off = lift + np.array([0.0, 0.0, ARGS.appendix_raise])
        for attr in prim.GetAttributes():
            name = attr.GetName().lower()
            if not (name == "points" or name.endswith("points")):
                continue
            val = attr.Get()
            if not val:
                continue
            arr = np.array([[v[0], v[1], v[2]] for v in val], np.float64)
            arr = arr @ R.T + off
            attr.Set([Gf.Vec3f(*p) for p in arr.tolist()])
            n_attrs += 1
        ext = prim.GetAttribute("extent")
        if ext and ext.Get():
            lo, hi = np.array(ext.Get()[0]), np.array(ext.Get()[1])
            corners = np.array([[x, y, z] for x in (lo[0], hi[0])
                                for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
            corners = corners @ R.T + off
            ext.Set([Gf.Vec3f(*corners.min(axis=0).tolist()),
                     Gf.Vec3f(*corners.max(axis=0).tolist())])
    return n_attrs


def transform_specimen(stage) -> None:
    """Pose the specimen: spin organs+field AND the phantom+endoscope set about Z;
    lift everything onto the plate. The D405 alone keeps identity orientation
    (handled in add_realsense_405 -- it translates with the ports, no rotation).

    Two mechanisms, deliberately different:
    - organs + field: the spin+lift is BAKED INTO THE POINT DATA, because PhysX
      deformables ignore ancestor xformOps once the sim initialises (they step
      from cooked world-space points) -- with an ancestor op they visibly snap
      back to the unrotated pose on play.
    - phantom + endoscope (static rigid colliders, transforms respected): plain
      ancestor RotateZ + lift xformOps.
    """
    for path in ("/World/organs", "/World/colliders/field"):
        n = bake_rot_lift_points(stage, path)
        log(f"  baked spin+lift into {n} point array(s) under {path}")

    phantom = stage.GetPrimAtPath("/World/colliders/phantom")
    if phantom and phantom.IsValid() and not UsdGeom.Xformable(phantom).GetOrderedXformOps():
        xf = UsdGeom.Xformable(phantom)
        xf.AddRotateZOp().Set(ARGS.rotate_phantom)
        xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, ARGS.lift))
    if phantom and phantom.IsValid():
        # white moulded plastic, not the collider default's dark grey -- it is
        # the abdominal wall that surrounds the whole scope view
        col = _parse_rgb(ARGS.phantom_color, PHANTOM_COLOR, "--phantom-color")
        pv = UsdGeom.Mesh(phantom).CreateDisplayColorPrimvar(UsdGeom.Tokens.constant)
        pv.Set([Gf.Vec3f(*col)])
        log(f"  phantom recoloured: displayColor {col} (linear)")
    else:
        log("  ! missing: /World/colliders/phantom")

    # endoscope: the pose fitted to the real recording (see ENDO_POSE_TRANSLATE).
    # Authored as translate + ONE rotateXYZ op -- separate rotY/rotX/rotZ ops, as
    # the GUI writes them, trip omni.usd's 'too many single axis rotation ops'
    # warning.
    endo = stage.GetPrimAtPath("/World/colliders/endoscope")
    if endo and endo.IsValid() and not UsdGeom.Xformable(endo).GetOrderedXformOps():
        import numpy as np

        xf = UsdGeom.Xformable(endo)
        t_op = xf.AddTranslateOp()
        t_op.Set(Gf.Vec3d(*ENDO_POSE_TRANSLATE))
        rot = list(ENDO_POSE_ROTXYZ)
        rot[0] += float(ARGS.endo_pitch)
        rot[1] += float(ARGS.endo_yaw)
        rot[2] += float(ARGS.endo_roll)
        xf.AddRotateXYZOp().Set(Gf.Vec3f(*rot))
        if ARGS.endo_pitch or ARGS.endo_yaw or ARGS.endo_roll:
            log(f"  endoscope aimed pitch {ARGS.endo_pitch:+.1f} yaw "
                f"{ARGS.endo_yaw:+.1f} roll {ARGS.endo_roll:+.1f} deg -> "
                f"rotateXYZ ({rot[0]:.3f}, {rot[1]:.3f}, {rot[2]:.3f})")
        if (abs(ARGS.endo_advance) > 1e-9 or abs(ARGS.endo_drop) > 1e-9
                or ARGS.endo_offset):
            # Advance along the shaft, not along the boresight: this is a
            # 30-degree scope, so the two differ by 30 degrees, and what the
            # operator actually does is slide the shaft through its port.
            # shaft_dir comes back in the scope's LOCAL frame; the rotation just
            # authored takes it to world. Translation does not change it, so
            # reading it before applying the advance is exact, not a linearisation.
            _, shaft_local, _ = endoscope_tip_frame(stage)
            shaft_w = _endoscope_world_matrix(stage)[:3, :3] @ np.asarray(shaft_local)
            shaft_w = shaft_w / np.linalg.norm(shaft_w)
            t_new = np.asarray(ENDO_POSE_TRANSLATE) + ARGS.endo_advance * shaft_w
            t_new = t_new + np.asarray(
                _parse_rgb(ARGS.endo_offset, (0.0, 0.0, 0.0), "--endo-offset"))
            t_new[2] -= float(ARGS.endo_drop)
            t_op.Set(Gf.Vec3d(*t_new.tolist()))
            log(f"  endoscope advanced {ARGS.endo_advance*1000:+.0f} mm along its "
                f"shaft ({shaft_w[0]:+.3f}, {shaft_w[1]:+.3f}, {shaft_w[2]:+.3f})"
                f"{f', dropped {ARGS.endo_drop*1000:.0f} mm' if ARGS.endo_drop else ''}"
                f" -> translate {tuple(round(float(x), 5) for x in t_new)}")
    elif not (endo and endo.IsValid()):
        log("  ! missing: /World/colliders/endoscope")

    # the organ support plane: lift only (a plane; spinning it is meaningless)
    gp = stage.GetPrimAtPath("/World/groundPlane")
    if gp and gp.IsValid():
        _lift_existing(gp)
    log(f"  organs+field baked at {ARGS.rotate_organs:+.0f} deg (capture frame) + "
        f"lift; phantom spun {ARGS.rotate_phantom:+.0f} deg; endoscope at fitted "
        f"pose T{ENDO_POSE_TRANSLATE} rotXYZ{ENDO_POSE_ROTXYZ}; lift "
        f"{ARGS.lift*1000:.0f} mm (D405 stays unrotated)")


def _parse_rgb(text, fallback, flag: str):
    """'R,G,B' -> (r, g, b) floats, or `fallback` when the flag was not given."""
    if text is None:
        return tuple(fallback)
    parts = [p for p in str(text).replace(" ", "").split(",") if p]
    if len(parts) != 3:
        raise SystemExit(f"{flag} wants R,G,B; got {text!r}")
    return tuple(float(p) for p in parts)


def recolour_tissue(stage) -> None:
    """Pull the organs' and field's baked capture colour toward a target tone.

    Organ colour is not a constant anywhere: it is a per-vertex displayColor
    primvar baked from the reconstruction plys, i.e. the real D405 capture of
    that rabbit, which came out olive-green under the capture lighting. A
    laparoscope looking at the same tissue reads pale cream. This re-authors the
    primvars on the OPENED stage -- no deformable re-cook, exactly the mechanism
    PHANTOM_COLOR uses.

    Per vertex, in LINEAR space (which is how RTX reads displayColor):

        out = lerp(c, luma(c) * tint, desat) * gain

    Blending toward `luma * tint` rather than toward a flat colour is the point:
    every vertex keeps its OWN brightness, so the surface detail the capture
    carries survives even at desat = 1 -- only the hue is replaced. A flat fill
    would give correctly-coloured plasticine.
    """
    import numpy as np

    d, g = float(ARGS.tissue_desat), float(ARGS.tissue_gain)
    if d <= 0.0 and abs(g - 1.0) < 1e-9:
        return
    tint = np.asarray([float(x) for x in ARGS.tissue_tint.split(",")], np.float64)
    if tint.shape != (3,):
        raise SystemExit(f"--tissue-tint wants R,G,B; got {ARGS.tissue_tint!r}")

    targets = []
    organs = stage.GetPrimAtPath("/World/organs")
    if organs and organs.IsValid():
        for organ in organs.GetChildren():
            m = stage.GetPrimAtPath(organ.GetPath().AppendChild("render"))
            if m and m.IsValid():
                targets.append(m)
    if not ARGS.no_tint_field:
        fld = stage.GetPrimAtPath("/World/colliders/field")
        if fld and fld.IsValid():
            targets.append(fld)

    if not targets:
        log("  ! retint: no organ render meshes or field found")
        return

    log(f"  retinting tissue: desat {d:.2f} toward "
        f"({tint[0]:.3f}, {tint[1]:.3f}, {tint[2]:.3f}) linear, gain {g:.2f}")
    for prim in targets:
        pv = UsdGeom.Mesh(prim).GetDisplayColorPrimvar()
        vals = pv.Get() if pv else None
        if not vals:
            log(f"    ! {prim.GetPath()}: no displayColor to retint")
            continue
        c = np.asarray([[v[0], v[1], v[2]] for v in vals], np.float64)
        luma = c @ np.asarray([0.2126, 0.7152, 0.0722])
        out = np.clip(((1.0 - d) * c + d * luma[:, None] * tint[None, :]) * g, 0.0, 1.0)
        pv.Set([Gf.Vec3f(*row) for row in out.tolist()])
        log(f"    {prim.GetPath().name if prim.GetPath().name != 'render' else prim.GetPath().GetParentPath().name}: "
            f"{len(c)} verts, mean rgb "
            f"({c.mean(axis=0)[0]:.3f},{c.mean(axis=0)[1]:.3f},{c.mean(axis=0)[2]:.3f}) -> "
            f"({out.mean(axis=0)[0]:.3f},{out.mean(axis=0)[1]:.3f},{out.mean(axis=0)[2]:.3f})")


def force_gpu_dynamics(stage) -> None:
    """Assert GPU dynamics on every physics scene -- deformables are GPU-only.

    scene_soft.usd authors the flags on /World/physicsScene, but the runtime can
    still end up parsing a scene without them (omni.physx re-creates or re-reads
    scenes on play), which kills every deformable with 'Deformable Body feature is
    only supported on GPU'. Same belt-and-braces as build_deformable_usd.py:
    stamp the flags on every UsdPhysics.Scene that exists.
    """
    found = False
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Scene):
            continue
        px = PhysxSchema.PhysxSceneAPI.Apply(prim)
        px.CreateEnableGPUDynamicsAttr().Set(True)
        px.CreateBroadphaseTypeAttr().Set("GPU")
        px.CreateSolverTypeAttr().Set("TGS")
        # The organs' ~1.6 mm tets need dt = 1/480 (build_deformable_usd.py steps
        # at 1/(60*8) for the same reason). GUI play uses the scene's own rate; at
        # the 60 Hz default the FEM elements invert on instrument contact and the
        # GPU pipeline crashes outright.
        px.CreateTimeStepsPerSecondAttr().Set(ARGS.physics_hz)
        # headroom for the deformable-vs-tool contact burst when the jaws touch
        px.CreateGpuCollisionStackSizeAttr().Set(2 * 67108864)
        px.CreateGpuMaxSoftBodyContactsAttr().Set(2 * 1048576)
        log(f"  physics scene {prim.GetPath()}: GPU dynamics on, "
            f"{ARGS.physics_hz:.0f} Hz, widened GPU contact buffers")
        found = True
    if not found:
        log("  ! no UsdPhysics.Scene on the stage")


def stiffen_shell_contact(stage) -> None:
    """Firm up the shell material so the instrument cannot punch through it.

    scene_soft.usd's /World/materials/shell (bound to the field, phantom and
    endoscope colliders) carries compliantContactStiffness=1e4 N/m -- soft
    enough that the teleop drives (maxForce 87 Nm, ~100 N at the tip) compress
    the contact spring 10+ mm and sail through the zero-thickness field mesh
    with only a few mm of tracking lag. At 3e5 N/m the same saturated push
    stops within ~0.3 mm, so the field actually blocks the tool and the
    tracking-error haptics get something to render. The organs load the shell
    with ~0.1 N at rest (grams of tissue), so they sit on the stiffer spring
    exactly as before.
    """
    prim = stage.GetPrimAtPath("/World/materials/shell")
    if not (prim and prim.IsValid()):
        log("  ! no /World/materials/shell to stiffen")
        return
    api = PhysxSchema.PhysxMaterialAPI.Apply(prim)
    old_k = api.GetCompliantContactStiffnessAttr().Get()
    api.CreateCompliantContactStiffnessAttr().Set(ARGS.shell_stiffness)
    api.CreateCompliantContactDampingAttr().Set(ARGS.shell_damping)
    log(f"  shell contact stiffened: {old_k:.0f} -> {ARGS.shell_stiffness:.0f} N/m "
        f"(damping {ARGS.shell_damping:.0f}) so the tool cannot punch through "
        f"the field")


def dim_scene_lights(stage) -> None:
    """Scale down the dome/key lights inherited from scene_soft.usd.

    They were tuned for organ close-ups; at cell scale the room renders far too
    bright, so scale both by --scene-light-scale.
    """
    for path in ("/World/domeLight", "/World/keyLight"):
        prim = stage.GetPrimAtPath(path)
        if not (prim and prim.IsValid()):
            continue
        attr = prim.GetAttribute("inputs:intensity")
        if not (attr and attr.Get() is not None):
            attr = prim.GetAttribute("intensity")
        if attr and attr.Get() is not None:
            old = float(attr.Get())
            attr.Set(old * ARGS.scene_light_scale)
            log(f"  {path} intensity {old:.0f} -> {old * ARGS.scene_light_scale:.0f}")


def remove_ground_collision(stage) -> None:
    """Delete /World/groundPlane (the organ support plane) entirely.

    build_deformable_usd.py adds a physics ground plane under the organs; in the
    assembled cell the table/plate carry the support, so neither its infinite
    collider nor its visual grid (the CollisionMesh, which rides up with the
    organ offset and floats mid-air) has any business here. A finite support
    box (add_organ_support) replaces it -- the infinite plane, lifted onto the
    organs, would slice through the robot bases at that height.
    """
    gp = stage.GetPrimAtPath("/World/groundPlane")
    if not (gp and gp.IsValid()):
        log("  ! no /World/groundPlane to remove")
        return
    stage.RemovePrim(gp.GetPath())
    log("  removed /World/groundPlane (collider + visual grid)")


def add_organ_support(stage) -> None:
    """A thin, finite, hidden collision box just under the organs -- the cavity
    floor they rest on.

    The field is an open shell with the surgical opening cut through it, so once
    the organs actually simulate (they used to be frozen by a cooking bug, so
    this never mattered) all but the one over solid shell fall straight through
    the opening -- measured 30+ m of free fall. build_deformable_usd.py solved
    exactly this with a ground plane just beneath the lowest organ vertex, but
    the cell deletes that plane (an infinite plane, lifted onto the organs,
    would cut through the robot bases). So author the same support as a FINITE
    box spanning the organ footprint, collision-only (invisible), bound to the
    firm-but-soft shell material so the organs settle rather than bounce.

    Must run AFTER transform_specimen -- it reads the baked (spun+lifted) organ
    render points to size and place itself.
    """
    import numpy as np

    org = stage.GetPrimAtPath("/World/organs")
    if not (org and org.IsValid()):
        log("  ! no /World/organs to support")
        return
    v = []
    for kid in org.GetChildren():
        r = kid.GetChild("render")
        a = r.GetAttribute("points").Get() if r else None
        if a:
            v.extend([[p[0], p[1], p[2]] for p in a])
    if not v:
        log("  ! organ render points not found -- no support box")
        return
    v = np.asarray(v)
    lo, hi = v.min(axis=0), v.max(axis=0)
    m = 0.04                                   # 4 cm margin past the footprint
    top = float(lo[2]) - 0.001                 # top face 1 mm under lowest vertex
    # A THICK solid slab, not a thin sheet: a thin triangle-mesh box lets the
    # soft bodies tunnel and ejects some of them (measured: 4 of 6 fell through
    # a 1 cm sheet). The slab runs from the top face down past the plate, so
    # there is no far side to tunnel to, and it is a convex box (convexHull is
    # exact) -- far more stable than an open triangle mesh.
    bottom = -0.02
    thick = top - bottom
    cx, cy = float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2)
    sx, sy = float(hi[0] - lo[0] + 2 * m), float(hi[1] - lo[1] + 2 * m)
    cz = (top + bottom) / 2.0

    path = "/World/organSupport"
    add_box(stage, path, (cx, cy, cz), (sx, sy, thick), (0.30, 0.30, 0.32))
    prim = stage.GetPrimAtPath(path)
    UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set(
        UsdPhysics.Tokens.convexHull)
    PhysxSchema.PhysxCollisionAPI.Apply(prim)
    pc = PhysxSchema.PhysxCollisionAPI(prim)
    pc.CreateContactOffsetAttr().Set(0.004)
    pc.CreateRestOffsetAttr().Set(0.001)
    shell = stage.GetPrimAtPath("/World/materials/shell")
    if shell and shell.IsValid():
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            UsdShade.Material(shell), UsdShade.Tokens.weakerThanDescendants,
            "physics")
    log(f"  organ support slab: {sx*100:.0f}x{sy*100:.0f} cm, top z={top*1000:.1f} mm "
        f"(under organs {lo[2]*1000:.1f}..{hi[2]*1000:.1f} mm), hidden convex collider")


def add_trocar_ports(stage):
    """Draw a small sphere at every trocar hole; the two used ports stand out.

    The spheres are authored in the phantom's local frame and parented under a
    group that carries the same spin+lift as the phantom, so they track the dome.
    Returns the final world positions (for placing the robots against ports 2,3).
    """
    local = port_positions_local()
    world = port_positions_world()

    grp = UsdGeom.Xform.Define(stage, "/World/trocarPorts")
    gxf = UsdGeom.Xformable(grp.GetPrim())
    gxf.AddRotateZOp().Set(ARGS.rotate_phantom)     # same spin+lift as the phantom
    gxf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, ARGS.lift))
    log("  trocar ports:")
    for i, (lp, wp) in enumerate(zip(local, world)):
        used = i in USED_PORT_IDX
        radius = 0.005 if used else 0.0035     # 7 mm holes; used ones a touch bigger
        if used:
            side = "left" if wp[0] < 0 else "right"
            color = LEFT_COLOR if wp[0] < 0 else RIGHT_COLOR
        else:
            side, color = "", AMBER
        path = f"/World/trocarPorts/port_{i+1:02d}"
        s = UsdGeom.Sphere.Define(stage, path)
        s.CreateRadiusAttr().Set(radius)
        s.CreateExtentAttr().Set([Gf.Vec3f(-radius, -radius, -radius),
                                  Gf.Vec3f(radius, radius, radius)])
        s.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set([Gf.Vec3f(*color)])
        UsdGeom.Xformable(s.GetPrim()).AddTranslateOp().Set(
            Gf.Vec3d(float(lp[0]), float(lp[1]), float(lp[2])))
        tag = f"  <- port {i+1} = {side.upper()} (used)" if used else ""
        log(f"    port {i+1:2d}  world ({wp[0]*1000:+7.1f}, {wp[1]*1000:+7.1f}, "
            f"{wp[2]*1000:+7.1f}) mm{tag}")
    return world


def show_phantom(stage):
    """Un-hide the phantom dome so the trocar markers can be read against it.

    build_deformable_usd.py authors the phantom shell invisible (it encloses the
    organs). For trocar verification the dome is exactly what we want to see, so
    flip it back on.
    """
    prim = stage.GetPrimAtPath("/World/colliders/phantom")
    if prim and prim.IsValid():
        UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(UsdGeom.Tokens.inherited)
        log("  phantom dome shown (--hide-phantom to keep it hidden)")
    else:
        log("  ! no /World/colliders/phantom to show")


def cook_harvest(stage):
    """Cook the deformables and harvest their rest sim-mesh nodes.

    Requirement (user): with no applied force the organs must barely move; the
    appendix must remain limp enough to grasp and lift. That is implemented by
    pinning sim-mesh vertices (author_pins) -- but the sim meshes only exist
    after PhysX's async cooking runs, so this initializes physics IN THE BUILD
    and pumps the app until they appear, then returns {organ: rest node array}
    plus the appendix base point.

    The auto sphere-overlap attachment path (create_auto_deformable_attachment)
    was tried first and silently bonded NOTHING for 3 of the 6 organs --
    identical authoring, on-surface anchors, healthy cooked meshes, yet
    bowel2/bowel3/stomach fell wholesale while cecum/bowel1/appendix held.
    Hence the explicit-vertex path.

    THIS STAGE IS SACRIFICIAL: physics advances during the pump and (with CPU
    readback) writes state back into the stage, so the caller must export the
    cell BEFORE calling this and re-open that file afterwards -- never save
    the stage this ran on.

    Returns {organ: node array}; organs whose cook failed are missing.
    """
    import numpy as np

    org = stage.GetPrimAtPath("/World/organs")
    if not (org and org.IsValid()):
        log("  ! no /World/organs to pin")
        return {}

    # ---- cook: physics must parse for the async cooker to fill /sim --------
    # Physics ADVANCES during app updates once initialize_physics has run,
    # timeline playing or not -- so under gravity the organs start falling
    # while we wait for the cooker, and the harvested node positions would be
    # a FALLEN pose (first attempt pinned the organs several mm low). Zero the
    # scene gravity for the cook pass; this stage is discarded afterwards (the
    # caller re-opens the exported file), so nothing leaks into the cell.
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Scene):
            UsdPhysics.Scene(prim).CreateGravityMagnitudeAttr().Set(0.0)

    from isaacsim.core.api import SimulationContext

    sim = SimulationContext(stage_units_in_meters=1.0)
    force_gpu_dynamics(stage)          # boot reset the scene attrs; cook on GPU
    for prim in stage.Traverse():      # boot also re-authors gravity -- re-zero
        if prim.IsA(UsdPhysics.Scene):
            UsdPhysics.Scene(prim).CreateGravityMagnitudeAttr().Set(0.0)
    sim.initialize_physics()
    names = [k.GetName() for k in org.GetChildren()]
    nodes = {}
    for round_ in range(20):           # up to ~20x30 updates of cooking time
        for _ in range(30):
            simulation_app.update()
        nodes = {}
        for n in names:
            p = stage.GetPrimAtPath(f"/World/organs/{n}/sim")
            a = p.GetAttribute("points").Get() if p else None
            if a:
                nodes[n] = np.array([[q[0], q[1], q[2]] for q in a])
        if len(nodes) == len(names):
            break
    log(f"  cook pass: sim meshes for {len(nodes)}/{len(names)} organs "
        f"({(round_ + 1) * 30} app updates, zero-g)")
    import omni.timeline as _tl
    _tl.get_timeline_interface().stop()
    return nodes


def _author_vtx_attachment(stage, path, sim_path, src1_path, v, sel,
                           src1_origin=None) -> None:
    """One OmniPhysicsVtxXformAttachment: sim verts `sel` held at their rest
    positions, expressed in src1's frame (src1_origin = its translation)."""
    import numpy as np

    origin = np.zeros(3) if src1_origin is None else np.asarray(src1_origin)
    att = stage.DefinePrim(path, "OmniPhysicsVtxXformAttachment")
    att.CreateAttribute("omniphysics:attachmentEnabled",
                        Sdf.ValueTypeNames.Bool).Set(True)
    att.CreateAttribute("omniphysics:vtxIndicesSrc0",
                        Sdf.ValueTypeNames.IntArray).Set([int(i) for i in sel])
    att.CreateAttribute("omniphysics:localPositionsSrc1",
                        Sdf.ValueTypeNames.Point3fArray).Set(
                            [Gf.Vec3f(*(v[i] - origin).tolist()) for i in sel])
    att.CreateRelationship("omniphysics:src0").SetTargets([Sdf.Path(sim_path)])
    att.CreateRelationship("omniphysics:src1").SetTargets([Sdf.Path(src1_path)])


def _author_grip_cluster(stage, idx, sim_path, v, sel):
    """A kinematic grip handle + rigid weld on sim verts `sel`.

    idx 0 -> /World/organPins/appendixGrip (+ grip_appendix); idx>0 ->
    appendixGrip_<idx> (+ grip_appendix_<idx>). Authored ENABLED so the
    attachment is registered at cook time; the teleop disables the non-anchor
    clusters at start and re-enables one when the jaws latch it (toggling a
    registered attachment is the proven path -- one authored disabled might
    never register). Returns (centroid, n_verts)."""
    import numpy as np

    c = v[sel].mean(axis=0)
    hname = "appendixGrip" if idx == 0 else f"appendixGrip_{idx}"
    aname = "grip_appendix" if idx == 0 else f"grip_appendix_{idx}"
    h = UsdGeom.Sphere.Define(stage, f"/World/organPins/{hname}")
    h.CreateRadiusAttr().Set(0.003)
    h.CreateExtentAttr().Set([Gf.Vec3f(-0.003, -0.003, -0.003),
                              Gf.Vec3f(0.003, 0.003, 0.003)])
    UsdGeom.Xformable(h.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(*[float(x) for x in c]))
    UsdGeom.Imageable(h.GetPrim()).CreateVisibilityAttr().Set(
        UsdGeom.Tokens.invisible)
    UsdPhysics.RigidBodyAPI.Apply(h.GetPrim()).CreateKinematicEnabledAttr().Set(True)
    UsdPhysics.CollisionAPI.Apply(
        h.GetPrim()).CreateCollisionEnabledAttr().Set(False)
    _author_vtx_attachment(stage, f"/World/organPins/{aname}", sim_path,
                           f"/World/organPins/{hname}", v, sel, src1_origin=c)
    return c, len(sel)


def add_appendix_support(stage) -> None:
    """An invisible shelf conforming to the appendix's underside.

    The field was reconstructed from the D405 looking straight DOWN, so there
    is no geometry UNDER the appendix -- through the surgical opening it hangs
    over a hole and kPa-soft tissue drapes straight down. This builds a small
    static collision heightfield 2.5 mm below the appendix's lower surface
    (grid over its footprint, each vertex at the local minimum of the render
    verts above it), so the body can sag half a millimetre and then rests --
    it never leaves its captured drape, but lifts freely off the shelf.
    """
    import numpy as np

    r = stage.GetPrimAtPath("/World/organs/appendix/render")
    a = r.GetAttribute("points").Get() if (r and r.IsValid()) else None
    if not a:
        log("  ! no appendix render points -- no support shelf")
        return
    v = np.array([[p[0], p[1], p[2]] for p in a])
    lo, hi = v.min(axis=0), v.max(axis=0)
    pad, cell, drop = 0.006, 0.004, 0.0025
    xs = np.arange(lo[0] - pad, hi[0] + pad + cell, cell)
    ys = np.arange(lo[1] - pad, hi[1] + pad + cell, cell)
    zfloor = float(lo[2]) - drop
    pts, zmap = [], {}
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            near = v[(np.abs(v[:, 0] - x) < 1.5 * cell)
                     & (np.abs(v[:, 1] - y) < 1.5 * cell)]
            z = (float(near[:, 2].min()) - drop) if len(near) else zfloor
            zmap[(i, j)] = len(pts)
            pts.append((float(x), float(y), z))
    faces = []
    for j in range(len(ys) - 1):
        for i in range(len(xs) - 1):
            a0, b0 = zmap[(i, j)], zmap[(i + 1, j)]
            c0, d0 = zmap[(i + 1, j + 1)], zmap[(i, j + 1)]
            faces += [(a0, b0, c0), (a0, c0, d0)]
    m = UsdGeom.Mesh.Define(stage, "/World/organSupport_appendix")
    m.CreatePointsAttr().Set([Gf.Vec3f(*p) for p in pts])
    m.CreateFaceVertexIndicesAttr().Set([i for f in faces for i in f])
    m.CreateFaceVertexCountsAttr().Set([3] * len(faces))
    m.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    arr = np.array(pts)
    m.CreateExtentAttr().Set([Gf.Vec3f(*arr.min(axis=0).tolist()),
                              Gf.Vec3f(*arr.max(axis=0).tolist())])
    prim = m.GetPrim()
    UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set(
        UsdPhysics.Tokens.none)
    # tighter offsets than the shelf-to-tissue gap, or the shelf would push
    # the appendix up at spawn instead of catching it after ~0.5 mm
    PhysxSchema.PhysxCollisionAPI.Apply(prim)
    pc = PhysxSchema.PhysxCollisionAPI(prim)
    pc.CreateContactOffsetAttr().Set(0.0005)
    pc.CreateRestOffsetAttr().Set(0.0)
    shell = stage.GetPrimAtPath("/World/materials/shell")
    if shell and shell.IsValid():
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            UsdShade.Material(shell), UsdShade.Tokens.weakerThanDescendants,
            "physics")
    log(f"  appendix support shelf: {len(pts)} verts, {len(faces)} tris, "
        f"{drop*1000:.1f} mm under the underside")


def author_pins(stage, nodes) -> None:
    """Write the OmniPhysicsVtxXformAttachment pins into a CLEAN stage.

    Explicit sim-mesh vertex indices held at their rest positions (src1 is the
    identity /World/organPins Xform, so local positions are world positions):
      - each organ: every sim node in the lower --pin-band fraction of its
        height (the resting side); the exposed top stays free.
      - appendix (user spec, appendectomy): pinned ONLY at its end FARTHEST
        from the left robot; the near end is bonded to a small KINEMATIC
        handle sphere (appendixGrip) that the teleop drives while the jaws
        hold it -- the grasp assist (soft tissue between rigid jaws slips on
        friction alone, so the grasp is made a constraint instead). At rest
        the handle sits at the tip's captured position, so it doubles as the
        tip's anchor; the mid-body rests on the support shelf
        (add_appendix_support).
    The runtime cook is content-deterministic, so the indices stay valid;
    teleop logs the per-organ node counts, so a mismatch would be visible.
    """
    import numpy as np

    UsdGeom.Xform.Define(stage, "/World/organPins")
    # left robot base, as build_cell places it
    ports = port_positions_world()
    mid = ports[list(USED_PORT_IDX)].mean(axis=0)
    robot_base = np.array([-ARGS.spacing / 2.0,
                           float(mid[1]) - ARGS.port_offset, 0.0])

    for n, v in nodes.items():
        sim_path = f"/World/organs/{n}/sim"
        if n == "appendix":
            d_base = np.linalg.norm(v - robot_base, axis=1)
            far_pt, near_pt = v[d_base.argmax()], v[d_base.argmin()]
            # far end: fixed to the world, like the other organs' pins
            df = np.linalg.norm(v - far_pt, axis=1)
            sel_far = np.where(df < max(ARGS.pin_radius,
                                        float(df.min()) + 0.004))[0]
            _author_vtx_attachment(stage, f"/World/organPins/pin_{n}",
                                   sim_path, "/World/organPins", v, sel_far)
            # grip clusters: partition the (non-far-pinned) body into DISJOINT
            # segments along its long axis, each with a kinematic grip handle +
            # rigid weld, so the jaws can latch ANYWHERE along the appendix --
            # not only at the tip (user: "make every part grabbable, like the
            # middle"). Disjoint (a Voronoi split along the axis) so no sim vert
            # is welded to two handles: the near-tip anchor and a mid grab can
            # then both be active without fighting over shared vertices.
            far_set = set(int(i) for i in sel_far)
            body = np.array([i for i in range(len(v)) if i not in far_set])
            axis = far_pt - near_pt
            L = float(np.linalg.norm(axis))
            axis = axis / L if L > 1e-9 else np.array([1.0, 0.0, 0.0])
            t = (v[body] - near_pt) @ axis           # 0 at near end .. L at far
            span = float(t.max() - t.min()) if len(t) else 0.0
            K = int(max(3, min(7, round(span / 0.010))))  # ~1 cm segments
            edges = np.linspace(t.min() - 1e-6, t.max() + 1e-6, K + 1)
            bins = np.digitize(t, edges) - 1
            gi, gl = 0, []
            for b in range(K):                       # near end (b=0) -> far
                sel = body[bins == b]
                if len(sel) < 3:
                    continue
                c, ncl = _author_grip_cluster(stage, gi, sim_path, v, sel)
                gl.append((gi, ncl, c))
                gi += 1
            log(f"  pinned {n:9s} far end {len(sel_far)} nodes at "
                f"({far_pt[0]*1000:+.1f},{far_pt[1]*1000:+.1f},"
                f"{far_pt[2]*1000:+.1f}) mm; {gi} grip cluster(s) over "
                f"{span*1000:.0f} mm: " + ", ".join(
                    f"#{i}={ncl}n@({c[0]*1000:+.0f},{c[1]*1000:+.0f},"
                    f"{c[2]*1000:+.0f})" for i, ncl, c in gl))
            continue
        lo, hi = v[:, 2].min(), v[:, 2].max()
        # bowel2 has an unpinned/unsupported region that oscillates forever at
        # rest (no material tuning -- stiffness, damping, iterations, contact
        # filtering -- settled it; it is a rigid-ish free mode). Pin more of it.
        band = PIN_BAND_CAP.get(n, ARGS.pin_band)
        sel = np.where(v[:, 2] < lo + (hi - lo) * band)[0]
        tag = f"lower {band:.0%} band"
        if len(sel) == 0:
            sel = np.array([int(np.argmin(v[:, 2]))])
            tag += " (empty -> lowest node)"
        _author_vtx_attachment(stage, f"/World/organPins/pin_{n}",
                               sim_path, "/World/organPins", v, sel)
        log(f"  pinned {n:9s} {len(sel):4d}/{len(v):4d} sim nodes [{tag}]")

    add_appendix_support(stage)


def calm_organs(stage) -> None:
    """Stop the at-rest squirm: no phantom contact gap, no field spawn overlap.

    Persistent contact forces were animating the 'resting' organs (the
    user-observed trembling that eventually ejected them from the cavity):
      1. Contact offsets wider than the anatomical packing: at 3 mm contact /
         0.8 mm rest offset every neighbouring pair sits permanently inside
         the offset band, so the solver pushes them apart forever. Tighten to
         --organ-contact-offset / --organ-rest-offset.
      2. The field collider overlaps the organs at spawn (they were captured
         embedded against it) -- filtered below unless --organ-field-collision.
    (Gravity sag is handled by the pin attachments; the schema's per-body
    disableGravity was tried and is inert on these deformables.)
    """
    org = stage.GetPrimAtPath("/World/organs")
    if not (org and org.IsValid()):
        log("  ! no /World/organs to calm")
        return
    for kid in org.GetChildren():
        root = kid.GetPrim() if hasattr(kid, "GetPrim") else kid
        for cname in ("coll", "sim"):
            c = root.GetChild(cname)
            if not (c and c.IsValid()):
                continue
            pc = PhysxSchema.PhysxCollisionAPI.Apply(c)
            if c.GetAttribute("physxCollision:contactOffset").Get() is not None:
                pc.CreateContactOffsetAttr().Set(ARGS.organ_contact_offset)
                pc.CreateRestOffsetAttr().Set(ARGS.organ_rest_offset)
        # stiffer contact resolution: a strong position drive used to punch
        # the tool tip deep into a pushed organ; more solver iterations hold
        # the contact so it dents instead (paired with the tip rest-offset skin)
        it = root.GetAttribute("physxDeformable:solverPositionIterationCount")
        if not it:
            it = root.CreateAttribute(
                "physxDeformable:solverPositionIterationCount",
                Sdf.ValueTypeNames.Int)
        it.Set(int(ARGS.organ_iterations))
    if not ARGS.organ_field_collision:
        field = stage.GetPrimAtPath("/World/colliders/field")
        if field and field.IsValid():
            fp = UsdPhysics.FilteredPairsAPI.Apply(field)
            rel = fp.CreateFilteredPairsRel()
            for kid in org.GetChildren():
                rel.AddTarget(kid.GetPath())
            log("  organ<->field collision filtered (spawn overlap ejects; "
                "--organ-field-collision restores)")
    if ARGS.filter_organ_organ:
        # optional: filter organ<->organ contact. Not needed for the bowel2 buzz
        # (that was a numerical instability, fixed by YOUNGS_CAP), but available
        # if organ<->organ contact is ever undesirable. Each body still collides
        # with the tool and the field.
        kids = list(org.GetChildren())
        for a in kids:
            ap = a.GetPrim() if hasattr(a, "GetPrim") else a
            fp = UsdPhysics.FilteredPairsAPI.Apply(ap)
            rel = fp.CreateFilteredPairsRel()
            for b in kids:
                if b.GetPath() != a.GetPath():
                    rel.AddTarget(b.GetPath())
        log(f"  organ<->organ collision filtered ({len(kids)} bodies; stops the "
            f"at-rest contact wobble; --organ-organ-collision restores)")
    # organ material tuning: firmer + denser tissue (5 kPa/1050 was so soft &
    # light the tool sank in and the tissue flew off it -- user), modest
    # friction (high friction made a poked organ ride up with the tip), and
    # damping against the perpetual at-rest wobble.
    mats = stage.GetPrimAtPath("/World/materials")
    for kid in (mats.GetChildren() if mats and mats.IsValid() else []):
        if not kid.HasAttribute("omniphysics:youngsModulus"):
            continue
        api = PhysxSchema.PhysxDeformableBodyMaterialAPI.Apply(kid)
        api.CreateElasticityDampingAttr().Set(ARGS.organ_damping)
        api.CreateDynamicFrictionAttr().Set(float(ARGS.organ_friction))
        # authoritative omniphysics pair (the physx one is a legacy mirror)
        for fa, val in (("omniphysics:staticFriction", ARGS.organ_friction),
                        ("omniphysics:dynamicFriction", ARGS.organ_friction),
                        ("omniphysics:density", ARGS.organ_density)):
            a = kid.GetAttribute(fa)
            if a:
                a.Set(float(val))
        # Per-organ stiffness cap: bowel2 (and mildly bowel3) go NUMERICALLY
        # UNSTABLE at 30 kPa -- their cooked mesh has fine/sliver elements whose
        # stable timestep at 480 Hz falls below 1/480 s, so they buzz ~1-2 mm/s
        # at rest forever. This is not a contact or elastic-damping problem
        # (filtering organ<->organ contact and 10x damping + 64 iters all left
        # it unchanged); the only lever is a lower modulus for those bodies.
        # The stable organs keep the firm 30 kPa the user asked for.
        nm = kid.GetName()
        if nm == "appendix":
            E = ARGS.appendix_youngs
        else:
            E = YOUNGS_CAP.get(nm, ARGS.organ_youngs)
        if E > 0:
            kid.GetAttribute("omniphysics:youngsModulus").Set(float(E))
    log(f"  organ materials: E appendix {ARGS.appendix_youngs/1000:.0f} / "
        f"others {ARGS.organ_youngs/1000:.0f} kPa (caps "
        + ",".join(f"{k}:{v/1000:.0f}" for k, v in YOUNGS_CAP.items())
        + f"), density {ARGS.organ_density:.0f}, friction {ARGS.organ_friction}, "
        f"damping {ARGS.organ_damping}, iterations {ARGS.organ_iterations}")
    log(f"  organs calmed: contact/rest offsets {ARGS.organ_contact_offset*1000:.1f}/"
        f"{ARGS.organ_rest_offset*1000:.1f} mm")


def show_field(stage):
    """Un-hide the field -- the rabbit's body wall / abdomen the organs sit in.

    build_deformable_usd.py authors it invisible so the organs are visible from
    OUTSIDE the specimen; the cell's cameras (D405, endoscope) sit INSIDE the
    phantom looking at the surgical site, where the field is the photographed
    tissue that surrounds the organs. It stays a static rigid collider (an open
    457k-face scanned shell cannot be a PhysX volume deformable) with the
    firm shell contact material; only its visibility changes here. GUI unhides
    done by hand die with every rebuild -- this authors it into the cell.
    """
    prim = stage.GetPrimAtPath("/World/colliders/field")
    if prim and prim.IsValid():
        UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(UsdGeom.Tokens.inherited)
        log("  field (body wall) shown (--hide-field to keep it hidden)")
    else:
        log("  ! no /World/colliders/field to show")


def phantom_apex_world_z():
    """World Z of the phantom's top after its placement (base on z=0) + lift.

    The phantom was scaled mm->m and shifted so its base sits at z=0; the specimen
    lift then raises it. The 180 deg Z-spin does not touch Z, so apex = height + lift.
    """
    import numpy as np
    import trimesh
    v = np.asarray(trimesh.load(ROOT / "phantom.stl", process=False).vertices, np.float64)
    height = (v[:, 2].max() - v[:, 2].min()) * 0.001
    return height + ARGS.lift


def add_realsense_405(stage):
    """A RealSense D405 in the phantom's top hole: XY at the midpoint of the four
    highest (top) ports, Z at the dome apex, looking straight down (-Z) with its
    image axes parallel to world X/Y.

    A USD camera images along its local -Z with +Y up, so an identity rotation
    already looks -Z with X/Y world-aligned -- exactly the requested pose.
    """
    import numpy as np

    pw = port_positions_world()
    top4 = np.argsort(pw[:, 2])[-4:]                 # the four highest ports
    cam_xy = pw[top4][:, :2].mean(axis=0)
    cam_z = phantom_apex_world_z() - ARGS.camera_drop
    cam_pos = (float(cam_xy[0]), float(cam_xy[1]), float(cam_z))

    root = UsdGeom.Xform.Define(stage, "/World/sensors/realsense_d405")
    UsdGeom.Xformable(root.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*cam_pos))

    cam = UsdGeom.Camera.Define(stage, "/World/sensors/realsense_d405/d405")
    W, H = D405_RES
    if ARGS.d405_intrinsics and ARGS.d405_intrinsics != "nominal":
        # exact pinhole of the real unit: fx -> aperture, ppx/ppy -> aperture
        # offsets (USD: offset is from the image centre, in aperture units)
        fx, fy, ppx, ppy = (float(v) for v in ARGS.d405_intrinsics.split(","))
        hap = W * D405_FOCAL / fx
        vap = H * D405_FOCAL / fy
        cam.CreateHorizontalApertureOffsetAttr().Set(float((ppx - W / 2) * hap / W))
        cam.CreateVerticalApertureOffsetAttr().Set(float(-(ppy - H / 2) * vap / H))
        log(f"  d405 intrinsics from calibration: fx={fx:.1f} fy={fy:.1f} "
            f"ppx={ppx:.1f} ppy={ppy:.1f}")
    else:
        # nominal datasheet model: 87 deg HFOV, principal point at the centre
        hap = 2.0 * D405_FOCAL * np.tan(np.radians(D405_HFOV_DEG) / 2.0)
        vap = hap * (H / W)                          # match the 848:480 sensor aspect
    cam.CreateFocalLengthAttr().Set(D405_FOCAL)
    cam.CreateHorizontalApertureAttr().Set(float(hap))
    cam.CreateVerticalApertureAttr().Set(float(vap))
    cam.CreateClippingRangeAttr().Set(Gf.Vec2f(0.001, 1000.0))
    # record the real sensor resolution for downstream renderers
    p = cam.GetPrim()
    p.CreateAttribute("d405:width", Sdf.ValueTypeNames.Int).Set(D405_RES[0])
    p.CreateAttribute("d405:height", Sdf.ValueTypeNames.Int).Set(D405_RES[1])

    # a small dark housing sitting on top of the lens (extends up +Z from the optic)
    bx, by, bz = D405_BODY
    add_box(stage, "/World/sensors/realsense_d405/body",
            (0.0, 0.0, bz / 2.0), D405_BODY, D405_COLOR)

    vfov = 2.0 * np.degrees(np.arctan(vap / (2.0 * D405_FOCAL)))
    log(f"  RealSense D405 ('d405') at ({cam_pos[0]*1000:+.1f}, {cam_pos[1]*1000:+.1f}, "
        f"{cam_pos[2]*1000:+.1f}) mm looking -Z ({ARGS.camera_drop*1000:.0f} mm below "
        f"apex); ports {sorted((top4+1).tolist())} are the top four; "
        f"FOV {D405_HFOV_DEG:.0f}x{vfov:.0f} deg, {D405_RES[0]}x{D405_RES[1]}")


def cut_phantom_view_hole(stage):
    """Open the phantom's top hole so nothing blocks the D405 looking down.

    The real dome has a ~3 cm opening at the apex, but the scanned mesh came out
    thick enough that the shell (and the camera dropped 5 mm into it) occludes the
    view. Delete every phantom face whose centroid lies inside a vertical cylinder
    around the camera axis in the upper dome -- for the camera it is as if the
    shell simply is not there (visuals and collision both, as the faces are gone).
    Authored in the phantom's LOCAL frame (the prim carries the spin+lift).
    """
    import numpy as np

    prim = stage.GetPrimAtPath("/World/colliders/phantom")
    if not (prim and prim.IsValid()):
        log("  ! no phantom to cut the view hole through")
        return
    mesh = UsdGeom.Mesh(prim)
    pts = np.array([[v[0], v[1], v[2]] for v in mesh.GetPointsAttr().Get()])
    idx = np.array(mesh.GetFaceVertexIndicesAttr().Get(), np.int64)
    cnt = np.array(mesh.GetFaceVertexCountsAttr().Get(), np.int64)
    if not (cnt == 3).all():
        log("  ! phantom is not all-triangles; skipping view hole")
        return
    tris = idx.reshape(-1, 3)

    # camera axis in the phantom's local frame: the ports' local midpoint
    pl = port_positions_local()
    top4 = np.argsort(pl[:, 2])[-4:]
    centre = pl[top4][:, :2].mean(axis=0)

    # A face blocks the hole if its XY projection INTERSECTS the circle -- the
    # scanned mesh has triangles large enough that a centroid-in-circle test
    # leaves the axis covered. Test: any vertex inside the circle, OR the circle
    # centre inside the triangle, OR any edge passing within the radius.
    R = ARGS.view_hole_radius
    v2 = pts[tris][:, :, :2] - centre                # (F,3,2), circle at origin
    vert_in = (np.linalg.norm(v2, axis=2) < R).any(axis=1)

    def edge_near(a, b):
        ab = b - a
        t = np.clip(-(a * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-30), 0.0, 1.0)
        near = a + ab * t[:, None]
        return np.linalg.norm(near, axis=1) < R

    edge_in = (edge_near(v2[:, 0], v2[:, 1]) | edge_near(v2[:, 1], v2[:, 2])
               | edge_near(v2[:, 2], v2[:, 0]))

    def cross_z(a, b):
        return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]

    d0 = cross_z(v2[:, 1] - v2[:, 0], -v2[:, 0])
    d1 = cross_z(v2[:, 2] - v2[:, 1], -v2[:, 1])
    d2 = cross_z(v2[:, 0] - v2[:, 2], -v2[:, 2])
    centre_in = ((d0 >= 0) & (d1 >= 0) & (d2 >= 0)) | ((d0 <= 0) & (d1 <= 0) & (d2 <= 0))

    in_cyl = (vert_in | edge_in | centre_in)
    in_cyl &= pts[tris][:, :, 2].max(axis=1) > 0.12  # upper dome only
    keep = tris[~in_cyl]
    mesh.GetFaceVertexIndicesAttr().Set([int(i) for i in keep.flatten()])
    mesh.GetFaceVertexCountsAttr().Set([3] * len(keep))
    log(f"  cut D405 view hole: removed {int(in_cyl.sum())} phantom faces within "
        f"r={ARGS.view_hole_radius*1000:.0f} mm of the camera axis (upper dome)")


def endoscope_tip_frame(stage):
    """Endoscope tooltip geometry, in the endoscope's LOCAL frame:
    (tip, shaft_dir, organ_dir).

    The tip camera and light are parented UNDER /World/colliders/endoscope so they
    ride along when the scope moves, which means their transforms must be authored
    in the endoscope's local frame -- the raw mesh coordinates (the prim's own
    RotateZ+lift then carries them to world). The organ centroid (a world-space
    quantity) is pulled back into that local frame for the comparisons.

    tip = the shaft end nearer the organs. shaft_dir = unit shaft axis pointing
    OUT of the tip (a 0-degree scope camera looks along it). organ_dir = from the
    tip toward the organ centroid (what the light illuminates).
    """
    import numpy as np
    import trimesh

    bc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    rng = bc.ComputeWorldBound(stage.GetPrimAtPath("/World/organs")).ComputeAlignedRange()
    organ_w = (np.array(rng.GetMin()) + np.array(rng.GetMax())) / 2.0
    # world -> endoscope local, through whatever transform the prim actually
    # carries (the measured translate+rotateXYZ pose)
    organ_c = _endoscope_world_matrix_inv(stage) @ np.append(organ_w, 1.0)
    organ_c = organ_c[:3]

    v = np.asarray(trimesh.load(ROOT / "mesh" / "endoscope.ply", process=False).vertices,
                   np.float64) * 0.001
    c = v.mean(axis=0)
    axis = np.linalg.svd(v - c)[2][0]
    proj = (v - c) @ axis
    ends = [v[proj.argmin()], v[proj.argmax()]]
    if np.linalg.norm(ends[0] - organ_c) < np.linalg.norm(ends[1] - organ_c):
        tip, tail = ends[0], ends[1]
    else:
        tip, tail = ends[1], ends[0]
    shaft_dir = tip - tail
    shaft_dir = shaft_dir / np.linalg.norm(shaft_dir)
    organ_dir = organ_c - tip
    organ_dir = organ_dir / np.linalg.norm(organ_dir)
    return tip, shaft_dir, organ_dir


def _endoscope_world_matrix(stage):
    """The endoscope prim's local-to-world matrix, column-vector convention."""
    import numpy as np
    prim = stage.GetPrimAtPath("/World/colliders/endoscope")
    M = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)
    # Gf matrices are row-vector; transpose into column convention.
    return np.array([[M[i][j] for j in range(4)] for i in range(4)]).T


def _endoscope_world_matrix_inv(stage):
    import numpy as np
    return np.linalg.inv(_endoscope_world_matrix(stage))


def endoscope_local_to_world(stage, p):
    """Map a point in the endoscope's local frame to world."""
    import numpy as np
    return (_endoscope_world_matrix(stage) @ np.append(p, 1.0))[:3]


def _minus_z_rotation(direction, roll_deg: float = 0.0):
    """The Gf.Rotation taking local -Z (USD camera/light boresight) onto
    `direction`, with an optional roll about that boresight."""
    d = Gf.Vec3d(float(direction[0]), float(direction[1]),
                 float(direction[2])).GetNormalized()
    rot = Gf.Rotation(Gf.Vec3d(0, 0, -1), d)
    if roll_deg:
        # Gf composes left-to-right (row vectors): roll in local frame first,
        # then align the boresight.
        rot = Gf.Rotation(Gf.Vec3d(0, 0, 1), roll_deg) * rot
    return rot


def _orient_minus_z(prim, direction, roll_deg: float = 0.0) -> None:
    """Orient a prim so its local -Z points along `direction` -- authored as ONE
    orient op (a second rotation op alongside an orient trips omni.usd's
    'Incompatible xformOpOrder' warning)."""
    _author_orient(prim, _minus_z_rotation(direction, roll_deg))


def _author_orient(prim, rot) -> None:
    q = rot.GetQuat()
    UsdGeom.Xformable(prim).AddOrientOp().Set(
        Gf.Quatf(float(q.GetReal()), Gf.Vec3f(*[float(x) for x in q.GetImaginary()])))


def bevelled_tip_basis(shaft_dir, roll_deg: float = 180.0,
                       tilt_deg: float = None, clock_deg: float = None):
    """(right, up, forward) for a scope tip ground at `tilt_deg` to the plane
    perpendicular to the shaft.

    Starts from the basis a 0-degree scope would have -- boresight along
    `shaft_dir`, `roll_deg` about it -- then rolls by `clock_deg` about the shaft
    to point the bevel, and swings the boresight `tilt_deg` toward image DOWN.
    A 0-degree scope (tilt 0) reproduces the old basis exactly.
    """
    import numpy as np

    tilt = ENDO_TIP_ANGLE_DEG if tilt_deg is None else tilt_deg
    clock = ENDO_BEVEL_CLOCK_DEG if clock_deg is None else clock_deg
    M = Gf.Matrix3d(_minus_z_rotation(shaft_dir, roll_deg))
    # Gf is row-vector: local axis i maps to row i.
    right = np.array([M[0][j] for j in range(3)])
    up = np.array([M[1][j] for j in range(3)])
    fwd = -np.array([M[2][j] for j in range(3)])
    c, s = np.cos(np.radians(clock)), np.sin(np.radians(clock))
    right, up = c * right + s * up, -s * right + c * up
    ct, st = np.cos(np.radians(tilt)), np.sin(np.radians(tilt))
    fwd, up = ct * fwd - st * up, st * fwd + ct * up
    return right, up, fwd


def _orient_from_basis(prim, right, up, fwd) -> None:
    """Author the orient op for a prim whose local +X/+Y/-Z are right/up/fwd."""
    rows = [right, up, [-f for f in fwd]]
    M = Gf.Matrix3d(*[float(rows[i][j]) for i in range(3) for j in range(3)])
    _author_orient(prim, M.ExtractRotation())


def add_endoscope_light(stage):
    """A laparoscope tip light inside the phantom: a small warm-white SphereLight
    with a wide ShapingAPI cone at the endoscope tooltip.

    Parented UNDER the endoscope prim. Colour, intensity, cone and radius are as
    tuned in the GUI on the rig; radius/diffuse/specular use physical values (the
    GUI session had dragged them negative by accident).

    The POSE is no longer the GUI one. On a 30-degree scope the fibre exits the
    same bevelled end face as the objective, so the illumination axis is the
    VIEW axis, not the shaft axis and not a separately dragged direction: the
    light sits at the tooltip and is aimed down `bevelled_tip_basis`'s forward.
    The old GUI orientation was hand-aimed at the organs from the scope's
    previous pose and does not survive re-posing the scope.
    """
    tip, shaft_dir, _ = endoscope_tip_frame(stage)
    right, up, fwd = bevelled_tip_basis(shaft_dir)
    light = UsdLux.SphereLight.Define(
        stage, "/World/colliders/endoscope/endoscope_light")
    light.CreateRadiusAttr(ENDO_LIGHT_RADIUS)
    light.CreateIntensityAttr(ARGS.endoscope_intensity)
    light.CreateExposureAttr(ENDO_LIGHT_EXPOSURE)
    light.CreateColorAttr(Gf.Vec3f(*ENDO_LIGHT_COLOR))
    light.CreateEnableColorTemperatureAttr(False)
    light.CreateNormalizeAttr(True)
    light.CreateDiffuseAttr(ENDO_LIGHT_DIFFUSE)
    light.CreateSpecularAttr(ENDO_LIGHT_SPECULAR)
    cone = ENDO_CONE_DEG if ARGS.endo_cone_deg is None else ARGS.endo_cone_deg
    shaping = UsdLux.ShapingAPI.Apply(light.GetPrim())
    shaping.CreateShapingConeAngleAttr(cone)
    shaping.CreateShapingConeSoftnessAttr(ENDO_CONE_SOFTNESS)
    xf = UsdGeom.Xformable(light.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(float(tip[0]), float(tip[1]), float(tip[2])))
    _orient_from_basis(light.GetPrim(), right, up, fwd)

    log(f"  endoscope light at the tooltip, aimed down the {ENDO_TIP_ANGLE_DEG:.0f}-deg "
        f"view axis, intensity {ARGS.endoscope_intensity:.0f} "
        f"diffuse {ENDO_LIGHT_DIFFUSE} specular {ENDO_LIGHT_SPECULAR} "
        f"radius {ENDO_LIGHT_RADIUS} cone {cone}")


def author_endoscope_distortion(cam, W: int, H: int) -> None:
    """Give the scope camera a real wide-angle barrel, not a pinhole.

    A laparoscope is a wide-angle objective and its picture is barrelled: the
    centre is magnified and the scale falls off toward the edge, so a shaft
    entering near the tip is fat and thins toward the distal end. A pinhole
    renders that geometry flat, which is what makes the sim frame read
    orthographic next to a real one.

    RTX can render this itself -- `omni:lensdistortion:model = opencvPinhole`
    plus the OpenCV coefficients -- so it goes in the projection, not into a
    post-hoc warp of an already-rendered pinhole image (which would resample
    what the renderer sampled at the wrong density and blur the periphery).
    The attribute set is exactly what
    `isaacsim.sensors.camera.Camera.set_opencv_pinhole_properties` writes.

    A second thing falls out of it. The aperture path forces SQUARE pixels
    (measured by check_projection.py), so the calibration's 2 % fx/fy anisotropy
    could not be expressed at all; this model takes fx and fy separately, so the
    measured pair is authored as measured.

    NOTE the calibrated tower measured LOW distortion and was deliberately
    undistorted to a pinhole. k1 = -0.20 is therefore a chosen wide-angle LOOK,
    not that measurement, and it puts the render out of agreement with
    `start_pose.project()`, which stays a pinhole -- so the tip pixels the build
    logs are pinhole predictions and drift from the rendered position toward the
    edge. `--endo-k1 0` restores the calibrated pinhole.
    """
    import numpy as np

    if not ARGS.endo_k1 and not ARGS.endo_k2:
        return
    if ARGS.endo_intrinsics and ARGS.endo_intrinsics != "nominal":
        fx, fy, ppx, ppy = (float(v) for v in ARGS.endo_intrinsics.split(","))
    else:
        fx = fy = W / (2.0 * np.tan(np.radians(ENDO_CAM_HFOV_DEG) / 2.0))
        ppx, ppy = W / 2.0, H / 2.0
    p = cam.GetPrim()
    p.CreateAttribute("omni:lensdistortion:model",
                      Sdf.ValueTypeNames.Token).Set("opencvPinhole")
    pre = "omni:lensdistortion:opencvPinhole:"
    p.CreateAttribute(pre + "imageSize", Sdf.ValueTypeNames.Int2).Set(Gf.Vec2i(W, H))
    for name, val in (("cx", ppx), ("cy", ppy), ("fx", fx), ("fy", fy),
                      ("k1", ARGS.endo_k1), ("k2", ARGS.endo_k2),
                      ("k3", 0.0), ("k4", 0.0), ("k5", 0.0), ("k6", 0.0),
                      ("p1", 0.0), ("p2", 0.0),
                      ("s1", 0.0), ("s2", 0.0), ("s3", 0.0), ("s4", 0.0)):
        p.CreateAttribute(pre + name, Sdf.ValueTypeNames.Float).Set(float(val))
    log(f"  endoscope lens: opencvPinhole barrel k1={ARGS.endo_k1:+.3f} "
        f"k2={ARGS.endo_k2:+.3f}, fx={fx:.1f} fy={fy:.1f} "
        f"({100*(fy/fx-1):+.1f} % anisotropy now expressible), "
        f"principal point ({ppx:.1f}, {ppy:.1f})")


def add_endoscope_camera(stage):
    """The scope camera at the endoscope tip, with the tip's angulation.

    A 0-degree laparoscope has no angulation and its optical axis IS the shaft
    axis. This scope is a 30-degree one: the objective sits on an end face ground
    at ENDO_TIP_ANGLE_DEG to the plane perpendicular to the shaft, so the
    boresight leaves the tip that many degrees off the shaft, deflected toward
    image DOWN (see ENDO_BEVEL_CLOCK_DEG). Parented UNDER the endoscope prim
    (transform authored in its local frame), so the camera rides along when the
    scope moves.
    """
    import numpy as np

    tip, shaft_dir, _ = endoscope_tip_frame(stage)
    right, up, fwd = bevelled_tip_basis(shaft_dir)

    cam = UsdGeom.Camera.Define(stage, "/World/colliders/endoscope/endoscope_cam")
    W, H = ENDO_CAM_RES
    if ARGS.endo_intrinsics and ARGS.endo_intrinsics != "nominal":
        # exact pinhole of the real tower: fx -> aperture, ppx/ppy -> aperture
        # offsets.
        #
        # The offset signs and scales are NOT the naive ones. Measured against
        # the rendered image (isaac/check_projection.py: a 3x3 emissive grid at
        # a known depth, centroids fitted in the PNG), the RTX viewport puts the
        # principal point at
        #     cx = W/2 - hoff*(W/hap),   cy = H/2 + voff*(W/hap)
        # so the previous authoring -- +(ppx-W/2)*hap/W and -(ppy-H/2)*vap/H --
        # landed it MIRRORED about the image centre, 57 px across and 126 px
        # down from the calibrated point. start_pose.aperture_offsets inverts
        # the measured mapping instead.
        #
        # The renderer also uses SQUARE pixels (fy = fx), ignoring
        # verticalAperture, so the calibration's 2 % fx/fy anisotropy cannot be
        # expressed here; verticalAperture is authored to the square-pixel value
        # so the USD describes what is actually drawn.
        fx, fy, ppx, ppy = (float(v) for v in ARGS.endo_intrinsics.split(","))
        hap = W * ENDO_CAM_FOCAL / fx
        vap = hap * H / W
        sys.path.insert(0, str(ROOT / "isaac"))
        from start_pose import aperture_offsets
        hoff, voff = aperture_offsets(ppx, ppy, W, H, hap)
        cam.CreateHorizontalApertureOffsetAttr().Set(float(hoff))
        cam.CreateVerticalApertureOffsetAttr().Set(float(voff))
        log(f"  endoscope intrinsics from calibration: fx={fx:.1f} fy={fy:.1f} "
            f"ppx={ppx:.1f} ppy={ppy:.1f}")
        log(f"    -> rendered pinhole fx=fy={W*ENDO_CAM_FOCAL/hap:.1f} "
            f"(square pixels; calibrated fy {fy:.1f} is {100*(fy/fx-1):+.1f} % "
            f"and cannot be authored), principal point on ({ppx:.1f}, {ppy:.1f}) "
            f"via aperture offsets ({hoff:+.4f}, {voff:+.4f})")
    else:
        hap = 2.0 * ENDO_CAM_FOCAL * np.tan(np.radians(ENDO_CAM_HFOV_DEG) / 2.0)
        vap = hap * (H / W)
    cam.CreateFocalLengthAttr().Set(ENDO_CAM_FOCAL)
    cam.CreateHorizontalApertureAttr().Set(float(hap))
    cam.CreateVerticalApertureAttr().Set(float(vap))
    cam.CreateClippingRangeAttr().Set(Gf.Vec2f(0.001, 1000.0))
    p = cam.GetPrim()
    p.CreateAttribute("endocam:width", Sdf.ValueTypeNames.Int).Set(ENDO_CAM_RES[0])
    p.CreateAttribute("endocam:height", Sdf.ValueTypeNames.Int).Set(ENDO_CAM_RES[1])
    author_endoscope_distortion(cam, W, H)

    UsdGeom.Xformable(p).AddTranslateOp().Set(
        Gf.Vec3d(float(tip[0]), float(tip[1]), float(tip[2])))
    # The minimal rotation that takes -Z onto the shaft leaves the roll arbitrary,
    # and it came out upside-down; the 180 deg roll about the boresight is folded
    # into the SAME orient quaternion (a separate RotateZ op after an orient op
    # trips omni.usd's "Incompatible xformOpOrder, too many single axis rotation
    # ops") -- and so is the tip angulation, by bevelled_tip_basis.
    _orient_from_basis(p, right, up, fwd)

    w = endoscope_local_to_world(stage, tip)
    hfov = 2.0 * np.degrees(np.arctan(hap / (2.0 * ENDO_CAM_FOCAL)))
    vfov = 2.0 * np.degrees(np.arctan(vap / (2.0 * ENDO_CAM_FOCAL)))
    log(f"  endoscope {ENDO_TIP_ANGLE_DEG:.0f}-deg camera riding the scope, tip at "
        f"({w[0]*1000:+.1f}, "
        f"{w[1]*1000:+.1f}, {w[2]*1000:+.1f}) mm world, boresight {ENDO_TIP_ANGLE_DEG:.0f} "
        f"deg off the shaft (bevel clocked {ENDO_BEVEL_CLOCK_DEG:.0f} deg), "
        f"FOV {hfov:.1f}x{vfov:.1f} deg, {W}x{H}")


# --------------------------------------------------------------------------
# FR3 URDF -> a self-contained robot USD we can reference twice
# --------------------------------------------------------------------------

def retarget_urdf() -> Path:
    """Rewrite fr3.urdf's mesh paths onto the bundled franka_description meshes.

    fr3.urdf references ./meshes/fr3/{visual,collision}/<link>.{dae,stl}, which
    are not shipped beside it. The franka_description meshes that come with the
    URDF importer carry the identical link names, so a path swap is all it needs.
    """
    mg = Path(get_extension_path_from_name("isaacsim.robot_motion.motion_generation"))
    src = mg / "motion_policy_configs" / "FR3" / "fr3.urdf"
    imp = Path(get_extension_path_from_name("isaacsim.asset.importer.urdf"))
    meshes = imp / "data" / "urdf" / "robots" / "franka_description" / "meshes"
    if not src.exists():
        raise FileNotFoundError(f"FR3 urdf not found: {src}")
    if not (meshes / "visual" / "link0.dae").exists():
        raise FileNotFoundError(f"franka meshes not found under {meshes}")

    text = src.read_text()
    text = text.replace("./meshes/fr3/visual/", f"{meshes}/visual/")
    text = text.replace("./meshes/fr3/collision/", f"{meshes}/collision/")

    out = ROOT / "out" / "fr3_local.urdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    log(f"  retargeted FR3 urdf -> {out}  (meshes: {meshes})")
    return out


def build_arm_tool_urdf(T) -> Path:
    """FR3 (hand removed) + one of the motorised tools on the bare flange.

    Both arms are built by this one function from a TOOLS spec; the left
    grasper and the right cutter are the same instrument family and differ
    only in the numbers (T.vis_rpy is turned 180 deg about Z on the cutter,
    its tip is 21.8 mm shorter, and its blades open 50 deg instead of 74).
    Keeping one builder is deliberate -- two copies of this would drift.

    Neither arm has a Franka hand: the unit bolts straight onto the flange
    (fr3_link8). So the hand/finger subtree is stripped from the retargeted
    FR3 and one rigid `tool_flange` link -- carrying the CAD meshes -- is
    added coincident with fr3_link8, plus a massless `tool_tip` frame at the
    measured SE(3) T_flange->tip. This is the single URDF both engines share.

    The endtip is articulated: the two blades/jaws are their own links on
    revolute joints about the clevis pin, 0 = closed as exported, +half the
    full opening each. Everything else is one rigid body, position-driven.
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(retarget_urdf())
    root = tree.getroot()
    for el in list(root):
        name = el.get("name")
        if (el.tag == "link" and name in HAND_LINKS) or \
           (el.tag == "joint" and name in HAND_JOINTS):
            root.remove(el)

    if not T.mesh_dir.is_dir():
        raise FileNotFoundError(f"{T.name} meshes not found: {T.mesh_dir}")
    meshes = sorted(p for p in T.mesh_dir.glob("*.stl")
                    if p.name not in set(T.jaw_meshes.values()))
    if not meshes:
        raise FileNotFoundError(f"no STL meshes under {T.mesh_dir}")
    have = {p.name for p in meshes}
    lost = [m for m in LAPARO_EXTRA_MESHES if m not in have]
    if lost:
        raise FileNotFoundError(f"{T.name}: casing meshes missing: {lost}")

    # per-part materials: this arm's own body colour on the housing + shaft
    # (T.body_mat), silver metal endtip, dark fasteners. Every entry is emitted
    # on both arms even though each uses only one body colour -- the two URDFs
    # import to separate stages, so the unused one costs a Looks prim.
    for mname, rgb in TOOL_MATERIALS.items():
        m = ET.SubElement(root, "material", {"name": mname})
        ET.SubElement(m, "color", {"rgba": "%g %g %g 1" % rgb})

    link = ET.SubElement(root, "link", {"name": "tool_flange"})
    inr = ET.SubElement(link, "inertial")
    ET.SubElement(inr, "origin", {"xyz": "%g %g %g" % T.com, "rpy": "0 0 0"})
    ET.SubElement(inr, "mass", {"value": "%g" % T.mass})
    ET.SubElement(inr, "inertia", T.inertia)
    vis_xyz = "%g %g %g" % T.vis_xyz
    vis_rpy = "%g %g %g" % T.vis_rpy
    scale = "%g %g %g" % ((LAPARO_MESH_SCALE,) * 3)
    n_col = 0
    for mesh in meshes:
        fn = str(staged_mesh_path(mesh))
        mname = mesh_material_name(mesh.stem, T.body_mat)
        for tag in ("visual", "collision"):
            if tag == "collision" and mesh.name not in LAPARO_COLLIDE_MESHES:
                continue
            node = ET.SubElement(link, tag)
            ET.SubElement(node, "origin", {"xyz": vis_xyz, "rpy": vis_rpy})
            geo = ET.SubElement(node, "geometry")
            ET.SubElement(geo, "mesh", {"filename": fn, "scale": scale})
            if tag == "visual":
                ET.SubElement(node, "material", {"name": mname})
            else:
                n_col += 1

    # the two blades: revolute about the clevis pin, 0 = closed (as exported),
    # +half the full opening each. The blade link frame is the flange frame
    # TRANSLATED to the pivot (no rotation), so the same one visual origin
    # places the mesh after subtracting the pivot, and the joint axis is the
    # pin axis expressed directly in the flange frame. POSITIVE q opens BOTH,
    # with the opposite signs carried by the two axes.
    import math as _math
    half_open = _math.radians(T.open_total_deg) / 2.0
    jaw_off = tuple(v - p for v, p in zip(T.vis_xyz, T.pivot_xyz))
    for side, sign in (("left", 1.0), ("right", -1.0)):
        jl = ET.SubElement(root, "link", {"name": f"{T.link_prefix}_{side}"})
        inr = ET.SubElement(jl, "inertial")
        ET.SubElement(inr, "origin", {"xyz": "%g %g %g" % JAW_COM,
                                      "rpy": "0 0 0"})
        ET.SubElement(inr, "mass", {"value": "%g" % JAW_MASS})
        ET.SubElement(inr, "inertia", JAW_INERTIA)
        fn = str(T.mesh_dir / T.jaw_meshes[side])
        for tag in ("visual", "collision"):
            node = ET.SubElement(jl, tag)
            ET.SubElement(node, "origin", {"xyz": "%g %g %g" % jaw_off,
                                           "rpy": vis_rpy})
            geo = ET.SubElement(node, "geometry")
            ET.SubElement(geo, "mesh", {"filename": fn, "scale": scale})
            if tag == "visual":
                ET.SubElement(node, "material", {"name": "stainless_steel"})
        j = ET.SubElement(root, "joint", {"name": f"{T.joint_prefix}_{side}",
                                          "type": "revolute"})
        ET.SubElement(j, "parent", {"link": "tool_flange"})
        ET.SubElement(j, "child", {"link": f"{T.link_prefix}_{side}"})
        ET.SubElement(j, "origin", {"xyz": "%g %g %g" % T.pivot_xyz,
                                    "rpy": "0 0 0"})
        ET.SubElement(j, "axis", {"xyz": "%g %g %g" %
                                  tuple(sign * a for a in T.pin_axis)})
        ET.SubElement(j, "limit", {"lower": "0", "upper": "%.6f" % half_open,
                                   "effort": "20", "velocity": "20"})
        ET.SubElement(j, "dynamics", {"damping": "0.01", "friction": "0"})

    # (no synthetic shaft cylinder any more: laparo_shaft.stl is on its own a
    # clean 5 mm tube coaxial with flange Z, so the importer's convex hull of
    # it IS the tube and threads a trocar exactly.)

    # fixed joint: the tool_flange frame IS the bare flange (identity transform)
    j = ET.SubElement(root, "joint", {"name": "fr3_link8_to_tool", "type": "fixed"})
    ET.SubElement(j, "parent", {"link": "fr3_link8"})
    ET.SubElement(j, "child", {"link": "tool_flange"})
    ET.SubElement(j, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    # massless tip frame at the measured SE(3) T_flange->tip
    ET.SubElement(root, "link", {"name": "tool_tip"})
    jt = ET.SubElement(root, "joint",
                       {"name": "tool_flange_to_tip", "type": "fixed"})
    ET.SubElement(jt, "parent", {"link": "tool_flange"})
    ET.SubElement(jt, "child", {"link": "tool_tip"})
    ET.SubElement(jt, "origin", {"xyz": "%g %g %g" % T.tip_xyz,
                                 "rpy": "%g %g %g" % T.tip_rpy})

    out = ROOT / T.urdf
    out.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(out), xml_declaration=True, encoding="utf-8")
    log(f"  built {T.name} urdf -> {out}  ({len(meshes)} static tool meshes, "
        f"{n_col} collider(s) + 2 articulated {T.link_prefix}s "
        f"(0..{T.open_total_deg/2:.2f} deg each, "
        f"{T.open_total_deg:.0f} deg full opening), hand removed, "
        f"tip {T.tip_xyz[2]*1000:.1f} mm out along flange +Z)")
    return out


def build_left_grasper_urdf() -> Path:
    return build_arm_tool_urdf(TOOLS["left"])


def build_right_cutter_urdf() -> Path:
    return build_arm_tool_urdf(TOOLS["right"])


def author_tool_contact(stage) -> None:
    """High-friction, rounded-skin contact on the distal grasper colliders.

    Two fixes authored into the left-arm USD:
      1. Grip: the imported jaw/tip mesh colliders carry the importer's ~0.5
         default friction, so a soft pinch slides. Bind a high-friction
         physics material (--jaw-friction) so the tips bite tissue.
      2. No burial: a hard push used to sink the sharp mesh tip INSIDE an
         organ. A collision rest offset (--tip-rest-offset) gives the tip a
         thin shell it holds outside the tissue surface, so it dents rather
         than penetrates; --tip-contact-offset catches a fast push earlier.
    Applied to the MESH colliders under the jaw links and the distal tool
    cluster. The thin shaft cylinder is left alone -- it threads the trocar
    and must stay slim.
    """
    root = stage.GetDefaultPrim()
    root_path = root.GetPath().pathString if root else ""

    def _friction_material(name, mu, combine):
        m = UsdShade.Material.Define(stage, f"{root_path}/{name}")
        p = m.GetPrim()
        UsdPhysics.MaterialAPI.Apply(p)
        UsdPhysics.MaterialAPI(p).CreateStaticFrictionAttr().Set(mu)
        UsdPhysics.MaterialAPI(p).CreateDynamicFrictionAttr().Set(mu)
        PhysxSchema.PhysxMaterialAPI.Apply(p).CreateFrictionCombineModeAttr(
        ).Set(combine)
        return m

    # the grasper's jaws BITE (combine "max", so the tip's friction wins over
    # whatever tissue offers)
    mat = _friction_material("graspTipMaterial", ARGS.jaw_friction, "max")
    # The cutter's blades must NOT. Scissors that hold tissue are pliers, and
    # that is what the cutter was doing: the pass-through only lifts the
    # colliders while the blades MOVE, so a close that ends with tissue between
    # them puts two solid, 0.6-friction faces on the appendix and carries it
    # around -- indistinguishable from the grasper's grip (user, twice). The
    # combine mode matters as much as the value: "min" means this surface sets
    # the pair's friction no matter how grippy the organ material is.
    blade_mat = _friction_material("bladeTipMaterial", ARGS.blade_friction,
                                   "min")
    # The URDF importer applies CollisionAPI to Xform prims named
    # node_STL_BINARY_ under a separate /colliders/<link>/<part>/ tree; the
    # distal grip surfaces are the two jaws (/jaw_left, /jaw_right) and the
    # clevis/pin cluster on the flange (DISTAL_COLLIDER_STEMS). The long
    # shaft and the housing box are left alone -- the shaft threads the trocar
    # and must stay slim, so only the endtip gets the friction + skin.
    n, hit, slick = 0, [], []
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        path = prim.GetPath().pathString
        is_blade = any(f"/blade_{side}/" in path for side in ("left", "right"))
        is_jaw = is_blade or any(f"/jaw_{side}/" in path
                                 for side in ("left", "right"))
        is_clevis = ("/tool_flange/" in path and
                     any(f"/{s}/" in path for s in DISTAL_COLLIDER_STEMS))
        if not (is_jaw or is_clevis):
            continue
        if prim.IsA(UsdGeom.Cylinder):          # never the shaft
            continue
        pc = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        pc.CreateContactOffsetAttr().Set(ARGS.tip_contact_offset)
        pc.CreateRestOffsetAttr().Set(ARGS.tip_rest_offset)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            blade_mat if is_blade else mat,
            UsdShade.Tokens.weakerThanDescendants, "physics")
        n += 1
        (slick if is_blade else hit).append(prim.GetPath().GetParentPath().name)
    log(f"  endtip contact: {n} collider(s), contact/rest offset "
        f"{ARGS.tip_contact_offset*1000:.1f}/{ARGS.tip_rest_offset*1000:.1f} mm"
        + (f"; GRIP mu {ARGS.jaw_friction} [{', '.join(hit)}]" if hit else "")
        + (f"; SLICK mu {ARGS.blade_friction} [{', '.join(slick)}] "
           f"-- the blades push tissue but cannot hold it" if slick else ""))


def deinstance_endtip_colliders(stage) -> None:
    """Make the endtip links' mesh colliders editable prims.

    The URDF importer marks each link's `collisions` scope INSTANCEABLE. That
    leaves the colliders under it as instance proxies, which a plain
    Usd.PrimRange does not visit at all and which cannot carry an authored
    opinion even when reached -- so the runtime cannot switch one off. That
    switch is what the cutter's pass-through needs (teleop `_step_blade_phase`).

    It has to happen HERE, not at runtime: de-instancing changes composition,
    and doing that once physics has parsed the stage invalidates the
    articulation's tensor view outright ("Failed to get DOF positions from
    backend"). Done at build time it is just how the stage ships.

    Both arms are de-instanced, though only the cutter uses it -- two mesh
    prims per arm is nothing, and having the two instruments differ in
    structure is a trap for the next thing that walks them.
    """
    done = []
    for prefix in ("jaw", "blade"):
        for side in ("left", "right"):
            for prim in stage.Traverse():
                if prim.GetName() != "collisions":
                    continue
                if prim.GetParent().GetName() != f"{prefix}_{side}":
                    continue
                if prim.IsInstanceable():
                    prim.SetInstanceable(False)
                    done.append(f"{prefix}_{side}")
    if done:
        log(f"  endtip colliders de-instanced: {', '.join(done)} "
            f"(so collisionEnabled can be toggled live)")


def author_tool_metal_look(stage) -> None:
    """Turn the imported tool materials into real metals.

    The URDF importer lands each <material> as `/<root>/Looks/material_<name>`
    carrying a single `diffuse_color_constant` input. That is an unlit-looking
    rough dielectric: the "stainless steel" endtip comes out a flat pale grey
    that reads as white plastic, which is the complaint this answers. Metal is
    not a colour -- it is metallic=1 plus a low roughness, so the jaws take
    their brightness from the scope light they reflect.

    Which input names to write depends on the shader the importer produced:
    OmniPBR/MDL takes `metallic_constant`, UsdPreviewSurface takes `metallic`.
    Writing both is not free -- Hydra logs "Parameter 'metallic' ... not
    available in the MDL representation" for every stray input on every frame
    -- so the shader is asked which it is.
    """
    n = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdShade.Material):
            continue
        name = prim.GetName()
        if not name.startswith("material_"):
            continue
        spec = TOOL_PBR.get(name[len("material_"):])
        if spec is None:
            continue
        metal, rough, rgb = spec
        for kid in Usd.PrimRange(prim):
            shader = UsdShade.Shader(kid)
            if not shader:
                continue
            mdl = bool(shader.GetSourceAsset("mdl")) or \
                bool(shader.GetPrim().GetAttribute("info:mdl:sourceAsset"))
            inputs = ((("diffuse_color_constant", Sdf.ValueTypeNames.Color3f,
                        Gf.Vec3f(*rgb)),
                       ("metallic_constant", Sdf.ValueTypeNames.Float, metal),
                       ("reflection_roughness_constant",
                        Sdf.ValueTypeNames.Float, rough),
                       ("specular_level", Sdf.ValueTypeNames.Float, 0.5))
                      if mdl else
                      (("diffuseColor", Sdf.ValueTypeNames.Color3f,
                        Gf.Vec3f(*rgb)),
                       ("metallic", Sdf.ValueTypeNames.Float, metal),
                       ("roughness", Sdf.ValueTypeNames.Float, rough)))
            for key, tname, val in inputs:
                shader.CreateInput(key, tname).Set(val)
        n.append(f"{name[len('material_'):]}"
                 f"(metal {metal:.0f}, rough {rough:.2f})")
    log(f"  tool PBR: {len(n)} material(s) -> " + ", ".join(sorted(n)))


def author_jaw_drives(stage) -> None:
    """Angular position drives on the two jaw joints of the freshly imported
    left-arm stage, so plain GUI play holds the gripper closed. Teleop
    overrides these through the articulation controller. USD angular-drive
    units are torque per DEGREE."""
    n = 0
    for prim in stage.Traverse():
        if prim.GetName() not in ENDTIP_JOINTS:
            continue
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        drv = UsdPhysics.DriveAPI.Apply(prim, "angular")
        drv.CreateTypeAttr("force")
        drv.CreateStiffnessAttr(JAW_DRIVE_STIFFNESS)
        drv.CreateDampingAttr(JAW_DRIVE_DAMPING)
        drv.CreateMaxForceAttr(JAW_DRIVE_MAX_FORCE)
        drv.CreateTargetPositionAttr(0.0)          # closed
        n += 1
    log(f"  authored position drives on {n} endtip joint(s) "
        f"(k={JAW_DRIVE_STIFFNESS}/deg, closed target)")


def import_urdf_to_usd(local_urdf: Path, dest: Path, label: str = "FR3",
                       jaw_drives: bool = False) -> None:
    """Import a URDF and export a referenceable robot USD (defaultPrim set)."""
    status, cfg = omni.kit.commands.execute("URDFCreateImportConfig")
    cfg.merge_fixed_joints = False       # keep the gripper joints
    cfg.fix_base = True                  # arm is bolted to its stand
    cfg.import_inertia_tensor = True
    cfg.distance_scale = 1.0             # urdf is already in metres
    cfg.convex_decomp = False
    # optional knobs across importer versions -- set defensively
    for attr, val in (("self_collision", False),
                      ("create_physics_scene", False),
                      ("make_default_prim", True),
                      ("parse_mimic", True)):
        try:
            setattr(cfg, attr, val)
        except Exception:
            pass

    omni.usd.get_context().new_stage()
    status, prim_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=str(local_urdf),
        import_config=cfg,
        get_articulation_root=True,
    )
    log(f"  imported {label} -> {prim_path}  (status={status})")

    stage = omni.usd.get_context().get_stage()
    # The referenceable root is the top-level prim of whatever path came back.
    root = "/" + prim_path.strip("/").split("/")[0]
    root_prim = stage.GetPrimAtPath(root)
    stage.SetDefaultPrim(root_prim)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    if jaw_drives:
        author_jaw_drives(stage)
        author_tool_contact(stage)
        author_tool_metal_look(stage)
        deinstance_endtip_colliders(stage)

    dest.parent.mkdir(parents=True, exist_ok=True)
    stage.GetRootLayer().Export(str(dest))
    log(f"  wrote robot USD -> {dest}  (defaultPrim {root})")


def import_robot_usd(dest: Path) -> None:
    """Import the plain FR3 (with hand) to a referenceable robot USD."""
    import_urdf_to_usd(retarget_urdf(), dest, label="FR3")


# --------------------------------------------------------------------------
# assemble the cell
# --------------------------------------------------------------------------

def add_box(stage, path, center, size, color) -> None:
    """A visual box (no collision; the bases are fixed and the specimen is baked)."""
    cx, cy, cz = center
    hx, hy, hz = size[0] / 2.0, size[1] / 2.0, size[2] / 2.0
    corners = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
               (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
    pts = [Gf.Vec3f(cx + sx * hx, cy + sy * hy, cz + sz * hz) for sx, sy, sz in corners]
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
             (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr().Set(pts)
    m.CreateFaceVertexIndicesAttr().Set([i for f in faces for i in f])
    m.CreateFaceVertexCountsAttr().Set([4] * 6)
    m.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    m.CreateDoubleSidedAttr().Set(True)
    m.CreateExtentAttr().Set([Gf.Vec3f(cx - hx, cy - hy, cz - hz),
                              Gf.Vec3f(cx + hx, cy + hy, cz + hz)])
    m.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set([Gf.Vec3f(*color)])


def make_metal_material(stage, path, color, roughness=0.30):
    """A UsdPreviewSurface metal -- displayColor alone can't read as silver, so the
    table needs a real metallic/rough PBR surface for the RTX renderer to reflect."""
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(1.0)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    mat.CreateSurfaceOutput().ConnectToSource(
        shader.ConnectableAPI(), "surface")
    return mat


def add_table(stage, base_y: float) -> None:
    """The aluminium table (top = z=0), the 1 cm specimen plate, and a room floor.

    The table is centred in X on the arms and in Y between the base line and the
    specimen, so both arms and the phantom sit on the top.

    The Y-depth AUTO-grows with the stand-off (--table-depth 0): the 80 cm depth
    was sized for --port-offset 0.27; each cm the bases move back in -Y (a bigger
    offset) is added to the depth so the base back-margin stays constant, rounded
    to 5 cm (0.27->80, 0.46->100 cm). Without this the RCM-optimal 0.46 m offset
    slides the FR3 bases onto -- and past -- the old 80 cm table's back edge.
    """
    tx, _ty, tz = TABLE_SIZE
    if ARGS.table_depth > 0.0:
        ty = ARGS.table_depth
    else:
        ty = round((0.80 + max(0.0, ARGS.port_offset - 0.27)) / 0.05) * 0.05
    table_size = (tx, ty, tz)
    # midway between the base line and the specimen, then nudged toward the
    # phantom (+Y) by --table-shift
    table_cy = base_y / 2.0 + ARGS.table_shift
    add_box(stage, "/World/table", (0.0, table_cy, -tz / 2.0), table_size, ALU)
    alu_mat = make_metal_material(stage, "/World/materials/aluminium", ALU)
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath("/World/table")).Bind(alu_mat)
    add_box(stage, "/World/floor", (0.0, table_cy, -tz - 0.005),
            (6.0, 6.0, 0.01), FLOOR_COLOR)
    if ARGS.lift > 0:
        add_box(stage, "/World/specimenPlate",
                (0.0, 0.0, ARGS.lift / 2.0),
                (PLATE_SIZE[0], PLATE_SIZE[1], ARGS.lift),
                _parse_rgb(ARGS.plate_color, PLATE_COLOR, "--plate-color"))
    log(f"  table {tx*100:.0f}x{ty*100:.0f}x{tz*100:.0f} cm, top at z=0, "
        f"centred y={table_cy:+.3f}; plate {ARGS.lift*1000:.0f} mm")


def add_robot(stage, side: str, x: float, y: float, yaw_deg: float, robot_ref: str):
    # Placement lives on a parent Xform; the reference goes on a child, so our
    # translate/rotate never collide with the xformOps the referenced robot
    # brings in on its own root prim.
    group = f"/World/robots/fr3_{side}"
    xf = UsdGeom.Xform.Define(stage, group)
    xf.AddTranslateOp().Set(Gf.Vec3d(x, y, 0.0))
    xf.AddRotateZOp().Set(yaw_deg)
    child = stage.DefinePrim(f"{group}/robot")
    child.GetReferences().AddReference(robot_ref)
    log(f"  fr3_{side:5s} base at ({x:+.3f}, {y:+.3f}, 0.000)  yaw {yaw_deg:+.0f} deg")
    return group


ARM_START = {
    "left":  dict(urdf="out/fr3_left_grasper.urdf", port_sign=-1.0),
    "right": dict(urdf="out/fr3_right_cutter.urdf", port_sign=+1.0),
}


def start_pose_targets(stage):
    """Solve the bimanual start pose against THIS stage.

    Both instruments sit exactly `--start-insert` metres past their own trocar
    port, aimed so the tips land near the middle of the endoscope image and at
    least `--start-min-sep` apart. See isaac/start_pose.py for the aim, which is
    the only free variable -- the depth is a hard constraint.

    Returns {arm: dict(q, ...)} (radians, fr3_joint1..7) plus a summary dict.
    """
    import math
    import numpy as np
    sys.path.insert(0, str(ROOT / "isaac"))
    from laparo_ik import (Chain, JAW_PIN_AXIS_F, SHAFT_AXIS_F,
                           elbow_up_gradient, ik_step, perp_unit,
                           port_inward_normal, rcm_manipulability, rodrigues,
                           task_error)
    from start_pose import endoscope_camera, project, solve_start_tips

    def w_of(prim):
        M = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)
        return np.array([[M[i][j] for j in range(4)] for i in range(4)]).T

    cam = endoscope_camera(stage)
    tp = stage.GetPrimAtPath("/World/trocarPorts")
    if not (tp and tp.IsValid()):
        log("  ! no trocarPorts group -- start pose skipped")
        return {}, {}
    R_ports = w_of(tp)[:3, :3]

    ports, arm_of_port = {}, {}
    for arm, cfg in ARM_START.items():
        for i in (k + 1 for k in USED_PORT_IDX):     # 0-based -> port numbers
            prim = stage.GetPrimAtPath(f"/World/trocarPorts/port_{i:02d}")
            if prim and prim.IsValid() and w_of(prim)[0, 3] * cfg["port_sign"] > 0:
                ports[i] = (w_of(prim)[:3, 3], R_ports @ port_inward_normal(i))
                arm_of_port[i] = arm
    if len(ports) != 2:
        log(f"  ! found {len(ports)} used ports, need 2 -- start pose skipped")
        return {}, {}

    # organ surfaces (rest pose) so the tips do not start buried in tissue
    organ_pts = []
    org = stage.GetPrimAtPath("/World/organs")
    if org and org.IsValid():
        for kid in org.GetChildren():
            rp = kid.GetChild("render")
            pa = rp.GetAttribute("points").Get() if rp else None
            if not pa:
                continue
            M = w_of(rp)
            v = np.asarray([[p[0], p[1], p[2]] for p in pa], np.float64)
            organ_pts.append((M[:3, :3] @ v.T).T + M[:3, 3])
    organ_pts = np.vstack(organ_pts) if organ_pts else None

    if ARGS.start_aim == "approach":
        from start_pose import spawn_tips
        sol = spawn_tips(ports, arm_of_port, ARGS.start_insert)
    else:
        sol = solve_start_tips(cam, ports, insert=ARGS.start_insert,
                               min_sep=ARGS.start_min_sep,
                               max_tilt_deg=ARGS.start_max_tilt,
                               v_frac=ARGS.start_v_frac, organ_pts=organ_pts,
                               min_clear=ARGS.start_min_clear)
        sol.pop("_meta")

    # Seed. The two arms are mirror images about x = 0 (bases at -+spacing/2,
    # both yawed +90 deg), but they are the SAME right-handed chain, so an exact
    # mirror pose does not exist. Seeding the right arm with the mirrored left
    # solution still lands it in the mirror-LIKE local optimum instead of a
    # different one -- without it, both arms start from the same seed, the
    # posture level walks each to its own manipulability optimum, and the two
    # elbows end up visibly unalike.
    from start_pose import SEED, MIRROR

    ROLL_OFF = {"left": ARGS.start_roll_left, "right": ARGS.start_roll_right}

    def solve(arm, tip, p_rcm, q_seed):
        """Delegates to start_pose.solve_start_q -- the SAME call the teleop
        makes. These were separate copies once, and an elbow lift added here
        changed nothing in the live pose, because the teleop re-solves rather
        than trusting the stage."""
        from start_pose import solve_start_q
        chain = Chain(ROOT / ARM_START[arm]["urdf"])
        T_base = w_of(stage.GetPrimAtPath(f"/World/robots/fr3_{arm}"))
        q, info = solve_start_q(
            chain, T_base, tip, p_rcm, seed=q_seed,
            roll_deg=ROLL_OFF.get(arm, 0.0),
            elbow_lift=ARGS.start_elbow_lift,
            elbow_point=ARGS.start_elbow_point)
        return dict(info, q=q, chain=chain, T_base=T_base, rcm=p_rcm,
                    manip=float(rcm_manipulability(chain, q, T_base, p_rcm)))

    out = {}
    first = ARGS.start_seed_from if ARGS.start_seed_from != "none" else None
    order = sorted(sol, key=lambda k: arm_of_port[k] != first)
    q_prev = None
    for port_no in order:
        r = sol[port_no]
        arm = arm_of_port[port_no]
        seed = SEED if (first is None or arm == first or q_prev is None) \
            else q_prev * MIRROR
        res = solve(arm, r["tip"], ports[port_no][0], seed)
        if arm == first:
            q_prev = res["q"]
        u, v, _ = project(cam, res["tip"])
        out[arm] = dict(res, port=port_no, uv=(u, v), tilt=r["tilt_deg"])
    sep = float(np.linalg.norm(out["left"]["tip"] - out["right"]["tip"])) \
        if len(out) == 2 else float("nan")
    return out, dict(cam=cam, sep=sep)


def pose_arms_at_start(stage) -> None:
    """Bake the solved start pose into the stage: joint state + drive target on
    fr3_joint1..7 of both arms.

    Both are written -- the joint STATE is where PhysX starts the body, the
    DRIVE TARGET is where it is asked to stay. Setting only the state makes the
    arm snap back to its old target the instant play is pressed."""
    import numpy as np
    poses, info = start_pose_targets(stage)
    if not poses:
        return
    cam = info["cam"]
    for arm, r in sorted(poses.items()):
        grp = stage.GetPrimAtPath(f"/World/robots/fr3_{arm}")
        by_name = {p.GetName(): p for p in Usd.PrimRange(grp)
                   if p.IsA(UsdPhysics.RevoluteJoint)}
        n = 0
        for j in range(7):
            prim = by_name.get(f"fr3_joint{j+1}")
            if prim is None:
                continue
            deg = float(np.degrees(r["q"][j]))
            st = PhysxSchema.JointStateAPI.Apply(prim, "angular")
            st.CreatePositionAttr().Set(deg)
            st.CreateVelocityAttr().Set(0.0)
            drv = UsdPhysics.DriveAPI.Get(prim, "angular")
            if not drv:
                drv = UsdPhysics.DriveAPI.Apply(prim, "angular")
                drv.CreateTypeAttr("force")
            drv.CreateTargetPositionAttr().Set(deg)
            n += 1
        log(f"  fr3_{arm:5s} start pose: {n} joints, tip "
            f"{np.round(r['tip'], 4).tolist()} = {r['depth']*1000:.1f} mm past "
            f"port {r['port']}, scope image ({r['uv'][0]:.0f}, {r['uv'][1]:.0f}) px "
            f"of {cam['W']}x{cam['H']} [frame centre "
            f"({cam['W']//2}, {cam['H']//2})]")
        log(f"                  shaft {r['tilt']:.1f} deg off the port normal, "
            f"IK tip err {r['tip_err']*1000:.2f} mm / RCM err "
            f"{r['rcm_err']*1000:.2f} mm, manip {r['manip']:.1e}, "
            f"limit margin {np.degrees(r['margin']):.1f} deg")
        log(f"                  tool +y vs world +z {r['y_up_deg']:.1f} deg "
            f"(best the shaft allows: {r['y_best_deg']:.1f} deg), "
            f"q={np.round(r['q'], 3).tolist()}")
    log(f"  tip-to-tip separation {info['sep']*1000:.1f} mm "
        f"(requested >= {ARGS.start_min_sep*1000:.0f} mm)")


def add_camera(stage):
    cam = UsdGeom.Camera.Define(stage, "/World/cellCam")
    cam.CreateFocalLengthAttr().Set(24.0)
    cam.CreateClippingRangeAttr().Set(Gf.Vec2f(0.01, 1000.0))
    xf = UsdGeom.Xformable(cam.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(0.0, -1.9, 1.4))
    xf.AddRotateXYZOp().Set(Gf.Vec3f(58.0, 0.0, 0.0))  # look toward +Y, tilted down
    log("  cellCam at (0, -1.9, 1.4) looking +Y")

    # A low, head-on FRONT camera at roughly the height and distance the rig
    # photos are taken from, so the simulated start pose can be compared with a
    # photograph of the real cell directly instead of by impression.
    fc = UsdGeom.Camera.Define(stage, "/World/frontCam")
    fc.CreateFocalLengthAttr().Set(28.0)
    fc.CreateClippingRangeAttr().Set(Gf.Vec2f(0.01, 1000.0))
    xf = UsdGeom.Xformable(fc.GetPrim())
    # From BEYOND the trocar looking back at the robots -- the viewpoint the
    # rig photographs are taken from, with the phantom in the foreground and
    # both arms behind it. rotateXYZ (75, 0, 180): the X term aims the boresight
    # along -Y and 15 deg down, the Z term rolls the image upright (without it
    # the camera's up vector points into the floor).
    xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 1.9, 0.95))
    xf.AddRotateXYZOp().Set(Gf.Vec3f(75.0, 0.0, 180.0))
    log("  frontCam at (0, 1.9, 0.95) looking -Y (rig-photo viewpoint: from "
        "beyond the trocar, back toward the robots)")


def build_cell(stage) -> None:
    import numpy as np

    UsdGeom.Xform.Define(stage, "/World/robots")

    # 1) spin + lift the phantom/organs so the used ports face the arms and the
    #    specimen sits on its 1 cm plate.
    transform_specimen(stage)
    recolour_tissue(stage)
    stiffen_shell_contact(stage)
    dim_scene_lights(stage)
    if ARGS.organ_support == "plane":
        # keep the deformable scene's own infinite ground plane as the support
        # (lifted onto the organs by transform_specimen); hide its floating grid.
        # The lifted plane sits ~52 mm up, but the arms clear it -- their bases
        # are fixed (static, no plane contact) and the moving links stay above
        # it, verified by a full teleop run.
        gp = stage.GetPrimAtPath("/World/groundPlane")
        cm = stage.GetPrimAtPath("/World/groundPlane/CollisionMesh")
        if cm and cm.IsValid():
            UsdGeom.Imageable(cm).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
        log(f"  organ support = infinite ground plane kept "
            f"({'found' if gp and gp.IsValid() else 'MISSING'}), visual grid hidden")
    elif ARGS.organ_support == "slab":
        remove_ground_collision(stage)
        add_organ_support(stage)
    else:                                  # none
        remove_ground_collision(stage)
        log("  organ support = none (organs will fall away once they simulate)")

    # 1b) remove the contact forces that kept the resting organs squirming.
    #     (The pin attachments are authored in a later cook pass -- see
    #     cook_and_pin_organs in main().)
    calm_organs(stage)

    # 2) the field (abdomen around the organs) -- shown, still a rigid collider.
    if not ARGS.hide_field:
        show_field(stage)

    # 2a) trocar markers, riding the same spin+lift as the phantom.
    if not ARGS.no_trocars:
        add_trocar_ports(stage)
        if not ARGS.hide_phantom:
            show_phantom(stage)

    # 2b) overhead RealSense D405 in the phantom's top hole; open the hole in the
    #     (too thick) scanned shell so nothing occludes its view.
    if not ARGS.no_camera:
        add_realsense_405(stage)
        cut_phantom_view_hole(stage)

    # 2c) laparoscope tip light + 0-deg scope camera inside the phantom.
    if not ARGS.no_endoscope_light:
        add_endoscope_light(stage)
    add_endoscope_camera(stage)

    # 3) robots: the base line sits --port-offset (0.23 m, XY) in -Y from the
    #    midpoint of the two used ports (2,3). Bases stay symmetric about X=0.
    ports = port_positions_world()
    mid = ports[list(USED_PORT_IDX)].mean(axis=0)
    base_y = float(mid[1]) - ARGS.port_offset
    x = ARGS.spacing / 2.0
    robot_ref = "./" + Path(ARGS.robot_usd).name
    log(f"  used-ports midpoint y={mid[1]:+.3f} m -> base line y={base_y:+.3f} m "
        f"({ARGS.port_offset*100:.0f} cm apart in XY)")
    # both arms face the phantom (+Y). base +X -> world +Y is a +90 deg yaw.
    # both arms are hand-less: left carries the grasper, right the cutter.
    left_ref = "./" + Path(LEFT_ROBOT_USD).name
    right_ref = ("./" + Path(RIGHT_ROBOT_USD).name if not ARGS.plain_right_arm
                 else robot_ref)
    add_robot(stage, "left", -x, base_y, 90.0, left_ref)
    add_robot(stage, "right", +x, base_y, 90.0, right_ref)

    # 4) table + plate under everything.
    if not ARGS.no_table:
        add_table(stage, base_y)

    add_camera(stage)

    # 5) both arms into the surgical start pose: instruments 10 cm through
    #    their own port, tips near the middle of the endoscope image.
    if not ARGS.no_start_pose:
        pose_arms_at_start(stage)


def main() -> int:
    robot_usd = ROOT / ARGS.robot_usd
    if not (ARGS.reuse_robot and robot_usd.exists()):
        log("importing FR3:")
        import_robot_usd(robot_usd)
    else:
        log(f"reusing robot USD {robot_usd}")

    left_usd = ROOT / LEFT_ROBOT_USD
    if not (ARGS.reuse_robot and left_usd.exists()):
        log("importing left grasper (FR3 no-hand + motorised grasper):")
        import_urdf_to_usd(build_left_grasper_urdf(), left_usd,
                           label="left grasper", jaw_drives=True)
    else:
        log(f"reusing left grasper USD {left_usd}")

    right_usd = ROOT / RIGHT_ROBOT_USD
    if ARGS.plain_right_arm:
        log("right arm: plain FR3 with the stock hand (--plain-right-arm)")
    elif not (ARGS.reuse_robot and right_usd.exists()):
        log("importing right cutter (FR3 no-hand + motorised cutter):")
        import_urdf_to_usd(build_right_cutter_urdf(), right_usd,
                           label="right cutter", jaw_drives=True)
    else:
        log(f"reusing right cutter USD {right_usd}")

    scene = ROOT / ARGS.scene
    if not scene.exists():
        log(f"scene not found: {scene} -- run isaac/build_deformable_usd.py first")
        simulation_app.close()
        return 1

    log(f"opening scene {scene}")
    omni.usd.get_context().open_stage(str(scene))
    stage = omni.usd.get_context().get_stage()

    log("assembling cell:")
    build_cell(stage)
    force_gpu_dynamics(stage)

    out_path = ROOT / ARGS.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stage.GetRootLayer().Export(str(out_path))
    log(f"\nwrote {out_path}")

    if not ARGS.no_organ_pins:
        # The cook pass advances physics and writes state back into its stage,
        # so it runs on the already-exported copy in memory (sacrificial);
        # the clean file is then re-opened and ONLY the pins are added.
        nodes = cook_harvest(stage)
        from isaacsim.core.api import SimulationContext
        SimulationContext.clear_instance()
        omni.usd.get_context().open_stage(str(out_path))
        stage = omni.usd.get_context().get_stage()
        author_pins(stage, nodes)
        stage.GetRootLayer().Save()
        log(f"pins written into {out_path}")

    if ARGS.view:
        try:
            from omni.kit.viewport.utility import get_active_viewport
            get_active_viewport().set_active_camera("/World/cellCam")
        except Exception as exc:
            log(f"could not set viewport camera ({exc})")
        # Re-assert right before interaction: pressing play parses the physics
        # scene as it stands in THIS session, and a scene introduced or re-read
        # without the GPU flags kills every deformable organ.
        force_gpu_dynamics(stage)
        log("\nrunning -- close the window to quit")
        while simulation_app.is_running():
            simulation_app.update()
        simulation_app.close()
        return 0

    # headless: render one frame from cellCam to verify placement
    png = ROOT / ARGS.verify_png
    try:
        from omni.kit.viewport.utility import (capture_viewport_to_file,
                                               get_active_viewport)
        vp = get_active_viewport()
        vp.set_active_camera("/World/cellCam")
        for _ in range(60):
            simulation_app.update()
        capture_viewport_to_file(vp, str(png))
        for _ in range(30):
            simulation_app.update()
        log(f"wrote verification render -> {png}")
    except Exception as exc:
        log(f"could not capture verification render ({exc})")
    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
