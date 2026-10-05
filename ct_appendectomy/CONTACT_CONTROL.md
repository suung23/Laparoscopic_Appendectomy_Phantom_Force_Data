# Dynamic contact control

The CT articulation path now uses a measured-reaction supervisor and a
torque-limited dynamic gripper. No tissue attachment is enabled and no jaw
pose is teleported during control. The original Rosota IK/controller remains
unchanged underneath the supervisor; the fixed-port aperture checks still run
at every physics tick.

## Signals and actuation

- PhysX tool-mount reaction is read by joint-name metadata, including fixed
  joints (`joint_index + 1`). The channel fails on unavailable/nonfinite data.
- A 64-tick static tare follows settling. The force residual is filtered at
  480 Hz with alpha 0.2. A latched arm hold triggers above the larger of 0.15 N
  and eight times the maximum calibration residual. Reset clears the latch.
- This is a **locally tared reaction residual**, not a calibrated external
  tissue wrench: gravity orientation changes, acceleration and jaw motion
  remain confounders. Do not use it as ground-truth force for an RL reward.
- Each jaw's PhysX drive is capped at **0.00045 N.m**. Its measured effort is
  reported separately. Effort divided by a nominal 15 mm lever is explicitly
  a load proxy, not a measured tissue force.
- Closure is limited to 30 included-angle degrees/s; opening to 90 degrees/s.
  The target stays within one degree of the measured included angle under
  resistance. A measured effort of **0.0003 N.m** latches closure inhibition
  and opens the included-angle target by 0.25 degrees to unload. The latch
  prevents repeated re-closing when reaction falls; an explicit opening
  request clears it. Opening remains available after an arm hold.
- Translation requests are limited to 3.6 mm/s in this commissioning profile.
- RL observations include measured jaw positions and efforts, six-axis mount
  residuals, per-arm stop flags and separate jaw-contact hold flags. Both arms
  have independent channels, so a policy can explicitly reopen a blocked jaw.

These are simulator engineering settings, not clinical thresholds. A latched
position hold is not full Cartesian impedance/admittance control. Calibrated
external wrench estimation over orientation, contact-force tracking, tissue
damage laws and successful sustained bimanual grasp remain separate gates.

## Reproduction

Use the same Isaac Python and FR3 URDF as `run-rosota-ct.ps1`. Run
`ct_appendectomy/run_rosota_ct.py` with `--contact-probe --steps 360 --headless`
and a fresh `--out` directory. The script requests 150 samples of slow axial
insertion, 150 samples of closure and 60 samples of reopening. A contact stop
may shorten actual insertion. This is a scripted commissioning experiment,
not a learned policy or a successful appendectomy demonstration.

Add `--surface-approach` to target the nearest live tissue nodes plus 2 mm of
requested overtravel. This geometric probing policy uses privileged simulator
state; it is not image-based autonomous grasping. The controller may stop
before reaching the requested target. `--contact-travel-mm 0` without
`--surface-approach` provides a free-space open/close control.

## Rejected setting

The first surface-targeted experiment used a 0.003 N.m drive cap. It produced
a tet inversion during left jaw closure at control sample 226. The mount
reaction residual stayed below 0.046 N, so a mount-force stop alone did not
protect the tissue. The left jaw effort crossed 0.0003 N.m at 7.4146 s while
the minimum signed volume ratio was still 0.5999; this motivated the direct
jaw-effort latch and lower drive cap. The failed report is retained locally
at `out/ct_appendectomy/contact_surface_v1/report.json`.

With the lower cap and jaw latch, the 300-sample repeat completed 4,800
physics checks with zero inversions. Left closure latched at 7.4021 s and
the jaw reopened to 73.995 degrees. The minimum signed volume ratio across
the run was 0.4721. The right jaw had only reached 15.45 degrees when the
three-second close window ended, so that run does not establish right contact.
The probe now allows five seconds of closure (150 samples); `--close-samples`
can override it. These are engineering commissioning results, not a validated
clinical grasp or a claim of force-tracking accuracy.

The extended 360-sample repeat passed 5,760 physics checks: zero inversions,
minimum signed volume ratio 0.5939, left jaw hold at 7.3875 s, and both jaws
reopened to 73.995 degrees. The right jaw reached zero opening without a
contact latch; that contact site remains unvalidated.

The 210-sample free-space negative control (`--approach-samples 0
--contact-travel-mm 0 --steps 210`) passed another 3,360 physics checks.
Both jaws fully closed and reopened with zero false latches. Left peak jaw
effort was 0.000000635 N.m in free space versus 0.000315 N.m at the surface.
Thus the tested left closure stop is distinguishable from insufficient motor
torque in free space. Machine-readable results, failure evidence and hashes
are in `validation/ct-contact-feedback.json`. The recorded video is a scripted
engineering probe; no learned policy or sustained grasp is claimed.

Free-space commissioning uses the same command without `--contact-probe`.
Inspect report status and every-tick `physics_trace.json`, not just process
exit code. Tet inversions or trocar violations invalidate the run.

## Hash-verified force/deformation event review

`analyze_contact_trial.py` checks each local report/trace hash against
`validation/ct-contact-feedback.json` before deriving tick-aligned summaries.
The six-trial summary is `validation/contact-force-deformation-event-analysis.json`.
The failed 0.003 N.m contact trial first crossed the 0.0003 N.m jaw-effort
threshold at trace time 7.4146 s, 51 physics ticks (0.1063 s) before the first
inverted tetrahedron at 7.5208 s. At threshold crossing, the minimum signed
volume ratio was 0.5999; the final observed minimum reached -0.1984. The peak
locally tared mount residual was 0.0455 N, below its 0.15 N arm-stop threshold.
This is why the jaw-effort latch was added; it does not prove that the chosen
threshold protects biological tissue.

In the capped/latching surface trials v2 and v3, the left jaw limit occurred at
7.4021 s and 7.3875 s, with zero inverted tetrahedra. Their minimum signed
volume ratios still fell to 0.4721 and 0.5939, so absence of inversion is only
a mesh-integrity observation, not a tissue-safety pass. The free-space control
peaked at 6.35e-7 N.m and produced no jaw trip. Neither surface trial validates
right-hand tissue contact. Reproduce the hash-checked analysis with
`python ct_appendectomy/analyze_contact_trial.py --out
validation/contact-force-deformation-event-analysis.json` using the local
asset Python environment. Joint effort and tared mount reaction remain
simulator signals, not measured external tissue force.
