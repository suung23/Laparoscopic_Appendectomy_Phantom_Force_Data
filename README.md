# Laparoscopic Appendectomy Phantom — Force Dataset

Six-axis force measurements taken on a silicone laparoscopic appendectomy phantom with
an instrumented laparoscopic grasper, together with the acquisition code, the driver
needed to read the DAQ on Linux, and the measurement protocol.

- **Author** — Suung23 <wjdtjd0203@snu.ac.kr>
- **Session** — 2026-10-07, specimen `A01`
- **Instrument** — laparoscopic grasper with 350 Ω foil strain gauges, CAS LCT-PRO
  amplifiers, NI USB-6009 at 1 kHz
- **Protocol** — [`docs/PHANTOM_SPEC_PROTOCOL.md`](docs/PHANTOM_SPEC_PROTOCOL.md)

> **Read [Limitations](#limitations) before using the numbers.** Forces are valid as
> ratios and rankings, not as absolute newtons, and the direction channel is not usable.

## What is measured

Eight linear gauges on the shaft form four half bridges (bending *x*/*y* at two axial
stations); a rosette gives shaft torsion and a gauge on the trigger lever gives handle
grasp force. The quantity used throughout is the tip bending magnitude

```
F_tip = hypot(bending_gripper_x, bending_gripper_y)
```

from the distal station only, so trocar reaction does not enter: the instrument is not
passed through a trocar for phantom characterisation.

There is **no axial (thrust) channel and no displacement channel**. Every load is
applied perpendicular to the shaft, and stiffness in N/mm cannot be derived.

## Results

Peak `F_tip` per trial, mean ± SD over n trials, specimen A01.

### Lift force vs height (§3.3)

| Height | n | Peak F_tip | Slope from previous |
|---|---|---|---|
| 5 mm | 5 | 1.77 ± 0.46 N | — |
| 10 mm | 5 | 2.79 ± 0.43 N | +0.205 N/mm |
| 15 mm | 5 | 2.84 ± 0.44 N | +0.010 N/mm |
| 20 mm | 5 | 2.15 ± 0.21 N | **−0.138 N/mm** |

The curve rises steeply to 10 mm, flattens by 15 mm, and **falls at 20 mm** while its
spread halves (SD 0.44 → 0.21). Two readings are consistent with that: the attachments
yielded between 15 and 20 mm, or the lifted geometry changed the direction of the load
so that less of it projects onto the bending plane. The data alone do not separate them.

### Directional resistance to an attempted move (§3.4)

| Direction | n | Peak F_tip | Relative |
|---|---|---|---|
| **Right** | 5 | **5.25 ± 0.81 N** | **2.87×** |
| Left | 5 | 2.22 ± 0.45 N | 1.21× |
| Posterior | 5 | 1.99 ± 0.37 N | 1.09× |
| Anterior | 5 | 1.83 ± 0.42 N | 1.00× |

The phantom is strongly anisotropic: moving to the right costs nearly three times what
the other three directions cost, which the three remaining directions barely separate.
This follows the build — the mesentery is attached on one side and the supports sit
underneath only.

**Displacement was not controlled.** Directions differ in how far the tissue can travel
before it stops, so each trial is "the force developed while attempting to move in that
direction", not "the force at a fixed displacement". The operator's effort is the only
controlled variable, so treat the ratios, not the absolute values, as the result.

### Grasp onset (§3.2)

| Channel | n | Peak |
|---|---|---|
| `F_tip` (lateral force from asymmetric grasp) | 10 | 2.03 ± 0.73 N |
| `gripping` (handle actuation force) | 10 | 4.95 ± 1.14 N |

### Slip threshold (§4)

| | n | Peak F_tip | drop_ratio |
|---|---|---|---|
| Anterior traction, 1 kHz | 5 | **2.33 ± 0.27 N** | 0.82 |

n = 5 matches the per-stage target used in this session; every stage was collected to
n = 5 (grasp has 10 because it was recorded before the target was standardised).

```
slip threshold / anterior move resistance = 2.33 / 1.83 = 1.28
```

**The grasp holds only 28 % more than the task already demands** in that direction, so
anterior retraction on this phantom runs close to the slipping limit.

Pull direction deviates from the protocol: §4.2 asks for vertical traction, but pulling
vertically to the slip point risked tearing the attachments, and this session was
non-destructive. Slip peaks are therefore paired with `move_F`, the same direction —
**not** with `lift_*`, which is vertical.

## Layout

```
code/            acquisition and analysis (start at code/phantom_experiment.py)
driver/          NI USB-6009 driver for Linux (NI-DAQmx Base in Docker, kernel 6.8 port)
docs/            measurement protocol, calibration report
data/            force HDF5 + per-trial CSV + per-frame video timestamps
data/video_clips/  one representative trial per stage, 960 px — tracked
data/video/      full-session mp4 — not tracked, see below
```

Only material belonging to this phantom dataset is included. Other work that shared the
same acquisition folder (ICRA 2027 force-logger drafts, CDE 2026 report and poster, the
in-vivo sessions, the trocar decomposition analysis) was deliberately left out.

### Force HDF5 — one row per sample

| Dataset | Shape | Meaning |
|---|---|---|
| `timestamp` | `()` f8 | Unix epoch, UTC seconds. The clock every stream shares |
| `forces` | `(6,)` f4 | Six axes after decoupling, tare and low-pass — what the GUI drew |
| `raw` | `(6,)` f4 | Channel volts, so the data can be recomputed with another matrix |
| `stage` | `()` i2 | Index into `attrs["stage_names"]` |
| `trial` | `()` i4 | `> 0` inside that trial; `< 0` the pad window just after trial `|n|` |

Only trial windows were recorded (`attrs["record_mode"] == "trials_only"`), plus a
0.5 s pad after each trial so a peak just before the stop key is not clipped. The one
exception is session `161659`, taken before that change, which also holds the gaps;
filter it with `trial > 0`.

### Per-trial CSV

One row per trial with peak / mean / p95 `F_tip`, the angle at peak, `dF/dt` over the
0.5 s before peak, the drop ratio, grasp and torsion peaks, and the acquisition
settings. `travel_mm` is optional and mostly empty — see the note on displacement above.

### Video

`data/*_cam.h5` holds `timestamp[i]`, the absolute time of frame *i*, on the same clock
as the force data; mp4 has no standard per-frame absolute time, so use this to align.
Frames were written only inside trial windows, so **the mp4 is a concatenation of
segments and playback time is not proportional to wall clock** — gaps appear as jumps in
`timestamp`.

Full-session mp4 files are 126–745 MB each, over GitHub's 100 MB per-file limit, so
`data/video/` is excluded from git and kept separately.

`data/video_clips/` holds what is tracked instead: **one representative trial per stage**
— the trial whose peak `F_tip` is closest to that stage's median — cut from the session
mp4 by frame index and re-encoded at 960 px (3.7 MB for all ten). File names carry the
stage and the trial number, so each clip maps back to a row of the per-trial CSV and to
a `trial` value in the force HDF5. Per-frame sidecar timestamps are tracked for every
session, clipped or not.

## Reproducing

```bash
conda activate surgitag

# terminal 1 — DAQ driver (Linux, Docker, privileged)
cd driver/linux_daqmxbase && ./run_driver.sh --device Dev1 --channels 8 --rate 1000

# terminal 2 — experiment GUI (starts the driver itself if it is not up)
cd code && ./run_phantom_session.sh --specimen A01
```

`run_phantom_session.sh` runs Phase 1 (100 Hz, block mean, 5 Hz low-pass) and then, on
confirmation, the slip phase (1 kHz, subsampled, no filter). The two phases cannot share
one process: the rate and decimation are fixed when the socket source is built.

## Limitations

1. **Absolute newtons are not established.** The decoupling matrix in `code/calib.json`
   was fitted on 2026-06-30 in a free-cantilever setup. A different load point changes
   the moment arm and scales every value by an unknown constant. Ratios, rankings and
   within-session comparisons hold; absolute specifications do not. Recalibration at the
   contact point (protocol appendix A-1) needs a push-type scale, which was unavailable.
2. **The direction channel is unusable.** θ clusters at 130–165° for all four move
   directions, which should differ by 90°. The `bending_gripper_y` half bridge was
   rewired on 2026-10-02 and the calibration was never refitted. Report magnitude
   ratios; do not plot the polar histogram the protocol describes.
3. **Move displacement is not controlled** (see §3.4 above).
4. **Slip was measured anteriorly, not vertically** (see §4 above).
5. **One specimen.** The supports are hand-built and the protocol rates their
   reproducibility as low, while they are what sets the lift and retraction reaction.
   These numbers describe specimen A01; at least three specimens are needed before they
   can be quoted as phantom specifications.
6. **n = 5 per stage**, below the n ≥ 10 that the protocol asks for grasp and slip.
7. **No MCU kinematics.** Grasp onset timing in §3.2 is defined on the hall-sensor
   `grip` stream, which was not connected; onset is approximated from the `gripping`
   trace instead.

## Data edits

Three trials were removed and one session relabelled, all with the reason recorded in
the HDF5 attributes (`removed_note`, `stage_relabel_note`) and the CSV `note` column.
Unmodified copies sit beside each edited file as `.orig`.

| Session | Edit |
|---|---|
| `163214` | Trials 1–4 relabelled to `grasp`, 5–9 to `lift_10`; the GUI stage selector had lagged the physical action |
| `164339` | Trial 14 removed — grasp lost mid-trial |
| `165306` | Trial 14 removed — shortest run, drop ratio not computable |
| `171401` | Trial 1 removed — practice run |
| `170625` | Kept but **excluded from analysis**: recorded at Phase 1 settings, which smooth away the slip peak |

## License and citation

To be set by Rosota Intelligence Institute before release.
