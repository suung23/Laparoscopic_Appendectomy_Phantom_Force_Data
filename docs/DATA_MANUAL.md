# Data manual — Laparoscopic appendectomy phantom force dataset

Complete reference for this dataset: what is in every file, what every field means, which
channels to use, what the numbers say, and what will break a naive loader. Self-contained
— you should not need another document to work with the data.

Verified against the files in `data/`: **7 force sessions, 60 trial rows, 96,740 force
samples**, specimen `A01`, session 2026-10-07.

---

## 0. The headline, first

> ### The production channel set is `bending_trocar_x` + `bending_trocar_y`
>
> **Those two strain gauges only.** The hardware team is developing the instrumented
> instrument around that pair deliberately: fewer gauges on the shaft means less to
> route, less to sterilise, and the **smallest possible disturbance to the operating
> field**. Everything that ships reads those two channels.
>
> **But every result in this dataset's README is computed on the *gripper* pair.** The
> two are not interchangeable — §7.2 recomputes all of it on the production pair and
> shows what survives the move and what does not. Read that before reusing any number.

Second thing to know: **forces are valid as ratios and rankings, not as absolute
newtons**, and **the direction channel is unusable on both pairs**. Details in §8.

---

## 1. What was measured

A six-axis force measurement on a silicone laparoscopic appendectomy phantom, taken with
an instrumented laparoscopic grasper, plus synchronised endoscope-style video.

| | |
|---|---|
| **Specimen** | `A01` — one hand-built silicone phantom (appendix, cecum, mesoappendix, Sil-Poxy bonds) |
| **Session** | 2026-10-07, 16:16–17:30 KST (07:17–08:15 UTC) |
| **Instrument** | laparoscopic grasper, 8 × 350 Ω foil strain gauges |
| **Amplifier** | CAS LCT-PRO |
| **DAQ** | NI USB-6009, 8 channels |
| **Sampling** | 1 kHz hardware; stored at 100 Hz (block mean) or 1 kHz (subsampled) per session |
| **Video** | 1920×1080 h264, measured 29.75–29.82 fps, 0 dropped frames |
| **Clock** | one shared clock — unix epoch seconds, UTC — across force, video and trial records |

Eight linear gauges form **four half bridges** (bending *x* and *y* at two axial
stations: distal "gripper" and proximal "trocar"); a rosette gives shaft **torsion**, and
a gauge on the trigger lever gives handle **grasp** force.

**The instrument is not passed through a trocar in these sessions.** That is a deliberate
choice for phantom characterisation — it keeps trocar friction and reaction out of the
measurement. It is also the single biggest caveat on the production channels (§7.3).

What the session covers, mapped to the measurement protocol:

| Protocol § | Stage(s) | Question |
|---|---|---|
| §3.2 | `grasp` | what force does closing on the appendix take? |
| §3.3 | `lift_05` `lift_10` `lift_15` `lift_20` | how does lift force grow with lift height (mm)? |
| §3.4 | `move_L` `move_R` `move_F` `move_B` | is the phantom anisotropic — does direction matter? |
| §4 | `slip_dry` | at what traction force does the grasp slip? |

Stages defined in the recorder but **never used** in this session: `idle`, `slip_gel`,
`rupture`.

---

## 2. File inventory

```
data/
  phantom_A01_20261007_<session>.h5             force, one row per sample
  phantom_A01_20261007_<session>_cam.h5         per-frame absolute timestamps
  phantom_A01_20261007_<session>_trials.csv     one row per trial
  phantom_A01_20261007_<session>.h5.orig        pre-edit backup (4 sessions)
  phantom_A01_20261007_<session>_trials.csv.orig
  calib_snapshots_20260630_2108*.csv            calibration raw snapshots (160 total)
  video_clips/<stage>_trial<n>.mp4              one representative trial per stage, 960 px
  video/phantom_A01_20261007_<session>_cam.mp4  full session — NOT in git (126–745 MB each)
code/           acquisition + analysis; wiring.py and calib.json are authoritative
driver/         NI USB-6009 driver for Linux (NI-DAQmx Base in Docker, kernel 6.8 port)
docs/           this manual, the protocol, the calibration report, the overview figure
```

`.orig` backups exist for `163214`, `164339`, `165306`, `171401` — the four sessions with
recorded edits (§9). Diff against them to audit any edit.

---

## 3. Session map

The key lookup: which file holds which stage. `trial` numbers are **per session** and
restart at 1.

| Session | Rate | Stages (trial numbers) | Samples | Notes |
|---|---|---|---|---|
| `161659` | 100 Hz | `grasp` t1–6 (6) | 20,020 | **`all` mode** — holds the gaps between trials. Filter `trial != 0` |
| `163214` | 100 Hz | `grasp` t1–4 (4), `lift_10` t5–9 (5) | 13,250 | stages **relabelled** — `attrs["stage_relabel_note"]` |
| `164339` | 100 Hz | `lift_05` t1–5, `lift_15` t6–10, `move_L` t11–16 (5) | 17,400 | trial 14 removed |
| `165306` | 100 Hz | `move_R` t1–5, `move_F` t6–10, `move_B` t11–16 (5) | 9,340 | trial 14 removed |
| `170625` | 100 Hz | `slip_dry` t1–5 (5) | 5,670 | **excluded from analysis** — Phase-1 filtering smooths the slip peak away |
| `171030` | 100 Hz | `lift_20` t1–5 (5) | 6,060 | |
| `171401` | **1000 Hz** | `slip_dry` t2–6 (5) | 25,000 | the slip session that counts. `cutoff_hz = 0` (unfiltered), `decimate = subsample`, `traction_direction = anterior_horizontal` |
| `173014` | — | — | — | **`_trials.csv` header only, no `.h5`** — aborted session. Skip it |

- **`grasp` and `slip_dry` each span two sessions.** Every other stage sits in exactly
  one. For `slip_dry`, only `171401` is analysed.
- **`move_L` and `move_B` show t11–16 but hold 5 trials** — trial 14 was removed from
  each. Trial numbers are not contiguous; never assume `max(trial) == n`.

---

## 4. Force HDF5

### 4.1 Datasets — one row per sample

| Dataset | Shape | dtype | Meaning |
|---|---|---|---|
| `timestamp` | `(N,)` | f8 | unix epoch seconds, UTC. The clock every stream shares |
| `forces` | `(N, 6)` | f4 | six axes after decoupling → tare → low-pass. What the GUI drew |
| `raw` | `(N, 6)` | f4 | channel volts, so the data can be recomputed with another matrix |
| `stage` | `(N,)` | i2 | index into `attrs["stage_names"]` |
| `trial` | `(N,)` | i4 | see the sign convention below |

```python
import h5py, numpy as np

with h5py.File("data/phantom_A01_20261007_171401.h5", "r") as h:
    t      = h["timestamp"][:]
    forces = h["forces"][:]
    raw    = h["raw"][:]
    stage  = h["stage"][:]
    trial  = h["trial"][:]
    names  = list(h.attrs["stage_names"])     # stage[i] -> names[stage[i]]
    keys   = list(h.attrs["force_keys"])      # column order of forces/raw
```

**The `trial` sign convention:**

| value | meaning |
|---|---|
| `> 0` | inside that trial |
| `< 0` | the 0.5 s **pad window after** trial `abs(trial)`, kept so a peak just before the stop key is not clipped |
| `== 0` | outside any trial — appears **only** when `record_mode == "all"` (session `161659`) |

So `trial == k` selects trial *k*'s window (what the CSV metrics were computed on), and
`np.abs(trial) == k` adds its pad. Only `161659` was recorded continuously; the rest used
`record_mode == "trials_only"`, so the file holds trial windows plus pads and nothing else
(`attrs["samples_skipped_between_trials"]` records how much was dropped).

### 4.2 Attributes

| Attribute | Meaning |
|---|---|
| `specimen` | `A01` |
| `rate`, `actual_rate` | configured / achieved sample rate (Hz) |
| `cutoff_hz` | low-pass corner. `0.0` = unfiltered (`171401` only) |
| `decimate` | `average` (block mean, 100 Hz sessions) or `subsample` (`171401`) |
| `tare_enabled` | `True` throughout — zero taken before each trial block |
| `calib_mode`, `calib_path` | `decoupled`, and the `calib.json` used |
| `force_keys`, `raw_keys` | column order of `forces` / `raw` (§5) |
| `force_units` | `['N','N','N','N','N','N·m']` — **stored mojibake** for the torsion entry; trust §5, not this |
| `stage_names` | the 13-stage vocabulary; `stage` indexes into it |
| `class_keys` | failure-site vocabulary for the CSV `label` column — **never used** (§6) |
| `num_samples` | row count |
| `start_unix`, `end_unix`, `start_iso_utc` | session bounds |
| `record_mode`, `pad_sec` | `trials_only` + `0.5`. **Absent in `161659`** |
| `samples_skipped_between_trials` | how many samples `trials_only` discarded |
| `trials_<stage>` | per-stage trial count in this session |
| `trials_csv`, `video_file`, `video_h5` | the sibling files |
| `protocol` | `PHANTOM_SPEC_PROTOCOL.md Phase 1-3` |
| `protocol_deviation` | free text — present on `171401` (traction direction) |
| `traction_direction` | `anterior_horizontal` on `171401` |
| `removed_trials`, `removed_note` | which trials were deleted and why (§9) |
| `stage_relabeled`, `stage_relabel_map`, `stage_original_counts`, `stage_relabel_note` | the `163214` relabel, both versions kept (§9) |
| `values_note`, `trial_label_note` | inline restatement of the conventions above |

---

## 5. The six channels

`attrs["force_keys"]` order, fixed by [`../code/wiring.py`](../code/wiring.py):

| i | `force_keys` | `raw_keys` | AI | Bridge | Unit | Use it? |
|---|---|---|---|---|---|---|
| 0 | `bending_gripper_x` | `bgx` | A0 | half | N | reference — the README's `F_tip` |
| 1 | `bending_trocar_x` | `btx` | A1 | half | N | **PRODUCTION (§0, §7.2)** |
| 2 | `bending_gripper_y` | `bgy` | A4 | half | N | reference — the README's `F_tip` |
| 3 | `bending_trocar_y` | `bty` | A5 | half | N | **PRODUCTION (§0, §7.2)** |
| 4 | `gripping` | `grip` | A2 | quarter | N | yes — handle actuation force |
| 5 | `torsion` | `tor` | A6 | quarter | **N·m** | with care — quarter bridge, low SNR |

### 5.1 Bending magnitude — two stations, two different quantities

```python
gx, gy = keys.index("bending_gripper_x"), keys.index("bending_gripper_y")
tx, ty = keys.index("bending_trocar_x"),  keys.index("bending_trocar_y")

F_tip  = np.hypot(forces[:, gx], forces[:, gy])   # distal  — what the README reports
F_troc = np.hypot(forces[:, tx], forces[:, ty])   # proximal — the production pair
```

`F_tip` is the distal station alone, chosen so trocar reaction cannot enter. `F_troc` is
the same formula on the proximal pair. **They are not interchangeable** — §7.2.

### 5.2 Two warnings that change what you can compute

1. **Direction is invalid on *both* pairs — magnitude only.** The README documents this
   for `bending_gripper_y`: its half bridge was rewired 2026-10-02 and the calibration was
   never refitted, so θ clusters at 130–165° for all four move directions when they should
   differ by 90°. Recomputing θ on the **trocar** pair does not rescue it — the four
   directions span only **29°** there (vs 49° on the gripper pair), with *tight* per-stage
   SD: precise, and systematically wrong. `hypot(x, y)` is usable on either pair;
   `atan2(y, x)` — and `theta_at_peak_deg` in the CSV — is not. Do not plot the polar
   histogram the protocol describes.

2. **`torsion` is the only non-newton channel** (N·m). `force_units` records this but the
   stored string is mojibake.

**No axial/thrust channel and no displacement channel exist.** Every load is perpendicular
to the shaft, so stiffness in N/mm cannot be derived from this data at all.

### 5.3 Recomputing forces from raw

`forces` is `raw` after decoupling → tare → low-pass. Because `raw` (channel volts) is
stored, the whole chain can be redone with a different matrix:

```python
import json
c = json.load(open("code/calib.json"))
W = np.array(c["W"]); b = np.array(c["b"])   # note: "clean = W.T @ signals + b"
clean = raw @ W + b                          # row-vector form of the same thing
```

`W` is **block-structured, not dense** — `wiring.py` `ALLOWED_SIGNALS` encodes the
physics: `gripping` reads only A2 and nothing else reads A2; gripper bending reads only
the gripper gauges; trocar bending only the trocar gauges; no gripper↔trocar coupling;
`torsion` reads everything except `grip`. Fitted from the 160 snapshots in
`data/calib_snapshots_*.csv` on 2026-06-30.

---

## 6. Per-trial CSV — every column

One row per trial. 24 columns (23 in the four earliest sessions — `travel_mm` was added
later; see §9.4). All metrics are computed on **`F_tip` (the gripper pair)** over the
`trial == k` window, in [`../code/phantom_experiment.py`](../code/phantom_experiment.py).

| Column | Type | Meaning |
|---|---|---|
| `trial` | int | trial number **within this session**, restarts at 1. Not contiguous where trials were removed |
| `specimen` | str | `A01` |
| `stage` | str | one of `attrs["stage_names"]` — the manoeuvre being measured |
| `stage_section` | str | protocol section: `§3.2` grasp, `§3.3` lift, `§3.4` move, `§4` slip |
| `label` | enum | post-hoc **failure-site** class. `unset` in all 60 rows (§6.1) |
| `start_iso` | ISO 8601 | trial start, UTC, human-readable |
| `start_unix` | f8 | trial start on the shared clock |
| `end_unix` | f8 | trial end (excludes the 0.5 s pad) |
| `duration_s` | f4 | `end_unix - start_unix` |
| `n_samples` | int | force samples in the trial window |
| `peak_F_tip` | N | `max(F_tip)` — the headline metric for every stage |
| `peak_unix` | f8 | timestamp at the peak. **Use this to find the video frame** (§7.1) |
| `theta_at_peak_deg` | deg | `atan2(F_y, F_x)` at the peak — **INVALID, do not use** (§5.2) |
| `mean_F_tip` | N | `mean(F_tip)` over the window |
| `p95_F_tip` | N | 95th percentile of `F_tip` — peak-like but robust to a single spike |
| `dFdt_median_pre05s` | N/s | median `dF/dt` over the 0.5 s **ending at the peak**. A proxy for traction *speed*, since speed was not otherwise controlled. Blank if fewer than 3 samples in that window |
| `drop_ratio` | 0–1 | `(peak - min(F_tip after peak)) / peak`. **≈1 = the grasp was fully lost** (slip or rupture); small = load was held. This is how a slip trial is identified |
| `gripping_peak` | N | `max(abs(grip))` — **absolute** value |
| `gripping_mean` | N | `mean(grip)` — **signed**, not absolute. Negative values are normal and are not errors |
| `torsion_peak` | N·m | `max(abs(tor))` — absolute value |
| `calib_path` | str | `calib.json` — the matrix in force at recording time |
| `rate` | Hz | 100 or 1000 — **matters**: §4 slip analysis is only valid at 1000 |
| `cutoff_hz` | Hz | low-pass corner, `0` = none |
| `travel_mm` | f4 | manual displacement entry. **Present in 4 sessions' headers, empty everywhere** (§8, limitation 3) |
| `note` | str | free text; carries the reason when a trial was edited or removed |

### 6.1 The `label` column and `class_keys`

`attrs["class_keys"]` defines a failure-site vocabulary from protocol §5.2:

| key | meaning |
|---|---|
| `unset` | unclassified |
| `bond_app_cec` | ① Sil-Poxy appendix–cecum bond failed |
| `appendix_base` | ② appendix base ruptured |
| `meso` | ③ mesoappendix system failed |
| `slip` | slipped / aborted mid-manoeuvre |

**It was never used — `label` is `unset` in all 60 trial rows.** This session was
non-destructive, so there was no failure site to classify. Do not build on this field;
use `drop_ratio` to detect loss of grasp instead.

---

## 7. Video, and joining the streams

All streams share one clock (unix epoch seconds, UTC).

### 7.1 Camera HDF5 and the full-session mp4

`*_cam.h5` holds exactly one dataset: `timestamp[i]` = absolute time of frame *i* in the
mp4. **mp4 carries no per-frame absolute time, so this is the only correct way to align.**

```python
with h5py.File(f"data/phantom_A01_20261007_{sess}_cam.h5", "r") as c:
    ft = c["timestamp"][:]
frame_at_peak = int(np.searchsorted(ft, float(row["peak_unix"])))
```

Attributes: `num_frames`, `measured_fps`, `frame_width`/`frame_height` (1920×1080),
`codec`, `dropped_frames` (0 in every session), `wb_applied` + `wb_gain_bgr` (white
balance already applied), `start_unix`/`end_unix`.

**Frames were written only inside trial windows.** The mp4 is a concatenation of segments:
playback time is *not* proportional to wall clock, and gaps appear as jumps in `ft`.
**Never convert a timestamp to a frame index by multiplying by fps.**

Full-session mp4s (126–745 MB) exceed GitHub's 100 MB limit and are kept outside git in
`data/video/`. The per-frame sidecar timestamps are tracked for every session regardless.

### 7.2 Force h5 ↔ trials.csv

Join on `trial`, within the same session:

```python
import csv
rows = list(csv.DictReader(open(f"data/phantom_A01_20261007_{sess}_trials.csv")))
for r in rows:
    m = trial == int(r["trial"])
    seg = forces[m]
    assert abs(t[m][0]  - float(r["start_unix"])) < 0.1
    assert abs(t[m][-1] - float(r["end_unix"]))   < 0.1   # trial==k excludes the pad
```

Verified: max start delta 0.063 s, max end delta 0.069 s across `171401`.

### 7.3 Clips ↔ trials

`data/video_clips/` holds one clip per stage: the trial whose peak `F_tip` is closest to
that stage's median (rule implemented in
[`../code/make_trial_overview.py`](../code/make_trial_overview.py)). Filenames carry stage
and trial number but **not the session**, which is ambiguous for the two stages that span
two sessions. Recomputing the rule resolves nine of ten:

| clip | session | clip | session |
|---|---|---|---|
| `lift_05_trial2` | `164339` | `move_L_trial11` | `164339` |
| `lift_10_trial6` | `163214` | `move_R_trial3` | `165306` |
| `lift_15_trial9` | `164339` | `move_F_trial7` | `165306` |
| `lift_20_trial5` | `171030` | `move_B_trial11` | `165306` |
| `slip_dry_trial3` | `171401` — median *within the analysed session*, not pooled | `grasp_trial2` | **ambiguous** — trial 2 exists in both `161659` and `163214`; pooled median points to `161659` |

---

## 8. Results

### 8.1 As published — on the reference (gripper) pair

Peak `F_tip` per trial, mean ± SD over n trials, specimen A01.

**Lift force vs height (§3.3)**

| Height | n | Peak `F_tip` | Slope from previous |
|---|---|---|---|
| 5 mm | 5 | 1.77 ± 0.46 N | — |
| 10 mm | 5 | 2.79 ± 0.43 N | +0.205 N/mm |
| 15 mm | 5 | 2.84 ± 0.44 N | +0.010 N/mm |
| 20 mm | 5 | 2.15 ± 0.21 N | **−0.138 N/mm** |

Rises steeply to 10 mm, flattens by 15 mm, **falls at 20 mm** while its spread halves
(SD 0.44 → 0.21). Two readings fit: the attachments yielded between 15 and 20 mm, or the
lifted geometry rotated the load out of the bending plane. The data do not separate them.

**Directional resistance (§3.4)**

| Direction | n | Peak `F_tip` | Relative |
|---|---|---|---|
| **Right** | 5 | **5.25 ± 0.81 N** | **2.87×** |
| Left | 5 | 2.22 ± 0.45 N | 1.21× |
| Posterior | 5 | 1.99 ± 0.37 N | 1.09× |
| Anterior | 5 | 1.83 ± 0.42 N | 1.00× |

Strongly anisotropic — moving right costs nearly three times what the other three cost,
and those three barely separate. This follows the build: the mesentery is attached on one
side and the supports sit underneath only.

**Grasp onset (§3.2)**

| Channel | n | Peak |
|---|---|---|
| `F_tip` (lateral force from asymmetric grasp) | 10 | 2.03 ± 0.73 N |
| `gripping` (handle actuation force) | 10 | 4.95 ± 1.14 N |

**Slip threshold (§4)** — 1 kHz, anterior traction, session `171401`

| | n | Peak `F_tip` | `drop_ratio` |
|---|---|---|---|
| Anterior traction | 5 | **2.33 ± 0.27 N** | 0.82 |

```
slip threshold / anterior move resistance = 2.33 / 1.83 = 1.28
```

**The grasp holds only 28 % more than the task already demands** in that direction, so
anterior retraction on this phantom runs close to the slipping limit.

### 8.2 On the production (trocar) pair — what survives

Same trials, same windows, `F_troc` instead of `F_tip` (`170625` excluded):

| Stage | n | Reference (`F_tip`) | **Production (`F_troc`)** | ratio |
|---|---|---|---|---|
| `grasp` | 10 | 2.03 ± 0.73 N | 1.90 ± 0.32 N | 0.94 |
| `lift_05` | 5 | 1.77 ± 0.46 N | 1.85 ± 0.31 N | 1.05 |
| `lift_10` | 5 | 2.79 ± 0.43 N | 2.18 ± 0.37 N | 0.78 |
| `lift_15` | 5 | 2.84 ± 0.44 N | 2.96 ± 0.28 N | 1.04 |
| `lift_20` | 5 | 2.15 ± 0.21 N | 2.19 ± 0.50 N | 1.02 |
| `move_L` | 5 | 2.22 ± 0.45 N | 2.20 ± 0.43 N | 0.99 |
| `move_R` | 5 | **5.25 ± 0.81 N** | **3.78 ± 0.50 N** | **0.72** |
| `move_F` | 5 | 1.83 ± 0.42 N | 2.64 ± 0.12 N | 1.44 |
| `move_B` | 5 | 1.99 ± 0.37 N | 2.25 ± 0.09 N | 1.13 |
| `slip_dry` | 5 | 2.33 ± 0.27 N | 2.38 ± 0.18 N | 1.02 |

**The pair is live and usable.** Magnitudes land within 0.72–1.44× of the reference pair
rather than near zero, so the proximal station is genuinely loaded. Baseline noise is
comparable (0.016 N vs 0.012 N over the first 0.2 s of each window).

**It is more repeatable.** Per-stage coefficient of variation is lower on the production
pair in 7 of 10 stages, and sharply so where the reference pair is noisiest:

| | `move_F` | `move_B` | `grasp` | `lift_15` | `slip_dry` | `lift_20` |
|---|---|---|---|---|---|---|
| gripper CV | 0.231 | 0.187 | 0.359 | 0.154 | 0.116 | **0.100** |
| trocar CV | **0.045** | **0.042** | **0.168** | **0.094** | **0.077** | 0.227 |

Plausibly because the proximal station integrates over the whole shaft and cares less
about exactly where on the jaw the tissue sits. `lift_20` is the one clear reversal.

**But the headline anisotropy does not survive.** Right-vs-anterior attenuates from
**2.87× to 1.43×**. The *ranking* holds — right is still the most expensive direction —
but the effect size roughly halves. **Any threshold or safety limit derived from 2.87×
must be re-derived on `F_troc`.**

### 8.3 Before the production channels can be trusted in the OR

1. **Re-measure with the instrument actually through a trocar.** It is not, here — by
   design. In the OR these same gauges will see trocar friction and reaction *added* to
   tissue load, and that term is precisely what the distal station was chosen to avoid.
   Nothing in this dataset characterises it.
2. **Refit the calibration for the trocar pair at the contact point.** The 2026-06-30 fit
   is free-cantilever; a trocar-only instrument changes the moment arm that matters.
3. **Fix the direction channel if direction is needed.** Both pairs are magnitude-only
   today (§5.2). A trocar-only instrument with no usable θ reports *how hard*, never
   *which way*.
4. **Re-derive, do not rescale.** The ratios in §8.2 are specimen-A01, one session,
   n = 5. They are not conversion factors.

### 8.4 Limitations that bound every number above

1. **Absolute newtons are not established.** The decoupling matrix was fitted in a
   free-cantilever setup; a different load point changes the moment arm and scales every
   value by an unknown constant. Ratios, rankings and within-session comparisons hold;
   absolute specifications do not. Recalibration at the contact point (protocol appendix
   A-1) needs a push-type scale, which was unavailable.
2. **The direction channel is unusable** — on both pairs (§5.2).
3. **Move displacement was not controlled.** Directions differ in how far the tissue can
   travel before it stops, so each trial is "the force developed while *attempting* to move
   in that direction", not "the force at a fixed displacement". Operator effort is the only
   controlled variable.
4. **Slip was measured anteriorly, not vertically.** Protocol §4.2 asks for vertical
   traction; pulling vertically to the slip point risked tearing the attachments and this
   session was non-destructive. Slip peaks are therefore paired with `move_F` (same
   direction), **not** with `lift_*`, which is vertical.
5. **One specimen.** The supports are hand-built and the protocol rates their
   reproducibility as low, while they are what sets the lift and retraction reaction. These
   numbers describe A01; **at least three specimens** are needed before quoting them as
   phantom specifications.
6. **n = 5 per stage**, below the n ≥ 10 the protocol asks for grasp and slip.
7. **No MCU kinematics.** Grasp onset in §3.2 should be defined on the hall-sensor `grip`
   stream, which was not connected; onset is approximated from the `gripping` trace.

---

## 9. Data edits, and pitfalls that break loaders

### 9.1 Recorded edits

Three trials removed and one session relabelled, every reason stored in the HDF5 attrs
(`removed_note`, `stage_relabel_note`) and the CSV `note`. Unmodified copies sit beside
each edited file as `.orig`.

| Session | Edit |
|---|---|
| `163214` | Trials 1–4 relabelled to `grasp`, 5–9 to `lift_10` — the GUI stage selector lagged the physical action |
| `164339` | Trial 14 removed — grasp lost mid-trial |
| `165306` | Trial 14 removed — shortest run, `drop_ratio` not computable |
| `171401` | Trial 1 removed — practice run |
| `170625` | Kept but **excluded from analysis** — recorded at Phase-1 settings, which smooth away the slip peak |

Stage labels in `163214` are **operator-asserted, not as-recorded**;
`attrs["stage_relabel_map"]` and `stage_original_counts` hold both versions.

### 9.2 `.orig` files match the data globs

`data/*.h5` and `data/*_trials.csv` both catch `.orig`. **Exclude `*.orig` explicitly** or
you will load each edited session twice.

### 9.3 `161659` is structurally different

No `record_mode` / `pad_sec` attrs — it predates that change — and it is the only session
with `trial == 0` samples. Always filter on `trial` rather than assuming every sample
belongs to a trial.

### 9.4 The CSV schema is ragged

`travel_mm` appears in the header of `170625`, `171030`, `171401`, `173014` only — 24
columns there, 23 in the four earlier sessions. **Do not assume a fixed column count.**
Use `csv.DictReader` and `.get()`, never positional indexing.

### 9.5 `173014` has no force data

Globbing `data/*_trials.csv` yields 8 sessions; only 7 have `.h5`. Guard for a missing
`.h5` and for a header-only CSV.

### 9.6 Other
- Trial numbers are **not contiguous** in `164339`, `165306`, `171401`.
- `force_units` is **mojibake** for the torsion entry.
- `gripping_mean` is **signed**; `gripping_peak` and `torsion_peak` are **absolute**.

---

## 10. Recipes

**Iterate every trial of a stage, correctly** — excludes `.orig`, the excluded session,
and the CSV with no force data:

```python
import glob, os, csv, h5py, numpy as np

EXCLUDE_SESSIONS = {"170625"}          # slip at Phase-1 settings

def sessions():
    for f in sorted(glob.glob("data/phantom_A01_*_trials.csv")):
        if f.endswith(".orig"):
            continue
        sess = os.path.basename(f).split("_")[3]
        if sess in EXCLUDE_SESSIONS:
            continue
        if not os.path.exists(f"data/phantom_A01_20261007_{sess}.h5"):
            continue
        yield sess, f

def trials_of(stage, pair="trocar"):
    """pair='trocar' = the production channels; 'gripper' = the README's reference."""
    for sess, csv_path in sessions():
        rows = [r for r in csv.DictReader(open(csv_path)) if r["stage"] == stage]
        if not rows:
            continue
        with h5py.File(f"data/phantom_A01_20261007_{sess}.h5", "r") as h:
            keys = list(h.attrs["force_keys"])
            ix = keys.index(f"bending_{pair}_x")
            iy = keys.index(f"bending_{pair}_y")
            F  = np.hypot(h["forces"][:, ix], h["forces"][:, iy])
            tr = h["trial"][:]
            for r in rows:
                yield sess, int(r["trial"]), F[tr == int(r["trial"])], r
```

**Reproduce the anisotropy on both pairs** (verifies §8.2):

```python
for pair in ("gripper", "trocar"):
    pk = {d: [f.max() for _, _, f, _ in trials_of(f"move_{d}", pair)] for d in "LRFB"}
    print(pair, {d: round(float(np.mean(v)), 2) for d, v in pk.items()},
          "R/F =", round(np.mean(pk["R"]) / np.mean(pk["F"]), 2))
# gripper {'L': 2.22, 'R': 5.25, 'F': 1.83, 'B': 1.99} R/F = 2.87
# trocar  {'L': 2.2,  'R': 3.78, 'F': 2.64, 'B': 2.25} R/F = 1.43
```

**Find the video frame at a trial's peak:**

```python
for sess, tn, F, r in trials_of("slip_dry"):
    with h5py.File(f"data/phantom_A01_20261007_{sess}_cam.h5", "r") as c:
        ft = c["timestamp"][:]
    print(sess, tn, "frame", int(np.searchsorted(ft, float(r["peak_unix"]))),
          "drop_ratio", r["drop_ratio"])
```

**Rebuild the overview figure** ([`fig_trial_overview.png`](fig_trial_overview.png)):

```bash
python code/make_trial_overview.py      # needs data/video/ (untracked full mp4s)
```

**Re-record a session** — Phase 1 (100 Hz, block mean, 5 Hz low-pass) then, on
confirmation, the slip phase (1 kHz, subsampled, unfiltered). The two phases cannot share
one process: rate and decimation are fixed when the socket source is built.

```bash
conda activate surgitag
# terminal 1 — DAQ driver (Linux, Docker, privileged)
cd driver/linux_daqmxbase && ./run_driver.sh --device Dev1 --channels 8 --rate 1000
# terminal 2 — experiment GUI (starts the driver itself if it is not up)
cd code && ./run_phantom_session.sh --specimen A01
```
