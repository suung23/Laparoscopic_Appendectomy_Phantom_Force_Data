"""Force-to-language labelling for VLA training on the A01 phantom force dataset.

WHAT THIS IS FOR
A force-aware VLA needs the force channel as *text* its language encoder can
read. The naive route -- "F_tip = 2.47 N" -- is wrong for this dataset, and the
reason decides the whole design:

    external/phantom_force_A01/README.md, Limitation 1:
      the decoupling matrix was fitted on 2026-06-30 in a free-cantilever setup;
      a different load point scales EVERY value by an unknown constant.
      Ratios, rankings and within-session comparisons hold. Absolute N do not.

So newtons are not a label. A ratio is. Every force word emitted here is a
fraction of one measured in-dataset reference, so the unknown calibration
constant cancels in the label exactly as it cancels in the dataset's own
publishable Phase 2 result.

    r = F_tip / F_slip_ref        F_tip = hypot(bending_gripper_x, _y)

`F_slip_ref` is the median peak F_tip of the dry slip trials (§4) -- the tension
at which the tissue actually escaped the jaws. r = 1 therefore means "as much
tension as this grasp was able to hold", which is the quantity a force-aware
policy has to respect. r > 1 is real and informative: the 15 mm lifts exceed the
anterior slip limit, which is consistent with the README's reading that the
attachments yielded between 15 and 20 mm.

THE EVENT AXIS IS NAMED FOR WHAT IS MEASURED
A collapse in tension is detectable; whether it was an escaped grasp or the
operator letting go is not, from magnitude alone -- trough persistence does not
separate the stages (move_F troughs outlast slip_dry ones). So the event is
`traction_loss`, with the handle gauge's direction of change as a hedged subtype.
Calling every collapse a "slip" would have put a confident word on an inference.

THREE ORTHOGONAL AXES, NOT ONE FLAT VOCABULARY
Level (how much), trend (which way it is going) and event (what just happened)
are emitted separately and composed into the utterance. A flat phrase set would
force the model to learn "firm and rising" as a token unrelated to "firm and
steady"; composed axes let it learn the factors.

WHAT IS DELIBERATELY NOT LABELLED
  * Direction. README Limitation 2: theta clusters at 130-165 deg for all four
    move directions, which should differ by 90 deg. The `bending_gripper_y` half
    bridge was rewired and never refitted. No label here says left/right/up/down,
    even though `move_L/R/F/B` stage names exist -- the stage name is the
    operator's intent, not a reading of the direction channel.
  * Grasp force in newtons. `gripping` is the handle lever gauge: 0.5 N
    resolution, -1.57 N contamination under pure torsion (protocol §1). It is
    carried through as `handle_actuation_N_nominal` for reference and used for
    one thing only -- the DIRECTION of its change across a collapse, which
    separates tissue escaping closed jaws from a deliberate release. No label
    quotes it as a grasp force.
  * Stiffness, displacement, thrust. No such channel exists.
  * The 261021 surgilog session. It carries video + kinematics but its force is
    a no-load recording: 1.4-2.2 mN against ~100 mN resolution, and its averaging
    converges to zero, i.e. there is no signal underneath (DATASET_MANUAL §6.1).
    `refuse_unloaded_session()` exists so that path cannot silently produce force
    words; that session may contribute `no_load` labels and nothing else.

OUTPUT
  out/vla_force_labels/<episode>.jsonl   one record per 10 Hz label tick,
                                         carrying the mp4 frame index
  out/vla_force_labels/episodes.jsonl    one caption per trial (clip-level)
  validation/force-language-labeling.json the spec, thresholds and distributions
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'external' / 'phantom_force_A01' / 'data'
UNLOADED = ROOT / 'external' / 'phantom_261021'
OUT = ROOT / 'out' / 'vla_force_labels'
REPORT = ROOT / 'validation' / 'force-language-labeling.json'

# Column order is attrs['force_keys']; the distal (gripper) station is live in
# this dataset, unlike the 261021 instrument where those gauges were removed.
IDX_BGX, IDX_BGY, IDX_GRIP = 0, 2, 4

LABEL_HZ = 10.0          # VLA chunk rate; video is 30 fps, force 100 Hz
TREND_WINDOW_S = 0.3     # dF/dt regression window
DWELL_TICKS = 2          # dropping to a lower band must hold 0.2 s before it is believed
HYSTERESIS = 0.10        # boundary margin when dropping, fraction of the boundary value

# Level bands on r = F_tip / F_slip_ref. The lowest boundary is the measurement
# floor: the bending axes read ~0.1 N of noise (protocol §3.1), which is 0.05 r,
# so 0.07 keeps the no-load class strictly below anything resolvable.
BANDS = [
    (0.00, 'no_load',           '무하중'),
    (0.07, 'light_contact',     '가벼운 접촉'),
    (0.25, 'secure_traction',   '안정 견인'),
    (0.60, 'high_traction',     '강한 견인'),
    (1.00, 'beyond_grip_limit', '파지 한계 초과'),
]
BAND_EDGES = [b[0] for b in BANDS]

# Trend thresholds in r per second: fraction of the grip limit gained per second.
TREND_RISE_FAST, TREND_RISE = 0.50, 0.10
TRENDS = {
    'rising_fast':   ('is rising sharply',  '급격히 증가하고'),
    'rising':        ('is still rising',    '증가하고'),
    'steady':        ('is steady',          '유지되고'),
    'releasing':     ('is easing off',      '감소하고'),
    'dropping_fast': ('is collapsing',      '급락하고'),
}

# Traction loss: F_tip collapses below SLIP_DROP_RATIO of the peak it held over
# the preceding SLIP_LOOKBACK_S, and does not climb back to SLIP_RECOVER_RATIO of
# that peak within SLIP_PERSIST_S. The peak is taken over a trailing window, not
# from the start of the trial: a move trial reciprocates about once every two
# seconds (protocol 3.4), and against a running maximum every later trough would
# read as a fall from the first cycle's peak.
#
# The event is called traction_loss, not slip, because magnitude alone cannot say
# which it was. The trough-persistence distribution does not separate the stages
# at all -- move_F troughs last longer (median 3.2 s) than slip_dry ones (2.2 s)
# -- so "a big drop that stays down" is a true statement about the tension and
# nothing more.
SLIP_DROP_RATIO, SLIP_MIN_BAND = 0.60, 2
SLIP_LOOKBACK_S, SLIP_PERSIST_S, SLIP_RECOVER_RATIO = 1.60, 0.50, 0.85

# What the subtype rests on. Measured across every traction-loss crossing, the
# handle gauge moves in opposite directions by stage (N nominal, post - pre):
#
#   slip_dry   n=13  median +1.36   38 % negative
#   lift_*     n= 7  median -1.63  100 % negative
#   move_*     n=21  median -1.16   50-100 % negative
#   grasp      n=14  median -0.85   71 % negative
#
# The operator kept squeezing while the tissue escaped in the slip stage, and let
# go everywhere else. CAVEAT: `gripping` carries -1.07 to -1.20 N of bending
# contamination at 4.9 N and -1.57 N under 0.23 N.m of torsion (protocol 1), and
# bending IS falling across the drop, so relief of that contamination pushes the
# reading up by a few tenths of a newton on its own. The separation here is
# larger than that, but the subtype is a cue, not a measurement, and it is
# emitted as `handle_held` / `handle_opened` rather than as a newton value.
HANDLE_WINDOW_S, HANDLE_LAG_S, HANDLE_DEADBAND_N = 0.40, 0.30, 0.50

LEVEL_EN = {
    'no_load': 'there is no measurable load on the instrument tip',
    'light_contact': 'the tip is barely touching the tissue, {pct} percent of the '
                     'tension this grasp can hold',
    'secure_traction': 'the tissue is held under secure traction at {pct} percent '
                       'of the tension this grasp can hold',
    'high_traction': 'traction is high, {pct} percent of the tension this grasp '
                     'can hold, close to the limit',
    'beyond_grip_limit': 'traction has passed the tension this grasp can hold '
                         '({pct} percent); the tissue can slip out of the jaws',
}
LEVEL_KO = {
    'no_load': '기구 끝단에 유의미한 하중이 없고',
    'light_contact': '조직에 가볍게 닿은 상태로 이 파지가 버틸 수 있는 장력의 {pct}퍼센트이며',
    'secure_traction': '조직을 안정적으로 견인하는 중이며 이 파지가 버틸 수 있는 장력의 {pct}퍼센트이고',
    'high_traction': '견인력이 높아 이 파지가 버틸 수 있는 장력의 {pct}퍼센트로 한계에 가깝고',
    'beyond_grip_limit': '견인력이 이 파지가 버틸 수 있는 장력을 넘어 {pct}퍼센트에 이르렀고 조직이 겸자에서 빠질 수 있으며',
}
ADVICE = {
    'high_traction': ('hold or reduce the traction', '견인력을 유지하거나 줄일 것'),
    'beyond_grip_limit': ('reduce the traction now; the grasp is about to fail',
                          '즉시 견인력을 줄일 것. 파지가 곧 풀린다'),
}
EVENT_EN = {
    'contact_onset': 'contact with the tissue has just been made',
    'traction_loss': 'the tension just collapsed from {pct} percent of what this '
                     'grasp could hold',
    'release': 'the load has just been released',
}
EVENT_KO = {
    'contact_onset': '조직과 접촉이 막 시작되었다',
    'traction_loss': '장력이 버틸 수 있는 한계의 {pct}퍼센트에서 막 급락했다',
    'release': '하중이 막 해제되었다',
}
# Appended to a traction_loss utterance. The handle cue is weak (see above), so
# the wording stays hedged and never asserts a measured jaw state.
SUBTYPE_EN = {
    'handle_held': ' while the handle stayed closed, which is what tissue escaping '
                   'the jaws looks like',
    'handle_opened': ' as the handle opened, which is what a deliberate release '
                     'looks like',
    'handle_unclear': '; these channels cannot say whether the jaws opened',
}
SUBTYPE_KO = {
    'handle_held': ' 손잡이는 닫힌 상태였다. 조직이 겸자에서 빠져나갈 때의 양상이다',
    'handle_opened': ' 손잡이가 열렸다. 의도적으로 놓을 때의 양상이다',
    'handle_unclear': ' 겸자가 열렸는지는 이 채널들로 판단할 수 없다',
}


def refuse_unloaded_session(path: Path) -> None:
    """Hard stop before any force word is produced from a no-load recording.

    DATASET_MANUAL.md §6.1 shows the 261021 trocar pair averaging down as
    1/sqrt(n) out to a 2 s window, i.e. converging on zero, and shows that
    z-scoring it yields a plausible-looking +-3 sigma trace indistinguishable
    from the physically disconnected channels. A label in newtons passes schema
    validation and looks right on a plot, so it propagates without warning.
    """
    raise ValueError(
        f'{path}: no-load session. Live channels are bending_trocar_x/y at '
        '1.4-2.2 mN against ~100 mN resolution (1/45), spectrally and '
        'statistically indistinguishable from the gauge-removed channels. '
        'Usable here only as the no_load class; force magnitude, trend and '
        'slip labels must not be derived from it (DATASET_MANUAL.md 6.1).'
    )


def band_of(r: float, current: int | None = None) -> int:
    """Level index for r. Escalation is immediate; de-escalation gets hysteresis.

    The asymmetry is deliberate. A label that lags a rising force would tell a
    policy the grasp is still safe after it has stopped being safe, so crossing
    upward is believed at once. Crossing downward only has to be believed for
    flicker control, so it costs HYSTERESIS of the boundary (and, in the caller,
    DWELL_TICKS of persistence).
    """
    k = int(np.searchsorted(BAND_EDGES, r, side='right') - 1)
    if current is None or k >= current:
        return k
    return current if r > BAND_EDGES[current] * (1.0 - HYSTERESIS) else k


def trend_of(rdot: float) -> str:
    if rdot >= TREND_RISE_FAST:
        return 'rising_fast'
    if rdot >= TREND_RISE:
        return 'rising'
    if rdot <= -TREND_RISE_FAST:
        return 'dropping_fast'
    if rdot <= -TREND_RISE:
        return 'releasing'
    return 'steady'


def slope_per_s(r: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Centred least-squares slope of r over TREND_WINDOW_S, in r per second."""
    rate = 1.0 / np.median(np.diff(t)) if len(t) > 2 else 100.0
    half = max(1, int(round(TREND_WINDOW_S * rate / 2)))
    out = np.zeros_like(r)
    for i in range(len(r)):
        a, b = max(0, i - half), min(len(r), i + half + 1)
        if b - a < 3:
            continue
        tt = t[a:b] - t[a:b].mean()
        out[i] = float(tt @ (r[a:b] - r[a:b].mean()) / (tt @ tt))
    return out


def find_traction_losses(r: np.ndarray, grip: np.ndarray,
                         t: np.ndarray) -> dict[int, tuple[float, str]]:
    """{index: (peak_r it fell from, handle subtype)} for each sustained collapse."""
    rate = 1.0 / np.median(np.diff(t)) if len(t) > 2 else 100.0
    look = max(2, int(round(SLIP_LOOKBACK_S * rate)))
    hold = max(1, int(round(SLIP_PERSIST_S * rate)))
    hw, lag = max(1, int(round(HANDLE_WINDOW_S * rate))), int(round(HANDLE_LAG_S * rate))
    out: dict[int, tuple[float, str]] = {}
    i = 0
    while i < len(r):
        peak = float(r[max(0, i - look):i + 1].max())
        if (band_of(peak) >= SLIP_MIN_BAND
                and r[i] < peak * SLIP_DROP_RATIO
                and float(r[i:i + hold].max()) < peak * SLIP_RECOVER_RATIO):
            pre, post = grip[max(0, i - lag - hw):max(1, i - lag)], grip[i:i + hw]
            if len(pre) and len(post):
                d = float(post.mean() - pre.mean())
                sub = ('handle_opened' if d <= -HANDLE_DEADBAND_N else
                       'handle_held' if d >= HANDLE_DEADBAND_N else 'handle_unclear')
            else:
                sub = 'handle_unclear'
            out[i] = (peak, sub)
            j = np.where(r[i:] >= peak * SLIP_RECOVER_RATIO)[0]
            i += int(j[0]) + 1 if len(j) else len(r)
        else:
            i += 1
    return out


def render(band_name: str, trend: str, r: float, event: str | None,
           peak_r: float | None, subtype: str | None) -> dict:
    pct = int(round(r * 100))
    level_en, level_ko = LEVEL_EN[band_name], LEVEL_KO[band_name]
    if band_name == 'no_load':
        obs_en = level_en + ' and it ' + TRENDS[trend][0]
        obs_ko = level_ko + ' ' + TRENDS[trend][1] + ' 있다'
    else:
        obs_en = level_en.format(pct=pct) + ', and it ' + TRENDS[trend][0]
        obs_ko = level_ko.format(pct=pct) + ' ' + TRENDS[trend][1] + ' 있다'
    rec = {'utterance_en': obs_en + '.', 'utterance_ko': obs_ko + '.'}
    if event:
        p = int(round((peak_r if peak_r is not None else r) * 100))
        en, ko = EVENT_EN[event].format(pct=p), EVENT_KO[event].format(pct=p)
        if event == 'traction_loss' and subtype:
            en, ko = en + SUBTYPE_EN[subtype], ko + '.' + SUBTYPE_KO[subtype]
        rec['event_utterance_en'] = en + '.'
        rec['event_utterance_ko'] = ko + '.' 
    if band_name in ADVICE:
        rec['advice_en'], rec['advice_ko'] = ADVICE[band_name]
    return rec


def read_session(path: Path) -> dict:
    with h5py.File(path) as f:
        a = f.attrs
        return dict(
            episode=path.stem,
            stage_names=[str(s) for s in a['stage_names']],
            rate=float(a['rate']), cutoff_hz=float(a['cutoff_hz']),
            specimen=str(a['specimen']),
            forces=f['forces'][()], stage=f['stage'][()],
            trial=f['trial'][()], timestamp=f['timestamp'][()],
            removed_trials=str(a.get('removed_trials', '')),
        )


def frame_lookup(path: Path):
    """Map an absolute time to the nearest mp4 frame index, or None if no sidecar."""
    cam = path.with_name(path.stem + '_cam.h5')
    if not cam.exists():
        return lambda _t: None
    with h5py.File(cam) as f:
        ts = f['timestamp'][()]
    return lambda t: int(np.argmin(np.abs(ts - t)))


def slip_reference(sessions: list[dict]) -> tuple[float, dict]:
    """Median peak F_tip over the dry slip trials -- the ratio anchor."""
    peaks = []
    for s in sessions:
        ft = np.hypot(s['forces'][:, IDX_BGX], s['forces'][:, IDX_BGY])
        for t in np.unique(s['trial'][s['trial'] > 0]):
            sel = s['trial'] == t
            if s['stage_names'][int(s['stage'][sel][0])].startswith('slip_dry'):
                peaks.append(float(ft[sel].max()))
    if not peaks:
        raise ValueError('no slip_dry trials: the ratio anchor cannot be established')
    return float(np.median(peaks)), {
        'trials': len(peaks), 'peaks_N_nominal': sorted(round(p, 4) for p in peaks),
        'median_N_nominal': round(float(np.median(peaks)), 4),
        'spread_N_nominal': [round(float(np.min(peaks)), 4), round(float(np.max(peaks)), 4)],
    }


def label_trial(s: dict, t_id: int, f_slip: float, to_frame) -> tuple[list[dict], dict]:
    sel = s['trial'] == t_id
    t = s['timestamp'][sel]
    ft = np.hypot(s['forces'][sel, IDX_BGX], s['forces'][sel, IDX_BGY])
    grip = s['forces'][sel, IDX_GRIP]
    r = ft / f_slip
    rdot = slope_per_s(r, t)
    losses = find_traction_losses(r, grip, t)
    stage = s['stage_names'][int(s['stage'][sel][0])]

    step = max(1, int(round(s['rate'] / LABEL_HZ)))
    recs, cur_band, dwell, cand = [], None, 0, None
    prev_band = None
    for i in range(0, len(r), step):
        k = band_of(float(r[i]), cur_band)
        if cur_band is None or k > cur_band:
            cur_band, cand, dwell = k, None, 0      # escalate at once
        elif k < cur_band:
            cand, dwell = (k, dwell + 1) if cand == k else (k, 1)
            if dwell >= DWELL_TICKS:
                cur_band, cand, dwell = k, None, 0
        else:
            cand, dwell = None, 0

        window = [j for j in range(i, min(len(r), i + step)) if j in losses]
        event, peak_r, subtype = None, None, None
        if window:
            event = 'traction_loss'
            peak_r, subtype = losses[window[0]]
        elif prev_band == 0 and cur_band >= 1:
            event = 'contact_onset'
        elif prev_band is not None and prev_band >= 1 and cur_band == 0:
            event = 'release'
        prev_band = cur_band

        name, name_ko = BANDS[cur_band][1], BANDS[cur_band][2]
        trend = trend_of(float(rdot[i]))
        rec = {
            'episode': s['episode'], 'specimen': s['specimen'], 'trial': int(t_id),
            'stage': stage, 'stage_is_operator_intent': True,
            't_unix': float(t[i]), 't_rel_s': round(float(t[i] - t[0]), 3),
            'frame_index': to_frame(float(t[i])),
            'F_tip_ratio': round(float(r[i]), 4),
            'F_tip_ratio_dot_per_s': round(float(rdot[i]), 4),
            'F_tip_N_nominal': round(float(ft[i]), 4),
            'handle_actuation_N_nominal': round(float(grip[i]), 4),
            'band': cur_band, 'band_name': name, 'band_name_ko': name_ko,
            'trend': trend, 'event': event, 'event_subtype': subtype,
            'force_validity': 'ratio_only_unknown_scale',
            'direction_labelled': False,
        }
        rec.update(render(name, trend, float(r[i]), event, peak_r, subtype))
        recs.append(rec)

    peak = float(r.max())
    hold = r[int(len(r) * 0.6):]
    held = [v[1] for v in losses.values() if v[1] == 'handle_held']
    if losses:
        tail = (', then the tension collapses with the handle still closed'
                if held else ', then the tension collapses as the handle opens')
    else:
        tail = f' and settles near {int(round(float(np.median(hold)) * 100))} percent'
    caption = (f'{stage}: traction rises to {int(round(peak * 100))} percent of the '
               f'tension this grasp can hold' + tail)
    episode = {
        'episode': s['episode'], 'trial': int(t_id), 'stage': stage,
        'duration_s': round(float(t[-1] - t[0]), 3), 'n_labels': len(recs),
        'peak_ratio': round(peak, 4), 'mean_ratio': round(float(r.mean()), 4),
        'peak_band': BANDS[band_of(peak)][1],
        'traction_loss_events': len(losses),
        'traction_loss_handle_held': len(held),

        'caption_en': caption + '.',
        'force_validity': 'ratio_only_unknown_scale',
    }
    return recs, episode


def clip_duration_s(path: Path) -> float | None:
    """Clip length from ffprobe, or None when ffprobe is not installed."""
    try:
        out = subprocess.run(['ffprobe', '-v', 'error', '-show_entries',
                              'format=duration', '-of', 'csv=p=0', str(path)],
                             capture_output=True, text=True, timeout=30)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    try:
        return float(out.stdout.strip())
    except ValueError:
        return None


def assign_clips(episodes: list[dict]) -> tuple[int, str]:
    """Attach each tracked clip to the one trial it was cut from.

    `data/video_clips/<stage>_trial<n>.mp4` names a stage and a trial number, but
    the trial number restarts in every session, so `slip_dry_trial3.mp4` matches a
    trial in two sessions and `grasp_trial2.mp4` in two more. Guessing costs a
    wrong video against a force trace, so the match is made on evidence: the clip
    was cut from that trial's frames, so its duration is the trial's duration.
    The two ambiguous names resolve cleanly -- grasp_trial2 is 10.12 s against
    trial windows of 10.69 s and 12.69 s, slip_dry_trial3 is 2.69 s against
    8.49 s and 2.70 s.

    Without ffprobe the fallback is the README's own selection rule (the trial
    whose peak is closest to the stage median), which agrees with the duration
    match on both ambiguous names but is an inference, so it is recorded as one.
    """
    for e in episodes:
        e['clip'] = None
    linked, method = 0, 'clip_duration'
    for path in sorted((DATA / 'video_clips').glob('*_trial*.mp4')):
        stage, _, num = path.stem.rpartition('_trial')
        cand = [e for e in episodes if e['stage'] == stage and e['trial'] == int(num)]
        if not cand:
            continue
        dur = clip_duration_s(path)
        if dur is None:
            method = 'stage_median_fallback'
            median = float(np.median([e['peak_ratio'] for e in episodes
                                      if e['stage'] == stage]))
            pick = min(cand, key=lambda e: abs(e['peak_ratio'] - median))
            pick['clip_matched_by'] = 'stage_median_fallback'
        else:
            pick = min(cand, key=lambda e: abs(e['duration_s'] - dur))
            pick['clip_matched_by'] = 'clip_duration'
            pick['clip_duration_s'] = round(dur, 3)
        pick['clip'] = str(path.relative_to(ROOT))
        pick['clip_candidates'] = len(cand)
        linked += 1
    return linked, method


def check_invariants(report: dict, episodes: list[dict]) -> None:
    """Fail loudly rather than write a report that contradicts its own claims."""
    if BAND_EDGES != sorted(BAND_EDGES):
        raise ValueError('band edges are not monotone')
    if BANDS[1][0] <= 0.1 / report['ratio_anchor']['value_N_nominal']:
        raise ValueError('the no_load edge is at or below the measurement floor')
    if report['coverage']['labels'] != sum(e['n_labels'] for e in episodes):
        raise ValueError('label count does not match the per-trial totals')
    if report['absolute_newtons_validated'] or report['direction_channel_used']:
        raise ValueError('this dataset supports neither claim')
    clipped = [e for e in episodes if e['clip']]
    if len({e['clip'] for e in clipped}) != len(clipped):
        raise ValueError('a clip is attached to more than one trial')


def main() -> None:
    paths = sorted(p for p in DATA.glob('phantom_A01_*.h5')
                   if not p.stem.endswith('_cam'))
    sessions = []
    for p in paths:
        s = read_session(p)
        if s['forces'].shape[0] == 0 or not np.any(s['trial'] > 0):
            continue
        s['path'] = p
        sessions.append(s)
    f_slip, anchor = slip_reference(sessions)

    # Exercise the guard rather than leaving it as a comment: the 261021 force
    # files are reachable from here, and nothing in the pipeline would notice if
    # they were quietly added to `paths`.
    refused = []
    for q in sorted(UNLOADED.rglob('NI6009/collected_data/*/*.h5')):
        try:
            refuse_unloaded_session(q)
        except ValueError as exc:
            refused.append(str(q.relative_to(ROOT)))
            reason = str(exc).split(': ', 1)[1]
            break
    else:
        reason = None

    OUT.mkdir(parents=True, exist_ok=True)
    episodes, hist, n_labels = [], np.zeros(len(BANDS), int), 0
    trend_hist: dict[str, int] = {k: 0 for k in TRENDS}
    for s in sessions:
        to_frame = frame_lookup(s['path'])
        lines = []
        for t_id in sorted(int(x) for x in np.unique(s['trial'][s['trial'] > 0])):
            recs, ep = label_trial(s, t_id, f_slip, to_frame)
            lines += recs
            episodes.append(ep)
            for r in recs:
                hist[r['band']] += 1
                trend_hist[r['trend']] += 1
        n_labels += len(lines)
        (OUT / f'{s["episode"]}.jsonl').write_text(
            ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in lines))
    linked, clip_method = assign_clips(episodes)
    (OUT / 'episodes.jsonl').write_text(
        ''.join(json.dumps(e, ensure_ascii=False) + '\n' for e in episodes))

    report = {
        'scope': 'Force-to-language labels for VLA training on phantom specimen A01; '
                 'relative-magnitude labels only, no absolute newtons and no direction',
        'absolute_newtons_validated': False,
        'direction_channel_used': False,
        'quantity': 'r = F_tip / F_slip_ref, F_tip = hypot(bending_gripper_x, bending_gripper_y)',
        'why_ratio': 'The decoupling matrix was fitted for a different load point, so every '
                     'newton value carries one unknown multiplicative constant '
                     '(phantom_force_A01/README.md limitation 1). A ratio to an in-dataset '
                     'reference cancels it, exactly as the dataset own Phase 2 result does.',
        'ratio_anchor': {'definition': 'median peak F_tip of the dry slip trials (protocol 4)',
                         'meaning': 'r = 1 is the tension at which the tissue escaped the jaws',
                         'value_N_nominal': round(f_slip, 4), **anchor},
        'label_axes': {
            'level': [{'band': i, 'r_from': BANDS[i][0],
                       'r_to': BANDS[i + 1][0] if i + 1 < len(BANDS) else None,
                       'name': BANDS[i][1], 'name_ko': BANDS[i][2]}
                      for i in range(len(BANDS))],
            'trend': {'unit': 'r per second', 'rising_fast': f'>= {TREND_RISE_FAST}',
                      'rising': f'>= {TREND_RISE}', 'steady': f'|.| < {TREND_RISE}',
                      'releasing': f'<= -{TREND_RISE}', 'dropping_fast': f'<= -{TREND_RISE_FAST}',
                      'window_s': TREND_WINDOW_S},
            'event': {
                'contact_onset': 'band 0 -> >=1',
                'release': 'band >=1 -> 0',
                'traction_loss': f'r falls below {SLIP_DROP_RATIO} of the peak held over '
                                 f'the preceding {SLIP_LOOKBACK_S} s, from a peak in band '
                                 f'{SLIP_MIN_BAND} or above, and does not climb back to '
                                 f'{SLIP_RECOVER_RATIO} of that peak within {SLIP_PERSIST_S} s',
                'why_not_called_slip':
                    'magnitude cannot tell an escaped grasp from a deliberate release. '
                    'Trough persistence does not separate the stages: move_F troughs last '
                    'longer (median 3.2 s) than slip_dry troughs (2.2 s). The event states '
                    'what is measured -- the tension collapsed and stayed down.',
                'subtype': {
                    'from': 'handle gauge change across the drop, '
                            f'+-{HANDLE_DEADBAND_N} N nominal deadband',
                    'handle_held': 'handle force rose or held: the pattern of tissue '
                                   'escaping closed jaws',
                    'handle_opened': 'handle force fell: the pattern of a deliberate release',
                    'evidence_N_nominal_median_by_stage': {
                        'slip_dry': 1.36, 'lift': -1.63, 'move': -1.16, 'grasp': -0.85},
                    'caveat': 'the handle gauge carries -1.07 to -1.20 N of bending '
                              'contamination at 4.9 N and -1.57 N under 0.23 N.m of torsion '
                              '(protocol 1), and bending is falling across the drop, so '
                              'relief of that contamination pushes the reading up by a few '
                              'tenths on its own. The separation above is larger than that, '
                              'but the subtype is a cue, not a measurement.'},
            },
        },
        'stability': {'label_rate_hz': LABEL_HZ, 'dwell_ticks': DWELL_TICKS,
                      'boundary_hysteresis': HYSTERESIS,
                      'asymmetric': 'escalation immediate, de-escalation needs dwell + hysteresis',
                      'why': 'at 100 Hz a raw threshold flickers across a boundary and the '
                             'language label would change several times a second; the '
                             'asymmetry keeps a rising force from being under-reported'},
        'noise_floor': {'bending_axis_N_nominal': 0.1,
                        'in_ratio_units': round(0.1 / f_slip, 4),
                        'lowest_band_edge': BANDS[1][0],
                        'note': 'no_load is kept strictly below anything the axis resolves'},
        'coverage': {'sessions': len(sessions), 'trials': len(episodes),
                     'labels': n_labels,
                     'clips_linked': linked, 'clip_match': clip_method,
                     'clip_files': len(list((DATA / 'video_clips').glob('*.mp4'))),
                     'band_distribution': {BANDS[i][1]: int(hist[i]) for i in range(len(BANDS))},
                     'trend_distribution': trend_hist,
                     'stages': sorted({e['stage'] for e in episodes})},
        'not_labelled': [
            'direction: theta clusters at 130-165 deg for all four move directions, so the '
            'bending_gripper_y calibration is stale (README limitation 2). move_L/R/F/B is '
            'carried as operator intent, never as a measured direction.',
            'grasp force in newtons: the gripping gauge is on the handle lever, 0.5 N '
            'resolution with -1.57 N torsion contamination (protocol 1)',
            'stiffness, displacement, axial thrust: no such channel exists',
            'the 261021 surgilog session: no-load recording, refuse_unloaded_session() blocks it',
        ],
        'remaining': [
            'Protocol appendix A-1 (push-type scale at the contact point) fixes the one '
            'constant that turns these ratios into absolute newtons; the band edges then '
            'stay as they are and only the anchor gains a unit.',
            'One specimen and n=5 per stage, below the protocol n>=10; the anchor should be '
            'remeasured per specimen before the bands are quoted as phantom specifications.',
            'A loaded session on the 261021 rig would give video + kinematics + force '
            'together; A01 has no MCU kinematics, so these labels carry no insertion depth '
            'or grip percentage.',
        ],
        'refused_inputs': {
            'sessions': len(list(UNLOADED.rglob('NI6009/collected_data/*/*.h5'))),
            'example': refused[0] if refused else None,
            'reason': reason,
            'may_still_contribute': 'the no_load class, and video + MCU kinematics, '
                                    'which are sound in that session',
        },
        'outputs': {'per_tick': str((OUT / '<episode>.jsonl').relative_to(ROOT)),
                    'per_trial': str((OUT / 'episodes.jsonl').relative_to(ROOT))},
    }
    check_invariants(report, episodes)
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(f'{n_labels} labels over {len(episodes)} trials in {len(sessions)} sessions')
    print('anchor F_slip_ref =', round(f_slip, 4), 'N nominal (ratio reference only)')
    for i in range(len(BANDS)):
        print(f'  {BANDS[i][1]:<18} {hist[i]:6d}  {100 * hist[i] / n_labels:5.1f}%')
    print('report ->', REPORT.relative_to(ROOT))


if __name__ == '__main__':
    main()
