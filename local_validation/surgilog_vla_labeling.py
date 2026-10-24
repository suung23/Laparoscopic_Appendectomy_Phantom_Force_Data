"""VLA labelling for the surgilog_261021 bimanual session: video + kinematics + language.

WHY THIS SESSION IS THE BASE
It is the only session that carries what a VLA needs as a triple: endoscope video,
both instruments' kinematics, and one shared absolute clock. 6973 frames with zero
drops, `camera/master_index` giving a 1:1 frame-to-sample key, MCU CRC and sequence
loss both zero. The phantom A01 force session has none of that -- no MCU, so no tip
pose, no insertion depth, no grip percentage -- and without proprioception its force
trace cannot be tied to a state or an action.

WHAT THIS SESSION CANNOT GIVE, AND WHAT IS DONE ABOUT IT
There is no usable force signal on either instrument:

  * NI6009, right gripper: only `bending_trocar_x/y` are wired, and in this session
    they sit at 1.4-2.2 mN against ~100 mN of practical resolution. Averaging follows
    1/sqrt(n) out to a 2 s window, so what it converges on is zero -- there is no
    signal underneath (DATASET_MANUAL 6.1).
  * SurgiTag loadcell, left cutter: uncalibrated uV/V, and the episode variation is
    slow thermal drift with no step at grasp or cut; `corr(delta_uvv, grip)` flips
    sign between episodes (DATASET_MANUAL 6.2).

So the force axis here is not estimated, it is DECLARED: every record carries
`force.state = "no_load"` with the measured detection limit. That is a true label --
the manual's one sanctioned use of this session's force channel -- and it is the
negative class a loaded session will be learned against. What it is not is a force
VOCABULARY: no magnitude word, no trend word, no slip word is emitted here, because
z-scoring this channel produces a convincing +-3 sigma trace that is
indistinguishable from the physically disconnected channels.

The force words themselves, and the band edges they sit on, are defined in
`force_language_labeling.py` on the A01 session, which does have load. That module
holds the schema; this one holds the state, the action and the video. They meet when
a loaded session is recorded on this rig -- `FORCE_SLOT` is the join.

WHAT IS LABELLED HERE, ALL FROM LIVE CHANNELS
  jaw          MCU `grip`, with saturation as its own class (36 % of valid samples
               sit at exactly 100 %, where the hall conversion clips)
  depth        MCU `dist`, trocar insertion depth in mm
  tip_motion   speed of MCU `rcm_position`
  cutter       SurgiTag `tof_mm` -- PROXIMITY, never called force or contact
  coordination which hand is moving, from tip speed and cutter angular rate

SESSION ZERO, VALIDITY, AND THE OTHER TRAPS THE MANUAL FLAGS
  * `rcm_qref` is per session: session A (17:10, 2 episodes) and session B (17:56-59,
    11 episodes) have different zeros, so `rcm_position` is comparable only within a
    session. Depth bands are therefore computed per session, and tip SPEED, being a
    difference, is comparable across both.
  * Five episodes (175635-175750) have 350-380 ms BLE gaps filled by linear
    interpolation. `imu/valid` masks them; ticks that fail it carry no cutter label.
  * `grip == 100` cannot be resolved further; the label says so rather than implying
    a known jaw angle.
  * No subtask or phase annotation exists in this dataset, so `stage` is absent here
    on purpose -- unlike the A01 labels, nothing in these records is operator intent.

OUTPUT
  out/vla_surgilog_labels/<episode>.jsonl   one record per 10 Hz tick: state, action,
                                            action chunk, frame index, utterance
  out/vla_surgilog_labels/episodes.jsonl    one summary per episode
  validation/surgilog-vla-labeling.json     the spec, thresholds and distributions
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = (ROOT / 'external' / 'phantom_261021' /
        'surgilog_261021_complete version of Phantom data')
SYNCED = BASE / 'SurgiTag' / 'synced_data' / '261021'
VIDEO = BASE / 'Endoscope' / 'collected_data' / '261021'
MCU_BOARD = 'mcu/board-11-l1x'
OUT = ROOT / 'out' / 'vla_surgilog_labels'
REPORT = ROOT / 'validation' / 'surgilog-vla-labeling.json'

LABEL_HZ = 10.0          # VLA tick; master is 100 Hz, video 29.77 fps
DIFF_WINDOW_S = 0.20     # centred window for every rate and speed
CHUNK = 5                # action-chunk length, i.e. 0.5 s of future action

# Episodes with BLE gaps of 350-380 ms, linearly interpolated in the merge
# (DATASET_MANUAL 6.3). Not dropped -- `imu/valid` already marks the filled
# samples -- but flagged, because a cutter label near a gap rests on a straight
# line someone drew, not on a measurement.
BLE_GAP_EPISODES = {'175635', '175653', '175713', '175731', '175750'}
# Session A and B have different RCM zeros (DATASET_MANUAL 6.6).
SESSION_A = {'171006', '171033'}

# Jaw bands on MCU `grip` (%). The top class exists because the hall-to-percent
# conversion clips: 36.2 % of valid samples read exactly 100.0, and inside that
# class the jaw angle is unknown. Finer angle needs `hall/adc` from the MCU file.
JAW_BANDS = [
    (0.0, 'jaw_open', '겸자 열림'),
    (10.0, 'jaw_part_closed', '겸자 반쯤 닫힘'),
    (60.0, 'jaw_closed', '겸자 닫힘'),
    (99.9, 'jaw_clamped_saturated', '겸자 완전 압착(센서 포화)'),
]
JAW_RATE = 25.0          # %/s; |d grip/dt| above this is a closing or opening motion

# Tip speed (mm/s) from `rcm_position`. Edges sit at the measured deciles:
# p10 = 3.0, p50 = 10.6, p90 = 78.8 mm/s pooled over the 13 episodes.
SPEED_BANDS = [
    (0.0, 'tip_still', '팁 정지'),
    (3.0, 'tip_slow', '팁 느리게 이동'),
    (15.0, 'tip_moderate', '팁 보통 속도 이동'),
    (60.0, 'tip_fast', '팁 빠르게 이동'),
]
DEPTH_RATE = 2.0         # mm/s on `dist`; measured p25/p75 are -3.8 / +4.2
# Cutter ToF (mm). Pooled range is 74-183 mm, quartiles 106 / 129 / 138.
TOF_BANDS = [
    (0.0, 'cutter_near', '커터 근접'),
    (105.0, 'cutter_mid', '커터 중간 거리'),
    (140.0, 'cutter_far', '커터 원거리'),
]
TOF_RATE = 15.0          # mm/s; measured quartiles are -15.1 / +14.0
CUTTER_SPIN = 30.0       # deg/s; |gyro| p95 is 32.2 deg/s

# The join to the loaded-session force vocabulary. `force_language_labeling.py`
# defines the bands; this session can only fill the bottom one, and says so with
# the number that bounds it.
FORCE_SLOT = {
    'state': 'no_load',
    'state_ko': '무하중',
    'measured': False,
    'detection_limit_mN_1s_average': 0.20,
    'detection_limit_mN_2s_average': 0.13,
    'basis': 'NI6009 bending_trocar_x/y, 1.4-2.2 mN against ~100 mN resolution; '
             'averaging follows 1/sqrt(n) to 2 s, converging on zero '
             '(DATASET_MANUAL 6.1)',
    'cutter_channel': 'SurgiTag loadcell is uncalibrated uV/V at drift level; '
                      'corr with grip flips sign between episodes '
                      '(DATASET_MANUAL 6.2)',
    'vocabulary_defined_in': 'local_validation/force_language_labeling.py',
    'vocabulary_source_session': 'phantom_force_A01 (loaded, ratio-valid)',
}

# Per record: the declaration and the number that bounds it, without the prose.
# The full slot, with its basis and the join to the loaded-session vocabulary,
# goes once into the episode summary and the report.
FORCE_SLOT_COMPACT = {k: FORCE_SLOT[k] for k in
                      ('state', 'measured', 'detection_limit_mN_1s_average')}

FORCE_SENTENCE_EN = ('No tissue load is measurable on either instrument in this '
                     'session; the detection limit is 0.2 mN at a one-second average')
FORCE_SENTENCE_KO = ('이 세션에서는 두 기구 모두 측정 가능한 조직 하중이 없다. '
                     '검출 한계는 1초 평균 0.2 mN이다')

# Utterances are built as whole sentences per axis, not as one chained clause:
# the grasper, the cutter, the coordination and the force declaration each stand
# on their own so a sentence can be dropped (an invalid cutter tick) without
# leaving the rest ungrammatical.
JAW_EN = {
    'jaw_open': 'the grasper jaws are open',
    'jaw_part_closed': 'the grasper jaws are part way closed',
    'jaw_closed': 'the grasper jaws are closed',
    'jaw_clamped_saturated': 'the grasper jaws are fully clamped, at the top of the grip '
                             'sensor range where the angle can no longer be told apart',
}
JAW_MOVE_EN = {'held': '', 'closing': ' and still closing', 'opening': ' and opening'}
# At saturation "still closing" would contradict the clause before it, which has
# just said the angle is no longer resolvable. The rate is real -- the jaw closed
# across the window -- so the wording changes, not the structured label.
JAW_MOVE_EN_SAT = {'held': '', 'closing': ', having just closed fully',
                   'opening': ' and opening'}
JAW_KO = {
    'jaw_open': '겸자가 열린 상태',
    'jaw_part_closed': '겸자가 반쯤 닫힌 상태',
    'jaw_closed': '겸자가 닫힌 상태',
    'jaw_clamped_saturated': '겸자가 완전히 압착된 상태(파지율 센서가 포화해 그 이상은 구분되지 않음)',
}
JAW_MOVE_KO = {'held': '이다', 'closing': '에서 더 닫히는 중이다', 'opening': '에서 열리는 중이다'}
JAW_MOVE_KO_SAT = {'held': '이다', 'closing': '로 방금 완전히 닫혔다',
                   'opening': '에서 열리는 중이다'}

SPEED_EN = {'tip_still': 'the tip is holding still', 'tip_slow': 'the tip is moving slowly',
            'tip_moderate': 'the tip is moving at a moderate speed',
            'tip_fast': 'the tip is moving quickly'}
SPEED_KO = {'tip_still': '팁은 정지한 채', 'tip_slow': '팁은 느리게 움직이며',
            'tip_moderate': '팁은 보통 속도로 움직이며', 'tip_fast': '팁은 빠르게 움직이며'}
DEPTH_EN = {'advancing': 'advancing into the trocar', 'withdrawing': 'withdrawing',
            'held': 'holding its insertion depth'}
DEPTH_KO = {'advancing': '트로카 안으로 전진한다', 'withdrawing': '후퇴한다',
            'held': '삽입 깊이를 유지한다'}

TOF_EN = {'cutter_near': 'the cutter is close in', 'cutter_mid': 'the cutter is at mid range',
          'cutter_far': 'the cutter is far out'}
TOF_MOVE_EN = {'held': '', 'approaching': ' and closing in', 'receding': ' and backing off'}
TOF_KO = {'cutter_near': '커터는 근접 거리에', 'cutter_mid': '커터는 중간 거리에',
          'cutter_far': '커터는 원거리에'}
TOF_MOVE_KO = {'held': ' 있다', 'approaching': ' 있으며 접근하는 중이다',
               'receding': ' 있으며 멀어지는 중이다'}
NO_CUTTER_EN = ('the cutter pose here is interpolated across a dropout, so it is not '
                'labelled')
NO_CUTTER_KO = '커터 자세는 이 구간이 결손 보간이라 라벨하지 않는다'

COORD_EN = {
    'both_active': 'both hands are working',
    'grasper_leads': 'the grasper is doing the moving',
    'cutter_leads': 'the cutter is doing the moving',
    'both_still': 'neither hand is moving',
}
COORD_KO = {
    'both_active': '양손이 모두 움직이고 있다', 'grasper_leads': '그리퍼가 움직임을 주도한다',
    'cutter_leads': '커터가 움직임을 주도한다', 'both_still': '양손 모두 정지해 있다',
}


def band_name(bands, value: float) -> str:
    edges = [b[0] for b in bands]
    return bands[int(np.searchsorted(edges, value, side='right') - 1)][1]


def centred_rate(y: np.ndarray, t: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """d y / d t over DIFF_WINDOW_S, NaN wherever either end is not valid.

    A plain gradient would silently interpolate across the BLE gaps and the NaN
    head and tail that the merge leaves behind, so both ends of every window are
    checked against the stream's own validity mask.
    """
    half = max(1, int(round(DIFF_WINDOW_S / float(np.median(np.diff(t))) / 2)))
    out = np.full(len(t), np.nan)
    for i in range(len(t)):
        a, b = max(0, i - half), min(len(t) - 1, i + half)
        if b > a and ok[a] and ok[b]:
            out[i] = (y[b] - y[a]) / (t[b] - t[a])
    return out


def centred_speed(p: np.ndarray, t: np.ndarray, ok: np.ndarray) -> np.ndarray:
    half = max(1, int(round(DIFF_WINDOW_S / float(np.median(np.diff(t))) / 2)))
    out = np.full(len(t), np.nan)
    for i in range(len(t)):
        a, b = max(0, i - half), min(len(t) - 1, i + half)
        if b > a and ok[a] and ok[b]:
            out[i] = float(np.linalg.norm(p[b] - p[a]) / (t[b] - t[a]))
    return out


def read_episode(path: Path) -> dict:
    with h5py.File(path) as f:
        m = f[MCU_BOARD]
        d = dict(
            episode=path.stem.replace('episode_', '').replace('_synced', ''),
            t=f['timestamp'][()],
            dist=m['dist'][()], grip=m['grip'][()],
            pos=m['rcm_position'][()], quat_rel=m['rcm_quat_rel'][()],
            mcu_valid=m['valid'][()],
            tof=f['imu/tof_mm'][()], gyro=f['imu/gyro_dps'][()],
            cutter_quat=f['imu/quat_wxyz'][()], imu_valid=f['imu/valid'][()],
            cam_frame=f['camera/frame_index'][()],
            cam_master=f['camera/master_index'][()],
            ni_forces=f['ni/forces'][()],
            imu_valid_pct=float(f.attrs['imu_valid_pct']),
            mcu_valid_pct=float(f.attrs[f'mcu_{MCU_BOARD.split("/")[-1]}_valid_pct']),
            lag_source=str(f.attrs['lag_source']),
        )
    d['mcu_ok'] = d['mcu_valid'] & np.all(np.isfinite(d['pos']), 1) & np.isfinite(d['dist'])
    d['imu_ok'] = d['imu_valid'] & np.isfinite(d['tof'])
    return d


def frame_for_master(ep: dict):
    """Nearest mp4 frame for a master index, from `camera/master_index`."""
    master, frames = ep['cam_master'], ep['cam_frame']
    return lambda i: int(frames[int(np.argmin(np.abs(master - i)))])


def session_of(episode: str) -> str:
    return 'A' if episode.split('_')[-1] in SESSION_A else 'B'


def depth_bands_per_session(eps: list[dict]) -> dict[str, tuple[float, float]]:
    """Depth is split at the terciles of each session's own range of `dist`.

    `rcm_qref` is re-zeroed between the sessions, so a shallow / deep word fitted
    on one does not transfer to the other. The rate of change does transfer and is
    labelled globally.
    """
    out = {}
    for s in ('A', 'B'):
        v = np.concatenate([e['dist'][e['mcu_ok']] for e in eps
                            if session_of(e['episode']) == s] or [np.zeros(1)])
        out[s] = (float(np.percentile(v, 33)), float(np.percentile(v, 67)))
    return out


def depth_word(dist: float, cuts: tuple[float, float]) -> tuple[str, str]:
    if dist < cuts[0]:
        return 'depth_shallow', '얕게 삽입된'
    if dist < cuts[1]:
        return 'depth_mid', '중간 깊이로 삽입된'
    return 'depth_deep', '깊게 삽입된'


def compose(jaw: str, jaw_move: str, depth_move: str, speed: str,
            tof: str | None, tof_move: str | None, coord: str) -> dict:
    sat = jaw == 'jaw_clamped_saturated'
    en = [JAW_EN[jaw] + (JAW_MOVE_EN_SAT if sat else JAW_MOVE_EN)[jaw_move],
          SPEED_EN[speed] + ', ' + DEPTH_EN[depth_move],
          (TOF_EN[tof] + TOF_MOVE_EN[tof_move or 'held']) if tof else NO_CUTTER_EN,
          COORD_EN[coord], FORCE_SENTENCE_EN]
    ko = [JAW_KO[jaw] + (JAW_MOVE_KO_SAT if sat else JAW_MOVE_KO)[jaw_move],
          SPEED_KO[speed] + ' ' + DEPTH_KO[depth_move],
          (TOF_KO[tof] + TOF_MOVE_KO[tof_move or 'held']) if tof else NO_CUTTER_KO,
          COORD_KO[coord], FORCE_SENTENCE_KO]
    return {
        'utterance_en': ' '.join(x[0].upper() + x[1:] + '.' for x in en),
        'utterance_ko': ' '.join(x + '.' for x in ko),
    }


def label_episode(ep: dict, cuts: tuple[float, float]) -> tuple[list[dict], dict]:
    t, ok, iok = ep['t'], ep['mcu_ok'], ep['imu_ok']
    speed = centred_speed(ep['pos'], t, ok)
    d_dist = centred_rate(ep['dist'], t, ok)
    d_grip = centred_rate(ep['grip'], t, ok)
    d_tof = centred_rate(ep['tof'], t, iok)
    spin = np.full(len(t), np.nan)
    spin[iok] = np.linalg.norm(ep['gyro'][iok], axis=1)
    to_frame = frame_for_master(ep)
    ble_gap = ep['episode'].split('_')[-1] in BLE_GAP_EPISODES

    step = max(1, int(round(1.0 / LABEL_HZ / float(np.median(np.diff(t))))))
    recs, counts = [], {'jaw': {}, 'speed': {}, 'coord': {}, 'cutter': {}}
    for i in range(0, len(t), step):
        if not ok[i] or not np.isfinite(speed[i]):
            continue
        nxt = min(len(t) - 1, i + step)
        jaw = band_name(JAW_BANDS, float(ep['grip'][i]))
        jaw_move = ('closing' if d_grip[i] > JAW_RATE else
                    'opening' if d_grip[i] < -JAW_RATE else 'held')
        depth_move = ('advancing' if d_dist[i] > DEPTH_RATE else
                      'withdrawing' if d_dist[i] < -DEPTH_RATE else 'held')
        sp = band_name(SPEED_BANDS, float(speed[i]))
        tof_b = band_name(TOF_BANDS, float(ep['tof'][i])) if iok[i] else None
        tof_move = (('approaching' if d_tof[i] < -TOF_RATE else
                     'receding' if d_tof[i] > TOF_RATE else 'held')
                    if iok[i] and np.isfinite(d_tof[i]) else None)
        hand = sp != 'tip_still'
        # The cutter carries no tip-position stream, only pose and ToF, so its
        # activity is "rotating fast OR changing range". Angular rate alone would
        # call a cutter that translates without turning idle.
        cut = bool(iok[i] and ((np.isfinite(spin[i]) and spin[i] > CUTTER_SPIN)
                               or (np.isfinite(d_tof[i]) and abs(d_tof[i]) > TOF_RATE)))
        coord = ('both_active' if hand and cut else 'grasper_leads' if hand else
                 'cutter_leads' if cut else 'both_still')

        rec = {
            'episode': ep['episode'], 'session': session_of(ep['episode']),
            't_unix': float(t[i]), 't_rel_s': round(float(t[i] - t[0]), 3),
            'master_index': i, 'frame_index': to_frame(i),
            'video': str((VIDEO / f'episode_cam0_{ep["episode"]}.mp4').relative_to(ROOT)),
            'state': {
                'insertion_depth_mm': round(float(ep['dist'][i]), 3),
                'grip_pct': round(float(ep['grip'][i]), 2),
                'grip_saturated': bool(ep['grip'][i] >= 99.9),
                'tip_xyz_mm_session_zero': [round(float(x), 3) for x in ep['pos'][i]],
                'tip_quat_rel_xyzw': [round(float(x), 5) for x in ep['quat_rel'][i]],
                'cutter_quat_wxyz': ([round(float(x), 5) for x in ep['cutter_quat'][i]]
                                     if iok[i] else None),
                'cutter_tof_mm': round(float(ep['tof'][i]), 2) if iok[i] else None,
                'cutter_spin_dps': round(float(spin[i]), 2) if np.isfinite(spin[i]) else None,
            },
            'action': {
                'd_tip_xyz_mm': [round(float(x), 3) for x in (ep['pos'][nxt] - ep['pos'][i])]
                                if ok[nxt] else None,
                'd_grip_pct': round(float(ep['grip'][nxt] - ep['grip'][i]), 2) if ok[nxt] else None,
                'horizon_s': round(float(t[nxt] - t[i]), 3),
            },
            'action_chunk': [
                {'d_tip_xyz_mm': [round(float(x), 3) for x in
                                  (ep['pos'][min(len(t) - 1, i + step * (k + 1))] -
                                   ep['pos'][min(len(t) - 1, i + step * k)])],
                 'd_grip_pct': round(float(ep['grip'][min(len(t) - 1, i + step * (k + 1))] -
                                           ep['grip'][min(len(t) - 1, i + step * k)]), 2)}
                for k in range(CHUNK)
                if ok[min(len(t) - 1, i + step * (k + 1))]
            ],
            'force': FORCE_SLOT_COMPACT,
            'labels': {
                'jaw': jaw, 'jaw_motion': jaw_move,
                'depth': depth_word(float(ep['dist'][i]), cuts)[0],
                'depth_motion': depth_move, 'tip_motion': sp,
                'cutter_proximity': tof_b, 'cutter_motion': tof_move,
                'coordination': coord,
            },
            'validity': {
                'mcu': True, 'cutter': bool(iok[i]),
                'in_ble_gap_episode': ble_gap,
                'depth_band_is_session_relative': True,
            },
        }
        rec.update(compose(jaw, jaw_move, depth_move, sp, tof_b, tof_move, coord))
        recs.append(rec)
        for key, val in (('jaw', jaw), ('speed', sp), ('coord', coord),
                         ('cutter', tof_b or 'unlabelled')):
            counts[key][val] = counts[key].get(val, 0) + 1

    # DATASET_MANUAL 6.1 quotes the span AFTER a 0.5 s moving average (1.4-2.2 mN
    # per axis). Reproducing it on the same statistic is what makes the no-load
    # claim checkable here; the raw span is several times larger and is noise.
    k = np.ones(50) / 50
    live = np.column_stack([np.convolve(ep['ni_forces'][:, c], k, 'valid')
                            for c in (1, 3)])
    summary = {
        'episode': ep['episode'], 'session': session_of(ep['episode']),
        'duration_s': round(float(t[-1] - t[0]), 3),
        'n_labels': len(recs), 'n_frames': int(len(ep['cam_frame'])),
        'video': str((VIDEO / f'episode_cam0_{ep["episode"]}.mp4').relative_to(ROOT)),
        'imu_valid_pct': round(ep['imu_valid_pct'], 2),
        'mcu_valid_pct': round(ep['mcu_valid_pct'], 2),
        'in_ble_gap_episode': ble_gap,
        'depth_range_mm': [round(float(ep['dist'][ep['mcu_ok']].min()), 2),
                           round(float(ep['dist'][ep['mcu_ok']].max()), 2)],
        'grip_saturated_pct': round(100 * float(np.mean(ep['grip'][ep['mcu_ok']] >= 99.9)), 2),
        'live_force_span_mN': {
            'bending_trocar_x': round(float(live[:, 0].max() - live[:, 0].min()) * 1000, 3),
            'bending_trocar_y': round(float(live[:, 1].max() - live[:, 1].min()) * 1000, 3),
            'window': '0.5 s moving average, as DATASET_MANUAL 6.1 reports it'},
        'force': FORCE_SLOT,
        'label_counts': counts,
    }
    return recs, summary


def check_invariants(report: dict, episodes: list[dict], recs_total: int) -> None:
    """Refuse to write a report that contradicts the session's own limits."""
    if report['force_axis']['measured']:
        raise ValueError('this session has no measured force')
    if recs_total != sum(e['n_labels'] for e in episodes):
        raise ValueError('label count does not match the per-episode totals')
    # The manual measured 1.4-2.0 mN on x and 1.6-2.2 mN on y over these 13
    # episodes. Reproducing that range is the check that the no-load claim and the
    # channel indices are both right; a wider span means something has changed and
    # the declared force axis must be revisited before any label is written.
    worst = max(max(e['live_force_span_mN']['bending_trocar_x'],
                    e['live_force_span_mN']['bending_trocar_y']) for e in episodes)
    if worst > 3.0:
        raise ValueError(f'live force span {worst:.2f} mN exceeds the 1.4-2.2 mN '
                         'DATASET_MANUAL 6.1 measured; the no-load declaration and the '
                         'channel indices both need re-checking before labelling')
    sessions = {e['session'] for e in episodes}
    if not sessions <= {'A', 'B'}:
        raise ValueError('unexpected session grouping')


def main() -> None:
    eps = [read_episode(p) for p in sorted(SYNCED.glob('*_synced.h5'))]
    if not eps:
        raise ValueError(f'no synced episodes under {SYNCED}')
    cuts = depth_bands_per_session(eps)

    OUT.mkdir(parents=True, exist_ok=True)
    summaries, total = [], 0
    pooled = {'jaw': {}, 'speed': {}, 'coord': {}, 'cutter': {}}
    for ep in eps:
        recs, summary = label_episode(ep, cuts[session_of(ep['episode'])])
        (OUT / f'{ep["episode"]}.jsonl').write_text(
            ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in recs))
        summaries.append(summary)
        total += len(recs)
        for key, counts in summary['label_counts'].items():
            for k, v in counts.items():
                pooled[key][k] = pooled[key].get(k, 0) + v
    (OUT / 'episodes.jsonl').write_text(
        ''.join(json.dumps(s, ensure_ascii=False) + '\n' for s in summaries))

    report = {
        'scope': 'VLA state-action-language labels for surgilog_261021: endoscope video, '
                 'both instruments kinematics, and a declared no-load force axis',
        'force_measured': False,
        'force_axis': FORCE_SLOT,
        'why_this_session': 'the only session carrying video, both instruments kinematics '
                            'and one shared absolute clock. 6973 frames with zero drops, '
                            'camera/master_index gives a 1:1 frame-to-sample key, MCU CRC '
                            'and sequence loss are both zero. The A01 force session has no '
                            'MCU at all, so no tip pose, insertion depth or grip percentage, '
                            'and its force trace cannot be tied to a state or an action.',
        'force_axis_is_declared_not_estimated':
            'Every record carries force.state = no_load with the measured detection limit. '
            'That is the manual one sanctioned use of this session force channel. No '
            'magnitude, trend or slip word is emitted: z-scoring this channel yields a '
            'convincing +-3 sigma trace indistinguishable from the physically disconnected '
            'channels, so a post-processed force label here would be manufactured, not '
            'recovered (DATASET_MANUAL 6.1 c).',
        'label_axes': {
            'jaw': {'from': 'MCU grip percent',
                    'bands': [{'from_pct': b[0], 'name': b[1], 'name_ko': b[2]}
                              for b in JAW_BANDS],
                    'motion_threshold_pct_per_s': JAW_RATE,
                    'saturation': 'the top band is the hall-to-percent clip; 36.2 percent '
                                  'of valid samples read exactly 100.0 and the jaw angle '
                                  'inside it is unknown. Finer angle needs MCU hall/adc '
                                  '(session range 1675-2456).'},
            'depth': {'from': 'MCU dist, trocar insertion depth in mm',
                      'bands': 'terciles of each session own range',
                      'session_cuts_mm': {k: [round(v[0], 2), round(v[1], 2)]
                                          for k, v in cuts.items()},
                      'motion_threshold_mm_per_s': DEPTH_RATE,
                      'why_per_session': 'rcm_qref is re-zeroed between session A and B '
                                         '(DATASET_MANUAL 6.6), so a shallow/deep word '
                                         'fitted on one does not transfer. The rate of '
                                         'change does transfer and is labelled globally.'},
            'tip_motion': {'from': 'speed of MCU rcm_position',
                           'bands': [{'from_mm_s': b[0], 'name': b[1], 'name_ko': b[2]}
                                     for b in SPEED_BANDS],
                           'measured_deciles_mm_s': {'p10': 3.03, 'p50': 10.56, 'p90': 78.76},
                           'cross_session_valid': 'yes -- a difference, so the per-session '
                                                  'zero cancels'},
            'cutter_proximity': {'from': 'SurgiTag tof_mm',
                                 'bands': [{'from_mm': b[0], 'name': b[1], 'name_ko': b[2]}
                                           for b in TOF_BANDS],
                                 'motion_threshold_mm_per_s': TOF_RATE,
                                 'naming': 'proximity only. The manual allows ToF, dist and '
                                           'grip as contact proxies on the explicit condition '
                                           'that they are not named force.'},
            'coordination': {'from': 'grasper tip speed, and cutter angular rate or '
                                     'range rate -- the cutter has no tip-position '
                                     'stream, so rotation alone would call a '
                                     'translating cutter idle',
                             'cutter_spin_threshold_dps': CUTTER_SPIN,
                             'cutter_range_rate_threshold_mm_per_s': TOF_RATE,
                             'classes': sorted(COORD_EN)},
        },
        'state_action': {
            'tick_hz': LABEL_HZ, 'diff_window_s': DIFF_WINDOW_S,
            'action': 'delta tip position (mm) and delta grip (percent) over the next tick',
            'action_chunk_len': CHUNK,
            'frame_key': 'camera/master_index, monotone and duplicate-free in all 13 episodes',
            'state_fields': ['insertion_depth_mm', 'grip_pct', 'tip_xyz_mm_session_zero',
                             'tip_quat_rel_xyzw', 'cutter_quat_wxyz', 'cutter_tof_mm',
                             'cutter_spin_dps'],
        },
        'validity_handling': {
            'rates': 'every rate and speed requires BOTH ends of its window to pass the '
                     'stream own validity mask, so no value is computed across a dropout '
                     'or across the NaN head and tail the merge leaves',
            'ble_gap_episodes': sorted(BLE_GAP_EPISODES),
            'ble_gap_effect': '350-380 ms gaps filled by linear interpolation; ticks failing '
                              'imu/valid carry no cutter label and say so in the utterance',
            'clean_episodes': ['171006', '171033', '175616', '175809', '175828', '175846',
                               '175906', '175925'],
            'uncorrected_lag': sorted({e['lag_source'] for e in eps}),
            'lag_effect': 'no tap test, so cross-modal timing is good to the clock residual '
                          '(0.2-0.9 ms MCU, 2.2-4.2 ms SurgiTag) but not to a measured '
                          'sensor-to-sensor lag; do not read causality at tens of ms',
        },
        'not_labelled': [
            'force magnitude, trend, or slip: no load in this session on either instrument',
            'subtask or phase: this dataset carries no such annotation, so unlike the A01 '
            'labels nothing in these records is operator intent',
            'session A to B tip position comparison: different RCM zero',
            'jaw angle above the grip saturation point',
        ],
        'coverage': {
            'episodes': len(summaries), 'labels': total,
            'frames': int(sum(s['n_frames'] for s in summaries)),
            'duration_s': round(sum(s['duration_s'] for s in summaries), 2),
            'label_distribution': pooled,
            'max_live_force_span_mN': round(max(
                max(s['live_force_span_mN']['bending_trocar_x'],
                    s['live_force_span_mN']['bending_trocar_y']) for s in summaries), 3),
            'manual_reported_span_mN': {'bending_trocar_x': [1.4, 2.0],
                                        'bending_trocar_y': [1.6, 2.2]},
        },
        'remaining': [
            'A loaded session on THIS rig is the one missing piece: it would fill the force '
            'bands defined in local_validation/force_language_labeling.py against the state '
            'and action labelled here. Until then the force axis trains only the no-load class.',
            'Protocol section 7 of the dataset manual gives the calibration the cutter '
            'loadcell needs (software half bridge, common-mode projection removal, then an '
            'affine decoupling fitted on LOADED data -- fitting it on this session '
            'overfits drift and makes leakage worse).',
            'A tap test would fix the uncorrected sensor-to-sensor lag and allow cross-modal '
            'reasoning at tens of ms.',
        ],
        'outputs': {'per_tick': str((OUT / '<episode>.jsonl').relative_to(ROOT)),
                    'per_episode': str((OUT / 'episodes.jsonl').relative_to(ROOT))},
    }
    check_invariants(report, summaries, total)
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')

    print(f'{total} labels over {len(summaries)} episodes, '
          f'{report["coverage"]["frames"]} frames, '
          f'{report["coverage"]["duration_s"]} s')
    for axis, counts in pooled.items():
        top = ', '.join(f'{k} {v}' for k, v in sorted(counts.items(), key=lambda x: -x[1]))
        print(f'  {axis:8} {top}')
    print(f'  force axis: {FORCE_SLOT["state"]} (declared; '
          f'max live span {report["coverage"]["max_live_force_span_mN"]} mN)')
    print('report ->', REPORT.relative_to(ROOT))


if __name__ == '__main__':
    main()
