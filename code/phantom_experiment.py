#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
팬텀 힘 스펙 측정 — 한 번에 전 Phase 를 돌리는 실험 GUI
========================================================

`PHANTOM_SPEC_PROTOCOL.md` 의 Phase 1~3 을 **하나의 연속 녹화**로 수집한다.
시편 1개를 물려 놓고 단계만 바꿔 가며 끝까지 진행하면 되고, 각 샘플에 단계(stage)
와 시행(trial) 번호가 함께 저장되므로 사후 분석에서 구간을 자동으로 갈라낼 수 있다.

    시편 1개당:  파지 개시 → 거상(4높이) → 이동(4방향) → 미끄러짐(젤 전/후) → 파단
                 ──────── Phase 1 ────────   ── Phase 2 ──   Phase 3

데이터 경로는 visualize_forces.py 를 그대로 재사용한다(소켓 → 블록평균 → calib 6x6
디커플링 → tare → LPF). 따라서 화면 값 == h5 의 forces 값이고, dataset_recorder.py
로 녹화한 기존 데이터와 완전히 같은 의미를 갖는다.

h5 스키마 (dataset_recorder.py + 라벨 2채널)
  timestamp (N,)    f8  PC 절대시각 (Unix epoch, UTC 초)
  forces    (N,6)   f4  화면 표시값과 동일 (calib+tare+LPF 후), 컬럼순서 = force_keys
  raw       (N,6)   f4  채널 원전압(V), 컬럼순서 = raw_keys
  stage     (N,)    i2  단계 코드 (attrs['stage_names'] 의 인덱스)
  trial     (N,)    i4  시행 번호 (0 = 시행 밖)

시행 CSV (분석 입력)
  각 시행이 끝날 때 한 줄. peak / mean / p95 / 방향 / 파단 전 dF/dt / 급락비 등.
  파단·미끄러짐 시행은 라디오로 **파단 부위를 사후 분류**해 같은 줄에 남긴다
  (grasper 하나로는 부위를 겨냥할 수 없으므로 — 프로토콜 §5.2).

조작키 (그래프 창에 포커스)
  [space] 시행 시작/종료      [n] 다음 단계        [b] 이전 단계
  [r]     녹화 시작/정지      [c] 영점(tare)
  [1..4]  마지막 시행 분류    [u] 마지막 시행 CSV 에서 취소
  [q]     안전 종료 (녹화 중이면 저장 후 종료)

Run:
    conda activate surgitag
    # 터미널 1 — 드라이버가 떠 있어야 한다
    #   cd linux_daqmxbase_realtime && ./run_driver.sh --device Dev1 --channels 8 --rate 1000
    # 터미널 2
    ./run_phantom_experiment.sh --specimen A01
    ./run_phantom_experiment.sh --simulate          # 하드웨어 없이 UI 확인
"""

import argparse
import csv
import os
import re
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import RadioButtons, Button, TextBox
import h5py

# 콘솔이 UTF-8 이 아닌 환경(예: 윈도우 cp949)에서도 한글/기호 출력이 깨지지 않도록.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from record_signal import RecordSignal
from phantom_camera import PhantomCamera


# ---------------------------------------------------------------- 한글 폰트
# matplotlib 기본(DejaVu Sans)에는 한글 글리프가 없어 전부 □ 로 나온다. HUD 는
# family="monospace" 라 더 확실히 깨진다 — 숫자 열을 맞춰야 하므로 '한글 고정폭'
# 폰트를 따로 고른다. 설치된 것 중 먼저 잡히는 것을 쓰고, 없으면 조용히 기본값.
_KO_SANS = ["NanumGothic", "NanumBarunGothic", "NanumSquareRound", "NanumSquare",
            "Noto Sans CJK KR", "Noto Sans CJK JP", "Malgun Gothic", "AppleGothic"]
_KO_MONO = ["NanumGothicCoding", "D2Coding", "Noto Sans Mono CJK KR",
            "Noto Sans Mono CJK JP"]


def _pick_font(cands):
    import matplotlib.font_manager as fm
    have = {f.name for f in fm.fontManager.ttflist}
    for c in cands:
        if c in have:
            return c
    return None


def setup_korean_font():
    """(일반 폰트, HUD 고정폭 폰트) 를 정하고 rcParams 에 반영한다."""
    import matplotlib
    sans = _pick_font(_KO_SANS)
    mono = _pick_font(_KO_MONO)
    if sans:
        matplotlib.rcParams["font.family"] = "sans-serif"
        matplotlib.rcParams["font.sans-serif"] = (
            [sans] + list(matplotlib.rcParams["font.sans-serif"]))
    else:
        print("[font] ! 한글 폰트를 못 찾았다 — 라벨이 □ 로 나온다.")
        print("[font]   sudo apt install fonts-nanum fonts-nanum-coding  후 "
              "matplotlib 폰트 캐시를 지울 것 (~/.cache/matplotlib)")
    # 한글 폰트는 유니코드 마이너스(−) 글리프가 없는 경우가 많다 → ASCII 하이픈으로
    matplotlib.rcParams["axes.unicode_minus"] = False
    if mono is None:
        mono = sans            # 고정폭이 없으면 정렬을 포기하고 한글 가독성을 택한다
    if sans or mono:
        print("[font] 일반=%s  HUD=%s" % (sans or "기본", mono or "기본"))
    return sans, mono

from wiring import (FORCES, RAW_CHANNELS, RAW_KEYS, UNIT, MIN_CHANNELS,
                    QUARTER_AXES)
from visualize_forces import (SocketSource, SimSource, compute_forces,
                              load_calibration, alpha_from_cutoff, onepole_lpf)

FORCE_KEYS = [k for k, _, _, _ in FORCES]
IDX_BGX = FORCE_KEYS.index("bending_gripper_x")
IDX_BGY = FORCE_KEYS.index("bending_gripper_y")
IDX_GRIP = FORCE_KEYS.index("gripping")
IDX_TOR = FORCE_KEYS.index("torsion")

CHUNK = 256
FLUSH_INTERVAL_S = 1.0

# --- 프로토콜 단계 (PHANTOM_SPEC_PROTOCOL.md 의 절 번호) ----------------------
# (코드명, 화면 표시, 프로토콜 절, 권장 시행 수)
STAGES = [
    ("idle",       "— 대기 —",            "—",    0),
    ("grasp",      "파지 개시",           "§3.2", 10),
    ("lift_05",    "거상 5 mm",           "§3.3", 5),
    ("lift_10",    "거상 10 mm",          "§3.3", 5),
    ("lift_15",    "거상 15 mm",          "§3.3", 5),
    ("lift_20",    "거상 20 mm",          "§3.3", 5),
    ("move_L",     "이동시도 좌",           "§3.4", 5),
    ("move_R",     "이동시도 우",           "§3.4", 5),
    ("move_F",     "이동시도 전",           "§3.4", 5),
    ("move_B",     "이동시도 후",           "§3.4", 5),
    ("slip_dry",   "미끄러짐 (젤 없음)",  "§4",   10),
    ("slip_gel",   "미끄러짐 (젤 도포)",  "§4",   10),
    ("rupture",    "파단 견인",           "§5",   8),
]
STAGE_NAMES = [s[0] for s in STAGES]
STAGE_LABELS = ["%s  %s" % (s[1], s[2]) for s in STAGES]
STAGE_TARGET = {s[0]: s[3] for s in STAGES}

# 단계별 '사용자가 할 동작' — PHANTOM_SPEC_PROTOCOL.md §3~5 를 그대로 옮긴 것.
# GUI 우측 패널과 단계 전환 시 콘솔에 띄워, 손을 보면서 순서를 틀리지 않게 한다.
STAGE_ACTION = {
    "idle": [
        "단계를 먼저 고르세요 ([n]/[b]).",
        "idle 에서는 [space] 를 눌러도 시행이 시작되지 않습니다.",
    ],
    "grasp": [
        "1. 파지 지점은 충수 원위 1/3 — 테이프 표식, 매번 동일하게.",
        "2. [space] 시작 → 무접촉에서 출발.",
        "3. 접촉 → 완전 파지까지 '일정 속도로 1회' 문다.",
        "4. 완전 파지 상태를 1초 유지 → [space] 종료.",
        "※ 거상 전이라 gripping 축 오염이 가장 적은 구간이다.",
    ],
    "move": [
        "1. 파지한 채 10 mm 거상 — 시도 내내 이 높이를 유지한다.",
        "2. [space] 시작 → %s 방향으로 '움직이려는 시도'.",
        "3. 조직이 버티면 거기까지. 억지로 끌지 말 것.",
        "4. 힘을 풀고 복귀 → [space] 종료.",
        "※ 재는 것은 '간 거리' 가 아니라 '그 방향으로 움직이려 할 때 드는 힘' 이다.",
        "※ 방향마다 갈 수 있는 거리가 다르다 — 거리를 맞추려 하지 말 것.",
        "※ 시도 강도를 방향 간에 비슷하게 유지하는 것이 유일한 통제 변수다.",
        "※ 특이사항은 아래 상자에 메모 (거리를 쟀다면 '12' 처럼 mm 로).",
    ],
    "slip_dry": [
        "1. 젤 없이 파지 (파지율을 매번 같게).",
        "2. [space] 시작 → 미끄러질 때까지 수직으로 견인.",
        "3. F_tip 이 급락하면 미끄러진 것 — 바로 [space] 종료.",
        "※ 판정은 gripping 이 아니라 F_tip 급락으로 한다.",
        "※ 보고값은 급락 직전의 F_tip peak.",
    ],
    "slip_gel": [
        "!! 이번 세션 범위가 아니다 (젤 측정 안 함).",
        "   [b] 로 되돌아가세요.",
    ],
    "rupture": [
        "!! 이번 세션 범위가 아니다 — 파괴 단계.",
        "   시편이 소모된다. [b] 로 되돌아가세요.",
    ],
}
_MOVE_DIR = {"move_L": "좌", "move_R": "우", "move_F": "전", "move_B": "후"}

# 이번 세션에서 안 하는 단계 — main() 이 --skip 으로 채운다.
SKIP = set()


def stage_action(name):
    """단계 코드 -> 행동 지시 줄 목록."""
    if name in SKIP:
        return ["!! 이번 세션 범위가 아니다 (--skip).",
                "   [b]/[n] 으로 다른 단계로 가세요."]
    if name in STAGE_ACTION:
        return STAGE_ACTION[name]
    if name.startswith("lift_"):
        mm = int(name.split("_")[1])
        return [
            "1. 파지를 유지한 채 [space] 시작.",
            "2. 수직으로 %d mm 거상 (베이스 시각 표식 기준)." % mm,
            "3. 그 높이에서 '3초 유지' — 이게 읽는 값이다.",
            "4. 하강 후 [space] 종료.",
            "※ 유지 구간 마지막 1초의 F_tip 평균을 쓴다. 짧으면 값이 안 나온다.",
        ]
    if name.startswith("move_"):
        return [l % _MOVE_DIR.get(name, "") if "%s" in l else l
                for l in STAGE_ACTION["move"]]
    return ["(행동 지시 없음)"]

# 파단 부위 사후 분류 (§5.2 의 ① ② ③ + 중도절단)
CLASSES = [
    ("unset",         "(미분류)"),
    ("bond_app_cec",  "① Sil-Poxy 충수–맹장"),
    ("appendix_base", "② 충수 기저부 파단"),
    ("meso",          "③ 간막 계통"),
    ("slip",          "미끄러짐 (중도절단)"),
]
CLASS_KEYS = [c[0] for c in CLASSES]
CLASS_LABELS = [c[1] for c in CLASSES]

CSV_FIELDS = [
    "trial", "specimen", "stage", "stage_section", "label",
    "start_iso", "start_unix", "end_unix", "duration_s", "n_samples",
    "peak_F_tip", "peak_unix", "theta_at_peak_deg",
    "mean_F_tip", "p95_F_tip",
    "dFdt_median_pre05s", "drop_ratio",
    "gripping_peak", "gripping_mean", "torsion_peak",
    "calib_path", "rate", "cutoff_hz", "travel_mm", "note",
]


# ---------------------------------------------------------------------------
# HDF5 확장 데이터셋 (dataset_recorder.py 와 같은 방식)
# ---------------------------------------------------------------------------
def _create_ds(h5, name, shape, dtype):
    return h5.create_dataset(
        name, shape=(0,) + shape, maxshape=(None,) + shape,
        dtype=dtype, chunks=(CHUNK,) + shape,
        compression="gzip", compression_opts=4,
    )


def _append_block(ds, block):
    block = np.asarray(block)
    n = block.shape[0]
    if n == 0:
        return
    ds.resize(ds.shape[0] + n, axis=0)
    ds[-n:] = block


def _trial_metrics(ts, ftip, grip, tor, theta=(), pre_window_s=0.5):
    """한 시행의 요약값. ts/ftip/grip/tor/theta 는 같은 길이의 1D 배열."""
    out = {k: "" for k in ("peak_F_tip", "peak_unix", "mean_F_tip", "p95_F_tip",
                           "theta_at_peak_deg", "dFdt_median_pre05s", "drop_ratio",
                           "gripping_peak", "gripping_mean", "torsion_peak")}
    if len(ftip) == 0:
        return out
    ts = np.asarray(ts, dtype=np.float64)
    ftip = np.asarray(ftip, dtype=np.float64)
    i_pk = int(np.argmax(ftip))
    peak = float(ftip[i_pk])
    out["peak_F_tip"] = "%.4f" % peak
    out["peak_unix"] = "%.6f" % ts[i_pk]
    out["mean_F_tip"] = "%.4f" % float(ftip.mean())
    out["p95_F_tip"] = "%.4f" % float(np.percentile(ftip, 95))
    if len(theta) > i_pk:
        out["theta_at_peak_deg"] = "%.1f" % float(theta[i_pk])
    if len(grip):
        out["gripping_peak"] = "%.4f" % float(np.max(np.abs(grip)))
        out["gripping_mean"] = "%.4f" % float(np.mean(grip))
    if len(tor):
        out["torsion_peak"] = "%.5f" % float(np.max(np.abs(tor)))

    # 파단 전 0.5 s 구간의 dF/dt 중앙값 — 견인 속도 대리값 (§5.4)
    pre = (ts >= ts[i_pk] - pre_window_s) & (ts <= ts[i_pk])
    if pre.sum() >= 3:
        d = np.diff(ftip[pre]) / np.maximum(np.diff(ts[pre]), 1e-9)
        out["dFdt_median_pre05s"] = "%.3f" % float(np.median(d))

    # peak 이후 급락비 — 1 에 가까우면 완전히 놓친 것(파단/미끄러짐), 작으면 유지
    after = ftip[i_pk:]
    if len(after) >= 2 and peak > 1e-9:
        out["drop_ratio"] = "%.3f" % float((peak - after.min()) / peak)
    return out


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="팬텀 힘 스펙 측정 실험 GUI (Phase 1~3 을 한 번에)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # --- 실험 식별 ---
    p.add_argument("--specimen", default=None,
                   help="시편 ID (예: A01). h5/CSV 파일명과 모든 행에 기록된다")
    p.add_argument("--out-root", default=None,
                   help="저장 루트 (기본: 이 스크립트 옆 phantom_data/)")
    # --- 데이터 경로 (visualize_forces.py 와 같은 의미) ---
    p.add_argument("--socket", default="/tmp/surgitag/daq.sock",
                   help="드라이버 소켓 경로")
    p.add_argument("--channels", type=int, default=8,
                   help="수집 채널 수 (wiring 매핑에 최소 %d 필요)" % MIN_CHANNELS)
    p.add_argument("--rate", type=float, default=100.0,
                   help="샘플레이트(Hz). 파단·미끄러짐 단계는 1000 권장")
    p.add_argument("--window", type=float, default=8.0, help="그래프 시간 윈도우(초)")
    p.add_argument("--calib", default=None,
                   help="디커플링 행렬 JSON. 미지정 시 스크립트 옆 calib_phantom.json → "
                        "calib.json 순으로 자동 사용")
    p.add_argument("--no-calib", action="store_true", help="raw 모드 강제")
    p.add_argument("--decimate", choices=["subsample", "average"], default="average",
                   help="드라이버 레이트 → rate 다운샘플 방식")
    p.add_argument("--cutoff", type=float, default=0.0,
                   help="전 축 저역통과 차단주파수(Hz). 0=끔. 조작 힘 단계는 5 권장")
    p.add_argument("--quarter-cutoff", type=float, default=None,
                   help="quarter 축(gripping·torsion)만 다른 차단주파수(Hz)")
    p.add_argument("--skip", default="slip_gel,rupture",
                   help="이번 세션에서 안 하는 단계(쉼표 구분). 진행표에서 '제외' 로 "
                        "표시되고 합계에서 빠지며, 들어가면 경고가 뜬다")
    p.add_argument("--record-all", action="store_true",
                   help="시행 사이 구간까지 전부 기록(예전 동작). 기본은 시행 구간만")
    p.add_argument("--pad-sec", type=float, default=0.5,
                   help="시행 종료 뒤 더 담을 시간(초). 블록이 0.1초라 정지 직전 peak 가 "
                        "잘리는 것을 막는다. 0 이면 끔")
    p.add_argument("--no-tare", action="store_true", help="시작 영점(tare) 끔")
    p.add_argument("--tare-sec", type=float, default=1.0, help="영점 평균 구간(초)")
    p.add_argument("--ymax", type=float, default=6.0, help="F_tip 축 최대(N)")
    p.add_argument("--camera", type=int, default=0,
                   help="엔도스코프 카메라 인덱스 (기본 0 = Medsurg). 못 열면 영상 없이 진행")
    p.add_argument("--no-camera", action="store_true",
                   help="영상 미리보기·녹화를 아예 끔")
    p.add_argument("--camera-wb", choices=["on", "off"], default="on",
                   help="소프트웨어 화이트밸런스 (이 카메라는 하드웨어 AWB 가 없다)")
    p.add_argument("--no-signal", action="store_true",
                   help="전역 녹화 신호(record_signal) 연동 끔")
    p.add_argument("--simulate", action="store_true", help="하드웨어 없이 UI 테스트")
    args = p.parse_args()

    rate = args.rate
    num_channels = args.channels
    if num_channels < MIN_CHANNELS:
        print("[error] 최소 %d채널 필요 (요청 %d)" % (MIN_CHANNELS, num_channels),
              file=sys.stderr)
        sys.exit(1)

    here = os.path.dirname(os.path.abspath(__file__))

    # --- 캘리브레이션: 팬텀 전용 행렬을 우선한다 (프로토콜 §7.1) ---
    calib_path = args.calib
    if calib_path is None and not args.no_calib:
        for cand in ("calib_phantom.json", "calib.json"):
            sib = os.path.join(here, cand)
            if os.path.exists(sib):
                calib_path = sib
                break
    if args.no_calib:
        calib_path = None
    if calib_path:
        forces_fn, _ = load_calibration(calib_path)
        print("[calib] 디커플링 행렬 적용: %s" % calib_path)
        if os.path.basename(calib_path) == "calib.json":
            print("[calib] [!] 자유 캔틸레버 기준 행렬이다. 접촉점 재캘리브레이션(부록 A-1) "
                  "전까지 N 은 상대값으로만 쓸 것")
    else:
        forces_fn = compute_forces
        print("[calib] raw 모드 — 값이 N 단위가 아님에 유의")

    chunk = max(1, int(round(rate / 10)))
    window_samples = max(chunk, int(round(args.window * rate)))

    # --- 데이터 소스 ---
    if args.simulate:
        print("[mode] simulation")
        source = SimSource(num_channels, rate, chunk)
        actual_rate = rate
    else:
        print("[mode] socket=%s channels=%d rate=%gHz" % (args.socket, num_channels, rate))
        try:
            source = SocketSource(args.socket, num_channels, rate, chunk,
                                  decim_mode=args.decimate)
        except Exception as e:
            print("소켓 연결 실패: %s" % e, file=sys.stderr)
            print("드라이버가 떠 있는지 확인: linux_daqmxbase_realtime/run_driver.sh",
                  file=sys.stderr)
            sys.exit(1)
        actual_rate = source.actual_rate
        print("  드라이버 %g Hz -> %s -> %.1f Hz"
              % (source._drv_rate, args.decimate, actual_rate))
    source.start()

    # --- 저장 경로 ---
    SKIP.clear()
    SKIP.update(x.strip() for x in args.skip.split(",") if x.strip())
    bad = SKIP - set(STAGE_NAMES)
    if bad:
        print("[skip] ! 모르는 단계 이름: %s" % ", ".join(sorted(bad)))
    if SKIP:
        print("[skip] 이번 세션 제외: %s" % ", ".join(
            s2[1] for s2 in STAGES if s2[0] in SKIP))

    save_dir = args.out_root or os.path.join(here, "phantom_data")
    os.makedirs(save_dir, exist_ok=True)
    spec = args.specimen or "nospec"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(save_dir, "phantom_%s_%s" % (spec, stamp))
    h5_path = base + ".h5"
    csv_path = base + "_trials.csv"
    print("📁 저장: %s" % save_dir)

    # --- 엔도스코프 (없거나 못 열면 영상 없이 계속한다) ---
    cam = None
    if not args.no_camera:
        c = PhantomCamera(args.camera, wb=(args.camera_wb == "on"))
        if c.open():
            c.start()
            cam = c
        else:
            print("[cam] ! 영상 없이 진행한다. 프로토콜 §5.5 는 분류 근거로 영상을 "
                  "요구하므로, 파단·미끄러짐 단계를 할 거면 카메라를 먼저 고칠 것")
    print("   h5  : %s" % os.path.basename(h5_path))
    print("   csv : %s" % os.path.basename(csv_path))

    # --- tare / LPF (visualize_forces 와 동일) ---
    tare_enabled = not args.no_tare
    tare_baseline = {k: None for k in FORCE_KEYS}
    tare_acc = {k: [] for k in FORCE_KEYS}
    tare_need = max(1, int(round(args.tare_sec * rate)))

    def _cutoff_for(k):
        if k in QUARTER_AXES and args.quarter_cutoff is not None:
            return args.quarter_cutoff
        return args.cutoff
    axis_alpha = {k: alpha_from_cutoff(_cutoff_for(k), rate) for k in FORCE_KEYS}
    lpf_state = {k: None for k in FORCE_KEYS}

    # --- 공유 상태 ---
    buffers = {k: deque([0.0] * window_samples, maxlen=window_samples)
               for k in FORCE_KEYS}
    ftip_buf = deque([0.0] * window_samples, maxlen=window_samples)
    t_axis = -np.arange(window_samples)[::-1] / rate

    rec = {'active': False, 'h5': None, 'index': 0, 'last_ts': None,
           'last_flush': None, 'ds': {}, 'pad_until': 0.0, 'pad_trial': 0,
           'kept': 0, 'skipped': 0}
    trial = {'active': False, 'num': 0, 'start_unix': None, 'start_iso': None,
             'ts': [], 'ftip': [], 'grip': [], 'tor': [], 'theta': []}
    ui = {'stage': 0, 'last_row': None, 'msg': '', 'live': (0.0, 0.0, 0.0)}
    counts = {name: 0 for name in STAGE_NAMES}
    pending = {'retare': False, 'mute_radio': False}
    rec_lock = threading.Lock()
    buf_lock = threading.Lock()
    stop_flag = threading.Event()

    # --- 시행 CSV ---
    csv_f = open(csv_path, "w", newline="", encoding="utf-8")
    csv_w = csv.DictWriter(csv_f, fieldnames=CSV_FIELDS)
    csv_w.writeheader()
    csv_f.flush()

    # ------------------------------------------------------------------ 녹화
    def start_recording():
        if rec['active']:
            return
        # 녹화를 껐다 다시 켜면 '새 파일' 을 만든다. 예전에는 같은 경로를 'w' 로
        # 다시 열어 앞서 받은 것을 통째로 날렸다.
        cur_base = base
        k = 1
        while os.path.exists(cur_base + ".h5"):
            k += 1
            cur_base = "%s_r%d" % (base, k)
        rec['base'] = cur_base
        cur_h5 = cur_base + ".h5"
        if cam is not None:
            cam.start_recording(cur_base + "_cam")
        h5 = h5py.File(cur_h5, 'w')
        start_unix = time.time()
        h5.attrs['specimen'] = spec
        h5.attrs['source'] = 'SIMULATION' if args.simulate else args.socket
        h5.attrs['rate'] = float(rate)
        h5.attrs['actual_rate'] = float(actual_rate)
        h5.attrs['channels'] = int(num_channels)
        h5.attrs['calib_mode'] = 'decoupled' if calib_path else 'raw'
        if calib_path:
            h5.attrs['calib_path'] = calib_path
        h5.attrs['force_keys'] = list(FORCE_KEYS)
        h5.attrs['force_units'] = [UNIT[k] for k in FORCE_KEYS]
        h5.attrs['raw_keys'] = list(RAW_KEYS)
        h5.attrs['stage_names'] = list(STAGE_NAMES)
        h5.attrs['class_keys'] = list(CLASS_KEYS)
        h5.attrs['tare_enabled'] = bool(tare_enabled)
        h5.attrs['cutoff_hz'] = float(args.cutoff)
        if args.quarter_cutoff is not None:
            h5.attrs['quarter_cutoff_hz'] = float(args.quarter_cutoff)
        h5.attrs['decimate'] = args.decimate
        h5.attrs['start_unix'] = start_unix
        h5.attrs['start_iso_utc'] = datetime.now(timezone.utc).isoformat()
        h5.attrs['timestamp_units'] = 'unix_epoch_seconds_utc'
        h5.attrs['protocol'] = 'PHANTOM_SPEC_PROTOCOL.md Phase 1-3'
        h5.attrs['trials_csv'] = os.path.basename(csv_path)
        if cam is not None:
            h5.attrs['video_file'] = os.path.basename(cur_base + "_cam.mp4")
            h5.attrs['video_h5'] = os.path.basename(cur_base + "_cam.h5")
        h5.attrs['record_mode'] = 'all' if args.record_all else 'trials_only'
        h5.attrs['pad_sec'] = float(args.pad_sec)
        h5.attrs['trial_label_note'] = (
            'trial > 0: inside that trial. trial < 0: pad window right after trial '
            '|trial| (kept so a peak just before [space] is not clipped). '
            'trial == 0 appears only when record_mode == "all".')
        h5.attrs['values_note'] = (
            'forces == what this GUI draws (after calib+tare+LPF). raw = channel volts. '
            'stage indexes attrs["stage_names"]; trial 0 means outside a trial.')
        ds = {
            'timestamp': _create_ds(h5, 'timestamp', (), 'f8'),
            'forces': _create_ds(h5, 'forces', (len(FORCE_KEYS),), 'f4'),
            'raw': _create_ds(h5, 'raw', (len(RAW_KEYS),), 'f4'),
            'stage': _create_ds(h5, 'stage', (), 'i2'),
            'trial': _create_ds(h5, 'trial', (), 'i4'),
        }
        with rec_lock:
            rec.update({'active': True, 'h5': h5, 'index': 0, 'last_ts': None,
                        'last_flush': start_unix, 'ds': ds})
        ui['msg'] = "REC 시작"
        print("\n[REC] 녹화 시작 -> %s" % h5_path)

    def stop_recording():
        with rec_lock:
            if not rec['active']:
                return
            rec['active'] = False
            if cam is not None:
                cam.set_gate(True)
                cam.stop_recording()
            if rec['h5']:
                rec['h5'].attrs['num_samples'] = rec['index']
                rec['h5'].attrs['end_unix'] = time.time()
                rec['h5'].attrs['samples_skipped_between_trials'] = rec['skipped']
                for name in STAGE_NAMES:
                    rec['h5'].attrs['trials_' + name] = counts[name]
                rec['h5'].flush()
                rec['h5'].close()
                rec['h5'] = None
        ui['msg'] = "REC 정지"
        print("[STOP] 녹화 정지 — %d 샘플 저장" % rec['index'])

    def toggle_recording(_=None):
        stop_recording() if rec['active'] else start_recording()

    def retare(_=None):
        pending['retare'] = True
        ui['msg'] = "영점(tare) 재설정"
        print("[tare] 현재 무부하 상태를 0 으로 잡는다")

    # ------------------------------------------------------------------ 시행
    def write_row(label="unset", note=""):
        st = STAGES[ui['stage']]
        row = {k: "" for k in CSV_FIELDS}
        row.update({
            "trial": trial['num'], "specimen": spec,
            "stage": st[0], "stage_section": st[2], "label": label,
            "start_iso": trial['start_iso'],
            "start_unix": "%.6f" % trial['start_unix'],
            "end_unix": "%.6f" % time.time(),
            "duration_s": "%.3f" % (time.time() - trial['start_unix']),
            "n_samples": len(trial['ftip']),
            "calib_path": os.path.basename(calib_path) if calib_path else "raw",
            "rate": "%g" % rate, "cutoff_hz": "%g" % args.cutoff, "note": note,
        })
        row.update(_trial_metrics(trial['ts'], trial['ftip'], trial['grip'],
                                  trial['tor'], trial['theta']))
        csv_w.writerow(row)
        csv_f.flush()
        ui['last_row'] = row
        counts[st[0]] += 1
        print("[trial %d] %-10s peak=%s N  mean=%s N  dF/dt=%s N/s  drop=%s  (%s)"
              % (row['trial'], row['stage'], row['peak_F_tip'] or "-",
                 row['mean_F_tip'] or "-", row['dFdt_median_pre05s'] or "-",
                 row['drop_ratio'] or "-", label))

    def toggle_trial(_=None):
        if ui['stage'] == 0:
            ui['msg'] = "단계를 먼저 고르세요 (idle 에서는 시행 불가)"
            return
        if not trial['active']:
            if not rec['active']:
                start_recording()
            trial.update({'active': True, 'num': trial['num'] + 1,
                          'start_unix': time.time(),
                          'start_iso': datetime.now(timezone.utc).isoformat(),
                          'ts': [], 'ftip': [], 'grip': [], 'tor': [],
                          'theta': []})
            ui['msg'] = "시행 %d 시작 — %s" % (trial['num'], STAGES[ui['stage']][1])
        else:
            trial['active'] = False
            rec['pad_until'] = time.time() + max(0.0, args.pad_sec)
            rec['pad_trial'] = trial['num']
            write_row()
            ui['msg'] = ("시행 %d 종료 — 분류 입력 [1..4]" % trial['num']
                         if STAGES[ui['stage']][0] in ("rupture", "slip_dry", "slip_gel")
                         else "시행 %d 종료" % trial['num'])

    def classify(idx):
        """마지막으로 기록한 CSV 행의 label 을 바꾼다 (파일을 다시 써서)."""
        if pending['mute_radio']:
            return
        if ui['last_row'] is None:
            ui['msg'] = "분류할 시행이 없습니다"
            return
        ui['last_row']['label'] = CLASS_KEYS[idx]
        _rewrite_csv()
        ui['msg'] = "시행 %s → %s" % (ui['last_row']['trial'], CLASS_LABELS[idx])
        print("[class] 시행 %s → %s" % (ui['last_row']['trial'], CLASS_KEYS[idx]))

    def undo_trial(_=None):
        if ui['last_row'] is None:
            ui['msg'] = "취소할 시행이 없습니다"
            return
        st = ui['last_row']['stage']
        counts[st] = max(0, counts[st] - 1)
        rows = _read_rows()
        rows = [r for r in rows if r['trial'] != ui['last_row']['trial']]
        _write_rows(rows)
        print("[undo] 시행 %s 삭제" % ui['last_row']['trial'])
        ui['msg'] = "시행 %s 취소" % ui['last_row']['trial']
        ui['last_row'] = rows[-1] if rows else None

    def _read_rows():
        csv_f.flush()
        with open(csv_path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def _write_rows(rows):
        nonlocal csv_w
        csv_f.seek(0)
        csv_f.truncate(0)
        csv_w = csv.DictWriter(csv_f, fieldnames=CSV_FIELDS)
        csv_w.writeheader()
        for r in rows:
            csv_w.writerow(r)
        csv_f.flush()

    def _rewrite_csv():
        rows = _read_rows()
        for r in rows:
            if r['trial'] == str(ui['last_row']['trial']):
                r['label'] = ui['last_row']['label']
        _write_rows(rows)

    def set_stage(i):
        if pending['mute_radio']:
            return
        ui['stage'] = max(0, min(len(STAGES) - 1, i))
        st = STAGES[ui['stage']]
        ui['msg'] = "단계: %s %s  (권장 n=%d, 현재 %d)" % (
            st[1], st[2], st[3], counts[st[0]])
        print("[stage] %s %s" % (st[1], st[2]))
        for line in stage_action(st[0]):
            print("         " + line)

    # ------------------------------------------------------- reader 스레드
    def reader():
        """소켓에서 받은 즉시 시각을 찍고, 처리해서 h5 에 append 한다.
        GUI 가 멈춰도 타임스탬프가 밀리지 않도록 전용 스레드로 돌린다."""
        while not stop_flag.is_set():
            try:
                data = source.read()
            except Exception as e:
                if not stop_flag.is_set():
                    print("\n[reader] 소스 종료: %s" % e, file=sys.stderr)
                return
            now = time.time()
            n = data.shape[1]

            forces = forces_fn(data)

            # tare: 시작 직후 무부하 평균을 0 으로
            if pending['retare']:
                for k in FORCE_KEYS:
                    tare_baseline[k] = None
                    tare_acc[k] = []
                pending['retare'] = False
            if tare_enabled:
                for k in FORCE_KEYS:
                    if tare_baseline[k] is None:
                        tare_acc[k].extend(np.asarray(forces[k]).tolist())
                        if len(tare_acc[k]) >= tare_need:
                            tare_baseline[k] = float(np.mean(tare_acc[k][:tare_need]))
                    if tare_baseline[k] is not None:
                        forces[k] = np.asarray(forces[k]) - tare_baseline[k]

            # 축별 one-pole 저역통과
            for k in FORCE_KEYS:
                a = axis_alpha[k]
                if a is not None:
                    forces[k], lpf_state[k] = onepole_lpf(forces[k], a, lpf_state[k])

            fmat = np.stack([np.asarray(forces[k], dtype=np.float64)
                             for k in FORCE_KEYS], axis=1)      # (n, 6)
            ftip = np.hypot(fmat[:, IDX_BGX], fmat[:, IDX_BGY])
            theta = np.degrees(np.arctan2(fmat[:, IDX_BGY], fmat[:, IDX_BGX]))

            # 샘플별 절대시각: 도착 시각에서 1/actual_rate 간격으로 역산
            dt = 1.0 / actual_rate
            ts = now - dt * np.arange(n - 1, -1, -1)
            if rec['last_ts'] is not None and ts[0] <= rec['last_ts']:
                ts = rec['last_ts'] + dt * np.arange(1, n + 1)

            # 시각화 버퍼
            with buf_lock:
                for j, k in enumerate(FORCE_KEYS):
                    buffers[k].extend(fmat[:, j].tolist())
                ftip_buf.extend(ftip.tolist())
                ui['live'] = (float(ftip[-1]), float(theta[-1]),
                              float(fmat[-1, IDX_GRIP]))

            # 시행 누적
            if trial['active']:
                trial['ts'].extend(ts.tolist())
                trial['ftip'].extend(ftip.tolist())
                trial['grip'].extend(fmat[:, IDX_GRIP].tolist())
                trial['tor'].extend(fmat[:, IDX_TOR].tolist())
                trial['theta'].extend(theta.tolist())

            # h5 기록 — 기본은 '시행 구간 + 후행 여유' 만 담는다 (저장공간).
            # trial 라벨: 시행 중 = +번호, 후행 여유 = -번호 (분석에서 골라 쓰라고 구분)
            in_trial = trial['active']
            in_pad = (not in_trial) and now < rec['pad_until']
            keep = args.record_all or in_trial or in_pad
            if in_trial:
                tlabel = trial['num']
            elif in_pad:
                tlabel = -rec['pad_trial']
            else:
                tlabel = 0
            if cam is not None:
                cam.set_gate(bool(keep))
            with rec_lock:
                if rec['active'] and rec['h5'] is not None and keep:
                    rec['kept'] += n
                    raws = np.stack([data[ch] for _, ch in RAW_CHANNELS], axis=1)
                    _append_block(rec['ds']['timestamp'], ts)
                    _append_block(rec['ds']['forces'], fmat.astype(np.float32))
                    _append_block(rec['ds']['raw'], raws.astype(np.float32))
                    _append_block(rec['ds']['stage'],
                                  np.full(n, ui['stage'], dtype=np.int16))
                    _append_block(rec['ds']['trial'],
                                  np.full(n, tlabel, dtype=np.int32))
                    rec['index'] += n
                    rec['last_ts'] = float(ts[-1])
                    if now - rec['last_flush'] >= FLUSH_INTERVAL_S:
                        rec['h5'].flush()
                        rec['last_flush'] = now
                elif rec['active'] and rec['h5'] is not None:
                    rec['skipped'] += n

    th = threading.Thread(target=reader, daemon=True)
    th.start()

    # --- 전역 녹화 신호(페달/다른 레코더와 동시 시작) ---
    signal = None
    if not args.no_signal:
        try:
            signal = RecordSignal(on_start=start_recording, on_stop=stop_recording,
                                  on_toggle=toggle_recording)
            signal.start()
            print("[signal] 전역 녹화 신호 연동 (페달/다른 레코더와 동시 시작)")
        except Exception as e:
            print("[signal] 연동 생략: %s" % e)

    # ------------------------------------------------------------------- GUI
    plt.style.use("default")
    _ko_sans, _ko_mono = setup_korean_font()
    fig = plt.figure(figsize=(18, 8.2))
    fig.canvas.manager.set_window_title("팬텀 힘 스펙 측정 — %s" % spec)
    # 왼쪽: 힘 그래프 / 가운데: 엔도스코프 / 오른쪽: 위젯·HUD
    gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.0],
                          left=0.045, right=0.365, top=0.93, bottom=0.08,
                          hspace=0.28)
    ax_f = fig.add_subplot(gs[0, 0])
    ax_g = fig.add_subplot(gs[1, 0], sharex=ax_f)

    # 엔도스코프 미리보기 (영상이 없으면 안내만 띄운다)
    ax_cam = fig.add_axes([0.395, 0.46, 0.265, 0.47])
    ax_cam.set_xticks([]); ax_cam.set_yticks([])
    ax_cam.set_title("엔도스코프", fontsize=10)
    nonlocal_im = [None]          # imshow 핸들을 클로저에서 들고 다닌다

    def _draw_cam(rgb, handle):
        if handle is None:
            return ax_cam.imshow(rgb, interpolation="nearest")
        handle.set_data(rgb)
        return handle

    if cam is None:
        ax_cam.text(0.5, 0.5, "영상 없음\n(--camera 로 인덱스 지정)",
                    ha="center", va="center", fontsize=9, color="0.45",
                    transform=ax_cam.transAxes)

    ax_ui = fig.add_axes([0.685, 0.045, 0.30, 0.885])
    ax_ui.axis("off")

    (ln_ftip,) = ax_f.plot(t_axis, np.zeros(window_samples), color="tab:blue",
                           lw=1.6, label="F_tip (N)")
    ax_f.axhline(0, color="0.8", lw=0.8)
    ax_f.set_ylim(0, args.ymax)
    ax_f.set_ylabel("F_tip (N)")
    ax_f.grid(alpha=0.3)
    ax_f.legend(loc="upper left", fontsize=9)
    span = ax_f.axvspan(0, 0, color="tab:green", alpha=0.12)
    span.set_visible(False)

    (ln_grip,) = ax_g.plot(t_axis, np.zeros(window_samples), color="tab:green",
                           lw=1.2, label="gripping (N, 손잡이)")
    ax_g.axhline(0, color="0.8", lw=0.8)
    ax_g.set_xlabel("시간 (s)")
    ax_g.set_ylabel("gripping (N)")
    ax_g.grid(alpha=0.3)
    ax_g.legend(loc="upper left", fontsize=8)
    ax_f.set_xlim(t_axis[0], 0)

    # 오른쪽 위젯 패널
    ax_stage = fig.add_axes([0.395, 0.055, 0.125, 0.375])
    ax_stage.set_title("단계 (n/b)", fontsize=9)
    rb_stage = RadioButtons(ax_stage, STAGE_LABELS, active=0)
    for lb in rb_stage.labels:
        lb.set_fontsize(7.5)
    rb_stage.on_clicked(lambda lab: set_stage(STAGE_LABELS.index(lab)))

    ax_cls = fig.add_axes([0.535, 0.215, 0.125, 0.215])
    ax_cls.set_title("마지막 시행 분류 (1..4)", fontsize=9)
    rb_cls = RadioButtons(ax_cls, CLASS_LABELS, active=0)
    for lb in rb_cls.labels:
        lb.set_fontsize(7.5)
    rb_cls.on_clicked(lambda lab: classify(CLASS_LABELS.index(lab)))

    ax_btn_tr = fig.add_axes([0.535, 0.170, 0.060, 0.048])
    btn_tr = Button(ax_btn_tr, "시행 ▶/■")
    btn_tr.on_clicked(toggle_trial)
    ax_btn_rec = fig.add_axes([0.600, 0.170, 0.060, 0.048])
    btn_rec = Button(ax_btn_rec, "REC")
    btn_rec.on_clicked(toggle_recording)

    ax_btn_c = fig.add_axes([0.535, 0.115, 0.060, 0.048])
    btn_c = Button(ax_btn_c, "영점(c)")
    btn_c.on_clicked(retare)
    ax_btn_u = fig.add_axes([0.600, 0.115, 0.060, 0.048])
    btn_u = Button(ax_btn_u, "취소(u)")
    btn_u.on_clicked(undo_trial)

    ax_note = fig.add_axes([0.535, 0.055, 0.125, 0.030])
    tb_note = TextBox(ax_note, "", initial="")
    ax_note.set_title("이동량mm / 메모 → Enter", fontsize=7.5)

    # 이동 단계는 목표 20 mm 를 물리적으로 다 못 가는 방향이 있다. 실제 이동량이
    # 없으면 방향별 힘을 비교할 수 없으므로, 앞머리 숫자를 travel_mm 로 받는다.
    #   "12"  -> travel_mm=12
    #   "12 걸림" -> travel_mm=12, note="걸림"
    #   "걸림"    -> note 만
    _TRAVEL_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(?:mm)?\b[\s,;:/-]*", re.I)

    def on_note(text):
        text = text.strip()
        if ui['last_row'] is not None and text:
            m = _TRAVEL_RE.match(text)
            travel, rest = ("", text)
            if m:
                travel, rest = m.group(1), text[m.end():].strip()
            ui['last_row']['note'] = rest
            if travel:
                ui['last_row']['travel_mm'] = travel
            rows = _read_rows()
            for r in rows:
                if r['trial'] == str(ui['last_row']['trial']):
                    r['note'] = rest
                    if travel:
                        r['travel_mm'] = travel
            _write_rows(rows)
            ui['msg'] = ("시행 %s: 이동량 %s mm%s" % (ui['last_row']['trial'], travel,
                                                   (" / " + rest) if rest else "")
                         if travel else "메모 저장")
            tb_note.set_val("")
    tb_note.on_submit(on_note)

    txt = ax_ui.text(0.0, 0.985, "", va="top", ha="left", fontsize=8.2,
                     family=(_ko_mono or "monospace"), transform=ax_ui.transAxes)

    def _silent_radio(rb, idx):
        """set_active 는 on_clicked 를 다시 부른다 — 키 입력에서는 그 재진입을 막는다."""
        pending['mute_radio'] = True
        try:
            rb.set_active(idx)
        finally:
            pending['mute_radio'] = False

    def on_key(ev):
        if ev.key == " ":
            toggle_trial()
        elif ev.key == "n":
            set_stage(ui['stage'] + 1)
            _silent_radio(rb_stage, ui['stage'])
        elif ev.key == "b":
            set_stage(ui['stage'] - 1)
            _silent_radio(rb_stage, ui['stage'])
        elif ev.key == "r":
            toggle_recording()
        elif ev.key == "c":
            retare()
        elif ev.key == "u":
            undo_trial()
        elif ev.key in ("1", "2", "3", "4"):
            i = int(ev.key)
            classify(i)
            _silent_radio(rb_cls, i)
        elif ev.key == "q":
            plt.close(fig)
    fig.canvas.mpl_connect("key_press_event", on_key)

    def update(_):
        with buf_lock:
            fv = np.fromiter(ftip_buf, dtype=np.float64, count=window_samples)
            gv = np.fromiter(buffers["gripping"], dtype=np.float64,
                             count=window_samples)
            live = ui['live']
        ln_ftip.set_ydata(fv)
        ln_grip.set_ydata(gv)
        top = max(args.ymax, float(fv.max()) * 1.15 if len(fv) else args.ymax)
        ax_f.set_ylim(0, top)
        g_lo, g_hi = float(gv.min()), float(gv.max())
        pad = max(0.5, (g_hi - g_lo) * 0.2)
        ax_g.set_ylim(g_lo - pad, g_hi + pad)
        ax_f.set_title("%s  %s   |   %s"
                       % (STAGES[ui['stage']][1], STAGES[ui['stage']][2],
                          "● 시행 %d 진행 중" % trial['num'] if trial['active']
                          else "대기"),
                       fontsize=11,
                       color="tab:red" if trial['active'] else "0.25")

        st = STAGES[ui['stage']]
        lines, got, need = [], 0, 0
        for s2 in STAGES[1:]:
            c = counts[s2[0]]
            if s2[0] in SKIP:
                lines.append("  %-20s %2d / -- 제외" % (s2[1], c))
                continue
            got += min(c, s2[3]); need += s2[3]
            lines.append("  %-20s %2d / %-2d %s"
                         % (s2[1], c, s2[3], "✓" if c >= s2[3] else ""))
        lines.append("  %-20s %2d / %-2d" % ("— 합계 —", got, need))
        done = "\n".join(lines)
        lr = ui['last_row']
        last = ("  시행 %s  %s\n  peak %s N   drop %s\n  라벨 %s"
                % (lr['trial'], lr['stage'], lr['peak_F_tip'] or "-",
                   lr['drop_ratio'] or "-", lr['label'])) if lr else "  (없음)"
        # 엔도스코프 미리보기 (최신 프레임만 그린다 — 느리면 건너뛸 뿐)
        if cam is not None:
            got = cam.latest()
            if got is not None:
                nonlocal_im[0] = _draw_cam(got[0], nonlocal_im[0])

        todo = "\n".join("  " + l for l in stage_action(st[0]))
        txt.set_text(
            "이번 단계에서 할 일\n%s\n"
            "\n실시간\n"
            "  F_tip     %7.3f N\n"
            "  방향 θ    %7.1f °\n"
            "  gripping  %7.3f N\n"
            "\n녹화\n"
            "  %s   샘플 %d\n"
            "\n진행 (권장 n)\n%s\n"
            "\n마지막 시행\n%s\n"
            "\n%s"
            % (todo, live[0], live[1], live[2],
               "● REC" if rec['active'] else "○ 정지", rec['index'],
               done, last, ui['msg'])
        )
        return ln_ftip, ln_grip, txt

    set_stage(0)
    print("\n조작: [space] 시행  [n/b] 단계  [r] 녹화  [c] 영점  "
          "[1..4] 분류  [u] 취소  [q] 종료\n")
    ani = FuncAnimation(fig, update, interval=50, blit=False,
                        cache_frame_data=False)
    try:
        plt.show()
    finally:
        if trial['active']:
            trial['active'] = False
            write_row(note="창 종료로 자동 마감")
        stop_flag.set()
        stop_recording()
        if cam is not None:
            cam.close()
        if signal is not None:
            try:
                signal.close()
            except Exception:
                pass
        try:
            source.close()
        except Exception:
            pass
        csv_f.close()
        print("\n저장 완료")
        print("  h5  : %s" % h5_path)
        print("  csv : %s" % csv_path)
        tot = sum(counts.values())
        print("  시행 %d건 — %s"
              % (tot, ", ".join("%s %d" % (s[1], counts[s[0]])
                                for s in STAGES[1:] if counts[s[0]])))


if __name__ == "__main__":
    main()
