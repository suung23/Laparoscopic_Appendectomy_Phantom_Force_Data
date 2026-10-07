#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
USB-6009 6-force visualization for the re-wired tool, fed from OUR Linux driver socket
(the same one visualizer.py uses).

새 wiring (자세한 채널/브릿지 표는 wiring.py 참고):
  half-bridge 4개가 L-R / T-B 빼기를 하드웨어에서 처리 → 채널 1개 = 힘 1개:
    A0 bending_gripper_x,  A1 bending_trocar_x,
    A4 bending_gripper_y,  A5 bending_trocar_y,
    A2 gripping (quarter), A6 torsion (quarter, 단위 N·m)

raw 모드는 채널 전압을 그대로 보여주고(게이지/배선 오차로 축 간 커플링 잔존),
calibrate.py 가 만든 calib.json(6x6 디커플링 행렬)을 적용하면 커플링이 제거되고
스케일이 가한 힘 단위(N, torsion N·m)로 맞춰진다.

캘리브레이션 스냅샷:
  저울로 '알고 있는 방향·크기'의 힘을 가한 상태에서 6개 힘 + 6개 raw 채널을 기록.
    1) 오른쪽 패널에서 가한 힘의 '방향'(라디오)과 '크기'(텍스트박스) 입력
    2) 's' 키/‘Snapshot’ 버튼 → 직전 --snap-window 초 평균을 CSV 한 줄로 저장
    3) 'u' 키/‘Undo’ 버튼 → 마지막 스냅샷 취소
  CSV 열: timestamp, t_rel_s, applied_dir, applied_force,
          <6 force keys...>, <6 raw keys: bgx,btx,bgy,bty,grip,tor>
  → calibrate.py 로 그 CSV 들을 넘기면 calib.json 갱신.

Run:
    conda activate surgitag
    python visualize_forces.py --socket /tmp/surgitag/daq.sock --rate 100 --window 5
    python visualize_forces.py --simulate          # no hardware, synthetic test
    python visualize_forces.py --simulate --out calib.csv   # 스냅샷 저장 경로 지정
"""

import argparse
import csv
import os
import socket
import struct
import sys
import time
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import RadioButtons, TextBox, Button

# --- matplotlib 3.11 버그 우회 -------------------------------------------------
# TextBox 가 'resize_event' 를 _resize 에 연결하는데, _resize 는 event.inaxes 를
# 읽는 데코레이터로 감싸여 있다. ResizeEvent 에는 inaxes 가 없어 창 크기 변경 때마다
# AttributeError 가 난다. _resize 는 stop_typing() 만 하므로 평범한 버전으로 교체.
def _safe_textbox_resize(self, _event):
    self.stop_typing()
TextBox._resize = _safe_textbox_resize

from wiring import (FORCES, RAW_CHANNELS, RAW_KEYS, AXES, CH_INDEX,
                    FORCE_CH, RIGHT_AXIS, UNIT, MIN_CHANNELS, QUARTER_AXES)

HEADER = struct.Struct("<4s i i i")   # magic, nchan, rate, chunk (driver header)


def recvall(sock, n):
    """Read exactly n bytes; return None if the connection closes."""
    parts, got = [], 0
    while got < n:
        b = sock.recv(n - got)
        if not b:
            return None
        parts.append(b)
        got += len(b)
    return b"".join(parts)


def connect(sockpath, timeout=30.0):
    """Retry until the driver's socket is up, then read and return the header."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.connect(sockpath)
        except (FileNotFoundError, ConnectionRefusedError, OSError):
            time.sleep(0.2)
            continue
        hdr = recvall(s, HEADER.size)
        if hdr is None:
            s.close(); time.sleep(0.2); continue
        magic, nchan, rate, chunk = HEADER.unpack(hdr)
        if magic != b"NIDQ":
            print(f"bad header magic: {magic!r}", file=sys.stderr)
            s.close(); time.sleep(0.2); continue
        return s, nchan, rate, chunk
    raise TimeoutError(f"could not connect to driver socket: {sockpath}")


# ---------------------------------------------------------------------------
# Data sources (driver socket / simulation) -- start(), read() -> (nch, chunk), close()
# (visualize_specific.py 와 동일)
# ---------------------------------------------------------------------------
class SocketSource:
    """Reads the in-container driver's Unix-socket stream, decimates the high
    acquisition rate down to `rate`, and serves (num_channels, chunk) blocks.

    decim_mode:
      'average' (기본) — 연속 decim 개 샘플의 평균을 1샘플로(빠른 DAQ 의 오버샘플 활용,
                  노이즈 √decim 감소 + 안티에일리어싱).
      'subsample' — drv_rate→rate 로 줄일 때 decim 개마다 raw 1개만 취함(평활 없음)."""

    def __init__(self, sockpath, num_channels, rate, chunk, timeout=30.0,
                 decim_mode="average"):
        self.num_channels = num_channels
        self.chunk = chunk
        self.rate = rate
        self.avg = (decim_mode == "average")
        self.sock, self._drv_nchan, self._drv_rate, self._drv_chunk = connect(sockpath, timeout)
        if num_channels > self._drv_nchan:
            self.sock.close()
            raise ValueError(f"driver streams {self._drv_nchan} channels, "
                             f"but {num_channels} requested")
        self.decim = max(1, int(round(self._drv_rate / rate)))
        self.actual_rate = self._drv_rate / self.decim
        self._frame = struct.Struct("<%dd" % (self._drv_nchan * self._drv_chunk))
        self._pending = [[] for _ in range(num_channels)]
        # decim 으로 나누어떨어지지 않고 남은 raw 샘플을 프레임 경계 너머로 이월
        self._carry = [np.empty(0, dtype=np.float64) for _ in range(num_channels)]

    def start(self):
        pass

    def read(self):
        """Block until `chunk` decimated samples per channel are available,
        return them as a (num_channels, chunk) float64 array."""
        ck = self._drv_chunk
        d = self.decim
        while len(self._pending[0]) < self.chunk:
            raw = recvall(self.sock, self._frame.size)
            if raw is None:
                raise RuntimeError("driver disconnected")
            arr = np.frombuffer(raw, dtype="<f8")     # GroupByChannel: ch*ck + i
            for ch in range(self.num_channels):
                seg = arr[ch * ck:(ch + 1) * ck]
                buf = (np.concatenate([self._carry[ch], seg])
                       if self._carry[ch].size else seg)
                n_full = buf.size // d                 # 만들 수 있는 출력 샘플 수
                if n_full:
                    blocks = buf[:n_full * d].reshape(n_full, d)
                    vals = blocks.mean(axis=1) if self.avg else blocks[:, 0]
                    self._pending[ch].extend(vals.tolist())
                self._carry[ch] = buf[n_full * d:].copy()   # 남은 샘플 이월
        out = np.empty((self.num_channels, self.chunk), dtype=np.float64)
        for ch in range(self.num_channels):
            out[ch] = self._pending[ch][:self.chunk]
            del self._pending[ch][:self.chunk]
        return out

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass


class SimSource:
    """하드웨어 없이 테스트하기 위한 가짜 신호 생성기.
    채널마다 다른 주파수의 사인파 + 노이즈를 만들어 낸다."""

    def __init__(self, num_channels, rate, chunk):
        self.num_channels = num_channels
        self.rate = rate
        self.chunk = chunk
        self.actual_rate = rate
        self._t = 0.0
        self._freqs = np.linspace(0.5, 5.0, num_channels)  # 채널별 주파수
        self._rng = np.random.default_rng(0)

    def start(self):
        pass

    def read(self):
        dt = 1.0 / self.rate
        t = self._t + np.arange(self.chunk) * dt
        self._t += self.chunk * dt
        sig = np.sin(2 * np.pi * self._freqs[:, None] * t[None, :])
        sig = sig * (1.0 + 0.4 * np.arange(self.num_channels)[:, None])  # 진폭 차등
        sig += 0.05 * self._rng.standard_normal(sig.shape)
        return sig

    def close(self):
        pass


# ---------------------------------------------------------------------------
# 힘 계산 -- 새 wiring 은 채널 1개가 힘 1개(1:1). raw 모드는 채널 전압을 그대로,
# calib 모드는 6x6 디커플링 행렬을 적용한다.
# ---------------------------------------------------------------------------
def compute_forces(data):
    """(num_channels, chunk) 전압 블록 → 6가지 힘의 dict (raw 모드).

    half-bridge 가 이미 빼기를 했으므로 force = 해당 채널 전압. 스케일/커플링 보정은
    --calib 의 디커플링 행렬이 담당한다(미적용 시 채널 raw 값 그대로)."""
    return {fk: data[ch] for fk, ch in FORCE_CH.items()}


def alpha_from_cutoff(cutoff_hz, rate):
    """one-pole 저역통과 계수 α = 1 - exp(-2π·fc/fs). fc<=0 이면 None(필터 끔)."""
    if cutoff_hz is None or cutoff_hz <= 0:
        return None
    return 1.0 - np.exp(-2.0 * np.pi * cutoff_hz / rate)


def onepole_lpf(x, alpha, state):
    """1D 청크에 one-pole 저역통과를 적용. 청크 경계 너머로 state 를 이어간다.
    반환: (filtered_array, new_state)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.empty(x.shape[0], dtype=np.float64)
    s = x[0] if state is None else state
    for i in range(x.shape[0]):
        s += alpha * (x[i] - s)
        y[i] = s
    return y, s


def load_calibration(path):
    """calibrate.py 가 저장한 calib.json 을 읽어 디커플링 변환 함수를 만든다.

        clean(6) = W^T · signals(6) + b
    반환: (forces_fn(data)->dict, axes 순서 리스트)."""
    import json
    with open(path) as f:
        c = json.load(f)
    idx = [CH_INDEX[s] for s in c["signals"]]   # raw 신호 순서 → 채널 인덱스
    W = np.asarray(c["W"], dtype=np.float64)     # (6 signals, 6 axes)
    b = np.asarray(c["b"], dtype=np.float64)     # (6 axes,)
    axes = c["axes"]

    def calibrated(data):
        g = np.stack([data[i] for i in idx])     # (6, chunk)
        clean = W.T @ g + b[:, None]             # (6, chunk)
        return {axes[j]: clean[j] for j in range(len(axes))}

    return calibrated, axes


# FORCES / RAW_CHANNELS 는 wiring.py 에서 가져온다 (새 wiring 단일 출처)
# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="NI USB-6009 6-force visualization (새 wiring: bending gripper/trocar x·y, gripping, torsion)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--socket", default="/tmp/surgitag/daq.sock",
                   help="driver socket path (linux_daqmxbase/daq_reader output)")
    p.add_argument("--channels", type=int, default=8,
                   help=f"수집할 아날로그 채널 수 (wiring 매핑에 최소 {MIN_CHANNELS} 필요)")
    p.add_argument("--rate", type=float, default=100.0,
                   help="채널당 샘플레이트 (Hz). 드라이버 스트림을 이 값으로 솎음")
    p.add_argument("--window", type=float, default=5.0,
                   help="그래프에 표시할 시간 윈도우 (초)")
    p.add_argument("--calib", default=None,
                   help="디커플링/스케일 보정 행렬 JSON (calibrate.py 산출). "
                        "지정 안 하면 스크립트 옆 calib.json 을 자동 사용")
    p.add_argument("--no-calib", action="store_true",
                   help="raw 모드 강제 (calib.json 이 있어도 무시)")
    p.add_argument("--ymin", type=float, default=None, help="왼쪽 세로축 최소 (힘 N)")
    p.add_argument("--ymax", type=float, default=None, help="왼쪽 세로축 최대 (힘 N)")
    p.add_argument("--torsion-ymax", type=float, default=0.5,
                   help="오른쪽 torsion 축 범위 ±값 (N·m). torsion 은 단위가 달라(N·m) "
                        "스케일이 작으므로 오른쪽 별도 축에 그린다")
    p.add_argument("--decimate", choices=["subsample", "average"], default="average",
                   help="1000Hz→rate 다운샘플 방식. average=블록평균(기본, 빠른 DAQ 의 "
                        "오버샘플을 활용해 노이즈↓·안티에일리어싱), subsample=raw 1개만 취함")
    p.add_argument("--cutoff", type=float, default=0.0,
                   help="모든 힘 축의 저역통과 차단주파수(Hz). 0=끔(기본, raw). 켜려면 값을 줌")
    p.add_argument("--quarter-cutoff", type=float, default=None,
                   help="quarter-bridge 축(gripping·torsion)만 다른 차단주파수(Hz)로 "
                        "덮어쓰기. 미지정 시 --cutoff 와 동일")
    p.add_argument("--no-tare", action="store_true",
                   help="시작 직후 무부하 영점(tare) 보정을 끔")
    p.add_argument("--tare-sec", type=float, default=1.0,
                   help="시작 직후 무부하 영점 평균 구간(초)")
    p.add_argument("--out", default=None,
                   help="스냅샷을 저장할 CSV 경로 (기본: calib_snapshots_<날짜시각>.csv)")
    p.add_argument("--snap-window", type=float, default=0.5,
                   help="스냅샷 시 평균 낼 직전 구간(초). 클수록 안정적이지만 느린 반응")
    p.add_argument("--simulate", action="store_true",
                   help="하드웨어 없이 가짜 신호로 시각화만 테스트")
    args = p.parse_args()

    rate = args.rate
    num_channels = args.channels
    if num_channels < MIN_CHANNELS:
        print(f"[error] wiring 매핑에는 최소 {MIN_CHANNELS}채널이 필요합니다 "
              f"(요청: {num_channels})", file=sys.stderr)
        sys.exit(1)
    short = dict(zip(AXES, RAW_KEYS))   # force key → 짧은 코드 (상태표시용)

    # --- 캘리브레이션(디커플링/스케일) 결정 ---
    calib_path = args.calib
    if calib_path is None and not args.no_calib:
        sibling = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calib.json")
        if os.path.exists(sibling):
            calib_path = sibling
    if args.no_calib:
        calib_path = None
    if calib_path:
        forces_fn, _ = load_calibration(calib_path)
        print(f"[calib] 디커플링 행렬 적용: {calib_path}")
        unit = "N"
    else:
        forces_fn = compute_forces
        print("[calib] raw 모드 (보정 없음). --calib 로 행렬 적용 가능")
        unit = "raw"
    # 보정 모드면 단위가 N (axial/grasping ~10, bending ~5, torsion ~0.2),
    # raw 모드면 작은 값 → 기본 세로축 범위를 모드에 맞춰 자동 선택
    if args.ymin is None:
        args.ymin = -12.0 if calib_path else -2.0
    if args.ymax is None:
        args.ymax = 12.0 if calib_path else 2.0

    # 채널별 한 번에 읽을 샘플 수 (대략 10fps 갱신)
    chunk = max(1, int(round(rate / 10)))
    window_samples = max(chunk, int(round(args.window * rate)))

    # --- 데이터 소스 준비 ---
    if args.simulate:
        print("[mode] simulation (synthetic signal)")
        source = SimSource(num_channels, rate, chunk)
    else:
        print(f"[mode] driver socket={args.socket}  channels={num_channels}  "
              f"display rate={rate} Hz")
        try:
            source = SocketSource(args.socket, num_channels, rate, chunk,
                                  decim_mode=args.decimate)
        except Exception as e:
            print(f"socket connect/setup failed: {e}", file=sys.stderr)
            sys.exit(1)
        if source.avg:
            print(f"  driver stream {source._drv_rate} Hz -> block-average x{source.decim} "
                  f"-> {source.actual_rate:.1f} Hz (noise ~×{source.decim**0.5:.1f} 감소)")
        else:
            print(f"  driver stream {source._drv_rate} Hz -> subsample x{source.decim} "
                  f"-> {source.actual_rate:.1f} Hz (raw, 후처리 없음)")

    # --- 링버퍼 (힘별 + raw 게이지별) ---
    t_axis = -np.arange(window_samples)[::-1] / rate
    buffers = {key: deque([0.0] * window_samples, maxlen=window_samples)
               for key, _, _, _ in FORCES}
    raw_recent = {ch: deque([0.0] * window_samples, maxlen=window_samples)
                  for _, ch in RAW_CHANNELS}
    snap_samples = max(1, int(round(args.snap_window * rate)))  # 스냅샷 평균 구간

    # --- 영점(tare) 상태: 무부하 구간 평균을 모아 기준값으로 빼준다 ---
    tare_enabled = not args.no_tare
    tare_baseline = {key: None for key, _, _, _ in FORCES}
    tare_acc = {key: [] for key, _, _, _ in FORCES}

    # --- 축별 저역통과(one-pole IIR): 노이즈 억제 ---
    #   quarter-bridge 축(gripping·torsion)은 --quarter-cutoff 로 따로 잡을 수 있음
    def _cutoff_for(k):
        if k in QUARTER_AXES and args.quarter_cutoff is not None:
            return args.quarter_cutoff
        return args.cutoff
    axis_cutoff = {k: _cutoff_for(k) for k, _, _, _ in FORCES}
    axis_alpha = {k: alpha_from_cutoff(c, rate) for k, c in axis_cutoff.items()}
    lpf_state = {k: None for k, _, _, _ in FORCES}

    # --- 스냅샷(캘리브레이션) 저장 준비 ---
    out_path = args.out or time.strftime("calib_snapshots_%Y%m%d_%H%M%S.csv")
    out_path = os.path.abspath(out_path)
    CSV_HEADER = (["timestamp", "t_rel_s", "applied_dir", "applied_force"]
                  + [k for k, _, _, _ in FORCES]
                  + [label for label, _ in RAW_CHANNELS])
    snapshots = []   # 메모리상 누적 (undo 가능). 매 변경 시 파일 전체를 다시 씀

    def write_all():
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(CSV_HEADER)
            w.writerows(snapshots)

    # --- 플롯 구성 (왼쪽: 그래프, 오른쪽: 캘리브레이션 UI) ---
    plt.style.use("default")
    # 's' 가 matplotlib 기본 "그림 저장" 단축키라서 스냅샷용으로 비워준다
    if "s" in plt.rcParams.get("keymap.save", []):
        plt.rcParams["keymap.save"].remove("s")

    fig = plt.figure(figsize=(13, 6.5))
    ax = fig.add_axes([0.06, 0.10, 0.56, 0.82])
    ax2 = ax.twinx()           # 오른쪽 축: RIGHT_AXIS(=torsion, 단위 N·m)
    right_color = next((c for k, _, c, _ in FORCES if k in RIGHT_AXIS), "tab:purple")

    # RIGHT_AXIS 힘은 단위(N·m)·스케일이 달라 오른쪽 축(ax2)에, 나머지는 왼쪽 축(ax)에
    lines = {}
    for key, label, color, style in FORCES:
        if axis_alpha[key] is not None:
            label = f"{label} (LPF {axis_cutoff[key]:g}Hz)"
        if key in RIGHT_AXIS:
            label = f"{label} ({UNIT[key]}, right)"
        tgt = ax2 if key in RIGHT_AXIS else ax
        (ln,) = tgt.plot(t_axis, np.zeros(window_samples),
                         lw=1.8, color=color, linestyle=style, label=label)
        lines[key] = ln

    ax.set_xlim(t_axis[0], 0)
    pad = 0.05 * (args.ymax - args.ymin)
    ax.set_ylim(args.ymin - pad, args.ymax + pad)
    ax.set_xlabel("time (s)")
    ax.set_ylabel(f"Force ({unit})")

    # 오른쪽 축(torsion): ±torsion_ymax 로 고정해 작은 신호도 잘 보이게
    ax2.set_ylim(-args.torsion_ymax, args.torsion_ymax)
    ax2.set_ylabel("Torsion (N·m)", color=right_color)
    ax2.tick_params(axis="y", labelcolor=right_color)
    ax2.spines["right"].set_color(right_color)

    src_label = "SIMULATION" if args.simulate else "USB-6009"
    mode_label = "decoupled" if calib_path else "raw"
    ax.set_title(f"NI {src_label} — {len(FORCES)} forces ({mode_label}) @ {rate:.0f} Hz")
    ax.grid(True, alpha=0.3)
    ax.axhline(0.0, color="black", lw=0.6, alpha=0.4)
    # 두 축의 선을 하나의 범례로 합침
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", ncol=3, fontsize=8)

    # 오른쪽 패널: 가한 힘의 '방향' 선택 + '크기' 입력 + 저장/취소 버튼
    directions = [key for key, _, _, _ in FORCES]
    ax_radio = fig.add_axes([0.70, 0.50, 0.18, 0.38])
    ax_radio.set_title("Applied direction", fontsize=10)
    radio = RadioButtons(ax_radio, directions)

    ax_text = fig.add_axes([0.80, 0.40, 0.10, 0.055])
    textbox = TextBox(ax_text, "Mag (N / N·m) ", initial="0")

    ax_save = fig.add_axes([0.70, 0.30, 0.12, 0.06])
    save_btn = Button(ax_save, "Snapshot (s)")
    ax_undo = fig.add_axes([0.83, 0.30, 0.07, 0.06])
    undo_btn = Button(ax_undo, "Undo (u)")

    live_txt = fig.text(0.70, 0.10, "", fontsize=9, family="monospace",
                        va="bottom")
    status_txt = fig.text(0.70, 0.025, f"saved → {out_path}", fontsize=8,
                          color="dimgray", va="bottom")

    def set_status(msg, color="black"):
        status_txt.set_text(msg)
        status_txt.set_color(color)
        fig.canvas.draw_idle()

    def avg_recent(buf):
        n = min(snap_samples, len(buf))
        return float(np.mean(list(buf)[-n:])) if n else float("nan")

    def take_snapshot(_event=None):
        direction = radio.value_selected
        raw_text = textbox.text.strip()
        try:
            applied = float(raw_text)
        except ValueError:
            set_status(f"⚠ 크기 '{raw_text}' 가 숫자가 아님 — 저장 안 함", "red")
            return
        fvals = {key: avg_recent(buffers[key]) for key, _, _, _ in FORCES}
        rvals = {label: avg_recent(raw_recent[ch]) for label, ch in RAW_CHANNELS}
        row = ([time.strftime("%Y-%m-%d %H:%M:%S"), round(time.time() - t0, 3),
                direction, applied]
               + [fvals[k] for k, _, _, _ in FORCES]
               + [rvals[label] for label, _ in RAW_CHANNELS])
        snapshots.append(row)
        write_all()
        meas = "  ".join(f"{short[k]}:{fvals[k]:+.3f}" for k, _, _, _ in FORCES)
        set_status(f"#{len(snapshots)} 저장  [{direction}={applied:g}]\n{meas}",
                   "darkgreen")
        print(f"[snap #{len(snapshots)}] {direction}={applied:g}  "
              f"forces={ {short[k]: round(fvals[k],4) for k,_,_,_ in FORCES} }")

    def undo_last(_event=None):
        if not snapshots:
            set_status("되돌릴 스냅샷이 없음", "red")
            return
        removed = snapshots.pop()
        write_all()
        set_status(f"마지막 스냅샷 삭제 (남은 {len(snapshots)}개)  "
                   f"[{removed[2]}={removed[3]}]", "darkorange")

    def on_key(event):
        if event.key == "s":
            take_snapshot()
        elif event.key == "u":
            undo_last()

    save_btn.on_clicked(take_snapshot)
    undo_btn.on_clicked(undo_last)
    fig.canvas.mpl_connect("key_press_event", on_key)

    t0 = time.time()
    source.start()

    def update(_frame):
        try:
            data = source.read()  # (num_channels, chunk)
        except Exception as e:
            print(f"읽기 오류: {e}", file=sys.stderr)
            return

        forces = forces_fn(data)
        for key in forces:
            vals = forces[key]
            # --- 영점(tare): 시작 직후 무부하 구간 평균을 빼서 0으로 맞춤 ---
            if tare_enabled:
                if tare_baseline[key] is None:
                    tare_acc[key].append(np.asarray(vals).copy())
                    base = float(np.concatenate(tare_acc[key]).mean())
                    if sum(a.size for a in tare_acc[key]) >= args.tare_sec * rate:
                        tare_baseline[key] = base   # 충분히 모이면 영점 확정
                    vals = vals - base
                else:
                    vals = vals - tare_baseline[key]
            # 축별 저역통과로 진동 억제 (청크 경계 너머 state 유지)
            if axis_alpha[key] is not None:
                vals, lpf_state[key] = onepole_lpf(vals, axis_alpha[key],
                                                   lpf_state[key])
            buffers[key].extend(vals)
            lines[key].set_ydata(buffers[key])
        for _, ch in RAW_CHANNELS:        # 스냅샷용 raw 전압 누적
            raw_recent[ch].extend(data[ch])

        live_txt.set_text(
            f"live avg ({args.snap_window:g}s):\n"
            + "\n".join(f"  {lbl:<11s}{avg_recent(buffers[k]):+8.3f}"
                        for k, lbl, _, _ in FORCES))

    # interval=0 → read()가 블로킹되며 자연스럽게 속도 조절됨.
    # blit=False: 위젯/상태 텍스트도 매 프레임 갱신되도록.
    ani = FuncAnimation(fig, update, interval=0, blit=False,
                        cache_frame_data=False)

    print(f"[snap] 's' 또는 'Snapshot' 버튼으로 기록 / 'u' 로 마지막 취소")
    print(f"[snap] CSV: {out_path}")
    try:
        plt.show()
    finally:
        source.close()
        print(f"종료됨. 스냅샷 {len(snapshots)}개 → {out_path}")


if __name__ == "__main__":
    main()
