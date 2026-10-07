#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NI USB-6009 6-force dataset recorder
====================================
Shows the exact same 6-force plots as visualize_forces.py, while recording each
sample together with the PC's absolute time (Unix epoch, UTC seconds) into an
HDF5 episode file.

Designed to pair with MCU/dataset_recorder.py: both recorders stamp each sample
with time.time() (Unix epoch), so the NI6009 "force" data and the MCU
"kinematics" data can later be aligned/merged on absolute time.

Timestamp principle (deterministic latency)
  A dedicated reader thread blocks on the driver socket and stamps time.time()
  the moment each chunk arrives, then processes (tare/LPF) and appends it to the
  HDF5 file right there. The GUI thread only draws the ring buffers. So a slow or
  stalled GUI (window drag, rendering hiccup) can no longer delay the timestamps:
  previously the stamp was taken inside the Matplotlib callback, so any GUI stall
  put "now" on samples that were acquired hundreds of ms earlier. The remaining
  latency (DAQ -> driver -> socket receipt) is invariant across episodes and can
  be compensated later. Receive-loop overload is shown on the GUI in color.

The recorded "force values" are exactly what visualize_forces.py draws on screen.
To guarantee they match, the post-processing logic (data source, calib 6x6
decoupling matrix, tare/zeroing, low-pass filter) is imported directly from
visualize_forces.py. In other words:
  value drawn on screen  ==  forces column in HDF5  (after calib + tare + LPF)
The raw channel voltages are also stored so the forces can be recomputed later
with a different calibration.

Keys
  [C] re-zero (tare) the current no-load values to force zero
  [R] toggle recording start/stop (shows REC on screen while recording)
  [Q] safe exit (auto-saves if recording, then quits)

Save location
  collected_data/episode_<datetime>.h5   (single device, so no dataset number)
  Pass --dataset <name> to save under a collected_data/<name>/ subfolder (optional).

Durability (against sudden termination)
  While recording, the HDF5 buffer is flushed to disk about once per second
  (FLUSH_INTERVAL_S). So even if the program is force-killed or power is lost,
  the file still opens cleanly and all data up to the last flush is recovered
  (worst-case loss ~= 1 second). A normal exit ([Q]/window close/Ctrl-C)
  finalizes num_samples/end_unix in the finally block before saving.
  (Without any flush, an incomplete HDF5 file is truncated and will not open.)

Run:
    conda activate surgitag
    python dataset_recorder.py --socket /tmp/surgitag/daq.sock --rate 100 --window 5
    python dataset_recorder.py --simulate            # test without hardware
    ./run_dataset_recorder.sh --rate 100 --window 5

The pipeline options (--rate/--calib/--cutoff/--tare-sec, etc.) have the same
meaning as in visualize_forces.py; use the same values and the screen and the
saved values match exactly.
"""

import argparse
import os
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import h5py

# 전역 녹화 신호(모든 레코더 동시 시작/중지). 저장소 최상위의 record_signal.py 를 쓴다.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from record_signal import RecordSignal
# 옵션: 로봇 워크스테이션(rosota_rollout)으로 UDP 스트리밍 (SURGITAG_UDP=1 일 때만)
from udp_stream import UdpStreamer
# 수신 루프 부하 감시 (GUI 에 문자+색으로 표시)
from load_monitor import LoadMonitor

from wiring import (FORCES, RAW_CHANNELS, RAW_KEYS, RIGHT_AXIS, UNIT,
                    MIN_CHANNELS, QUARTER_AXES)
# To guarantee identical force values, reuse the processing logic from visualize_forces.
from visualize_forces import (SocketSource, SimSource, compute_forces,
                              load_calibration, alpha_from_cutoff, onepole_lpf)

FORCE_KEYS = [k for k, _, _, _ in FORCES]        # order of saved force columns (= wiring AXES)
CHUNK = 256                                       # HDF5 chunk size (same as dataset_recorder)
# Flush the HDF5 buffer to disk at this interval so the file stays readable even on a
# sudden termination (force-kill / power loss). Worst-case loss = data since last flush
# (~= this value, in seconds). (Dying with no flush truncates the whole HDF5 file so it
# will not open at all -- verified experimentally.)
FLUSH_INTERVAL_S = 1.0


# ---------------------------------------------------------------------------
# HDF5 extendable-dataset helpers (same approach as MCU/dataset_recorder.py)
# ---------------------------------------------------------------------------
def _create_ds(h5, name, shape, dtype):
    return h5.create_dataset(
        name, shape=(0,) + shape, maxshape=(None,) + shape,
        dtype=dtype, chunks=(CHUNK,) + shape,
        compression="gzip", compression_opts=4,
    )


def _append_block(ds, block):
    """Append an (n, ...) block to the end of the dataset. Appends n samples at once."""
    block = np.asarray(block)
    n = block.shape[0]
    if n == 0:
        return
    ds.resize(ds.shape[0] + n, axis=0)
    ds[-n:] = block


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="NI USB-6009 6-force dataset recorder "
                    "(records the same values as visualize_forces, with absolute time, to HDF5)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- saving ---
    p.add_argument("--dataset", default=None,
                   help="dataset name (= save subfolder). If omitted, saves directly into collected_data")
    p.add_argument("--out-root", default=None,
                   help="dataset root folder (default: collected_data/ next to this script)")
    # --- data pipeline (same meaning as visualize_forces.py) ---
    p.add_argument("--socket", default="/tmp/surgitag/daq.sock",
                   help="driver socket path (linux_daqmxbase/daq_reader output)")
    p.add_argument("--channels", type=int, default=8,
                   help=f"number of analog channels to acquire (wiring mapping needs at least {MIN_CHANNELS})")
    p.add_argument("--rate", type=float, default=100.0,
                   help="per-channel sample rate (Hz). Driver stream is decimated to this value")
    p.add_argument("--window", type=float, default=5.0,
                   help="time window shown on the plot (seconds)")
    p.add_argument("--calib", default=None,
                   help="decoupling/scale calibration matrix JSON (from calibrate.py). "
                        "If not given, auto-uses calib.json next to this script")
    p.add_argument("--no-calib", action="store_true",
                   help="force raw mode (ignore calib.json even if present)")
    p.add_argument("--ymin", type=float, default=None, help="left y-axis min (force N)")
    p.add_argument("--ymax", type=float, default=None, help="left y-axis max (force N)")
    p.add_argument("--torsion-ymax", type=float, default=0.5,
                   help="right torsion axis range +/- value (N.m)")
    p.add_argument("--decimate", choices=["subsample", "average"], default="average",
                   help="driver-rate -> rate downsample mode (average=block mean, subsample=raw)")
    p.add_argument("--cutoff", type=float, default=0.0,
                   help="low-pass cutoff (Hz) for all force axes. 0=off (default, raw)")
    p.add_argument("--quarter-cutoff", type=float, default=None,
                   help="separate cutoff (Hz) for quarter-bridge axes only (gripping/torsion)")
    p.add_argument("--no-tare", action="store_true",
                   help="disable the no-load zero (tare) correction at startup")
    p.add_argument("--tare-sec", type=float, default=1.0,
                   help="no-load averaging window (seconds) at startup (or on 'C' key)")
    p.add_argument("--simulate", action="store_true",
                   help="test the recording pipeline with a synthetic signal (no hardware)")
    args = p.parse_args()

    rate = args.rate
    num_channels = args.channels
    if num_channels < MIN_CHANNELS:
        print(f"[error] wiring mapping needs at least {MIN_CHANNELS} channels "
              f"(requested: {num_channels})", file=sys.stderr)
        sys.exit(1)

    # --- resolve calibration (decoupling/scale) (same as visualize_forces.py) ---
    calib_path = args.calib
    if calib_path is None and not args.no_calib:
        sibling = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calib.json")
        if os.path.exists(sibling):
            calib_path = sibling
    if args.no_calib:
        calib_path = None
    if calib_path:
        forces_fn, _ = load_calibration(calib_path)
        print(f"[calib] applying decoupling matrix: {calib_path}")
        unit = "N"
    else:
        forces_fn = compute_forces
        print("[calib] raw mode (no calibration). Use --calib to apply a matrix")
        unit = "raw"
    if args.ymin is None:
        args.ymin = -12.0 if calib_path else -2.0
    if args.ymax is None:
        args.ymax = 12.0 if calib_path else 2.0

    # samples read per channel per chunk (about 10 fps refresh)
    chunk = max(1, int(round(rate / 10)))
    window_samples = max(chunk, int(round(args.window * rate)))

    # --- prepare save folder ---
    # Single device only, so a dataset number is not required.
    # With --dataset it goes into a subfolder; without it, straight into collected_data.
    dataset_num = args.dataset or ""
    save_root = args.out_root or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "collected_data")
    save_dir = os.path.join(save_root, dataset_num) if dataset_num else save_root
    os.makedirs(save_dir, exist_ok=True)
    print(f"[save] dataset folder: {save_dir}\n")

    # --- prepare data source (same as visualize_forces.py) ---
    if args.simulate:
        print("[mode] simulation (synthetic signal)")
        source = SimSource(num_channels, rate, chunk)
    else:
        print(f"[mode] driver socket={args.socket}  channels={num_channels}  "
              f"rate={rate} Hz")
        try:
            source = SocketSource(args.socket, num_channels, rate, chunk,
                                  decim_mode=args.decimate)
        except Exception as e:
            print(f"socket connect/setup failed: {e}", file=sys.stderr)
            sys.exit(1)
        if source.avg:
            print(f"  driver stream {source._drv_rate} Hz -> block-average x{source.decim} "
                  f"-> {source.actual_rate:.1f} Hz")
        else:
            print(f"  driver stream {source._drv_rate} Hz -> subsample x{source.decim} "
                  f"-> {source.actual_rate:.1f} Hz")
    actual_rate = source.actual_rate

    # --- ring buffers (per force) ---
    t_axis = -np.arange(window_samples)[::-1] / rate
    buffers = {key: deque([0.0] * window_samples, maxlen=window_samples)
               for key in FORCE_KEYS}

    # --- tare state: subtract the no-load average to zero it out (same as visualize_forces) ---
    tare_enabled = not args.no_tare
    tare_baseline = {key: None for key in FORCE_KEYS}
    tare_acc = {key: [] for key in FORCE_KEYS}

    # --- per-axis low-pass (one-pole IIR) (same as visualize_forces) ---
    def _cutoff_for(k):
        if k in QUARTER_AXES and args.quarter_cutoff is not None:
            return args.quarter_cutoff
        return args.cutoff
    axis_cutoff = {k: _cutoff_for(k) for k in FORCE_KEYS}
    axis_alpha = {k: alpha_from_cutoff(c, rate) for k, c in axis_cutoff.items()}
    lpf_state = {k: None for k in FORCE_KEYS}

    # --- recording state ---
    # 데이터 수신/처리/기록은 전용 reader 스레드가, start/stop 은 GUI 스레드가 수행하므로
    # rec 은 rec_lock 으로, 시각화 링버퍼는 buf_lock 으로 보호한다.
    rec = {'active': False, 'h5': None, 'h5_path': None, 'index': 0,
           'start_unix': None, 'last_ts': None, 'last_flush': None,
           'ds_ts': None, 'ds_forces': None, 'ds_raw': None}
    rec_lock = threading.Lock()
    buf_lock = threading.Lock()
    # tare 상태(tare_baseline/tare_acc)는 reader 스레드 소유 — 다른 스레드(키/UDP)는
    # 직접 만지지 않고 이 플래그만 세운다.
    pending = {'retare': False}

    def start_recording():
        if rec['active']:
            return
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        h5_path = os.path.join(save_dir, f"episode_{ts_str}.h5")
        suffix = 2
        while os.path.exists(h5_path):    # avoid second-resolution filename collisions
            h5_path = os.path.join(save_dir, f"episode_{ts_str}_{suffix}.h5")
            suffix += 1

        h5 = h5py.File(h5_path, 'w')
        start_unix = time.time()

        # --- absolute time and metadata ---
        h5.attrs['dataset_number']  = dataset_num
        h5.attrs['source']          = 'SIMULATION' if args.simulate else args.socket
        h5.attrs['rate']            = float(rate)
        h5.attrs['actual_rate']     = float(actual_rate)
        h5.attrs['channels']        = int(num_channels)
        h5.attrs['calib_mode']      = 'decoupled' if calib_path else 'raw'
        if calib_path:
            h5.attrs['calib_path']  = calib_path
        h5.attrs['force_keys']      = list(FORCE_KEYS)                 # forces column order
        h5.attrs['force_units']     = [UNIT[k] for k in FORCE_KEYS]
        h5.attrs['raw_keys']        = list(RAW_KEYS)                   # raw column order
        h5.attrs['tare_enabled']    = bool(tare_enabled)
        h5.attrs['tare_sec']        = float(args.tare_sec)
        h5.attrs['cutoff_hz']       = float(args.cutoff)
        if args.quarter_cutoff is not None:
            h5.attrs['quarter_cutoff_hz'] = float(args.quarter_cutoff)
        h5.attrs['decimate']        = args.decimate
        h5.attrs['start_unix']      = start_unix
        h5.attrs['start_iso_utc']   = datetime.now(timezone.utc).isoformat()
        h5.attrs['timestamp_units'] = 'unix_epoch_seconds_utc'
        h5.attrs['timestamp_note']  = ('timestamp is PC absolute time (Unix epoch, UTC), stamped by a '
                                       'dedicated reader thread the moment each chunk arrives from the '
                                       'driver socket (independent of GUI load), then back-computed per '
                                       'sample at 1/actual_rate spacing. '
                                       'Use it to align/merge with other datasets like MCU/dataset_recorder.')
        h5.attrs['values_note']     = ('forces are identical to what visualize_forces.py draws on screen '
                                       '(after calib+tare+LPF). raw is the channel voltages (V).')

        rec['ds_ts']     = _create_ds(h5, 'timestamp', (), 'f8')           # absolute time
        rec['ds_forces'] = _create_ds(h5, 'forces', (len(FORCE_KEYS),), 'f4')  # same as displayed
        rec['ds_raw']    = _create_ds(h5, 'raw', (len(RAW_KEYS),), 'f4')       # channel voltages

        # reader 스레드가 rec_lock 아래에서 active 를 확인/기록하므로, 상태 전환도
        # 같은 잠금 아래에서 원자적으로 수행한다.
        with rec_lock:
            rec.update({'active': True, 'h5': h5, 'h5_path': h5_path,
                        'index': 0, 'start_unix': start_unix, 'last_ts': None,
                        'last_flush': start_unix})
        print(f"\n[REC] recording started  ->  {h5_path}")

    def stop_recording():
        with rec_lock:                 # reader 스레드의 진행 중인 append 와 배타적으로
            if not rec['active']:
                return
            rec['active'] = False
            if rec['h5']:
                rec['h5'].attrs['num_samples'] = rec['index']
                rec['h5'].attrs['end_unix'] = time.time()
                rec['h5'].flush()
                rec['h5'].close()
                rec['h5'] = None
        print(f"\n[STOP] recording stopped -- {rec['index']} samples saved ({rec['h5_path']})\n")

    def retare():
        # tare 상태는 reader 스레드 소유 — 여기서는 요청 플래그만 세운다.
        pending['retare'] = True
        print("[tare] re-zeroing forces -- taking the current no-load state as zero.")

    # --- plot setup (same as the graph part of visualize_forces.py, snapshot UI removed) ---
    plt.style.use("default")
    fig = plt.figure(figsize=(13, 6.5))
    title_tag = f" [{dataset_num}]" if dataset_num else ""
    fig.canvas.manager.set_window_title(f"NI Force Recorder{title_tag}")
    ax = fig.add_axes([0.07, 0.10, 0.66, 0.80])
    ax2 = ax.twinx()           # right axis: RIGHT_AXIS (= torsion, unit N.m)
    right_color = next((c for k, _, c, _ in FORCES if k in RIGHT_AXIS), "tab:purple")

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

    ax2.set_ylim(-args.torsion_ymax, args.torsion_ymax)
    ax2.set_ylabel("Torsion (N.m)", color=right_color)
    ax2.tick_params(axis="y", labelcolor=right_color)
    ax2.spines["right"].set_color(right_color)

    src_label = "SIMULATION" if args.simulate else "USB-6009"
    mode_label = "decoupled" if calib_path else "raw"
    base_title = f"NI {src_label} recorder -- {len(FORCES)} forces ({mode_label}) @ {rate:.0f} Hz"
    ax.set_title(base_title)
    ax.grid(True, alpha=0.3)
    ax.axhline(0.0, color="black", lw=0.6, alpha=0.4)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", ncol=3, fontsize=8)

    # right info panel: key hints + live values + recording status + load status
    fig.text(0.76, 0.86, "[C] tare   [R] toggle record   [Q] quit",
             fontsize=8, color="dimgray", va="top")
    load_txt = fig.text(0.76, 0.92, "load: waiting for data", fontsize=8,
                        family="monospace", va="top", color="darkorange")
    live_txt = fig.text(0.76, 0.78, "", fontsize=9, family="monospace", va="top")
    rec_txt = fig.text(0.76, 0.14, "", fontsize=10, family="monospace",
                       va="bottom", fontweight="bold")

    def on_key(event):
        k = (event.key or "").lower()
        if k == "c":
            retare()
        elif k == "r":
            stop_recording() if rec['active'] else start_recording()
        elif k == "q":
            stop_recording()
            plt.close("all")

    fig.canvas.mpl_connect("key_press_event", on_key)

    # 전역 신호(record_signal.py) 로도 녹화 시작/중지를 받는다. 로컬 [R] 키와 동일한 동작.
    def _toggle_recording():
        stop_recording() if rec['active'] else start_recording()
    record_signal = RecordSignal(on_start=start_recording, on_stop=stop_recording,
                                 on_toggle=_toggle_recording)
    record_signal.start()

    # UDP 스트리밍 + 원격 tare (SURGITAG_UDP 미설정 시 전부 no-op)
    udp = UdpStreamer('ni6009')
    udp.set_tare_callback(retare)

    source.start()

    # =======================================================================
    # 전용 수신 스레드 — 소켓에서 청크가 도착하는 '즉시' 그 자리에서 타임스탬프를
    # 찍고, tare/LPF 후처리와 HDF5 기록까지 전부 이 스레드에서 수행한다.
    # GUI(Matplotlib)가 밀리거나 멈춰도 타임스탬프/기록에는 영향이 없다.
    # =======================================================================
    load_mon = LoadMonitor(nominal_hz=actual_rate, stall_sec=1.5)
    reader_stop = threading.Event()

    def reader_loop():
        sim_next = time.monotonic()
        while not reader_stop.is_set():
            try:
                data = source.read()  # (num_channels, chunk) — 하드웨어 소켓은 블로킹
            except Exception as e:
                if reader_stop.is_set():
                    return
                print(f"read error: {e}", file=sys.stderr)
                return
            if args.simulate:
                # SimSource.read() 는 즉시 반환하므로 실제 수집 속도로 페이싱한다.
                sim_next += chunk / actual_rate
                delay = sim_next - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    sim_next = time.monotonic()
            t_read = time.time()      # ★ 수신 직후의 절대시간 = 이 청크 마지막 샘플의 시각
            n = data.shape[1]
            load_mon.tick(n)

            # retare 요청 처리 (다른 스레드는 플래그만 세움 — tare 상태는 이 스레드 소유)
            if pending['retare']:
                pending['retare'] = False
                for key in FORCE_KEYS:
                    tare_baseline[key] = None
                    tare_acc[key] = []

            # --- exactly the same post-processing as visualize_forces.py (tare -> LPF) ---
            forces = forces_fn(data)
            proc = {}
            for key in FORCE_KEYS:
                vals = forces[key]
                if tare_enabled:
                    if tare_baseline[key] is None:
                        tare_acc[key].append(np.asarray(vals, dtype=np.float64).copy())
                        base = float(np.concatenate(tare_acc[key]).mean())
                        if sum(a.size for a in tare_acc[key]) >= args.tare_sec * rate:
                            tare_baseline[key] = base   # lock the zero once enough is collected
                        vals = vals - base
                    else:
                        vals = vals - tare_baseline[key]
                if axis_alpha[key] is not None:
                    vals, lpf_state[key] = onepole_lpf(vals, axis_alpha[key],
                                                       lpf_state[key])
                proc[key] = vals

            # 시각화 링버퍼 갱신 (GUI 는 buf_lock 아래에서 읽기만 한다)
            with buf_lock:
                for key in FORCE_KEYS:
                    buffers[key].extend(proc[key])

            # --- 워크스테이션 스트리밍 (화면/기록과 동일한 tare+LPF 후처리 값) ---
            if udp.enabled:
                udp.force_block(
                    t_read - (n - 1) / actual_rate, 1.0 / actual_rate,
                    np.column_stack([proc[k] for k in FORCE_KEYS]),
                    np.column_stack([data[ch] for _, ch in RAW_CHANNELS]),
                    tared=(tare_enabled
                           and all(tare_baseline[k] is not None
                                   for k in FORCE_KEYS)))

            # --- if recording, log this chunk together with absolute time ---
            if rec['active']:
                with rec_lock:
                    if not rec['active']:
                        continue
                    # the n samples of this chunk sit in [t_read-(n-1)/fs, t_read] -> back out
                    # per-sample time. The blocking socket read makes t_read naturally monotonic;
                    # the guard below only protects against chunk jitter / simulate overlap.
                    ts = t_read - (n - 1 - np.arange(n)) / actual_rate
                    if rec['last_ts'] is not None and ts[0] <= rec['last_ts']:
                        ts = ts + (rec['last_ts'] + 1.0 / actual_rate - ts[0])
                    rec['last_ts'] = float(ts[-1])
                    fblock = np.column_stack([proc[k] for k in FORCE_KEYS]).astype(np.float32)
                    rblock = np.column_stack([data[ch] for _, ch in RAW_CHANNELS]).astype(np.float32)
                    _append_block(rec['ds_ts'], ts.astype(np.float64))
                    _append_block(rec['ds_forces'], fblock)
                    _append_block(rec['ds_raw'], rblock)
                    rec['index'] += n
                    # periodic flush: even on a sudden power-off, the file opens and
                    # recovers up to the last flush.
                    if t_read - rec['last_flush'] >= FLUSH_INTERVAL_S:
                        rec['h5'].flush()
                        rec['last_flush'] = t_read

    reader = threading.Thread(target=reader_loop, daemon=True, name="daq_reader")
    reader.start()

    # =======================================================================
    # GUI 콜백 — 시각화 전용. 데이터 수신/기록과 완전히 분리되어 있어, 여기가
    # 느려져도 타임스탬프는 흔들리지 않는다 (부하는 load_txt 로 표시만 된다).
    # =======================================================================
    def update(_frame):
        # 전역 START/STOP/TOGGLE 신호 처리 (GUI 스레드에서 실행 — [R] 키와 동일 경로).
        # start/stop 콜백은 rec_lock 으로 reader 스레드와 배타 실행된다.
        record_signal.poll()

        with buf_lock:
            for key in FORCE_KEYS:
                lines[key].set_ydata(buffers[key])
            last_vals = {k: (buffers[k][-1] if buffers[k] else 0.0)
                         for k in FORCE_KEYS}

        # --- 수신 부하 상태 (문자+색) ---
        txt, _, mpl_color = load_mon.status()
        load_txt.set_text(txt)
        load_txt.set_color(mpl_color)

        # --- update status text ---
        live_txt.set_text(
            "live value:\n" + "\n".join(
                f"  {lbl:<11s}{last_vals[k]:+8.3f}"
                for k, lbl, _, _ in FORCES))
        if rec['active']:
            elapsed = time.time() - rec['start_unix']
            rec_txt.set_text(f"REC  {rec['index']} samples\n"
                             f"   {elapsed:5.1f}s\n   {os.path.basename(rec['h5_path'])}")
            rec_txt.set_color("red")
            ax.set_title(base_title + "   REC")
        else:
            rec_txt.set_text("idle -- press 'R' to record")
            rec_txt.set_color("dimgray")
            ax.set_title(base_title)

    ani = FuncAnimation(fig, update, interval=50, blit=False,
                        cache_frame_data=False)

    print("[keys] C=tare  R=record start/stop  Q=quit")
    print(f"[save] {save_dir}/episode_<time>.h5")
    try:
        plt.show()
    finally:
        reader_stop.set()      # 수신 스레드 종료 신호 (close 로 read 가 깨어나며 끝난다)
        stop_recording()
        source.close()
        print("Exiting.")


if __name__ == "__main__":
    main()
