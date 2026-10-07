#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 3 -- host-native visualizer for the NI USB-6009.

The driver (linux_daqmxbase/daq_reader, running in the Docker container) samples
all 8 channels at a high rate (1000 Hz) and streams raw float64 values over a
Unix-domain socket. This program connects to that socket, decimates the stream
down to a 100 Hz display rate, and plots it live with matplotlib.

It runs on the host (NOT in Docker), e.g. in the `surgitag` conda env:
    conda activate surgitag
    python visualizer.py --socket /tmp/surgitag/daq.sock --display-rate 100 --window 5

Socket framing (see linux_daqmxbase/daq_reader.c):
    header  : char magic[4]="NIDQ"; int32 nchan; int32 rate; int32 chunk
    frames  : nchan*chunk float64, grouped by channel
"""
import argparse
import socket
import struct
import sys
import threading
import time
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

HEADER = struct.Struct("<4s i i i")   # magic, nchan, rate, chunk


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


def main():
    p = argparse.ArgumentParser(description="USB-6009 visualizer (reads driver socket, plots)")
    p.add_argument("--socket", default="/tmp/surgitag/daq.sock", help="driver socket path")
    p.add_argument("--display-rate", type=float, default=100.0,
                   help="plot sample rate in Hz (driver's stream is decimated to this)")
    p.add_argument("--window", type=float, default=5.0, help="display time window (seconds)")
    p.add_argument("--vmin", type=float, default=-5.0, help="y-axis min voltage (V)")
    p.add_argument("--vmax", type=float, default=5.0, help="y-axis max voltage (V)")
    args = p.parse_args()

    print(f"[viz] connecting to socket: {args.socket}")
    sock, nchan, rate, chunk = connect(args.socket)

    # decimate the driver's stream (e.g. 1000 Hz) down to the display rate (100 Hz)
    decim = max(1, int(round(rate / args.display_rate)))
    disp_rate = rate / decim
    print(f"[viz] connected -- {nchan}ch @ {rate} Hz; decimate x{decim} -> {disp_rate:.1f} Hz display")

    window_samples = max(2, int(round(args.window * disp_rate)))
    t_axis = -np.arange(window_samples)[::-1] / disp_rate
    ydata = [deque([0.0] * window_samples, maxlen=window_samples) for _ in range(nchan)]
    lock = threading.Lock()
    state = {"alive": True, "phase": 0}

    frame_struct = struct.Struct("<%dd" % (nchan * chunk))

    def reader_thread():
        try:
            while state["alive"]:
                raw = recvall(sock, frame_struct.size)
                if raw is None:
                    print("[viz] driver connection closed", file=sys.stderr)
                    break
                flat = frame_struct.unpack(raw)  # GroupByChannel: ch*chunk + i
                # pick every `decim`-th sample, keeping phase across frame boundaries
                phase = state["phase"]
                idx = range((decim - phase) % decim, chunk, decim)
                state["phase"] = (phase + chunk) % decim
                with lock:
                    for ch in range(nchan):
                        base = ch * chunk
                        ydata[ch].extend(flat[base + i] for i in idx)
        finally:
            state["alive"] = False

    threading.Thread(target=reader_thread, daemon=True).start()

    fig, ax = plt.subplots(figsize=(11, 6))
    cmap = plt.get_cmap("tab10")
    lines = [ax.plot(t_axis, np.zeros(window_samples), lw=1.2,
                     color=cmap(ch % 10), label=f"AI{ch}")[0] for ch in range(nchan)]
    ax.set_xlim(t_axis[0], 0)
    pad = 0.05 * (args.vmax - args.vmin)
    ax.set_ylim(args.vmin - pad, args.vmax + pad)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title(f"NI USB-6009 -- {nchan}ch, {rate:.0f} Hz acquire / {disp_rate:.0f} Hz display")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", ncol=nchan, fontsize=8)

    def update(_frame):
        if not state["alive"]:
            ax.set_title("driver connection closed")
        with lock:
            for ch in range(nchan):
                lines[ch].set_ydata(ydata[ch])
        return lines

    # keep a reference — without it the animation is GC'd and never renders
    ani = FuncAnimation(fig, update, interval=20, blit=False, cache_frame_data=False)
    try:
        plt.show()
    finally:
        state["alive"] = False
        sock.close()
        print("[viz] stopped")


if __name__ == "__main__":
    main()
