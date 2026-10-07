#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
phantom_experiment.py 에 붙는 엔도스코프 영상 — 미리보기 + 녹화.

왜 Camera/camera_recorder.py 를 쓰지 않는가
  그 파일은 모듈 레벨에서 _parse_args() 를 돌리고 카메라 인덱스를 대화형으로 물어서
  import 가 불가능하다. 그래서 필요한 부분만 여기 다시 쓴다. 저장 형식(mp4 + 절대시간
  h5 사이드카, timestamp 데이터셋 이름·단위)은 그쪽과 맞춰 둬서 Annotation 계통이
  그대로 읽을 수 있다.

장치 하나는 프로세스 하나만 연다. 이 모듈이 카메라를 가지므로, 같은 카메라로
camera_recorder.py 를 동시에 돌리면 안 된다.

구조 (레코더들의 공통 원칙과 같다):
  grab 스레드  — 프레임을 받은 '즉시' time.time() 을 찍는다. GUI 가 멈춰도 안 밀린다.
  write 스레드 — 큐에서 꺼내 mp4 로 인코딩. 인코딩이 느려도 타임스탬프는 안 밀리고,
                 큐가 넘치면 버린 장수를 센다(= 조용한 손실을 만들지 않는다).
  GUI          — latest() 로 최근 프레임만 가져가 그린다. 느리면 건너뛸 뿐이다.
"""
import os
import queue
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone

import cv2
import h5py
import numpy as np

FFMPEG_BIN = shutil.which("ffmpeg")
X264_CRF = "20"
X264_PRESET = "veryfast"
CHUNK = 256
QUEUE_MAX = 60               # 1920x1080x3 ≈ 6 MB/프레임 → 약 370 MB 상한


class _FfmpegWriter:
    """시스템 ffmpeg 으로 H.264 인코딩. OpenCV 빌드에 H.264 가 없어도 재생 가능한 mp4."""

    codec = "h264 (libx264)"

    def __init__(self, path, fps, size):
        w, h = size
        self.p = subprocess.Popen(
            [FFMPEG_BIN, "-y", "-loglevel", "error",
             "-f", "rawvideo", "-pix_fmt", "bgr24",
             "-s", "%dx%d" % (w, h), "-r", "%.6f" % fps, "-i", "-",
             "-an", "-c:v", "libx264", "-preset", X264_PRESET, "-crf", X264_CRF,
             "-pix_fmt", "yuv420p", path],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL)

    def isOpened(self):
        return self.p.poll() is None

    def write(self, frame):
        try:
            self.p.stdin.write(frame.tobytes())
        except (BrokenPipeError, ValueError):
            pass

    def release(self):
        try:
            if self.p.stdin:
                self.p.stdin.close()
            self.p.wait(timeout=10)
        except Exception:
            try:
                self.p.kill()
            except Exception:
                pass


def _open_writer(path, fps, size):
    if FFMPEG_BIN:
        try:
            w = _FfmpegWriter(path, fps, size)
            time.sleep(0.15)
            if w.isOpened():
                return w, w.codec
            w.release()
        except Exception:
            pass
        print("[cam] ffmpeg 인코더 실패 — OpenCV 코덱으로 폴백")
    else:
        print("[cam] ffmpeg 없음 — OpenCV 코덱으로 저장 (재생 안 되면 apt install ffmpeg)")
    for cc in ("avc1", "mp4v"):
        w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*cc), fps, size)
        if w.isOpened():
            if cc != "avc1":
                print("[cam] '%s' 코덱으로 저장 — 기본 플레이어에서 안 열릴 수 있다 "
                      "(정렬용 h5 는 정상)" % cc)
            return w, cc
        w.release()
    return None, None


class PhantomCamera:
    """엔도스코프 한 대: 미리보기 프레임 공급 + 에피소드 녹화."""

    def __init__(self, index, width=1920, height=1080, req_fps=30.0,
                 wb=True, preview_width=480):
        self.index = int(index)
        self.req = (int(width), int(height), float(req_fps))
        self.wb_enabled = bool(wb)
        self.preview_width = int(preview_width)

        self.cap = None
        self.size = (0, 0)
        self.fps = float(req_fps)        # 실측으로 덮어쓴다
        self.name = "cam%d" % self.index

        self._stop = threading.Event()
        self._grab_thr = None
        self._write_thr = None
        self._latest = None              # (preview_rgb, ts)
        self._latest_lock = threading.Lock()

        self._q = queue.Queue(maxsize=QUEUE_MAX)
        self._rec = None                 # dict(writer, h5, ds, ...)
        self._rec_lock = threading.Lock()
        self._gate = True                # False 면 녹화 중이어도 프레임을 안 쓴다
                                         # (= 시행 사이 구간을 mp4 에 안 담는다)

        self.frames = 0
        self.dropped = 0                 # 큐가 넘쳐 버린 프레임 (녹화 중에만 의미)
        self.read_fail = 0
        self._wb_gain = None             # (bg, gg, rg) — 에피소드 내내 고정

    # ------------------------------------------------------------ 열기
    def open(self):
        w, h, fps = self.req
        cap = cv2.VideoCapture(self.index, cv2.CAP_V4L2)
        if not cap.isOpened():
            print("[cam] index=%d 를 열 수 없다" % self.index)
            return False
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, fps)
        for _ in range(10):
            cap.read()
        ok, fr = cap.read()
        if not ok or fr is None:
            print("[cam] index=%d 에서 프레임을 못 받는다" % self.index)
            cap.release()
            return False
        self.cap = cap
        self.size = (fr.shape[1], fr.shape[0])
        self.fps = self._measure_fps()
        print("[cam] index=%d  %dx%d  실측 %.2f fps  (요청 %.0f)"
              % (self.index, self.size[0], self.size[1], self.fps, fps))
        if abs(self.fps - fps) > 2.0:
            print("[cam] ! 표기와 실측이 다르다 — mp4 는 실측 fps 로 인코딩한다")
        return True

    def _measure_fps(self, n=60):
        """표기 fps 를 믿지 않는다. 실측값으로 mp4 를 인코딩해야 재생 속도가 맞는다."""
        ts = []
        for _ in range(n):
            ok, _f = self.cap.read()
            if ok:
                ts.append(time.time())
        if len(ts) < 10:
            return self.req[2]
        return (len(ts) - 1) / (ts[-1] - ts[0])

    # ------------------------------------------------------- 화이트밸런스
    def _gray_world(self, frame):
        b, g, r = [float(frame[..., i].mean()) + 1e-6 for i in range(3)]
        m = (b + g + r) / 3.0
        return (m / b, m / g, m / r)

    def _apply_wb(self, frame):
        if not self.wb_enabled or self._wb_gain is None:
            return frame
        out = frame.astype(np.float32)
        for i, gain in enumerate(self._wb_gain):
            out[..., i] *= gain
        return np.clip(out, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------ 스레드
    def start(self):
        if self.wb_enabled:
            ok, fr = self.cap.read()
            if ok:
                self._wb_gain = self._gray_world(fr)
                print("[cam] WB 게인 고정 B/G/R = %.3f / %.3f / %.3f" % self._wb_gain)
        self._grab_thr = threading.Thread(target=self._grab_loop, daemon=True)
        self._grab_thr.start()
        self._write_thr = threading.Thread(target=self._write_loop, daemon=True)
        self._write_thr.start()

    def _grab_loop(self):
        pw = self.preview_width
        ph = max(1, int(round(self.size[1] * pw / max(1, self.size[0]))))
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            ts = time.time()                      # 받은 '즉시' 찍는다
            if not ok or frame is None:
                self.read_fail += 1
                time.sleep(0.01)
                continue
            self.frames += 1
            frame = self._apply_wb(frame)

            if self._rec is not None and self._gate:
                try:
                    self._q.put_nowait((frame, ts))
                except queue.Full:
                    self.dropped += 1             # 조용히 넘기지 않는다

            small = cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_AREA)
            with self._latest_lock:
                self._latest = (cv2.cvtColor(small, cv2.COLOR_BGR2RGB), ts)

    def _write_loop(self):
        while not self._stop.is_set():
            try:
                frame, ts = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            with self._rec_lock:
                rec = self._rec
                if rec is None:
                    continue
                try:
                    rec['writer'].write(frame)
                    ds = rec['ds']
                    ds.resize(ds.shape[0] + 1, axis=0)
                    ds[-1] = ts
                    rec['n'] += 1
                    now = time.time()
                    if now - rec['last_flush'] > 1.0:      # crash-safe
                        rec['h5'].flush()
                        rec['last_flush'] = now
                except Exception as e:
                    print("[cam] 기록 오류: %s" % e)

    # ------------------------------------------------------------ 녹화
    def set_gate(self, on):
        """True 인 동안에만 프레임이 mp4 에 들어간다. 미리보기는 항상 돈다."""
        self._gate = bool(on)

    def latest(self):
        with self._latest_lock:
            return self._latest

    def start_recording(self, base_path):
        """base_path 는 확장자 없는 경로. <base>.mp4 / <base>.h5 를 만든다."""
        with self._rec_lock:
            if self._rec is not None:
                return
            mp4_path, h5_path = base_path + ".mp4", base_path + ".h5"
            writer, codec = _open_writer(mp4_path, self.fps, self.size)
            if writer is None:
                print("[cam] ❌ 인코더를 열 수 없어 영상 녹화를 시작하지 못했다")
                return
            h5 = h5py.File(h5_path, 'w')
            h5.attrs['camera_index'] = self.index
            h5.attrs['video_file'] = os.path.basename(mp4_path)
            h5.attrs['codec'] = codec
            h5.attrs['frame_width'] = self.size[0]
            h5.attrs['frame_height'] = self.size[1]
            h5.attrs['measured_fps'] = float(self.fps)
            h5.attrs['wb_applied'] = bool(self.wb_enabled)
            if self._wb_gain:
                h5.attrs['wb_gain_bgr'] = list(self._wb_gain)
            h5.attrs['start_unix'] = time.time()
            h5.attrs['start_iso_utc'] = datetime.now(timezone.utc).isoformat()
            h5.attrs['timestamp_units'] = 'unix_epoch_seconds_utc'
            h5.attrs['values_note'] = (
                'timestamp[i] = absolute time of frame i in the mp4 (same clock as '
                'the force h5). mp4 carries no per-frame absolute time; use this. '
                'Frames are written only while recording is gated on (trial windows), '
                'so the mp4 is a concatenation of segments: playback time is NOT '
                'proportional to wall clock. Gaps show up as jumps in timestamp[].')
            ds = h5.create_dataset('timestamp', shape=(0,), maxshape=(None,),
                                   dtype='f8', chunks=(CHUNK,),
                                   compression='gzip', compression_opts=4)
            self._rec = {'writer': writer, 'h5': h5, 'ds': ds, 'n': 0,
                         'last_flush': time.time(), 'mp4': mp4_path, 'h5_path': h5_path}
            self.dropped = 0
            print("[cam] 영상 녹화 시작 → %s" % os.path.basename(mp4_path))

    def stop_recording(self):
        with self._rec_lock:
            rec, self._rec = self._rec, None
        if rec is None:
            return
        # 큐에 남은 프레임을 마저 쓴다
        drained = 0
        while True:
            try:
                frame, ts = self._q.get_nowait()
            except queue.Empty:
                break
            rec['writer'].write(frame)
            ds = rec['ds']
            ds.resize(ds.shape[0] + 1, axis=0)
            ds[-1] = ts
            rec['n'] += 1
            drained += 1
        rec['writer'].release()
        rec['h5'].attrs['end_unix'] = time.time()
        rec['h5'].attrs['num_frames'] = rec['n']
        rec['h5'].attrs['dropped_frames'] = int(self.dropped)
        rec['h5'].close()
        msg = "[cam] 영상 저장 — %d 프레임" % rec['n']
        if drained:
            msg += " (큐 잔여 %d 포함)" % drained
        if self.dropped:
            msg += "  ⚠ 버림 %d" % self.dropped
        print(msg)

    def close(self):
        self.stop_recording()
        self._stop.set()
        for t in (self._grab_thr, self._write_thr):
            if t is not None:
                t.join(timeout=2.0)
        if self.cap is not None:
            self.cap.release()
