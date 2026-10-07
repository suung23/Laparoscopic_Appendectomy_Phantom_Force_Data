"""
load_monitor.py — 수신 루프 부하 측정 + GUI 표시용 상태 문자열/색
=================================================================
모든 레코더는 "센서에서 데이터를 받는 즉시 그 자리에서 타임스탬프를 찍는다"는
원칙으로 동작한다. 이 원칙이 깨지는 유일한 경우는 수신 루프가 부하로 밀려서
드라이버/OS 버퍼에 쌓인 '과거' 데이터에 '현재' 시각이 찍힐 때다.

이 모듈은 그 상황을 감지해 GUI 에 문자+색으로 보여 주기 위한 작은 도우미다.
  · tick(n)    : 수신 스레드가 데이터를 받을 때마다 호출 (n = 샘플/프레임 수)
  · status(...) : GUI 스레드가 호출 — (표시 문자열, BGR 색, matplotlib 색) 반환

판정 기준 (worst 가 최종 레벨):
  · 수신율 저하   — 실측 수신율 < nominal_hz * warn_ratio → SLOW,
                    < nominal_hz * bad_ratio → OVERLOAD
  · 수신 정지     — 마지막 수신 후 stall_sec 초과 → STALL(OVERLOAD 색)
  · 큐 적체       — status(queue_depth=...) 로 넘기면 임계 초과 시 경고
  · 드롭 발생     — status(drops=...) 로 넘기면 1개 이상 드롭 시 경고

스레드 안전: tick() 은 수신 스레드, status() 는 GUI 스레드에서 불려도 안전하다.
"""
import threading
import time

OK, WARN, BAD = 0, 1, 2
_BGR = {OK: (0, 220, 0), WARN: (0, 220, 255), BAD: (0, 0, 255)}       # OpenCV(BGR)
_MPL = {OK: "green",     WARN: "darkorange",  BAD: "red"}             # matplotlib


class LoadMonitor:
    def __init__(self, nominal_hz=None, warn_ratio=0.85, bad_ratio=0.6,
                 stall_sec=1.0, window_sec=1.0):
        self.nominal_hz = nominal_hz     # 기대 수신율(Hz). None 이면 수신율 판정 생략
        self.warn_ratio = warn_ratio
        self.bad_ratio = bad_ratio
        self.stall_sec = stall_sec
        self.window_sec = window_sec     # 수신율을 계산하는 슬라이딩 창 크기(초)
        self._lock = threading.Lock()
        self._win_t0 = None              # 현재 측정 창의 시작 (monotonic)
        self._win_n = 0                  # 현재 창에서 받은 샘플 수
        self._rate = None                # 직전 창에서 확정된 수신율(Hz)
        self._last = None                # 마지막 tick 시각 (monotonic)

    def set_nominal(self, hz):
        """실측으로 확정된 기대 수신율을 나중에 넣을 때 사용."""
        self.nominal_hz = hz

    def tick(self, n=1):
        """데이터를 받은 '즉시' 수신 스레드에서 호출한다."""
        now = time.monotonic()
        with self._lock:
            if self._win_t0 is None:
                self._win_t0 = now
            self._win_n += n
            self._last = now
            dt = now - self._win_t0
            if dt >= self.window_sec:
                self._rate = self._win_n / dt
                self._win_t0, self._win_n = now, 0

    def status(self, queue_depth=None, queue_warn=15, queue_bad=60,
               drops=0, extra=""):
        """(text, bgr_color, mpl_color) 를 반환한다. GUI 스레드에서 호출."""
        now = time.monotonic()
        with self._lock:
            rate, last = self._rate, self._last

        if last is None:
            return "load: waiting for data", _BGR[WARN], _MPL[WARN]

        level = OK
        parts = []

        # 1) 수신 정지 (가장 심각 — 과부하이거나 장치가 멈춘 것)
        stall = now - last
        if stall > self.stall_sec:
            level = BAD
            parts.append(f"STALL {stall:.1f}s")

        # 2) 수신율 저하 (버퍼에 쌓인 과거 데이터에 현재 시각이 찍히는 중일 수 있음)
        if rate is not None:
            if self.nominal_hz:
                if rate < self.nominal_hz * self.bad_ratio:
                    level = max(level, BAD)
                elif rate < self.nominal_hz * self.warn_ratio:
                    level = max(level, WARN)
                parts.append(f"{rate:.1f}/{self.nominal_hz:.0f}Hz")
            else:
                parts.append(f"{rate:.1f}Hz")

        # 3) 큐 적체 (기록 스레드가 밀리는 중 — 타임스탬프는 안전, 저장만 지연)
        if queue_depth is not None:
            if queue_depth >= queue_bad:
                level = max(level, BAD)
            elif queue_depth >= queue_warn:
                level = max(level, WARN)
            if queue_depth > 0:
                parts.append(f"q{queue_depth}")

        # 4) 드롭 (큐가 넘쳐 버린 프레임 수 — 데이터 손실이므로 항상 경고 이상)
        if drops:
            level = max(level, WARN if drops < 30 else BAD)
            parts.append(f"drop{drops}")

        if extra:
            parts.append(extra)

        label = {OK: "load ok", WARN: "load HIGH", BAD: "load OVERLOAD"}[level]
        text = label + ("  " + "  ".join(parts) if parts else "")
        return text, _BGR[level], _MPL[level]
