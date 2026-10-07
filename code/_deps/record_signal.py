"""
record_signal.py — 모든 레코더를 한 번에 시작/중지시키는 전역 신호
=================================================================
세 레코더(Camera / MCU / NI6009)는 이 작은 '제어 파일' 하나를 함께 감시합니다.
그래서 어느 터미널에서든 신호 한 번을 보내면 **실행 중인 모든 레코더가 동시에**
녹화를 시작하거나 중지합니다. 어떤 창이 포커스를 가졌는지와 무관합니다.
기존의 수동 [R] 키는 그대로 동작하며, 이 전역 신호는 '추가' 경로입니다.

전송 방식 — 왜 파일인가?
  · 네트워크(UDP)로 하면 localhost 유니캐스트는 한 프로세스에만 전달되고,
    멀티캐스트/브로드캐스트는 PC 가 오프라인이면(경로 없음) 실패할 수 있습니다.
  · 반면 작은 JSON 제어 파일(기본 /tmp/surgitag/record.cmd)을 각 레코더가
    폴링하면: 완전 오프라인에서도, conda/도커 경계를 넘어도, 아무 설정 없이 동작합니다.
    (NI6009 가 이미 쓰는 /tmp/surgitag 규약과 같은 위치를 씁니다.)
  · 각 명령에는 고유 nonce 가 붙습니다. 레코더는 '아직 못 본 nonce' 에만 반응하므로,
    이전 세션이 남긴 낡은 파일 때문에 실수로 녹화가 시작되는 일이 없습니다.
  · 모든 샘플에 어차피 절대시간이 찍히므로, 폴링에 따른 ~1프레임 지연은 나중 정렬에
    아무 영향이 없습니다(사람 눈에는 즉시).

신호 보내기 (아무 터미널에서나, 추가 의존성 없음):
    python3 record_signal.py start     # 실행 중인 모든 레코더가 녹화 시작
    python3 record_signal.py stop      # 모두 녹화 중지 + 저장
    python3 record_signal.py toggle    # 각자 토글(동기화에는 start/stop 권장)

발 페달(foot pedal)로 조종하기 — 페달 두 개와 릴레이 두 개를 한 아두이노에서:
    python3 record_signal.py pedal
    python3 record_signal.py pedal --gui          # + 화면에서 클릭해 소작기 조작
  · D7 = 기록 페달 : 밟을 때마다 전역 START↔STOP 을 번갈아 전송합니다.
                     한 번 밟으면 모든 레코더가 녹화를 시작하고, 다시 밟으면 중지·저장.
  · D8 = 에너지 페달: 에너지 기구(전기소작기)의 ON/OFF 를 모방합니다. 밟고 있는 동안이
                     ON 이고, 그 타이밍이 Energy/collected_datasets/ 에 카메라·MCU 와
                     같은 절대시간으로 실시간 기록됩니다(전역 녹화 구간과 동일 구간).
  · D12/D11 = CUT/COAG 릴레이: 전기소작기의 cut/coag 풋스위치를 **실제로 밟습니다**
                     (모방이 아니라 진짜 에너지). --gui 창에서 패드를 누르고 있는 동안만
                     닫히고, 그 구간도 위와 같은 파일에 mode=CUT/COAG 로 남습니다.
                     안전장치는 Energy/energy_relay.py · 펌웨어 주석을 보세요
                     (상호배타 · PC 무응답 시 자동 차단 · 최대 지속시간 · Esc 비상정지).
                     D8 페달을 실제 릴레이에 물리려면:  --pedal-relay cut|coag
                     릴레이를 아예 끄려면:              --no-relay
  · 두 페달이 아두이노 '한 대' 에 물려 있어 시리얼 포트를 두 프로세스가 나눠 쓸 수
    없습니다. 그래서 포트를 소유한 이 리스너가 두 가지를 함께 처리합니다.
  · 아두이노 펌웨어: Energy/pedal_energy_firmware/pedal_energy_firmware.ino
    (기존 MCU/pedal_switch/pedal_switch.ino 의 상위 호환 — D7 동작은 그대로)
    배선: 각 스위치 COM → D7 / D8,  NO(또는 NC) 중 하나 → GND.
  · 아두이노 포트는 자동으로 찾습니다('?' 질의에 'NAME,pedal...' 로 답하는 포트).
    직접 지정하려면:  python3 record_signal.py pedal --port /dev/ttyACM0
    에너지 기록만 끄려면:  python3 record_signal.py pedal --no-energy
    활성음만 끄려면:      python3 record_signal.py pedal --no-beep
  · D8 을 밟는 동안 PC 스피커로 '삐삐삐' 활성음이 울립니다 — 실제 에너지 기구가
    활성화 중에 내는 소리를 모방해, 리허설의 리듬이 실제와 같아지도록 합니다.
    소리를 미리 들어보려면:  python3 Energy/energy_beep.py --test
  · 시리얼이라 USB 입력장치 권한(sudo/input 그룹)이 필요 없습니다.
    (pyserial 필요. 에너지 기록에는 h5py/numpy 도 필요 — surgitag 환경에서 실행)

레코더 안에서 쓰기:
    from record_signal import RecordSignal
    sig = RecordSignal(on_start=start_recording, on_stop=stop_recording,
                       on_toggle=_toggle_recording)
    sig.start()
    ...  # GUI/메인 루프 안에서 (start/stop 을 소유한 바로 그 스레드에서):
    sig.poll()
"""
import json
import os
import sys
import tempfile
import threading
import time

VALID = ("START", "STOP", "TOGGLE")


def signal_path():
    """제어 파일 경로. NI6009 와 같은 SOCKDIR 규약을 따르며, SURGITAG_SIGNAL 로 덮어쓸 수 있다."""
    sockdir = os.environ.get("SOCKDIR", "/tmp/surgitag")
    return os.environ.get("SURGITAG_SIGNAL", os.path.join(sockdir, "record.cmd"))


def send(cmd):
    """제어 파일에 명령을 원자적으로 기록한다(폴링 중인 레코더가 반쪽 파일을 읽지 않도록)."""
    cmd = cmd.strip().upper()
    if cmd not in VALID:
        raise ValueError(f"command must be one of {VALID}, got {cmd!r}")
    path = signal_path()
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    payload = json.dumps({"cmd": cmd, "nonce": time.time()})
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".record.cmd.")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(payload)
        # mkstemp 은 0600 으로 만든다. 페달을 sudo(root)로 돌리면 파일이 root 소유가 되어
        # 일반 사용자로 실행 중인 레코더들이 못 읽는다. 누구나 읽도록 0644 로 풀어 준다.
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)          # 원자적 교체 → mtime 갱신
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def _read(path):
    """(cmd, nonce) 를 반환. 파일이 없거나 깨졌으면 None."""
    try:
        with open(path) as f:
            obj = json.load(f)
        return str(obj["cmd"]).upper(), float(obj["nonce"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


class RecordSignal:
    """제어 파일을 폴링하다가 '새 명령' 이 오면 콜백을 실행한다.

    poll() 은 start/stop 콜백을 소유한 스레드(= GUI/메인 루프)에서 호출해야 한다.
    그 콜백들은 HDF5/플롯 상태를 만지며, 그 상태는 그 스레드 밖에서 만지면 안전하지 않다.
    (그래서 백그라운드 스레드로 콜백을 부르지 않고, 루프에서 폴링만 한다 — [R] 키와 동일 경로.)
    """
    def __init__(self, on_start=None, on_stop=None, on_toggle=None, path=None):
        self.path = path or signal_path()
        self.on_start = on_start
        self.on_stop = on_stop
        self.on_toggle = on_toggle
        # 시작 시점에 이미 존재하는 파일(이전 세션 잔재)은 '이미 본 것'으로 처리해 무시한다.
        cur = _read(self.path)
        self._last_nonce = cur[1] if cur else None

    def start(self):
        """API 대칭용(파일 방식은 띄울 스레드가 없다). 감시 경로를 한 줄 안내한다."""
        print(f"[signal] 전역 녹화 신호 대기 중  ({self.path})")
        return self

    def poll(self):
        """새 명령이 있으면 해당 콜백을 실행한다. 없으면 즉시 반환.

        파일이 아주 작아 매 폴링마다 읽어도 비용이 무시할 수준이다. mtime 게이트는
        파일시스템의 mtime 해상도가 거칠면(같은 tick 에 두 명령) 명령을 놓칠 수 있어
        쓰지 않는다 — 정확성 판단은 오직 nonce 로 한다.
        """
        cur = _read(self.path)
        if cur is None:
            return
        cmd, nonce = cur
        if nonce == self._last_nonce:      # 아직 못 본 명령에만 반응(중복/잔재 무시)
            return
        self._last_nonce = nonce
        print(f"[signal] 수신: {cmd}")
        if cmd == "START" and self.on_start:
            self.on_start()
        elif cmd == "STOP" and self.on_stop:
            self.on_stop()
        elif cmd == "TOGGLE" and self.on_toggle:
            self.on_toggle()

    def close(self):
        pass


# =====================================================================
# 발 페달(foot pedal) 리스너 — 페달 두 개(D7 기록 / D8 에너지)를 함께 처리
# =====================================================================
# 저가형 발 페달(단순 스위치) 두 개를 아두이노 UNO 의 D7/D8 에 연결하고, 아두이노는
# Energy/pedal_energy_firmware/pedal_energy_firmware.ino 펌웨어로
#   · D7 을 밟는 순간          → "PRESS" (뗄 때 "RELEASE")
#   · D8 상태가 바뀌는 순간    → "ENERGY,<1|0>,<age_ms>"
# 를 시리얼에 보낸다(115200 baud, 프로젝트 공통).
#
# 이 리스너는 그 한 줄들을 읽어서
#   · "PRESS"  → 전역 START↔STOP 을 번갈아 전송 (모든 레코더 동시 시작/중지)
#   · "ENERGY" → Energy/energy_logger.py 로 넘겨 에피소드 h5 에 실시간 기록
# 을 동시에 한다. 포트가 하나뿐이라 두 일을 한 프로세스가 맡아야 한다.
#
# 에너지 에피소드의 시작/중지는 '전역 신호'(record.cmd)를 따라간다. 그래서 녹화를
# 이 페달로 시작하든, 다른 터미널의 `record_signal.py start` 로 시작하든, 카메라의
# [R] 키로 시작하든 언제나 카메라/MCU 와 같은 구간이 잘린다.
#
# pyserial 은 페달 모드에서만 필요하므로(핵심 send/RecordSignal 은 의존성 0),
# 아래 함수 안에서 지연 임포트한다. h5py/numpy 도 마찬가지로 에너지 기록에서만 쓴다.

PEDAL_BAUD = 115200
PEDAL_NAME = "pedal"          # 펌웨어가 '?' 질의에 "NAME,pedal[_energy]" 로 답한다
PEDAL_HELLO_SEC = 3.0         # 포트를 연 뒤 펌웨어 신원(=D8 지원 여부)을 기다리는 시간


def _probe_pedal(device, timeout=3.0):
    """포트를 잠깐 열어 '?' 를 보내고, 'NAME,pedal...' 회신이 오면 True.

    아두이노는 시리얼이 열릴 때 리셋되어 부팅에 ~2초가 걸리므로, 그동안 '?' 를
    주기적으로 다시 보내며 회신을 기다린다. (부팅 배너도 "NAME,..." 이라 빨리 잡힐 수 있다.)
    """
    import serial
    try:
        ser = serial.Serial(device, PEDAL_BAUD, timeout=0.2)
    except Exception:
        return False
    try:
        deadline = time.time() + timeout
        next_query = 0.0
        while time.time() < deadline:
            now = time.time()
            if now >= next_query:
                try:
                    ser.write(b"?")
                except Exception:
                    return False
                next_query = now + 0.5
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if line.startswith("NAME,") and PEDAL_NAME in line.split(",", 1)[1].strip().lower():
                return True
    finally:
        try:
            ser.close()
        except Exception:
            pass
    return False


def find_pedal_port():
    """아두이노 페달의 시리얼 포트를 자동 탐지한다(못 찾으면 None).

    USB 시리얼로 보이는 포트(usbmodem/usbserial/ttyACM/ttyUSB/Arduino)를 먼저
    후보로 올린 뒤, 각 후보에 '?' 를 보내 'NAME,pedal...' 로 답하는 포트를 고른다.
    """
    import serial.tools.list_ports
    ports = list(serial.tools.list_ports.comports())

    def looks_usb(p):
        s = f"{p.device} {p.description} {p.hwid}".lower()
        return any(k in s for k in ("usbmodem", "usbserial", "ttyacm", "ttyusb", "arduino"))

    ordered = [p.device for p in ports if looks_usb(p)]
    ordered += [p.device for p in ports if p.device not in ordered]
    for dev in ordered:
        if _probe_pedal(dev):
            return dev
    return None


def _pedal_hello(ser, timeout=PEDAL_HELLO_SEC):
    """포트를 연 직후 펌웨어 신원을 확인해 이름을 돌려준다(못 받으면 None).

    왜 미리 확인하나?
      아두이노는 포트를 열 때 리셋되어 부팅에 ~1.6초가 걸린다. 그 사이에 전역 START
      가 들어오면 'D8 을 지원하는 보드인지' 를 아직 몰라 에너지 에피소드를 놓친다.
      그래서 루프에 들어가기 전에 여기서 한 번 확실히 물어본다. (부팅 배너를 기다리는
      동안 '?' 도 함께 보내므로, 이미 부팅이 끝난 보드도 즉시 답한다.)
    """
    deadline = time.time() + timeout
    next_query = 0.0
    while time.time() < deadline:
        now = time.time()
        if now >= next_query:
            try:
                ser.write(b"?")
            except Exception:
                return None
            next_query = now + 0.5
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if line.upper().startswith("NAME,"):
            return line.split(",", 1)[1].strip()
    return None


def _make_energy_logger(port):
    """Energy/energy_logger.py 의 EnergyLogger 를 만든다. 실패하면 (None, 사유)."""
    here = os.path.dirname(os.path.abspath(__file__))
    energy_dir = os.path.join(here, "Energy")
    if energy_dir not in sys.path:
        sys.path.insert(0, energy_dir)
    try:
        from energy_logger import EnergyLogger
        return EnergyLogger(port=port), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def _make_beeper(enabled):
    """Energy/energy_beep.py 의 EnergyBeeper 를 만든다. 실패해도 None 을 돌려주고 넘어간다."""
    here = os.path.dirname(os.path.abspath(__file__))
    energy_dir = os.path.join(here, "Energy")
    if energy_dir not in sys.path:
        sys.path.insert(0, energy_dir)
    try:
        from energy_beep import EnergyBeeper
        return EnergyBeeper(enabled=enabled)
    except Exception as e:
        print(f"⚠️ 에너지 비프음을 켤 수 없습니다 ({type(e).__name__}: {e}). 소리 없이 진행합니다.")
        return None


def _make_relay(ser, write_lock, quiet=False):
    """이미 열린 포트에 얹어서 EnergyRelay 를 만든다. 실패하면 (None, 사유)."""
    here = os.path.dirname(os.path.abspath(__file__))
    energy_dir = os.path.join(here, "Energy")
    if energy_dir not in sys.path:
        sys.path.insert(0, energy_dir)
    try:
        from energy_relay import EnergyRelay
        return EnergyRelay(ser=ser, write_lock=write_lock, quiet=quiet), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def pedal_listen(port=None, debounce=0.20, debug=False, udp_only=False, energy=True,
                 beep=True, relay=True, gui=False, pedal_relay=None):
    """페달 두 개와 릴레이 두 개를 한 프로세스에서 다룬다 (Ctrl+C 종료).

    · port     : 강제로 쓸 시리얼 포트(생략하면 자동 탐지). 예) /dev/ttyACM0
    · debounce : 이 시간(초) 안에 들어온 연속 "PRESS" 는 한 번으로 친다(접점 떨림·중복 방지).
                 (펌웨어에서 이미 30ms 디바운스하지만, 소프트웨어에서 한 번 더 막아 준다.)
                 에너지 페달에는 적용하지 않는다 — 빠르게 켰다 끄는 것도 실제 조작이라
                 버리면 안 되고, 펌웨어 디바운스만으로 충분하다.
    · debug    : 아두이노가 보내는 모든 시리얼 줄을 그대로 출력해 진단을 돕는다.
    · udp_only : PRESS 를 UDP(SURGITAG_UDP)로만 흘리고 record.cmd 토글은 하지
                 않는다 — 로봇 롤아웃 모드(rosota_rollout 매니저가 시작/중지를
                 결정하고 record.cmd 도 매니저가 쓴다)에서 사용.
                 (에너지 기록은 이 모드에서도 record.cmd 를 따라 그대로 동작한다.)
    · energy   : D8 에너지 페달/릴레이의 h5 기록 사용 여부(기본 켬).
    · beep     : 에너지가 켜져 있는 동안 PC 스피커로 '삐삐삐' 활성음을 낼지(기본 켬).
                 실제 에너지 기구가 활성화 중에 내는 소리를 모방해, 리허설의 리듬이
                 실제와 같아지도록 한다. 기록 여부와 무관하게 켜지면 소리가 난다
                 (실제 장비도 녹화와 상관없이 울린다).
    · relay    : D12(CUT)/D11(COAG) 릴레이 제어를 켤지(기본 켬). 릴레이는 소작기의
                 풋스위치를 실제로 밟는다 — 모방이 아니라 진짜 에너지다.
    · gui      : 화면에서 클릭해 CUT/COAG 를 밟는 창을 띄운다(Energy/energy_gui.py).
                 이때 시리얼 읽기는 배경 스레드로 내려가고 메인 스레드는 tk 가 쓴다.
    · pedal_relay : 'cut' | 'coag' | None. D8 발 페달이 실제로 물 릴레이. 기본 None
                 이면 D8 은 지금까지처럼 '타이밍 모방 기록' 전용이다.
    """
    import serial
    from udp_stream import UdpStreamer  # SURGITAG_UDP 미설정 시 no-op
    udp = UdpStreamer('pedal')
    if udp_only and not udp.enabled:
        print("❌ --udp-only 인데 SURGITAG_UDP 가 설정되지 않았습니다. 예) SURGITAG_UDP=1")
        return 1

    dev = port or find_pedal_port()
    if not dev:
        print("❌ 아두이노 페달을 찾지 못했습니다. USB 연결을 확인하거나 --port 로 직접 지정하세요.")
        print("   포트 목록:  python3 -m serial.tools.list_ports")
        print("   (펌웨어 업로드: arduino-cli upload -p <포트> --fqbn arduino:avr:uno "
              "Energy/pedal_energy_firmware)")
        return 1

    try:
        ser = serial.Serial(dev, PEDAL_BAUD, timeout=0.2)
    except Exception as e:
        print(f"❌ {dev} 포트를 열 수 없습니다: {e}")
        return 1

    # --- 펌웨어 신원 확인 (루프에 들어가기 전에 D8/릴레이 지원 여부를 확정한다) ---
    firmware = _pedal_hello(ser)
    # 보드가 중간에 리셋되며 신원 배너를 다시 보낼 수 있어 dict 에 담아 둔다(중첩 함수에서 갱신).
    cap = {'energy': bool(firmware and "energy" in firmware.lower())}
    # None = '아직 모른다'(신원 응답이 없었다) — 이 경우는 막지 않고 경고만 한다.
    relay_capable = ("relay" in firmware.lower()) if firmware else None

    # 에너지 mode 코드와 사유 문구의 주인은 Energy/ 쪽이다 — 중복 정의하지 않고 가져온다.
    _energy_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Energy")
    if _energy_dir not in sys.path:
        sys.path.insert(0, _energy_dir)
    try:
        from energy_relay import MODE_CODE, MODE_PEDAL, MODE_OFF, REASON_TEXT
    except Exception:                    # Energy/ 가 없어도 페달 기능은 계속 쓴다
        MODE_CODE, MODE_PEDAL, MODE_OFF, REASON_TEXT = {}, 1, 0, {}

    # --- 에너지 활성음 (기록과 독립 — h5py 가 없어도 소리는 난다) ---
    beeper = _make_beeper(beep) if beep else None
    if beeper and not beeper.enabled:
        print(f"⚠️ 에너지 비프음을 낼 수 없습니다 ({beeper.reason}). 소리 없이 진행합니다.")
        beeper = None

    # --- 에너지 기록 준비 (h5py 가 없으면 페달/릴레이 기능만 계속 쓴다) ---
    logger = None
    if energy:
        logger, why = _make_energy_logger(dev)
        if logger is None:
            print(f"⚠️ 에너지 기록을 켤 수 없습니다 ({why}).")
            print("   h5py/numpy 가 있는 환경에서 실행하세요:  conda activate surgitag")
            print("   기록 페달(D7)만으로 계속 진행합니다.")
        else:
            logger.firmware = firmware or ""
            if not cap['energy']:
                print("⚠️ 이 보드의 펌웨어는 D8(에너지) 페달을 보내지 않습니다"
                      f" (신원: {firmware or '응답 없음'}). 에너지 기록은 꺼진 채로 진행합니다.")
                print(f"   업로드:  arduino-cli compile -u -p {dev} "
                      "--fqbn arduino:avr:uno Energy/pedal_energy_firmware")

    # --- 릴레이(D12=CUT, D11=COAG) 준비 ---
    # 이 포트를 읽는 것은 아래 루프뿐이고, 쓰는 쪽은 여럿(하트비트/GUI)이라 쓰기 락을 공유한다.
    write_lock = threading.Lock()
    relay_ctl = None
    if relay and relay_capable is False:
        print("⚠️ 이 보드의 펌웨어에는 릴레이(D12/D11)가 없습니다"
              f" (신원: {firmware}). 릴레이 제어를 끕니다.")
        print(f"   업로드:  arduino-cli compile -u -p {dev} "
              "--fqbn arduino:avr:uno Energy/pedal_energy_firmware")
    elif relay:
        relay_ctl, why = _make_relay(ser, write_lock)
        if relay_ctl is not None:
            relay_ctl.port = dev          # 포트를 이미 열어 넘겼으므로 이름만 알려 준다(표시용)
        if relay_ctl is None:
            print(f"⚠️ 릴레이 제어를 켤 수 없습니다 ({why}). 페달 기능만 계속합니다.")
        elif relay_capable is None:
            print("⚠️ 펌웨어 신원 응답이 없어 릴레이 지원 여부를 확인하지 못했습니다."
                  " 일단 켠 채로 진행합니다(--no-relay 로 끌 수 있습니다).")

    if pedal_relay:
        if relay_ctl:
            relay_ctl.set_pedal_map(pedal_relay)
        else:
            print(f"⚠️ --pedal-relay {pedal_relay} 를 무시합니다 (릴레이 제어가 꺼져 있음).")
            pedal_relay = None

    if gui and not relay_ctl:
        print("⚠️ 릴레이 제어가 없어 --gui 창을 열지 않습니다.")
        gui = False

    # --- 전역 신호를 따라 에너지 에피소드를 시작/중지한다 ---
    # 페달이 보낸 START 든, 다른 터미널·카메라 [R] 이 보낸 START 든 똑같이 여기로 온다.
    state = {'next_cmd': "START",   # 다음에 페달을 밟으면 보낼 명령
             'ep_start': None,      # 현재 에피소드 시작 시각(콘솔 t+ 표시용)
             'energy': 0}           # D8 페달의 현재 상태(1=밟음) — 중복 줄을 여기서 거른다

    # 에너지는 두 경로로 켜진다(D8 모방 페달, CUT/COAG 릴레이). 기록에 남는 '지금의
    # 에너지 상태' 는 하나뿐이므로 여기서 한 번 정리한다 — 진짜로 소작기를 무는
    # 릴레이가 우선이고, 릴레이가 열려 있을 때만 D8 모방을 본다.
    def _publish_energy(ts, latency_ms=0.0):
        mode_name = relay_ctl.active_mode if relay_ctl else None
        if mode_name:
            st, mode = 1, MODE_CODE.get(mode_name, MODE_PEDAL)
        elif state['energy']:
            st, mode = 1, MODE_PEDAL
        else:
            st, mode = 0, MODE_OFF
        if beeper:                      # 소리를 먼저 — 발/손에서 소리까지가 가장 짧아야 한다
            beeper.on() if st else beeper.off()
        if logger:
            logger.set_state(st, ts, latency_ms, mode=mode)

    def _t_plus(ts):
        """에피소드 시작 이후 경과시간 문자열(녹화 중이 아니면 빈 문자열)."""
        t0 = state['ep_start']
        return f"  (t+{ts - t0:6.2f}s)" if t0 else ""

    def _on_relay_change(mode, st, reason, ts, age_ms):
        """보드가 릴레이 상태를 보고할 때 (시리얼 읽기 스레드에서 호출된다)."""
        # 'pedal' 은 D8 이 물어서 바뀐 것이다. 바로 뒤따라오는 ENERGY 줄이 디바운스
        # 보정(age_ms)까지 실어 오므로, 기록은 그쪽에 맡기고 여기서는 출력만 한다.
        if reason != 'pedal':
            _publish_energy(ts)
        note = "" if reason in ('host', 'pedal') else f"  [{REASON_TEXT.get(reason, reason)}]"
        mark = f"⚡ {mode.upper()} ON " if st else f"   {mode.upper()} OFF"
        print(f"{mark}{_t_plus(ts) if logger and logger.active else ''}{note}", flush=True)

    if relay_ctl:
        relay_ctl.on_change = _on_relay_change

    def _on_start():
        state['next_cmd'] = "STOP"
        if logger and (cap['energy'] or relay_ctl):
            state['ep_start'] = time.time()
            logger.start_episode()

    def _on_stop():
        state['next_cmd'] = "START"
        state['ep_start'] = None
        if logger and logger.active:
            summary = logger.stop_episode()
            if summary:
                print(f"⏹  에너지 기록 중지 — {summary['duration_sec']:.1f}s 중 "
                      f"ON {summary['on_seconds']:.2f}s "
                      f"({summary['duty_ratio']*100:.1f}%, 작동 {summary['num_activations']}회"
                      f" — CUT {summary['num_cut']}회 {summary['cut_seconds']:.2f}s /"
                      f" COAG {summary['num_coag']}회 {summary['coag_seconds']:.2f}s /"
                      f" 모방 {summary['num_pedal']}회 {summary['pedal_seconds']:.2f}s)"
                      f"\n    {summary['path']}")

    def _on_toggle():
        _on_stop() if (logger and logger.active) else _on_start()

    sig = RecordSignal(on_start=_on_start, on_stop=_on_stop, on_toggle=_on_toggle)

    print(f"🦶 발 페달 대기 중 ({dev})")
    if udp_only:
        print("   [udp-only] PRESS 를 UDP 로만 전송합니다 (record.cmd 토글 없음"
              " — 롤아웃 매니저가 시작/중지를 결정).")
    else:
        print("   D7(기록) 페달을 밟으면 START↔STOP 을 번갈아 전송합니다. 종료: Ctrl+C")
    if logger and (cap['energy'] or relay_ctl):
        print(f"   에너지 기록 → {logger.save_dir}")
    if relay_ctl:
        print("   ⚡ 릴레이 D12=CUT / D11=COAG 준비됨 — 닫히면 소작기 페달을 실제로 밟습니다")
        if pedal_relay:
            print(f"   D8 페달 → {pedal_relay.upper()} 릴레이 (밟는 동안 실제로 소작합니다)")
        else:
            print("   D8 페달은 타이밍 모방 전용입니다 "
                  "(--pedal-relay cut|coag 로 실제 릴레이에 물릴 수 있습니다)")
    if beeper:
        print(f"   에너지가 켜져 있는 동안 활성음이 울립니다 ({beeper.player})")
    print(f"   펌웨어: {firmware or '(신원 응답 없음)'}")
    if debug:
        print("   [debug] 아두이노가 보내는 모든 줄을 출력합니다.")

    stop_evt = threading.Event()
    result = {'rc': 0}
    st_local = {'last_press': 0.0}

    def _serial_loop():
        """포트를 읽는 유일한 곳. --gui 일 때는 이 함수가 배경 스레드에서 돈다."""
        try:
            while not stop_evt.is_set():
                sig.poll()           # 전역 START/STOP 을 따라 에너지 에피소드를 여닫는다
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                if debug:
                    print(f"   [debug] {line}")

                # --- 펌웨어 식별 배너: 보드가 중간에 리셋되면 다시 온다 ---
                if line.upper().startswith("NAME,"):
                    name = line.split(",", 1)[1].strip()
                    if "energy" in name.lower():
                        cap['energy'] = True
                        if logger:
                            logger.firmware = name
                    if relay_ctl and pedal_relay:
                        relay_ctl.set_pedal_map(pedal_relay)   # 리셋되면 매핑이 날아간다
                    continue

                # --- 릴레이 상태 보고 (CUT/COAG) ---
                if relay_ctl and relay_ctl.handle_line(line, time.time()):
                    continue

                # --- D8 에너지 페달: 상태 변화를 기록한다 ---
                if line.upper().startswith("ENERGY,"):
                    cap['energy'] = True
                    now = time.time()
                    parts = line.split(",")
                    try:
                        st = 1 if int(parts[1]) else 0
                        age_ms = float(parts[2]) if len(parts) > 2 else 0.0
                    except (IndexError, ValueError):
                        print(f"   ⚠️ 해석할 수 없는 에너지 줄: {line!r}")
                        continue
                    if st == state['energy']:
                        continue      # 같은 상태 반복(부팅 초기 상태 등)은 조용히 무시
                    state['energy'] = st
                    edge = now - age_ms / 1000.0
                    _publish_energy(now, age_ms)
                    udp.note(f"energy {'on' if st else 'off'}")
                    tag = f" → {pedal_relay.upper()} 릴레이" if pedal_relay else ""
                    mark = f"⚡ 에너지 ON {tag}" if st else f"   에너지 OFF{tag}"
                    if logger and logger.active:
                        print(f"{mark}{_t_plus(edge)}")
                    elif logger:
                        print(f"{mark} — 녹화 중이 아니라 기록되지 않습니다"
                              " (다음 에피소드는 이 상태로 시작합니다)")
                    else:
                        print(mark)
                    continue

                # --- D7 기록 페달: 전역 START/STOP ---
                if line.upper() != "PRESS":     # RELEASE 등은 무시
                    continue
                now = time.time()
                if now - st_local['last_press'] < debounce:  # 접점 떨림·중복 눌림 무시
                    continue
                st_local['last_press'] = now
                udp.pedal('press', now)          # 워크스테이션에도 알림 (no-op 가능)
                if udp_only:
                    print("[pedal] PRESS → UDP")
                    continue
                cmd = state['next_cmd']
                send(cmd)
                print(f"[pedal] {cmd} 전송")
                # 여기서도 한 번 뒤집어 둔다(신호 폴링이 곧 같은 값으로 확정한다).
                state['next_cmd'] = "STOP" if cmd == "START" else "START"
        except KeyboardInterrupt:
            print("\n[pedal] 종료합니다.")
        except serial.SerialException as e:
            print(f"\n❌ 시리얼 연결이 끊겼습니다: {e}")
            result['rc'] = 1
        finally:
            stop_evt.set()

    try:
        if gui:
            t = threading.Thread(target=_serial_loop, daemon=True, name="pedal-serial")
            t.start()
            panel = None
            try:
                from energy_gui import EnergyPanel      # Energy/ 는 위에서 sys.path 에 넣었다
                # 비프음은 _publish_energy 가 이미 책임진다(주인을 하나로 둔다).
                panel = EnergyPanel(relay_ctl, logger=logger, beeper=None,
                                    signal_send=send)
                print("   🖥  릴레이 창을 열었습니다 — CUT/COAG 를 누르고 있는 동안 작동합니다"
                      " (Esc = 비상 정지)")
                panel.run()                             # 창을 닫을 때까지 여기서 돈다
            except Exception as e:
                print(f"❌ 릴레이 창을 열 수 없습니다 ({type(e).__name__}: {e}).")
                print("   창 없이 계속하려면 --gui 없이 실행하세요.")
                result['rc'] = 1
            finally:
                if panel:
                    panel.close()
                stop_evt.set()
                t.join(timeout=1.5)
        else:
            _serial_loop()
    except KeyboardInterrupt:
        print("\n[pedal] 종료합니다.")
        stop_evt.set()
    finally:
        if relay_ctl:
            relay_ctl.close()            # 무슨 일이 있어도 릴레이는 열고 끝낸다
        if beeper:
            beeper.close()
        if logger:
            logger.close()               # 녹화 중이었다면 저장하고 마감한다
        sig.close()
        try:
            ser.close()
        except Exception:
            pass
    return result['rc']


def _cli():
    args = sys.argv[1:]
    if args and args[0].lower() == "pedal":
        port = None
        debug = False
        udp_only = False
        energy = True
        beep = True
        relay = True
        gui = False
        pedal_relay = None
        rest = args[1:]
        i = 0
        while i < len(rest):
            if rest[i] in ("--port", "-p", "--device", "-d") and i + 1 < len(rest):
                port = rest[i + 1]; i += 2
            elif rest[i] == "--debug":
                debug = True; i += 1
            elif rest[i] == "--udp-only":
                udp_only = True; i += 1
            elif rest[i] in ("--no-energy", "--no-energie"):
                energy = False; i += 1
            elif rest[i] in ("--no-beep", "--silent"):
                beep = False; i += 1
            elif rest[i] == "--no-relay":
                relay = False; i += 1
            elif rest[i] == "--gui":
                gui = True; i += 1
            elif rest[i] == "--pedal-relay" and i + 1 < len(rest):
                pedal_relay = rest[i + 1].lower()
                if pedal_relay not in ("cut", "coag", "none"):
                    print(f"error: --pedal-relay 는 cut|coag|none 이어야 합니다 "
                          f"(받은 값: {rest[i + 1]!r})")
                    return 2
                if pedal_relay == "none":
                    pedal_relay = None
                i += 2
            else:
                print(f"error: 알 수 없는 인자 {rest[i]!r}")
                print("usage: python3 record_signal.py pedal "
                      "[--port /dev/ttyACM0] [--debug] [--udp-only] [--no-energy] "
                      "[--no-beep] [--gui] [--no-relay] [--pedal-relay cut|coag]")
                return 2
        return pedal_listen(port=port, debug=debug, udp_only=udp_only, energy=energy,
                            beep=beep, relay=relay, gui=gui, pedal_relay=pedal_relay)

    if len(args) != 1 or args[0].lower() in ("-h", "--help"):
        print("usage: python3 record_signal.py {start|stop|toggle|pedal}")
        print("  start  — 실행 중인 모든 레코더가 녹화 시작")
        print("  stop   — 모두 녹화 중지 + 저장")
        print("  toggle — 각자 토글(동기화에는 start/stop 권장)")
        print("  pedal  — D7 페달로 START↔STOP 전송 + D8 에너지 페달을 에피소드로 기록")
        print("           + D12(CUT)/D11(COAG) 릴레이로 전기소작기 조작")
        print("           옵션: --gui(클릭 조작 창) --pedal-relay cut|coag --no-relay")
        return 0 if args[:1] and args[0].lower() in ("-h", "--help") else 2
    try:
        path = send(args[0])
    except ValueError as e:
        print(f"error: {e}")
        return 2
    print(f"[signal] {args[0].upper()} 전송 → {path}")
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
