#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SurgiTagV2 → rosota_rollout UDP 스트리밍 헬퍼 (옵션 기능, 기본 꺼짐).

각 레코더(MCU/NI6009/pedal/Camera/…)가 자기 데이터를 읽는 김에 로봇
워크스테이션의 rosota_rollout(surgitag_bridge, 기본 udp:48761)로도 JSON
datagram 을 흘려보낸다. 레코더의 기존 동작(포트 선택, 시각화, h5 저장,
record.cmd 연동)은 그대로 — 이 모듈은 곁가지 송신만 담당한다.

활성화 (환경변수, 레코더 실행 전에):
    SURGITAG_UDP=1                      # 로컬 워크스테이션 (127.0.0.1:48761)
    SURGITAG_UDP=192.168.0.10           # 원격 호스트, 기본 포트
    SURGITAG_UDP=192.168.0.10:48761     # 호스트:포트
미설정 시 모든 호출은 no-op 이라 순수 데이터셋 수집 워크플로에 영향이 없다.

수신(제어) 채널: 같은 소켓으로 워크스테이션이 보내는 제어를 처리한다.
    ping   → pong (시계 동기)
    record → 이 호스트의 record.cmd 를 원자적으로 기록
             (record_signal.send — 로컬 레코더 일괄 시작/정지)
    tare   → set_tare_callback() 으로 등록된 콜백 호출 (NI6009 영점)

wire 포맷은 ~/franka/surgitag_link/wire.py 와 동일 (v=1 JSON datagram).
표준 라이브러리만 사용.
"""

import json
import os
import socket
import struct
import threading
import time

PROTOCOL_VERSION = 1
DEFAULT_PORT = 48761
MAX_DGRAM = 60000

# 내시경 프레임(JPEG)은 JSON datagram 한도(60KB)를 넘으므로 바이너리 조각으로
# 보낸다. 같은 포트를 쓰며, 수신측(bridge)은 'EJPG' 매직으로 구분해 재조립한다.
#   header: magic 4s | ver B | frame_id I | frag_idx H | n_frags H |
#           t_unix d | width H | height H | total_len I
FRAME_MAGIC = b'EJPG'
FRAME_HEAD = struct.Struct('<4sBIHHdHHI')
FRAME_FRAG_PAYLOAD = MAX_DGRAM - FRAME_HEAD.size


def _parse_dest(spec):
    spec = (spec or '').strip()
    if not spec:
        return None
    if spec in ('1', 'true', 'local', 'localhost'):
        return ('127.0.0.1', DEFAULT_PORT)
    if ':' in spec:
        host, port = spec.rsplit(':', 1)
        return (host, int(port))
    return (spec, DEFAULT_PORT)


class UdpStreamer:
    """모듈당 하나 생성. 비활성(dest 없음)이면 전 메서드가 no-op."""

    def __init__(self, module, dest=None):
        self.module = str(module)
        self.dest = _parse_dest(dest if dest is not None
                                else os.environ.get('SURGITAG_UDP', ''))
        self.enabled = self.dest is not None
        self._tare_cb = None
        self._counts = {}
        if not self.enabled:
            return
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('0.0.0.0', 0))
        self.sock.settimeout(0.5)
        threading.Thread(target=self._rx_loop, daemon=True,
                         name=f'udp_stream:{self.module}').start()
        threading.Thread(target=self._status_loop, daemon=True,
                         name=f'udp_status:{self.module}').start()
        print(f'[udp_stream] {self.module} → {self.dest[0]}:{self.dest[1]} '
              f'(SURGITAG_UDP)')

    # ------------------------------------------------------------------ send

    def _send(self, msg):
        if not self.enabled:
            return
        msg.setdefault('v', PROTOCOL_VERSION)
        msg.setdefault('t_host', time.time())
        msg.setdefault('module', self.module)
        try:
            data = json.dumps(msg, separators=(',', ':')).encode()
            if len(data) <= MAX_DGRAM:
                self.sock.sendto(data, self.dest)
                self._counts[msg['type']] = self._counts.get(msg['type'], 0) + 1
        except OSError:
            pass  # 송신 실패가 레코더를 방해해서는 안 된다

    def mcu(self, board, device_ms, quat_xyzw, dist_mm, grip_pct,
            t_sample=None):
        self._send({'type': 'mcu', 'board': str(board),
                    'device_ms': int(device_ms),
                    'quat_xyzw': [float(v) for v in quat_xyzw],
                    'dist_mm': float(dist_mm), 'grip_pct': float(grip_pct),
                    't_sample': float(t_sample if t_sample is not None
                                      else time.time())})

    def force_block(self, t0, dt, forces_n6, raw_n6, tared):
        rows = [[round(float(v), 6) for v in row] for row in forces_n6]
        self._send({'type': 'force', 't0': float(t0), 'dt': float(dt),
                    'tared': bool(tared), 'forces': rows,
                    'raw_v': [[round(float(v), 6) for v in row]
                              for row in raw_n6]})
        self._counts['force_samples'] = \
            self._counts.get('force_samples', 0) + len(rows)

    def pedal(self, event, t_sample=None):
        self._send({'type': 'pedal', 'event': str(event),
                    't_sample': float(t_sample if t_sample is not None
                                      else time.time())})

    def note(self, text):
        self._send({'type': 'status', 'note': f'{self.module}: {text}'})

    def frame_jpeg(self, t, jpeg_bytes, width, height):
        """내시경 JPEG 프레임을 조각내 전송 (다른 데이터와 같은 포트).

        720p q85 ≈ 100~250KB → 2~5 조각. 로컬/유선 LAN 전제 — 조각 유실 시
        수신측이 그 프레임만 버린다 (다음 프레임으로 자연 복구).
        """
        if not self.enabled:
            return
        data = bytes(jpeg_bytes)
        n = max(1, (len(data) + FRAME_FRAG_PAYLOAD - 1) // FRAME_FRAG_PAYLOAD)
        fid = self._counts.get('frame', 0) & 0xFFFFFFFF
        try:
            for i in range(n):
                chunk = data[i * FRAME_FRAG_PAYLOAD:(i + 1) * FRAME_FRAG_PAYLOAD]
                head = FRAME_HEAD.pack(FRAME_MAGIC, PROTOCOL_VERSION, fid,
                                       i, n, float(t), int(width),
                                       int(height), len(data))
                self.sock.sendto(head + chunk, self.dest)
            self._counts['frame'] = self._counts.get('frame', 0) + 1
        except OSError:
            pass

    def set_tare_callback(self, fn):
        self._tare_cb = fn

    # --------------------------------------------------------------- control

    def _rx_loop(self):
        while True:
            try:
                data, addr = self.sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                return
            try:
                msg = json.loads(data.decode())
                t = msg.get('type')
            except (ValueError, UnicodeDecodeError):
                continue
            try:
                if t == 'ping':
                    self._reply(addr, {'type': 'pong', 't0': msg.get('t0')})
                elif t == 'record':
                    action = str(msg.get('action', '')).lower()
                    import record_signal  # 같은 폴더 (지연 import)
                    path = record_signal.send(
                        {'start': 'START', 'stop': 'STOP'}[action])
                    self._reply(addr, {'type': 'record_done', 'ok': True,
                                       'action': action, 'path': path})
                elif t == 'tare':
                    if self._tare_cb is not None:
                        self._tare_cb()
                        self._reply(addr, {'type': 'tare_done', 'ok': True,
                                           'message': 'tare armed'})
                    else:
                        self._reply(addr, {'type': 'tare_done', 'ok': False,
                                           'message':
                                               f'{self.module}: no tare here'})
            except Exception as exc:  # noqa: BLE001 - 제어 실패도 격리
                self._reply(addr, {'type': f'{t}_done', 'ok': False,
                                   'message': repr(exc)})

    def _reply(self, addr, msg):
        msg.setdefault('v', PROTOCOL_VERSION)
        msg.setdefault('t_host', time.time())
        msg.setdefault('module', self.module)
        try:
            self.sock.sendto(json.dumps(msg, separators=(',', ':')).encode(),
                             addr)
        except OSError:
            pass

    def _status_loop(self):
        last = {}
        while True:
            time.sleep(1.0)
            snap = dict(self._counts)
            rates = {k: snap.get(k, 0) - last.get(k, 0)
                     for k in ('mcu', 'force_samples', 'frame')}
            last = snap
            self._send({'type': 'status',
                        'mcu_hz': rates['mcu'],
                        'force_hz': rates['force_samples'],
                        'frame_hz': rates['frame']})
