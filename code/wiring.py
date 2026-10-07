#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
새 wiring (재배선) 설정 -- visualize_forces.py 와 calibrate.py 가 공유한다.

이전과 달리 half-bridge 가 L-R / T-B 빼기를 하드웨어에서 이미 수행하므로,
각 아날로그 채널이 곧 하나의 힘 성분에 1:1 대응한다. 그래도 게이지/배선 오차로
축 간 커플링이 남아 있어, 6x6 디커플링 행렬(calibrate.py)로 커플링을 제거하고
스케일을 가한 힘 단위(N, torsion 은 N·m)에 맞춘다.

  AI | bridge          | gauges    | 힘 성분 (force key)        | 단위
  ---+-----------------+-----------+----------------------------+------
  A0 | half-bridge     | L1L, R1L  | bending_gripper_x          | N
  A1 | half-bridge     | L2L, R2L  | bending_trocar_x           | N
  A4 | half-bridge     | B1L, T1L  | bending_gripper_y          | N
  A5 | half-bridge     | B2L, T2L  | bending_trocar_y           | N
  A2 | quarter-bridge  | GL        | gripping                   | N
  A6 | quarter-bridge  | TR        | torsion                    | N·m

힘 컬럼(디커플링된 값)과 raw 채널 컬럼(원신호)을 CSV 에서 구분하려고,
raw 컬럼은 짧은 코드(bgx, btx, ...), 힘 컬럼은 풀네임을 쓴다.
"""

# (ai_index, raw_key, force_key, display_label, color, style, unit, on_right_axis)
WIRING = [
    (0, "bgx",  "bending_gripper_x", "Bend grip x",  "tab:blue",   "-", "N",   False),
    (1, "btx",  "bending_trocar_x",  "Bend troc x",  "tab:cyan",   "-", "N",   False),
    (4, "bgy",  "bending_gripper_y", "Bend grip y",  "tab:orange", "-", "N",   False),
    (5, "bty",  "bending_trocar_y",  "Bend troc y",  "tab:red",    "-", "N",   False),
    (2, "grip", "gripping",          "Gripping",     "tab:green",  "-", "N",   False),
    (6, "tor",  "torsion",           "Torsion",      "tab:purple", "-", "N·m", True),
]

# --- 파생 상수 (아래는 모두 WIRING 에서 유도) ---
RAW_KEYS     = [w[1] for w in WIRING]                  # CSV raw-signal 컬럼 순서
AXES         = [w[2] for w in WIRING]                  # 힘(디커플링 출력) 키 순서
CH_INDEX     = {w[1]: w[0] for w in WIRING}            # raw_key -> AI 채널 인덱스
FORCE_CH     = {w[2]: w[0] for w in WIRING}            # force_key -> AI 채널 인덱스
RAW_CHANNELS = [(w[1], w[0]) for w in WIRING]          # [(raw_key, ch)] 스냅샷 기록용
FORCES       = [(w[2], w[3], w[4], w[5]) for w in WIRING]  # (key, label, color, style)
RIGHT_AXIS   = {w[2] for w in WIRING if w[7]}          # 오른쪽(N·m) 축에 그릴 힘 키
UNIT         = {w[2]: w[6] for w in WIRING}            # force_key -> 단위
MIN_CHANNELS = max(w[0] for w in WIRING) + 1           # 읽어야 할 최소 채널 수 (A6 → 7)

# quarter-bridge 축(단일 게이지 → 저SNR 경향). half-bridge 보다 강하게 필터하고
# 싶을 때 visualize_forces.py 의 --quarter-cutoff 가 이 축들에만 적용된다.
QUARTER_AXES = {"gripping", "torsion"}

# --- 디커플링 행렬 구조 (각 힘이 참조할 수 있는 raw 신호) ----------------------
# 물리 구조를 강제해 full 6x6 대신 블록 구조로 최소제곱을 푼다 (calibrate.py 가 사용):
#   - gripping : A2(grip) 만 사용. 다른 힘과 커플링 없음 → 사실상 스케일만.
#   - gripping 외 어떤 힘도 A2(grip) 를 참조하지 않음.
#   - gripper 굽힘(x,y) : gripper 게이지 A0(bgx), A4(bgy) 만 (region 내 x↔y 커플링).
#   - trocar  굽힘(x,y) : trocar  게이지 A1(btx), A5(bty) 만 (region 내 x↔y 커플링).
#   - gripper↔trocar 사이 커플링 없음 (서로 다른 region, 채널 분리).
#   - torsion : grip 제외 전부 (bending 4 + tor).
_GRIP_BEND = ["bgx", "bgy"]              # A0, A4 (gripper region)
_TROC_BEND = ["btx", "bty"]              # A1, A5 (trocar region)
ALLOWED_SIGNALS = {
    "bending_gripper_x": _GRIP_BEND,
    "bending_gripper_y": _GRIP_BEND,
    "bending_trocar_x":  _TROC_BEND,
    "bending_trocar_y":  _TROC_BEND,
    "gripping":          ["grip"],
    "torsion":           ["bgx", "btx", "bgy", "bty", "tor"],
}

# --- 타깃 그룹 (캘리브레이션 시 같은 물리량으로 묶어 타깃 공유) -----------------
# gripper/trocar 굽힘은 같은 평면의 굽힘을 두 위치에서 잰 '같은 값'이라(사용자 확인),
# 같은 평면이면 같은 타깃을 준다 → 둘 다 가한 굽힘 전체값을 읽음(절반씩 나눠갖지 않음).
# 한 행의 applied_dir 가 어떤 그룹이면, 그 그룹의 모든 힘 컬럼 타깃 = 가한 크기.
TARGET_GROUP = {
    "bending_gripper_x": "bend_x",
    "bending_trocar_x":  "bend_x",
    "bending_gripper_y": "bend_y",
    "bending_trocar_y":  "bend_y",
    "gripping":          "gripping",
    "torsion":           "torsion",
}

# 자기 축 부하 행으로만 스케일을 적합할 축(완전 분리된 scale-only 축).
# gripping 게이지 A2 는 torsion(이 강하게)·bending 에 의해 오염돼(grip 신호가
# 그 부하들에 반응) 전체 행으로 적합하면 스케일이 눌린다(9.8N→7.5N). 그리핑 부하
# 행으로만 적합해 스케일을 바로잡는다. (대가: 순수 torsion 시 grip 가짜 출력 ~2-3N —
# A2 의 torsion 결합은 선형으로 제거 불가, 모든 신호를 써도 ~2N 남음.)
SELF_ROWS_ONLY = {"gripping"}
