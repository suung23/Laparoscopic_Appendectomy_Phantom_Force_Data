# 복강경 수술 시뮬레이터 — 토끼 개복 필드 · 양팔 FR3 · Isaac Sim PhysX

실제 토끼 개복 영상(RealSense D405)에서 재구성한 장기를 복강경 트레이닝 팬텀 안에
넣고, **FR3 두 대**가 복강경 기구(그래스퍼 / 커터)를 트로카를 통해 조작하는 시뮬레이션.

두 가지 방식으로 구동한다.

- **텔레옵** — 3D Systems **Touch** 햅틱 장치 두 대로 사람이 직접 조작. 시연 데이터
  수집과 환경 검증용. → [GUI_TELEOP.md](GUI_TELEOP.md)
- **VLA 정책 평가** — headless로 모델이 액션을 내고, 관측·영상·점수를 받는다.
  → [VLA_HANDOFF.md](VLA_HANDOFF.md)

장기는 **PhysX FEM soft body**다. 기구가 누르면 변형하고, 죠로 집으면 잡히고,
들어올리면 끌려온다.

![두 대의 FR3가 팬텀을 둘러싼 모습](docs/surgical_cell.png)

---

## 어디부터 읽을 것인가

| 당신이 하려는 일 | 읽을 문서 |
|---|---|
| **Touch로 직접 조작**하고 싶다 | [GUI_TELEOP.md](GUI_TELEOP.md) — 실행, 조작, 이 머신의 최적 설정 |
| **VLA 모델을 평가**하고 싶다 | [VLA_HANDOFF.md](VLA_HANDOFF.md) — 설치, 입출력 정의, 롤아웃, 결과 영상 |
| **RII 플랫폼으로 제출**하고 싶다 | [rii/README.md](rii/README.md) — 설치, 정책 어댑터, 제출, 이미지 빌드 |
| **실물 FR3 제어 코드**를 쓴다 | [FR3_CONTROL_HANDOFF.md](FR3_CONTROL_HANDOFF.md) — 무엇을 가져가고 무엇을 절대 가져가면 안 되는가 |
| 시뮬이 **왜 이렇게 생겼는지** 알고 싶다 | [SURGICAL_CELL.md](SURGICAL_CELL.md) — 씬 구성, 조직 물성, 카메라 캘리브레이션 |
| **제어기**를 만지려 한다 | [CONTROLLER_STUDY.md](CONTROLLER_STUDY.md) — 182 config 그리드 서치, 4개 파트 |
| **반응성 / 성능**이 문제다 | [TELEOP_RESPONSIVENESS.md](TELEOP_RESPONSIVENESS.md) — rtf 측정, 반박된 가설들 |
| 로봇을 **어디에 세울지** 정한다 | [PORT_OFFSET_STUDY.md](PORT_OFFSET_STUDY.md) — stand-off 0.46 m |
| **장기 자산**의 출처가 궁금하다 | [HANDOFF.md](HANDOFF.md) — 재구성 결과물 명세 |

---

## 빠른 실행

```bash
# 1. 씬 빌드 (한 번만). 순서가 중요하다 — 두 번째가 첫 번째의 산출물을 읽는다
~/isaacsim-5.1/python.sh isaac/build_deformable_usd.py   # 장기+팬텀 → out/scene_soft.usd
~/isaacsim-5.1/python.sh isaac/build_surgical_cell.py    # + FR3 2대·기구·테이블 → out/surgical_cell.usd

# 2a. 텔레옵 (Touch 2대)
./touch_bridge --device "Left Device"  --shm touch_bridge          # 터미널 1
./touch_bridge --device "Right Device" --shm touch_bridge_right    # 터미널 2
~/isaacsim-5.1/python.sh isaac/teleop_left_touch.py --arm both \
    --physics-hz 240 --ctrl-hz 120 --manip-hz 60                   # 터미널 3

# 2b. 장치 없이 동작 확인
~/isaacsim-5.1/python.sh isaac/teleop_left_touch.py --fake --seconds 12

# 2c. VLA 정책 평가 (headless)
~/isaacsim-5.1/python.sh isaac/vla_env.py --steps 60 --dump out/frames
```

## 저장소 구조

```
isaac/                Isaac Sim 쪽 전부 — 현재 작업의 본체
  build_deformable_usd.py   tet 메시 → PhysX FEM soft body USD
  build_surgical_cell.py    팬텀 + FR3 2대 + 기구 + 장기를 한 씬으로
  teleop_left_touch.py      Touch 햅틱 텔레옵 (--arm left|right|both)
  vla_env.py                VLA 평가 환경 (액션/관측 계약)
  laparo_control.py         툴팁 제어기 (ControlCfg / BEST)
  laparo_ik.py              FK, RCM IK, 자세 그래디언트
  trocar.py                 6 mm 구멍 기하 (bite, cone clamp)
  export_tool_urdf.py       Fusion 익스포트 → URDF
  sweep_*.py, plot_*.py     스터디용 스윕과 그림
  bench_step.py             물리 스텝 비용 벤치
  render_vla_latency.py     headless 추종 증거 영상 캡처
  plot_vla_latency.py       ↑ 를 mp4로 합성

tools/                재구성 파이프라인 (D405 영상 → 메시 → tet)
sim/                  Newton 씬 — 초기 검증용, 현재 경로 아님 (아래 참조)
rii/                  RII 플랫폼 통합 전체 — 가이드, 설정, 어댑터, 이미지, 제출 도구
mesh/ tet/            장기 표면 메시와 tet 메시
out/                  빌드 산출물 (USD, URDF, 로그, 영상)
docs/                 문서용 그림·영상
data/                 D405 녹화에서 추출한 프레임
touch_bridge.c        Touch → /dev/shm 1 kHz 브리지 (OpenHaptics)
```

---

## 환경

이 저장소가 개발·측정된 머신:

| | |
|---|---|
| CPU | Intel Core i7-14700KF (20 코어 / 28 스레드) |
| RAM | 62 GB |
| GPU | NVIDIA RTX 4090 24 GB, 드라이버 580.173.02 |
| OS | Linux 6.8.0-124-generic |
| Isaac Sim | 5.1.0-rc.19 (`~/isaacsim-5.1/python.sh`) |
| 햅틱 | 3D Systems Touch × 2, OpenHaptics 3.4 |

**GPU 필수.** PhysX deformable body는 GPU에서만 돈다 — CPU 폴백이 없다.
Isaac Sim의 `SimulationContext`가 부팅 시 씬을 CPU/60 Hz로 되돌려놓기 때문에
런타임에 다시 강제한다(`assert_scene_physics`).

파이썬은 세 개가 따로 논다.

| | 용도 |
|---|---|
| `~/isaacsim-5.1/python.sh` | Isaac Sim 전부 (`isaac/*.py`) |
| `.venv/bin/python` | 오프엔진 유틸, 제어기 단위 테스트 |
| `/usr/bin/python3` | matplotlib 그림 (`plot_*.py`) — Isaac 파이썬에 없다 |

---

## 장기는 어디서 왔는가

`tools/`의 재구성 파이프라인이 만들었다. 60초 D405 녹화(848×480 @30fps, 카메라 고정)
한 편에서:

```
db3 ──[extract_endonerf.py]──> data/     images / depth / masks
                                 │
                    [find_rest_frames.py]  기구 침범이 최소인 정지 프레임 선별
                                 │
                      [pick_points.py]     클릭 → prompts.json
                                 │
                     [segment_sam2.py]     장기별 마스크
                                 │
                     [build_volume.py]     워터타이트 볼륨 (단일 시점 inflation)
                                 │
                    [tetrahedralize.py]    tet 메시 (mm → m)
```

결과: 장기 6개(맹장 · 소장 1–3 · 위 · 충수), 합계 43,956 tets / 11,962 particles.
명세는 [HANDOFF.md](HANDOFF.md).

PhysX는 이 표면에서 **자체 hex 시뮬 메시를 다시 굽는다**(`build_deformable_usd.py`).
쿠킹 결과가 합계 **8,065 sim 노드**이고, 런타임 비용을 지배하는 건 이쪽 숫자다.

**뒷면은 관측된 적이 없다.** 카메라가 60초 내내 고정이라 장기 뒷면 데이터가 없고,
`build_volume.py`가 실루엣에 구면 프로파일 두께를 씌워 부풀린다. 이건 모델링
선택이지 측정이 아니다.

### 왜 Newton이 아니라 Isaac Sim PhysX인가

초기 검증은 Newton(`sim/scene.py`, `SolverVBD`)으로 했고 그 코드는 남아 있다.
현재 경로가 아닌 이유는 **Isaac Lab + Newton이 이 장기들을 구동하지 못하기**
때문이다 — soft-body 지원이 없고, 필요한 버전이 Isaac Sim 6.0이다.
FEM 장기 + 로봇 + 트로카가 한 씬에서 도는 것은 Isaac Sim 5.1 PhysX 경로뿐이다.

---

## 현재 상태

**되는 것**

- 장기 6개가 PhysX FEM으로 시뮬레이션되고, 기구로 누르면 변형하고 죠로 잡힌다
- FR3 두 대가 트로카를 RCM 제약으로 통과하며 양팔 동시 조작된다
- Touch 두 대로 텔레옵 — 클러치 래칫, 펜 트위스트 → 샤프트 롤, 추종 오차 햅틱
- 내시경 / D405 카메라가 실제 녹화와 맞춰진 화각·포즈로 렌더된다
- VLA 평가 환경이 액션 청크를 받아 관측·영상·점수를 낸다

**측정으로 확정된 수치** (근거는 각 문서에)

| | |
|---|---|
| 제어기 | `rig-answer-2026-07-30-p4` (soft RCM + velocity feed-forward, kp 5e5) |
| stand-off | 팬텀 ↔ 로봇 베이스 **0.46 m** |
| 시작 자세 | 양팔 각자 포트를 통해 100 mm 삽입, 팁 간격 ~46 mm |
| 자유 이동 추종 | 정적 droop 0.26 mm, 그 외 0.048 mm RMS |
| 실시간 배율 | **rtf 0.26–0.30** (한 팔, 권장 설정) |

**아직 안 된 것 / 한계**

1. **조직 물성이 문헌값이다.** E=20 kPa, ν=0.45는 장관 유사 조직의 문헌값이지 이
   녹화에서 역추정한 값이 아니다. 여기서 학습한 정책의 실물 전이는 보장되지 않는다.
2. **240 Hz 물리는 스크립트 에피소드에서만 검증됐다.** 기본값은 아직 480이다.
3. **rtf < 1** — 실시간이 아니다. 남은 최대 항목은 GPU 파이프라인 전환(24 %)과
   기구–장기 접촉 비용. [TELEOP_RESPONSIVENESS.md](TELEOP_RESPONSIVENESS.md) 참조.
4. **VLA 스크립트 정책의 리프트 구간에서 트로카 위반**이 관측된다(60 스텝 중 6).
   충수가 far-end world pin에 매달려 있어 그 근처를 잡고 들면 월드 앵커를 상대로
   당기는 것으로 보인다 — 에피소드 채점 전에 확인이 필요하다.
5. **뒷면 형상과 정지 상태**는 추정이다 (위 참조).
