# Rosota 제어·트로카·플랫폼 대조 기록

2026-10-04. CT 장면이 Rosota 제어 규약을 충족하는지 확인하기 위한 개발 기록이다.
현재 판정은 **부분 구현, 전체 규약 미충족**이다. 이전 영상의 7/7 목표 도달은
Rosota FR3 제어기 또는 실제 트로카/복벽 접촉 검증 결과가 아니다.

## 확인한 원본

- Simulator: `Rosota-Research/Rosota-Simulator`,
  `9135fe9eb8549d011124c8c57b7c86edeaaba735`.
- Platform: `Rosota-Research/rii-platform`,
  `cec7d11d628f1322086a7778f6bdbb1a7474fffa`.
- `platform.rosota.ai`는 실제 브라우저에서 RII 로그인 화면까지 확인했다.
  로그인된 실행 프로파일, 데이터셋, 배포 버전은 확인하지 않았다.
- 저장소 트리를 조회하고 관련 원본을 로컬 `.cache/rosota-audit/`에 읽기용으로
  저장했다. 비공개 원본과 바이너리를 새 공개 문서나 외부 서비스에 전송하지 않았다.
  기존 SOURCE_MANIFEST의 42개 원본 파일은 수정하지 않는다.

집중 검토한 부분은 `isaac/trocar.py`, `laparo_control.py`, `laparo_ik.py`,
`start_pose.py`, `build_surgical_cell.py`, `isaac/vla_env.py`,
`newton_cell/{arms,sim,vla_env,cell_spec,tool_urdf,fr3_urdf}.py`,
`FR3_CONTROL_HANDOFF.md`, `VLA_HANDOFF.md`, `CONTROLLER_STUDY.md`,
`rii/README.md`, `rii/adapter/{policy,run_eval}.py`와 플랫폼 실행·평가 계약이다.
저장소의 모든 파일을 실행 검증했다는 의미는 아니다.

## 1. RCM 오차 0은 포트 통과의 충분조건이 아니다

원본 `isaac/trocar.py:93`은 포트 평면과 원통 샤프트의 교차 타원을 계산한다.
구멍 반경 R=3 mm, 샤프트 반경 r=2.5 mm다. 중심을 정확히 통과해도
기울기 theta에서 필요한 반경은 `r/cos(theta)`다. 간섭 없는 기하학 한계는
약 33.56도이고, 0.15 mm 여유를 남기면 약 28.69도다.

현재 원본의 `TILT_CAP_DEG=60`은 탐색 범위를 넓힌 제어 상한이다.
원본 주석도 그 설정이 실제 구멍 통과를 보장하지 않는다고 명시한다.
FR3 핸드오프 문서의 28.69도와 기본 실행 코드의 60도를 구분해야 한다.
중심 RCM + 60도에서 간섭량은 2 mm다. 이 반례를 회귀 테스트에 넣었다.

이번 변경은 원본의 `port_geometry`를 그대로 재사용한다. CT 기구의 기본
cone은 0.15 mm 여유를 남기는 기하학 상한으로 변경했고, 관측 상태의
`bite`, 평면 교차 offset, tilt, 삽입 방향을 보고한다. 간섭은 환경 실패다.
단, CT 포트 위치/법선은 이전에 설정한 기구 프레임이며 복벽에서 측정한 값이
아니다. 이 검사는 해당 설정의 기하학 검사이고 복벽 힘 모델은 아니다.

## 2. 제어기는 원본을 재사용해야 한다

원본의 기준은 `laparo_control.BEST`와 `TipController.step()`이다.
soft RCM, command LPF 60 Hz, velocity feed-forward 1.0 및 LPF 30 Hz,
측정 관절 상태 reseed, joint 7 직접 roll, 관절 속도 균일 스케일을 사용한다.
루프 내부 순서까지 중요하다. 시뮬레이션의 kp=500000을 실기 게인으로 옮겨서는
안 된다는 것이 원본 FR3 핸드오프의 명시적 결론이다.

기존 CT `BimanualControl`은 5개 기구 변수로 USD kinematic body를 직접 배치한다.
FR3 Chain, 7관절 IK, 측정 관절 피드백과 articulation drive가 없다.
따라서 이름을 바꾸거나 속도값을 맞추는 것으로 같은 제어기가 되지 않는다.

새 `rosota_controller.py`는 원본 TipController와 IK를 직접 import한다.
strict profile은 `BEST.replace(tilt_cap_deg=0, manip_hz=60)`이며 변경점을
명시한다. 60 Hz manipulability 갱신은 원본 VLA 실행 설정을 따른다.
FR3 URDF와 실제 측정 q가 필요하고, 원본 파일은 수정하지 않았다.

`probe_rosota_controller.py`는 설치된 Isaac FR3의 관절 원점·한계와 Rosota
CAD tip 변환으로 **기구학 전용** 체인을 구성한다. 양팔 각각 480 tick에서
원본 제어기를 실행하여 삽입/롤 부호와 aperture를 검증한다.
이는 ideal joint follower 시험이다. CT 장면의 물리 articulation 연결이나
조직 하중 검증 완료를 뜻하지 않는다. 현재 CT 환경은 이 제어기를 사용한다고
표시하지 않으며 `rosota_controller_conformant=False`를 반환한다.

## 3. 정책 인터페이스와 좌표계

원본 Action은 `(2,K,7)`의 `[dx,dy,dz,rx,ry,rz,gripper]`다.
단위는 m/rad이고 그리퍼는 0~100의 절대 개방률이다. tip +z는 환자 바깥쪽이므로
**dz<0이 삽입**이다. rx/ry는 독립 자유도가 아니므로 실행하지 않고 rejected로
기록한다. rz만 roll이며 FR3 joint 7 방향과 반대다.

원본은 action sample 시작의 tool frame을 고정하고 30 Hz sample 하나를
480 Hz의 16 tick으로 분배한다. position command leash는 15 mm,
min_depth는 100 mm다. roll은 절대 목표·잔여량을 누적하고 관절 travel에서
anti-windup한다. 속도 제한 때문에 버려지는 회전을 숨겨서는 안 된다.

새 어댑터는 이 의미와 부호, frozen frame, 16 tick, reject 보고 및 roll
anti-windup을 구현한다. gripper output은 좌 74도/우 50도의 포함각 목표다.
힘 제한 폐구와 grasp latch는 물리 장면이 수행해야 하며 이 어댑터는 힘을 만들지 않는다.

기존 CT 환경의 10/16 normalized velocity action, 60 Hz, 30 mm 최소 삽입,
양손 동일 grasper는 여전히 legacy contract다. 원본의 우측 cutter, rig 좌표계,
D405 RGB-D, force channel도 동일하지 않다. 새 체크포인트 메타데이터에
이 구분과 제어 의존 파일 해시를 고정해 구형 정책을 조용히 재사용하지 않는다.

## 4. Newton 원본도 Isaac과의 동등성을 별도 검증해야 한다

검토한 pinned `newton_cell/sim.py:184`는 roll residual을 양수로 넘기지만
Isaac `ArmRig.apply`는 tip +z/샤프트 부호 차이 때문에 음수로 넘긴다.
Newton은 고정 `r_cmd0`를 쓰고, Isaac은 roll 목표 방향을 먼저 갱신한다.
Isaac의 joint travel anti-windup도 해당 Newton 경로에서 보이지 않는다.
또 Newton의 min_depth clamp는 법선 투영 거리이고 Isaac은 포트와 팁의 거리다.

따라서 README의 동일 인터페이스 설명만으로 같은 action에 같은 궤적이 나온다고
가정할 수 없다. 이는 소스 대조에서 찾은 불일치이며 이번에 upstream Newton
전체를 실행해 재현한 결과는 아니다. 해당 원본 파일을 수정하거나 이미 해결됐다고
표시하지 않는다. 엔진 비교 전에 동일 입력의 FK/roll/depth parity 시험이 필요하다.

## 5. 현재 내시경과 성공 판정의 한계

현재 EndoscopeControl은 optical centre에서 pan/tilt하며 translation ball 안에서
이동한다. 고정 trocar를 지나가는 긴 샤프트의 운동이 아니다.
`scope_trocar_constrained=False`를 명시했다. 스코프 port와 shaft-to-camera
외부 파라미터를 장면에 정의한 뒤 pivot으로부터 camera pose를 계산해야 한다.
기구와 카메라의 포트를 조직 목표 위치로부터 임의로 정하는 것으로는 해결되지 않는다.

원본에는 종료 한계(10 mm, 5 step 연속), excursion 집계(3 mm), 실제 aperture
간섭(bite)이 서로 다른 지표로 존재한다. 10 mm가 물리적으로 허용된다는 뜻은 아니다.
CT 과제의 목표 도달과 Rosota 호환/물리 검증 통과를 별도 항목으로 보고한다.
이번 CT geometry 검사는 control step 단위다. 실제 articulation을 연결한 뒤에는
모든 physics tick의 측정 shaft와 접촉 반력을 평가해야 한다.

## 6. 플랫폼은 제어기가 아니라 실행·평가 관리 계층이다

`rii-platform`은 FastAPI/PostgreSQL, Celery worker/Docker, Next.js UI 구조다.
실행은 immutable source/image/dataset, non-root, networkless가 기본이다.
소스는 `/workspace/source` read-only, 작업은 `/workspace/run`, 결과는
`/workspace/output`, 데이터는 `/datasets/<mount-key>`에 둔다.
metric은 유한 숫자의 JSONL이고 progress는 원자적 파일 교체로 갱신한다.

Simulator의 기존 `rii/adapter/run_eval.py`는 원본 `vla_rollout` summary를
플랫폼 metric으로 변환한다. 이미 RCM p95/max, bite p95, rubber 접촉 비율,
추종 오차, 힘 피크 등을 내보낸다. CT의 단일 coverage 값으로 대체하면 안 된다.

플랫폼 benchmark validator는 필수 metric 존재/중복과 실행 성공 등을 검사한다.
조직·트로카 안전 조건을 자동 추론하지 않는다. custom validation_rules도 현재
지원하지 않는다. 따라서 simulator scorer에서 물리 유효성을 success에 반영해야 한다.
동일 metric 이름을 여러 split/task로 보내면 필수 지표의 모호성으로 거부될 수 있다.

이 PC의 로컬 결과를 external-run API로 기록해도 `verified=False`다.
플랫폼에서 재현 실행한 결과와 구분되며 public board에 노출되지 않는다.
이번 작업에서는 API key를 읽거나 만들지 않았고, run 제출/결과 업로드도 하지 않았다.

## 다음 통합의 완료 조건

1. CT 복벽의 실제 port centre/normal 및 로봇 base-to-CT 변환을 장면 자산으로 저장하고 검증한다.
2. 양팔 FR3 + 좌 grasper/우 cutter articulation을 넣고 같은 원본 TipController를
   physics tick마다 호출한다. 명령 q와 측정 q를 따로 기록한다.
3. scope 전용 port, shaft/optics 외부 파라미터로 카메라 운동을 제한한다.
4. `(2,K,7)` action과 rig/tool 좌표계, rejection/backlog, force 및 camera 관측을 맞춘다.
5. 삽입·횡이동·롤 포화/역전·양팔 동시 동작·조직 하중에서 measured bite와
   FEM 상태를 연속 검증한다. 통과 전에는 Rosota-compliant/SOTA라고 표시하지 않는다.
6. 로컬 재현이 통과한 다음 플랫폼 image/dataset/metric version을 정해 동일 평가를 실행한다.

## 원본 링크

- [Simulator 제어 핸드오프](https://github.com/Rosota-Research/Rosota-Simulator/blob/9135fe9eb8549d011124c8c57b7c86edeaaba735/FR3_CONTROL_HANDOFF.md)
- [정책 입출력 계약](https://github.com/Rosota-Research/Rosota-Simulator/blob/9135fe9eb8549d011124c8c57b7c86edeaaba735/VLA_HANDOFF.md)
- [트로카 기하학](https://github.com/Rosota-Research/Rosota-Simulator/blob/9135fe9eb8549d011124c8c57b7c86edeaaba735/isaac/trocar.py)
- [플랫폼 Job 계약](https://github.com/Rosota-Research/rii-platform/blob/cec7d11d628f1322086a7778f6bdbb1a7474fffa/docs/job-contract.md)
- [플랫폼 benchmark 계약](https://github.com/Rosota-Research/rii-platform/blob/cec7d11d628f1322086a7778f6bdbb1a7474fffa/docs/benchmarks.md)
