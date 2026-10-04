# RII laparoscopic simulator

이 폴더가 현재 개발 기준 저장소입니다. C:의 기존 작업 위치를 유지합니다.
D: 이전 및 권한 변경 시도는 중단했습니다.

현재 복강 탐색 장면은 `IsaacSimulator.exe` 또는 `./run-cavity.ps1`로 실행합니다.
내시경 패널에서 회전·삽입·좌우 이동을, Rosota 집게 패널에서 개폐·전후 이동을 조작합니다.
충수에는 중력과 후방 지지면을 적용했습니다. 다른 CT 장기는 고정되어 있습니다.
자세한 범위는 `ct_appendectomy/CAVITY_EXPLORATION.md`, 검증 수치는
`validation/ct-cavity-exploration.json`에 있습니다. 기존 RL 실행기는 이전 시험 장면을 사용합니다.

## 파일 안내

- `LOCAL_STATUS.md`: 실제 검증 범위, 제한 사항, 남은 개발 단계
- `validate-local.ps1`: 로컬 GPU 조직 안정성 검증 실행
- `local_validation/`: 로컬 검증 코드
- `validation/tissue-baseline.json`: 기존 RTX 4090 검증 결과
- `SOURCE_MANIFEST.json`: 원본 커밋 및 42개 파일 해시
- `LOCAL_ENVIRONMENT.txt`: 실행 환경 패키지 버전
- `LOCAL_DEPENDENCY_LICENSES.json`: 의존성 라이선스 메타데이터
- `isaac/`, `newton_cell/`, `sim/`: 원본 시뮬레이션 코드
- `mesh/`, `tet/`, `phantom.stl`: 확보한 원본 기하 자산
- `README.md`, `APPENDECTOMY_TASK.md`: 원본 저장소 문서

## 실행

이 폴더에서 PowerShell로 `./validate-local.ps1`을 실행합니다.
다른 Python 환경을 사용할 경우 `-Python <python.exe 경로>`를 지정합니다.
현재 기본 환경은 작업 루트의 `work/venv`이며 Git에 포함하지 않습니다.

이 저장소는 부분 스냅샷입니다. 조직 안정성 검증을 통과했으나
전체 로봇·내시경 영상·물리 절단을 검증한 완성 시뮬레이터는 아닙니다.
원본 데이터와 문서의 출처 및 권리는 그대로 유지합니다.

## 주변 작업과 구분

같은 `outputs` 아래의 `appendectomy_lab`은 CT/해부학 뷰어,
`laparo_rl`은 초기 MuJoCo 실험입니다. 이 저장소와 별도로 보존합니다.
SSH 키, 임시 다운로드 파일, 캐시 및 가상환경은 이 저장소에 포함하지 않습니다.

GitHub 비공개 저장소: https://github.com/Rosota-Intelligence-Institute/rii-laparoscopic-simulator
이번 정리는 로컬 커밋이며 원격 업로드는 수행하지 않습니다.

## Isaac CT development

`ct_appendectomy/` contains the CT + appendix Isaac Sim pipeline.
Run `./run-ct.ps1 -RLSmoke` for GPU FEM and Gymnasium checks.
Run `./run-ct.ps1 -Forceps -Interactive` for the internal endoscope view and jaw-contact controls.
Run `./validate-ct.ps1` for the complete local CT integration suite.
See `ct_appendectomy/README.md` for prerequisites, API and limitations.

## Robot-agent episodes and learning

`./run-agent.ps1 -Policy scripted -Episodes 0` opens the automatic contact-lift agent with episode reset and a pause/resume panel. `RobotAgent.exe` is the terminal-free launcher built by `./build-launcher.ps1`; it replays a saved policy when available and otherwise uses the scripted baseline.

`./run-agent.ps1 -Headless -TrainSteps 2048 -DemoWarmStart` collects a real simulator demonstration, initializes a policy by imitation, then performs PPO updates and evaluation. Omit `-DemoWarmStart` to train from random initialization. Results/checkpoints stay local in `out/ct_appendectomy/agent`. See `ct_appendectomy/AGENT_TRAINING.md` for Gymnasium integration, RGB observations, replay, resume and limitations.

`DEVELOPMENT_PLAN.md` records the six development gates, execution evidence and remaining work. Current jaw-contact RL is a controlled lift task; live cutting, ligation, bleeding and a complete appendectomy episode remain unimplemented. The separate cut-topology/GPU benchmark is not wired into the Isaac procedure.
