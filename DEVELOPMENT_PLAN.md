# 충수절제 시뮬레이터 개발·검증 계획

2026-10-04. 목표는 CT 해부학 기반 복강 내시경 영상과 도구–조직 상호작용을 갖춘 appendectomy RL 환경이다. 완료는 아래 단계별 실행 증거로 판정한다. 외과적 타당성과 물성 정확성은 별도 전문가/실험 검증을 요구하며 코드 테스트로 대체하지 않는다.

## M1 — 해부학과 실제 내부 시야 (진행 중)

### 기본 에셋 재검토 — 사용자 지적 반영

Appendix300 두 원본 클립을 2초 간격 순차 프레임과 확대 프레임으로
재검토했다. 현재 CT+atlas 수술 영역은 최종 학습용 시각 에셋 기준에
미달한다. CT는 공간 배치의 참고로 유지하되, 기존 atlas를 검증된 최종
수술 에셋으로 취급하지 않는다. 다음 우선 작업은 충수–맹장 연결,
얇은 막과 지방 소엽의 구분, 주변 장 주름과 겹침을 갖춘 수술 영역의
재구성이다. 재질만 추가하는 작업보다 우선한다. 정지 형태와 견인 시
연결 조직이 펼쳐지는 동작을 함께 비교해야 한다. 두 단안 영상만으로
환자별 3D나 두께 정답을 얻었다고 주장하지 않는다.
근거: `validation/appendix300-backbone-review.json`.

중간 표면 자동 추출은 수술용 형상으로 채택하지 않는다. 단일 층 투영은
겹친 층을 잃었고, 다층 추출은 일부를 보존했으나 톱니 경계가 남았다.
0.75 mm → 0.25 mm 간격 변경에서 연결 성분이 1개 → 6개로 바뀌어
위상 안정성도 확인되지 않았다. 추가 격자 간격 조정 대신, 부착 경계와
접힘을 명시한 연결 표면을 구성해야 한다. 추출물은 위치 참고용으로만
유지한다. 근거: `validation/meso-multilayer-evaluation.json`.

### 2026-10-05 목표와 실행 우선순위 재확정

최종 목표는 실제 충수절제술의 시각과 도구–조직 역학을 함께 재현하는 Rosota RL 시뮬레이터다. 기존 활성 목표의 범위를 유지한다. 수치 시험, API 동작, 단일 이미지 개선을 전체 완성으로 대체하지 않는다. 실세계와의 동일성은 입증되지 않았으며, 오차 범위와 검증 범위를 명시한다.

1. **해부학부터 수정:** Appendix300에서 확인한 충수–맹장–충수간막 관계를 기준으로 현재 두꺼운 충수간막, 단순 덩어리 형태, 잔존 중첩을 수정한다. 서로 다른 사례를 한 환자의 정답 3D로 취급하지 않는다. 생성한 상세와 영상에서 관찰한 구조를 구분한다.
2. **시각 통합:** 실제 내시경 영상의 조직 층·혈관·지방·반사·그림자·가림을 참고해 수정하고, 동일 장면에서 양팔 접근–파지–견인–해제 영상을 렌더링한다. UV 얼룩이 확인된 최초 재질 후보는 미채택 상태를 유지한다. 정지 화면만으로 동작 품질을 판정하지 않는다.
3. **실측 역학:** 힘–변위, 압박·견인 후 복원, 응력 완화, 마찰·미끄러짐, 손상·절단 자료를 확보하고 출처·조직 종류·실험 조건을 기록한다. 보정용과 검증용 데이터를 분리한다. 자료 확보 전에는 물성을 추정값으로 표기하며 동영상에서 힘 정답을 추정해 만들지 않는다.
4. **동일 상태의 렌더·반력:** 보이는 조직 변형과 도구 반력은 같은 물리 상태를 사용한다. 별도 검증 솔버의 성공을 Isaac 통합 성공으로 표시하지 않는다. trocar/RCM 제약 아래 양팔 접촉을 검증한다.
5. **성능·일반화:** RTX4090에서 실제 렌더·제어·물리 지연을 측정한다. 검증 장면과 학습 장면을 분리하고, 여러 해부학·물성·카메라 조건의 held-out RL 성공률과 손상률을 보고한다. 속도를 얻기 위해 숨긴 물리 차이가 없어야 한다.

완료 증거에는 출처가 연결된 실제/시뮬레이션 비교 영상, 해부학 검토, 독립 실측 역학 비교, 양팔 제약·접촉 기록, 성능 프로파일, held-out RL 결과가 모두 필요하다. 시각·역학 허용오차는 측정 프로토콜과 실측 변동성을 근거로 사전 정의하며, 현재 결과에 맞춰 통과선을 낮추지 않는다. 사람이 느끼는 촉각 동일성은 햅틱 장치와 실물 비교가 확보되기 전 미검증으로 유지한다.

현재 다음 산출물은 **해부학 형태가 수정된 내시경 장면과 양팔 견인 비교 영상**이다. 오프라인 솔버 미세 최적화만 반복하는 작업보다 이를 우선한다. 외부 전송 없이 로컬에서 개발하며 개인정보와 자격증명은 저장소에 포함하지 않는다.

2026-10-05 재질 결함 수정: 미터 단위 메시의 작은 삼각형이 xatlas에서
면적 0인 UV로 처리되던 문제를 확인했다. UV 계산용 좌표만 정규화하고
표면 좌표 기반 텍스처 베이크를 추가했다. 충수 1,952개·충수간막 2,392개를
포함해 대상 메시의 UV 퇴화 삼각형은 모두 0이 됐다. 충수·충수간막의
경계 색 차이는 최초 후보보다 각각 약 98%·95% 감소했다. Isaac Sim
6.1.0.0의 동일 시야 렌더에서 삼각형 얼룩 제거를 확인했으나, 불투명한
충수간막·단순한 외형·해부학 정합 문제와 양팔 접촉 통합은 남아 있다.
전체 시각 합격은 아니다. 근거: `validation/appendix300-appearance-v4-review.json`,
`validation/surface-texture-bake-normalized.json`. 다음 작업은 해부학 연결과
양팔 동작 통합이며, 정지 화면 개선을 전체 시뮬레이터 완성으로 대체하지 않는다.

1. CT/atlas 좌표, 충수의 맹장 부착, 장기 중첩을 수치로 진단한다.
2. camera/port와 scope shaft를 분리하고 도구 진입 경로의 충돌을 검사한다.
3. 임시 body contour와 실제 복막·기복 상태를 구분한다. 카메라 조정으로 해결되지 않는 장기 가림은 노출 과제로 다룬다.
4. Isaac RGB·semantic segmentation·depth와 현재 시야 검사를 같은 장면에서 저장한다.
5. 합격: 투시/장기 숨김 없이 충수 작업 부위가 실제 RGB에 나타남, camera 조직 침투 없음, 재시작 시 동일 시야, geometry/optics 파라미터와 한계 기록. 임상적인 port 타당성은 전문가 확인 전 pending.

### 2026-10-06 Appendix300 영상 유래 재질 후보

`ct_appendectomy/appendix300_video_texture.py`는 로컬 `Center1_014.mp4`
12초 프레임에서 수작업 지정한 조직 ROI 두 곳을 추출해 appendix 및
mesoappendix USD 재질 레이어를 생성한다. Appendix300 DOI·CC BY 4.0,
원본 영상·프레임·텍스처 해시와 ROI 좌표를 출력 manifest에 기록한다.
초기 거울 대칭 반복은 표면에서 인공적으로 보여 폐기하고, 고정 seed의
중첩 패치 quilting으로 1024² 비대칭 텍스처 후보를 만들도록 바꿨다.
작은 중성색 반사를 일부 억제하지만 남은 하이라이트가 있고, diffuse/specular
성분을 복원했다는 뜻은 아니다. 동일 seed 재생성 시 두 텍스처 해시가
일치했다. CT와 지방층을 합성한 USD에 후보 레이어를 얹은 뒤 54개 메시의
점, 면 인덱스, 면 수, 노멀 불변성과 두 표면의 UV 코너 대응을 다시 확인했다.

이것은 비디오 프레임의 색을 옮긴 미검토 시각 후보다. ROI는 인스턴스
분할되지 않아 해당 영상 조직과 명명된 mesh가 해부학적으로 같은지
검증하지 않았다. quilting은 시각 통계만 확장하며 픽셀을 3D 위치에 정합하지
않는다. 조명 반사와 반복 통계가 남고 재질 파라미터도 보정되지 않았다.
Isaac Sim Python 실행은 Windows Application Control에
차단되어 실제 endoscope 렌더·운동 비교가 미완료다. 정적 장면 레이어에는
PhysX 속성이 없으므로 물리 불변성이나 접촉 충실도도 이 시험의 범위 밖이다.
원본 프레임, 추출 텍스처, 생성 USD는 로컬 `out/` 아래에만 두며 Git에서
제외한다. 코드와 요약 검증은 `validation/appendix300-texture-quilting-v2.json`에
기록했다. 이 후보는 M1 시각 합격이 아니다.

## M2 — 기구와 물리 접촉

두 기구의 shaft/jaw, 포트 기반 RCM 및 속도/행동 제한을 구현한다. 접촉 파지와 보조 attachment를 구분한다. 접근–닫기–견인–열기 시퀀스, 비접촉 negative control, 관통/미끄러짐/핀 drift를 측정한다. 실제 물성 데이터가 없으면 estimated로 유지한다.

### 2026-10-06 문헌 역학 자료로 정한 시험 우선순위

사람 충수의 ex-vivo shear-rheometry 연구에서 control, purulent, gangrenous
조직의 1 Hz storage modulus 요약치와 0–40% 압축 프로토콜을 provenance
파일에 정리했다. 현재 Isaac material의 E=12 kPa, ν=0.45를 선형 등방성
작은 변형률 선형 등방성 가정으로만 환산하면 전단 계수는 4.14 kPa다. 이는 논문 control 평균의
3.64배, purulent의 6.21배, gangrenous의 6.45배다. 이는 비교용 경고값이며
직접 파라미터 보정은 아니다. 논문 표본 수가 작고 조직은 동결·해동된
생체 외 시료이며, 주파수 의존성·손실 계수·큰 변형을 기존 단일탄성
PhysX FEM 값에 대응시킬 수 없어 production material은 변경하지 않았다.

별도 포르신 대장 tribology 연구는 1–4 N clamp, 2 mm/s·15 mm sliding,
윤활 상태와 jaw profile에 따른 sticking/slip 기준을 제공한다. 이는
충수 실측값이 아니므로 appendix 파라미터에 전용하지 않고, 접촉 구현의
외부 비교 과제로만 쓴다. 두 연구의 조건·한계·미도입 데이터는
`research/appendix-mechanics-reference.json` 및
`research/grasper-colon-friction-reference.json`에 기록했다. 현 12 kPa
값과 문헌 범위를 비교하는 코드는 `ct_appendectomy/appendix_rheology_prior.py`,
산출물은 `research/appendix-material-scale-comparison.json`이다. 다음 물성
작업은 이 조건에 맞춘 8 mm coupon의 동적 shear·압축 stiffening·stress
relaxation 시험이며, 실제 Isaac 실행과 force readback 없이는 통과로 간주하지 않는다.

## M3 — 조직 연결·절단·결찰

맹장–충수–mesoappendix–혈관의 연결 상태를 물리 표현에 연결한다. SOFA 및 MPM 후보를 동일 coupon 실험으로 비교한 뒤 solver를 결정한다. 실제 topology 분리, 절단 표면 갱신, ligation constraint, 비의도 손상, 초기 topology로의 reset을 검증한다. 시각적인 절단 gesture는 합격이 아니다.

## M4 — 수술 과제와 RL

노출·혈관 제어·base closure·절제·회수·최종 검사의 simulator state와 성공/실패를 정의한다. Gymnasium seed/reset/step, terminated/truncated, RGB와 privileged state 분리, 잘못된 action 거절을 검증한다. scripted 성공·실패 baseline 후 작은 학습 실험으로 환경이 실제 학습 가능한지 확인한다.

## M5 — 충실도와 성능

RTX4090 단일 환경 1280×720 end-to-end 30 Hz를 개발 목표로 측정한다. p50/p95, VRAM, physics/render/transfer 지연을 분리한다. 수술 영상과 광학/재질을 비교하고 해부학 변이 및 seed별 회귀 검사를 수행한다. 느린 물리 정밀 모드와 학습 모드의 차이를 기록한다.

## M6 — 배포 가능한 회사 자산

설치/실행 자동화, 버전 고정, asset provenance/개별 라이선스, 재현 가능한 평가 명령, 실패 기록, 문서를 정리한다. 개인정보·키·사용자 대화·원본 환자 식별정보는 저장소에 넣지 않는다. 사용자 요청 없는 공개 배포는 하지 않는다.

## 완료 현황

- 기존 재현 결과: PhysX sphere 접촉 및 attachment 견인, 견인 Gym API.
- 실패 확인: 실험 카메라의 충수 가시성 0 sampled pixels; CT/atlas overlap으로 전체 장기 접촉 불안정.
- 미완료: M1–M6 전체 합격. 진행 결과는 아래에 증거 파일과 함께 추가한다.

참고: research/REFERENCE_REVIEW.md, research/IMPLEMENTATION_DECISION.md.

## 2026-10-04 실행 증거

- M1: CT mask distance field에 따른 고정 부착점 회전 최적화, mesh overlap 검사, 내부 camera 탐색을 구현했다. 실제 RTX 1280×720에서 충수 9.44% 화면 점유율과 RGB/semantic/depth 출력을 확인했다 (`validation/ct-internal-view.json`, `ct-internal-view.png`). 원래 충수 표면 샘플 약 30%가 대장 안에 있었고, 수정 후 약 9%로 감소했다. 중첩이 남아 있어 해부학 정합 합격은 아니다 (`ct-fitted-overlap.json`).
- M2: 두 hinged jaw의 실제 충돌, inner ridges, close/hold/lift/open 및 negative control을 구현했다. 자동 deformable collision margin이 첫 프레임 큰 변위를 유발해 contact 0.5 mm / rest 0 mm로 명시했다. 실패한 실행 결과도 validation에 보존한다. RCM 수학 모델은 구현·단위 검증했으나 jaw 물리와의 통합은 아직 별도 작업이다.
- M3: offline planar tetrahedral split, 양쪽 닫힌 절단면, 체적 보존, velocity interpolation을 구현했다. 2 mm atlas voxel volume에서 연결 성분 1→2와 체적 오차 0을 확인했다 (`ct-cut-topology.json`). 이후 Newton CUDA에서 고정측 drift 0, 자유측 약 20.00045 mm 이동, inverted tetrahedra 0을 확인했다 (`ct-cut-gpu.json`). 실행 중 절단, finite blade, damage law, 혈관/결찰 및 Isaac 연결은 아직 구현하지 않았다.
- 추가 발견: 현재 PhysX 자동 생성 volume mesh는 절단 입력 topology 검사에서 non-manifold face를 포함했다 (`ct-cooked-topology-failed.json`). 그대로 절단 엔진에 투입하지 않고 별도 conforming volume을 사용했다. Cook/finite-state 성공은 절단용 mesh 적합성 인증이 아니다.
- M2/M4: 같은 profile의 접촉 RL 시험에서 scripted baseline이 189 action 후 약 7.60 mm lift로 성공했다. 초기 관측 reset 오차와 RGB 획득에 따른 상태 변화는 모두 0이었다. 잘못된 action, 종료 후 action, time-limit truncation 검사도 통과했다 (`ct-forceps-rl.json`). 접촉을 끈 동일 480-frame 대조 시험은 최대 변위 약 0.0338 mm, lift 0, grasp 성공 false였다 (`ct-forceps-negative-control.json`). 물성 식별, 실제 손상 한계, image policy 학습을 검증한 결과는 아니다.
- M6: 원본 42개 파일의 Git blob hash 일치, 해부학·카메라·RCM·체적·절단 관련 단위 시험 13개 통과. `validate-ct.ps1`은 같은 source scene hash의 렌더·접촉 RL·negative control·기존 견인 RL을 순서대로 실행하고 report freshness와 성공 상태를 확인한다. 실패 실험과 미완료 기능을 숨기지 않고 함께 보존한다.
- 통합 suite 4개 모두 통과 (`ct-integration-suite.json`). 데스크톱 launcher도 새 forceps probe를 통과했고 `Isaac Sim Python 5.1.0` 창이 Windows 목록에 나타났다. 다만 UI 제어 도구의 `failed to activate captured window`가 재시도에서도 발생해 이번 GUI의 사용자 화면 표시·슬라이더 조작은 검증하지 못했다. RTX 저장 영상 검증과 GUI 검증을 구분한다.

전체 M1–M6를 완료로 표시하지 않는다. 다음 통합 순서는 RCM/jaw 통합 → 정합·기복 및 조직 연결 → live topology update → ligation/vascular state → 전체 과제/학습/성능 검증이다.

## 남은 개발의 구체적 통과 조건

추가 M4 구현: 등록된 Gymnasium agent 환경, 자동 episode/reset, RGB+state 관측, PPO 학습·checkpoint 재개, GUI 정책 비교를 추가했다. 실제 2,048-step PPO 학습 및 저장/재로딩은 통과했으나 learned policy 평가는 0/3 성공이다 (`validation/ct-agent-training.json`). 초기 scripted 환경은 3/3 성공했고 현재 재파지 수정 후 demonstration도 성공했다. 이는 학습 가능한 실행 경로의 검증이며, 학습 수렴이나 전체 수술 성공은 아직 아니다.

1. 정합: 충수 base 연결을 유지하면서 허용된 접합부 외 조직 중첩을 제거한다. 현재 약 9% 표면 샘플 중첩은 실패로 유지한다. CT에 없는 환자별 충수 형상이나 실제 복막을 추측해 ground truth로 취급하지 않는다.
2. RCM/기구: 두 실제 shaft/jaw의 pose를 고정 port에서 계산하고 도구·장기 충돌을 켠 상태로 접근/파지/견인/해제를 검사한다. 현재 2-action fixture를 이 단계의 성공으로 대체하지 않는다.
3. 절단: 움직이는 조직에 유한 blade가 접촉할 때만 topology를 갱신하고 위치·속도·질량과 절단면을 전달한다. 현재 offline infinite-plane split은 입력 기하 검증용이다. mesh 재구성 후 state 연속성과 reset을 검사한다.
4. 결찰/혈관: mesoappendix 및 혈관의 기계적 연결, 혈관 개방 상태, base closure와 stump 누출을 simulator state에 연결한다. 혈관 제어 전 절단과 맹장 손상은 실패 episode가 되어야 한다.
5. 전체 과제: 노출부터 회수까지 scripted 성공과 각 단계별 실패 사례를 재현한다. 그 다음 실제 학습 정책을 별도 seed에서 평가하고 success rate, 손상, 수행 시간 및 카메라 가림을 기록한다.
6. 충실도/배포: 수술자 검토 및 물성/영상 기준 데이터에 대한 오차를 확인한다. 30 Hz 목표는 startup 제외 실제 action–physics–RGB loop p50/p95로 측정한다. 회사 자산으로 배포하기 전 모든 사용 자산의 권리·버전·재현 경로를 고정한다.

## 양손 다중 과제 RL 통합과 새로 발견한 실패

`ct_appendectomy/bimanual_env.py`에 10-action 양손 RCM, 31-state Gymnasium
환경을 추가했다. 접근 대기, 지정 패치 양손 파지, 목표 높이 견인을 각각
과제로 정의하고, 연속 30-step 성공 유지와 매 제어 단계의 FEM 뒤집힘 검사를
연결했다. `ct_appendectomy/MULTITASK.md`에 실행법과 관측/행동 계약을 기록한다.

기존 양손 시연의 lift/final 검사 통과는 전 구간 안정성 통과가 아니었다.
새 검사는 접근 중 국소 tetrahedron inversion을 발견했다. 속도 감소와
convex decomposition 실험 모두 접촉 과제를 통과시키지 못했다. 실패를
허용하는 방향으로 기준을 변경하지 않는다. 도구–충수 전체 경로의 접촉
위치와 FEM 품질을 먼저 수정해야 한다. 접근 대기는 조직 중심 파지가 아니라
포트 방향 18 mm 지점으로 분리했으며, 전체 수술 성공으로 집계하지 않는다.

또한 새 프로세스에서 PhysicsContext가 GPU broadphase를 덮어쓰는 문제를
수정했다. 중력 정착 시 조직 변형이 없는 백엔드는 reset 단계에서 거부한다.
이 수정 전 정적인 조직에서 나온 reach/grasp 성공 기록은 무효이다.

원본 42개 파일은 그대로 보존한다. 접촉 과제, 조직 연결, 절단·결찰,
환자별 일반화, 실시간 영상 관측 학습의 합격 전에는 SOTA 또는 완성된 수술
훈련기로 표현하지 않는다.

현재 실행 결과: 18 mm 접근 대기 scripted 1/1 성공(224 steps), PPO 1,024-step
실제 업데이트·checkpoint 재로딩 통과, learned policy 0/1(484 steps에서
tet inversion). RGB 읽기에 따른 FEM 좌표 변화 0, 제어 step p50 약 61 ms로
30 Hz 목표 미달이다. 관련 단위 검사 11개가 통과했다. 이 결과는
`validation/ct-bimanual-multitask.json`에 기록하며 M4/M5 전체 합격으로
표시하지 않는다.

## 양팔·내시경 탐색과 Newton 접촉 후보

`ExplorationEnv`에 16-action(양팔 10 + scope 6), 42-state 및 RGB 관측을
구현했다. 기하 기반 제어기는 충수 주변 7개 paired tool-tip 목표를 모두
방문했고 양팔 끝단이 각각 9.07 mm 이동했다. 내시경도 함께 움직였으며
FEM 뒤집힘은 관측되지 않았다. 실제 PPO 256-step 업데이트와 checkpoint
로드를 검사했으나 학습 정책은 초기 목표만 방문하고 시간 초과했다.

Newton 1.4 CUDA VBD의 별도 5,940-tet atlas 접촉 시험은 960 Hz / 32 iterations
설정에서 접촉 활성·비활성 모두 뒤집힘 0으로 통과했다. 기본 설정 및
CUDA Graph 재생은 실패하여 채택하지 않았다. 구형 도구 coupon 결과이며
실제 CAD 집게나 Isaac 장면에 통합된 엔진이라고 표현하지 않는다.
Isaac의 960 Hz / 128 iterations CAD 접촉 과제 역시 실패했다.

근거: `validation/ct-exploration-physics.json`, 실행법:
`ct_appendectomy/EXPLORATION.md`. 복강 전체 충돌 회피, 시각 기반 탐색,
다환자 일반화, 전체 수술 및 SOTA 성능은 여전히 미검증이다.

## Rosota CT articulation follow-up (2026-10-04)

Implemented a separate `RosotaCTEnv` with dynamic PhysX FR3 joint drives,
original Rosota controller feedback at 480 Hz, and a fixed-port scope camera.
The 90-sample / 1,440-physics-tick free-space commissioning run passed geometry
and FEM checks. See `ct_appendectomy/ROSOTA_CT_RUNTIME.md` and
`validation/rosota-ct-articulation.json`. The initial closer tool placement
failed with tissue inversion and remains a recorded contact failure; withdrawal
for commissioning is not a fix for grasp/contact. Ports/base poses remain
engineering placements, both tips are graspers, and full arm/scope collisions,
force-limited grasp, right cutter and complete policy observations remain open.

## Contact-feedback follow-up (2026-10-04)

The dynamic CT path now reads actual tool-mount reactions and jaw efforts.
It applies a 0.00045 N.m jaw drive cap, measured-effort closure latch at
0.0003 N.m, anti-windup, reopening, and a separate mount-reaction arm hold.
RL observations expose jaw position/effort and hold flags. A first surface
contact experiment failed with a tet inversion; reducing the cap and adding
the jaw latch resolved that tested left-contact case. The extended repeat
passed 5,760 physics checks over 12 simulated seconds, with left closure
inhibition and both jaws reopening. Right closure completed without a
contact latch; this is not demonstrated bimanual tissue grasping.

See `ct_appendectomy/CONTACT_CONTROL.md`. Calibrated tissue force, Cartesian
impedance, sustained traction, a second validated contact site, and image-only
autonomous grasp remain open. Static-tare mount residuals must not be treated
as ground-truth tissue-force rewards or clinical limits.
