# CT appendectomy 구현 결정 — 2026-10-04

상태: 연구 근거에 따른 개발 계획. 아래 목표는 달성 결과가 아니다. 참조 번호는 REFERENCE_REVIEW.md를 따른다.

## 유지할 기반

Isaac 5.1 / OpenUSD / RTX를 카메라·로봇·영상 계층으로 유지한다 [2,3]. 현재 PhysX FEM과 sphere contact는 regression baseline으로 보존한다. 하나의 조직에는 하나의 solver만 상태 소유권을 갖도록 한다. 외부 solver 도입 시 Isaac에 표면을 전송하며 동일 조직에 PhysX와 외부 solver를 중복 적용하지 않는다. 전송 시간과 동기화 비용도 end-to-end 성능에 포함한다.

## 1. 복강 장면과 영상부터 검증

- 수술 전 CT, atlas graft, 복막, 복벽, 장간막을 별도 자산 및 의미 레이블로 관리한다.
- 현재 10 mm inset body envelope와 180도 atlas 회전은 실험 상태로 보존한다. 실제 복막/기복/등록이라는 명칭을 부여하지 않는다.
- 배꼽측 진입을 포함한 port 기반 카메라 자세, shaft/RCM, 0/30도 scope 방향, 센서/렌즈 intrinsics를 명시한다. 수치가 측정값인지 초기 가정인지 기록한다.
- optics: 근거리 광원, exposure, specular/roughness, distortion, vignette를 순서대로 calibration한다. 붉은 색 재질만으로 실사성을 평가하지 않는다 [3,17–25].
- 정적 검사는 camera 내부 장기 침투, near clipping, 전체 프레임의 first-hit 물체, 충수/meso/맹장 가시 면적을 측정한다. 이것은 초기 작업 시야 검사이며 수술 중 의도된 occlusion을 모두 실패로 판단하지 않는다.
- 실제 Isaac RGB와 segmentation을 같은 프레임으로 저장하고 사용자가 보는 창에서도 확인한다. 정적 ray 검사는 렌더 검증을 대체하지 않는다.
- 기복은 복벽 경계와 장기 변위의 별도 모델로 만든다 [26,27]. skin contour 안에 있다는 검사만으로 복강 내부라는 판정을 내리지 않는다.

## 2. 재현 가능한 조직 solver 비교

우선 LapGym/SOFA의 tissue manipulation, dissection, ligating loop를 기준 과제로 재현한다 [1]. CRESSim-MPM에는 같은 작은 tissue coupon과 절단 경로를 입력해 비교한다 [7]. CRESSim-Neo는 GPU throughput 후보 [8]; NVIDIA surgical은 소스 접근 및 이용조건 확보 후 후보 [14,15]. 설치 환경은 현재 Isaac 환경과 분리한다.

각 실행은 backend/version, input mesh hash, timestep/substeps, material parameters, random seed, warmup 및 측정 구간을 기록한다. Windows RTX4090에서 실제 설치·실행되지 않은 후보는 tested로 표시하지 않는다.

동일 실험: 무부하 rest → indentation → jaw grasp/release → traction → cutting → loop compression. 측정: 힘–변위 곡선, unloading hysteresis, volume drift, penetration, grasp slip, cut topology/connectivity, reset 재현성. 재료 파라미터는 측정 데이터 또는 명시적 prior로 구분한다. 실제 force 자료가 없으면 물성 검증은 pending으로 둔다.

성능 목표는 우선 단일 환경 1280×720에서 end-to-end 30 Hz로 설정한다. 이는 제품 개발 목표이지 논문 또는 현재 실행 결과가 아니다. physics/render/transfer 각각의 median 및 p95 latency, VRAM peak를 보고한다. 이후에만 1/8/32 환경을 비교한다. CartPole 결과를 수술 throughput으로 환산하지 않는다.

## 3. Appendectomy 상태와 성공 기준

과제 상태: identify → expose → control mesoappendix/artery → secure base → divide → retrieve → inspect [32,33]. 순서의 임상적 적합성과 해부학 변이는 전문가 검토 대상으로 둔다. 이 문서는 환자 수술 지침이 아니다.

성공은 화면에서 충수가 사라짐이나 목표 지점 도달만으로 판정하지 않는다. 절단된 연결성, proximal closure 상태, artery control, unintended organ damage, specimen retrieval을 simulator state로 판정한다. 출혈 모델을 구현하지 않은 경우 hemostasis 기능은 unsupported로 표시한다. Clip/loop 상태의 시각적 표시와 실제 constraint를 일치시킨다.

RL API는 seed/reset/step/terminated/truncated를 지원하고 privileged state와 RGB 정책 관측을 분리한다. RGB segmentation 및 GT는 평가용으로 명확히 분리한다. Reset은 절단 전 topology, attachment, fluid 및 material state까지 복원해야 한다. 시각 domain randomization으로 잘못된 해부학을 숨기지 않는다.

## 현재 증거와 미해결 항목

기존 validation/ct-contact*.json은 sphere–appendix 접촉과 negative control의 증거다. 실제 jaws, mesentery mechanics, ligation, cutting, perfusion 또는 전체 수술의 증거가 아니다. CT 주변 장기 접촉 활성화는 기존 정합 중첩으로 실패했으며 수정 전 학습에 사용하지 않는다.

이번 연구의 첫 구현은 ct_appendectomy/audit_view.py이다. USD의 실제 카메라와 mesh transform으로 first-hit 화면 점유율을 계산한다. 결과는 out/에 저장하며, 실패도 그대로 보고한다. 이 검사 통과만으로 임상적 해부학이나 실사성을 인증하지 않는다.

## 실제 시야 검사 결과

2026-10-04 작업 중인 실험 USD를 48×27 pinhole ray로 검사했다. 충수·mesoappendix·충수동맥은 샘플 ray에서 모두 0%, 소장 51.62%, 대장 13.27%, body envelope 35.11%였다. 따라서 현재 자세는 초기 충수 작업 시야로 합격하지 못한다. 저해상도 샘플이므로 모든 픽셀에서 가시 면적이 정확히 0이라는 단정은 하지 않는다. RTX 영상 검증도 별도로 필요하다.

결과는 validation/ct-view-audit-experimental.json에 보존했다. scene SHA-256을 포함한다. 이 장면은 아직 커밋되지 않은 body envelope/atlas rotation 실험에서 만들어졌으며, 기존 접촉 검증 장면과 동일하지 않다. 해당 검사 결과로 기존 접촉 수치를 재검증했다고 주장하지 않는다.

다음 수정은 충수를 억지로 소장 앞에 이동시키는 방식이 아니라, 정합·기복 공간·포트 접근을 개선하고 필요한 노출 조작을 구현하는 것이다. 소장/대장을 숨기는 표시 모드는 디버그로만 허용하고 수술 시야 합격에 사용하지 않는다.

실행 예시 (asset Python 환경):

```powershell
python ct_appendectomy/audit_view.py --scene out/ct_appendectomy/ct_appendectomy.usda --out out/ct_appendectomy/view_audit.json --width 48 --height 27
python -m unittest local_validation.test_view_audit
```

별도 synthetic 장면에서 mesh의 월드 transform, 앞쪽 blocker의 차폐, invisible blocker 제거에 따른 target 노출을 검사했고 2개 테스트를 통과했다. Windows native rtree 경로에서 access violation이 발생해 검사기는 bounded-memory NumPy triangle intersection을 사용한다. 실시간 renderer가 아니라 오프라인 geometry 검사다.
