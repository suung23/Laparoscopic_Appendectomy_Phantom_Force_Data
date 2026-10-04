# 복강경 appendectomy simulator: 근거와 구현 결정

조사일: 2026-10-04. 목적: 현재 CT + atlas Isaac 실험을 실제 복강 내시경 수술 환경으로 발전시키기 위한 엔지니어링 조사. 논문·저자 저장소·공식 제품 문서를 우선했다. 아래는 범주별 핵심 레퍼런스 조사이며 모든 출판물의 전수조사나 동일 조건 성능 순위는 아니다. 이 문서의 구현 결정은 프로젝트 판단이며, 인용 연구가 우리 환경을 검증한 것은 아니다.

## 결론

Isaac Sim은 유지한다. OpenUSD, RTX 영상, 로봇 및 센서 계층에 활용하고, 현 PhysX 접촉 실험을 수술 전체의 물리 구현으로 간주하지 않는다. 조직의 절단·결찰·열손상은 전용 solver 후보를 동일 장면으로 비교한 뒤 선정한다. SOFA/LapGym을 공개된 수술 조작의 기준 구현으로, CRESSim-MPM을 GPU 절단 후보로 우선 평가한다. NVIDIA Newton surgical 모듈은 매우 관련성이 높지만 공개 배포 상태가 확보되기 전 핵심 의존성으로 삼지 않는다.

현재 body contour 10 mm inset은 복막이나 기복 모델이 아니다. Atlas 충수의 회전·이동도 환자별 registration이 아니다. 이를 외관 개선만으로 완료 처리하지 않는다. 환자 특이 충수가 없는 CT는 명시적으로 CT/atlas hybrid로 유지한다.

## A. 수술 물리와 RL: 직접 비교 대상

1. **LapGym, JMLR 2023 — SOFA.** 논문은 12개 복강경 RL 환경을 제시한다. 조직 견인, 박리, 정밀 절단, ligating loop가 직접 관련된다. 환경 구현은 MIT로 표시되지만 SOFA 및 플러그인의 별도 조건도 확인해야 한다. 전체 환자별 appendectomy를 제공한다는 의미는 아니다. 우선 적용: 과제 분해, 관측/보상, 절단·결찰 기초 실험. [논문](https://www.jmlr.org/papers/volume24/23-0207/23-0207.pdf), [코드](https://github.com/ScheiklP/sofa_env).

2. **ORBIT-Surgical, 2024 — Isaac Sim / Isaac Lab.** dVRK 기반 로봇 학습, GPU 병렬 환경과 수술 하위 과제가 강점이다. 현재 설치된 Isaac 5.1과 기존 예제의 버전 호환성은 별도 검증 대상이다. 코드 BSD-3-Clause 표기를 데이터 전체의 라이선스로 확대하지 않는다. [논문](https://arxiv.org/abs/2404.16027), [코드](https://github.com/orbit-surgical/orbit-surgical).

3. **SuFIA-BC, ICRA 2025 — Isaac 계열.** CT segmentation에서 mesh 정리·개선, texture 제작, OpenUSD 장기 모델로 이어지는 제작 방식이 우리의 시각 자산 파이프라인에 가장 직접적이다. 견인 등의 시각운동 imitation learning 연구이며 조직 절제 전체를 검증한 결과는 아니다. [프로젝트](https://orbit-surgical.github.io/sufia-bc/), [논문](https://arxiv.org/abs/2504.14857), [장기 자산 연결 release](https://github.com/orbit-surgical/orbit-surgical/releases/tag/v0.1.0). Release는 확인했으나 각 자산의 상업적 재사용 조건은 미확정이다.

4. **SurRoL, IROS 2021 — PyBullet.** dVRK 호환 surgical robot RL의 기준이다. 조작/정책 구조 참고에 적합하다. 기존 설치 지침의 오래된 Python/Gym 의존성을 현재 환경에 그대로 혼합하지 않는다. [공식 코드](https://github.com/med-air/SurRoL).

5. **AMBF — Bullet / CHAI3D.** 다물체, haptics, ROS 연결을 다루는 수술 로봇 시뮬레이션 기반. 로봇·힘 장치 연동 비교용이며 CT 기반 절단 엔진으로 바로 대체하는 선택은 아니다. [코드](https://github.com/WPI-AIM/ambf), [dVRK 시뮬레이션 안내](https://dvrk.readthedocs.io/main/pages/usage/simulation.html).

6. **Surgical Robotics Challenge의 Isaac port.** Isaac 기반 기구·수술 연습 장면 구현을 참고한다. 저장소의 Isaac/ROS 버전, deformable 및 thread 실행 모드를 분리 검증해야 한다. [코드](https://github.com/surgical-robotics-ai/isaac-sim-surgical-robotics-challenge).

7. **CRESSim-MPM, 2025 — CUDA MPM.** 절단·봉합을 겨냥한 공개 물리 후보. C API / Unity 연동 경로가 있지만 Isaac의 즉시 사용 가능한 플러그인은 아니다. 양방향 rigid coupling, topology와 렌더 표면 갱신 비용을 직접 측정해야 한다. 코드 BSD-3-Clause 표기. [논문](https://arxiv.org/abs/2502.18437), [코드](https://github.com/yafei-ou/CRESSim-MPM).

8. **CRESSim-Neo, 2026 preprint — GPU PBD.** rigid/soft/fluid/strand와 학습 연결 후보. 공개 C++17/Python 코드, Apache-2.0 표기, Windows MSVC 빌드 경로가 있다. 4090에서 보고한 대규모 CartPole 처리량을 복강경 처리량으로 사용하면 안 된다. [논문](https://arxiv.org/abs/2608.25192), [코드](https://github.com/yafei-ou/CRESSim-Neo).

9. **GPU PBD/MPM suturing RL, 2026.** 바늘 삽입과 추출을 위한 GPU 학습의 최신 비교군. 봉합 하위 과제의 성공률을 충수절제술 성공률로 전용하지 않는다. [논문](https://arxiv.org/abs/2607.27494).

10. **DiSECt, 2021.** 미분 가능한 절단과 물성 추정의 참고. 절단면과 힘 반응을 측정하는 실험 설계에 사용한다. 임의의 모든 수술 위상 변화가 자동 해결되는 것은 아니다. [논문](https://arxiv.org/abs/2105.12244), [프로젝트](https://diff-cutting-sim.github.io).

11. **SofaGym.** SOFA와 Gym 연결 및 soft robotics 환경의 보조 참고. 수술에 직접 대응하는 환경은 LapGym을 우선한다. [코드](https://github.com/SofaDefrost/SofaGym).

12. **MuJoCo flex.** 현재 MuJoCo는 연성체를 지원한다. 'MuJoCo는 rigid만 가능'이라는 배제 근거는 틀리다. 다만 flex 지원 자체가 절단·결찰·출혈까지 포함한 appendectomy 구현을 뜻하지 않는다. 해당 수술 기능과 GPU backend의 기능 동등성을 따로 확인해야 한다. 지금 엔진을 교체할 근거는 부족하다. [공식 문서](https://mujoco.readthedocs.io/en/stable/modeling.html).

13. **SofaUnity.** SOFA 물리와 Unity 시각 계층의 분리 사례. 공개 저장소가 있다고 전체 패키지가 자유로운 상업용 오픈소스인 것은 아니다. Asset Store EULA 조건을 별도 확인해야 한다. [코드](https://github.com/InfinyTech3D/SofaUnity).

## B. 최신 NVIDIA 경로: 발표와 실제 배포를 구분

14. **Medical Physics Simulation / Surgical Simulation Module, 2026.** 공식 발표는 Warp/Newton 기반 조직, 접촉, grasp, cutting, clipping, diathermy 및 기복된 복강의 담낭절제 예제를 설명한다. 이는 목표에 가까운 시스템 사례지만 appendectomy 검증은 아니다. 발표상 consumer GPU 실시간 수치를 우리 4090 성능으로 인용하지 않는다. [공식 발표](https://developer.nvidia.com/blog/developing-healthcare-robotics-with-gpu-native-medical-physics-simulation/).

15. **i4h 실제 배포 상태.** 조사일 문서는 surgical 폴더가 placeholder이며 omnisurg에서 이관 예정이라고 명시한다. 공개 catheter solver와 surgical package를 혼동하지 않는다. 외부 접근이 필요할 수 있으므로 당장 설치 완료 가능한 오픈소스 수술 엔진으로 계획하지 않는다. 문서의 Linux 요구사항도 Windows 직접 실행과 별개다. [공식 상태 문서](https://isaac-for-healthcare.github.io/medical-physics-simulation/), [저장소](https://github.com/isaac-for-healthcare/i4h-physics-simulation).

16. **Cosmos-H-Dreams, 2026.** 로봇 행동에 따른 수술 영상 생성 모델. 명시적 힘·절단 연결성·혈관 폐쇄 상태의 권위 있는 상태 저장소로 사용하지 않는다. 현재 공개 예제의 수술 종류와 GPU 성능 조건을 확인한 뒤 시각 domain adaptation 후보로만 취급한다. 코드와 모델 가중치 라이선스가 다르다. [코드](https://github.com/isaac-for-healthcare/Cosmos-H-Dreams), [모델 카드](https://huggingface.co/nvidia/Cosmos-H-Dreams), [논문](https://arxiv.org/abs/2608.24199).

## C. 실제 내시경 영상 및 3D 복원

17. **SimuScope, WACV 2025.** 복강경 수술 simulator의 annotation과 diffusion 기반 시각 변환을 결합한다. 논문은 담낭절제의 복강·기구·조직 상호작용을 다루므로 전체 장면 구성의 중요한 참고다. 공개 GitHub는 Training/Inference 중심이며 전체 물리 simulator 코드 공개를 확인한 것은 아니다. [논문](https://arxiv.org/abs/2412.02332), [코드](https://github.com/SanoScience/SimuScope).

18. **EndoNeRF, 2022.** 내시경 관측에서 변형 조직의 시각적 재구성. 보이지 않는 복강 전체와 절단 물리를 복구하는 도구로 간주하지 않는다. [코드](https://github.com/med-air/endonerf).

19. **Efficient EndoNeRF + simulation, 2024.** 복원 표면에서 시뮬레이션으로 연결하는 사례. 재구성 mesh의 폐곡면화와 FEM/MPM 연결을 검토할 직접 참고. [논문](https://arxiv.org/abs/2404.15339).

20. **EndoSurf, 2023.** 수술 영상으로부터 표면 및 변형 표현을 복원한다. CT와 수술 표면의 정합 연구에 참고. [논문](https://arxiv.org/abs/2307.11307), [코드](https://github.com/Ruyi-Zha/endosurf).

21. **EndoGS, 2024.** 동적 수술 장면 Gaussian 표현. appearance와 novel view 평가 후보. [논문](https://arxiv.org/abs/2401.11535), [코드](https://github.com/HKU-MedAI/EndoGS).

22. **EndoGaussian, 2024.** 효율적인 수술 장면 렌더링 연구. 논문의 영상 FPS는 접촉·절개·RL까지 포함한 전체 loop FPS가 아니다. [논문](https://arxiv.org/abs/2401.12561), [프로젝트](https://yifliu3.github.io/EndoGaussian/).

23. **SurgicalGS, MICCAI 2025.** 수술 장면 Gaussian 재구성 비교군. [논문](https://arxiv.org/abs/2410.09292).

24. **Surgical Gaussian Surfels, WACV 2026.** 최신 수술 표면·영상 품질 비교군. 물리적으로 조작 가능한 조직 상태와 별도의 평가 축이다. [논문](https://openaccess.thecvf.com/content/WACV2026/papers/Sunmola_Surgical_Gaussian_Surfels_Highly_Accurate_Real-time_Surgical_Scene_Rendering_using_WACV_2026_paper.pdf).

25. **EndoVis / SCARED.** stereo geometry, depth 및 calibration 검증의 데이터 후보. 수술 종류와 접근조건을 확인해야 하며 공개 페이지가 데이터의 무제한 상업 이용을 뜻하지 않는다. [공식 데이터 목록](https://opencas.dkfz.de/endovis/datasetspublications/), [SCARED](https://endovissub2019-scared.grand-challenge.org/).

## D. CT를 수술 중 복강으로 바꾸기

26. **Subject-specific pneumoperitoneum modelling, 2019.** 기복 전후 CT 및 변형 모델을 사용하는 접근. 우리의 단순 body contour inset을 대체할 모델링 근거다. 돼지/사람 결과와 해당 조건을 구분해야 한다. [논문](https://pmc.ncbi.nlm.nih.gov/articles/PMC6472552/).

27. **Patient-specific pneumoperitoneum simulation, 2019.** 압력에 따른 형상 변화 및 calibration을 다룬다. mesh 확장률을 실제 압력으로 곧바로 해석하지 않는다. [논문](https://pmc.ncbi.nlm.nih.gov/articles/PMC6736924/).

28. **PIVOTS, 2025.** 수술 전후 변형 정합 문제를 다루는 최신 비교군. 간 기반 결과를 충수 정합 검증으로 전용하지 않는다. [논문](https://arxiv.org/abs/2507.20337).

29. **OpenHELP, 2015.** CT 유래 장기와 복벽을 이용하는 물리적 훈련 phantom. 디지털 RL 엔진은 아니지만 복강 형상과 포트·시야 검증용 phantom 설계 참고다. [논문](https://pmc.ncbi.nlm.nih.gov/articles/PMC4607709/).

30. **TotalSegmentator.** 현재 장기 형상의 출처. 일반 복부 장기 segmentation과 환자별 충수 segmentation은 구별한다. 현 로컬 s0042 경로에는 충수 마스크가 없다. 사용 데이터 버전 및 라이선스는 기존 scene provenance를 유지한다. [논문](https://arxiv.org/abs/2208.05868), [코드](https://github.com/wasserth/TotalSegmentator).

31. **AOCR2024 appendix CT masks.** 공식 데이터 페이지는 appendicitis CT와 비염증성 대조 CT, abnormal appendix NIfTI masks를 명시한다. 5 mm slice 두께라 작은 구조의 3D 품질을 평가해야 한다. 라이선스가 Competition Rules로 되어 있어 상업 자산 편입은 아직 미승인 상태다. 정상 충수 및 mesoappendix/혈관 annotation이 확보됐다는 의미도 아니다. [데이터 설명](https://www.kaggle.com/competitions/aocr2024/data).

## E. 실제 appendectomy의 기능 기준

32. **LapSim Appendectomy.** sub-cecal/retro-cecal 변이, base 박리, loop/stapling, 절제와 회수라는 제품 기능을 제공한다. 우리가 비교할 전체 과제 범위다. 비공개 상용 제품이므로 내부 엔진이나 코드를 추정하지 않는다. [공식 제품](https://surgicalscience.com/simulators/lapsim/appendectomy-module/).

33. **LAP Mentor.** 전체 수술 및 훈련 curriculum의 상용 비교군. 사용자 경험과 과제 평가의 참고이며 코드 재사용 후보는 아니다. [제품](https://surgicalscience.com/simulators/lap-mentor/lap-mentor-vr/).

## 회사 자산에 반영할 때의 원칙

이번 조사는 외부 코드를 제품에 편입하지 않았다. 코드, mesh, CT, texture, pretrained weights 각각에 출처·버전·해시·개별 이용조건을 기록한다. 논문만 공개되거나 라이선스가 확인되지 않은 자산은 재사용 확정으로 표시하지 않는다. 계정정보, 개인키, 원본 사용자 대화와 비공개 저장소 내용은 외부 검색·게시 대상으로 사용하지 않는다.

## 다음 구현과 합격 기준

자세한 순서는 [구현 결정](IMPLEMENTATION_DECISION.md). SOTA 판단은 영상, 물리, 수술 범위, 실제 검증, 처리량을 분리한다. 현재 조사에서 모든 축을 동일 appendectomy benchmark로 비교한 근거는 확보하지 못했다.
