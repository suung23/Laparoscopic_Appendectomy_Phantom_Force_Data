# NI USB-6009 — 8채널 아날로그 수집 & 실시간 시각화 (Linux 네이티브)

NI USB-6009의 아날로그 입력 8채널을 **리눅스에서 네이티브로** 수집하고
실시간 그래프로 그립니다 (Windows / 가상머신 불필요).

구조는 **드라이버 ↔ 시각화** 두 부분으로 나뉩니다.

```
  [ 드라이버: linux_daqmxbase/ ]                 [ 시각화: 이 폴더 ]
  USB-6009 를 1000 Hz 로 읽어            Unix 소켓 ──▶  소켓을 100 Hz 로 읽어
  raw 값을 Unix 소켓으로 송출          /tmp/surgitag/    matplotlib 으로 그래프
  (Docker 컨테이너, --privileged)        daq.sock        (호스트, Docker 밖)
```

| 단계 | 코드 | 역할 |
|------|------|------|
| **Step 1–2 (드라이버)** | [`linux_daqmxbase/`](linux_daqmxbase/) | USB-6009 를 **1000 Hz** 로 읽어 raw 값을 소켓으로 출력 |
| **Step 3 (시각화)** | [`visualizer.py`](visualizer.py) (이 폴더) | 소켓 스트림을 **100 Hz** 로 솎아 실시간 그래프 |

> 리눅스에서 USB-6009 가 동작하는 이유와 커널 모듈 포팅 등 자세한 내용은
> [`linux_daqmxbase/README.md`](linux_daqmxbase/README.md) 참고. (NI 풀 DAQmx·Comedi 는
> USB-6009 를 지원하지 않아, NI-DAQmx Base + 커널 6.x 로 포팅한 모듈을 도커로 패키징합니다.)

> ### ⚙️ 커널에 맞는 드라이버 폴더 선택
> NI-KAL/NI-PAL **커널 모듈은 실행 중인 커널에 vermagic 이 묶여 있어**, 커널이 다르면
> 로드되지 않습니다. 현재 커널에 맞는 폴더를 쓰세요 (`uname -r` 로 확인):
>
> | 실행 커널 | 드라이버 폴더 |
> |---|---|
> | 일반(generic) 커널 `6.8.0-*-generic` | [`linux_daqmxbase/`](linux_daqmxbase/) |
> | **PREEMPT_RT 커널** `6.8.1-*-realtime` (Franka FR3 용) | [`linux_daqmxbase_realtime/`](linux_daqmxbase_realtime/) |
>
> 이 PC 는 Franka FR3 때문에 **PREEMPT_RT 커널(`6.8.1-1055-realtime`)** 로 전환되어,
> 지금은 **`linux_daqmxbase_realtime/`** 를 씁니다. 아래 «빠른 시작» 에서 폴더 이름만
> 바꾸면 사용법은 동일합니다. 두 폴더의 차이는 다음과 같습니다:
> - kernel-6.8 포팅 패치는 PREEMPT_RT 에서도 그대로 컴파일됨 → `build_modules.sh` 를
>   RT 커널에서 재실행해 모듈만 새로 빌드.
> - NI-DAQmx Base **유저스페이스는 커널 독립적** → 기존 `surgitag-daq:6009` 도커
>   이미지를 **그대로 재사용**하고, `run_driver.sh` 가 새 모듈(`kernel_modules/out`)을
>   컨테이너의 `/opt/mods` 에 **bind-mount** 해서 이미지 재빌드 없이 동작.
> - 실기기 검증 완료: 모듈 로드 → `Dev1` 인식 → 8채널 float64 스트리밍 정상.
> - 나중에 커널을 또 바꾸면 해당 폴더에서 `./kernel_modules/build_modules.sh` 만 재실행.
>   (`run_driver.sh` 는 모듈 vermagic 이 현재 커널과 다르면 경고를 출력합니다.)

---

## 빠른 시작

> 아래 예시는 일반 커널 기준으로 `linux_daqmxbase` 를 씁니다. **현재 이 PC(RT 커널)에서는
> `linux_daqmxbase` → `linux_daqmxbase_realtime` 로 폴더명만 바꿔** 그대로 실행하세요.

### 1) 설치 (한 번만, 다른 PC 에서도 동일)

```bash
cd linux_daqmxbase

sudo apt-get install -y docker.io build-essential binutils patch curl \
                        libarchive-tools linux-headers-$(uname -r)
sudo usermod -aG docker $USER          # 도커를 sudo 없이 — 이후 재로그인

./setup.sh --with-conda                # 베이스/모듈/드라이버 이미지 + 시각화 conda 환경
```

### 2) USB-6009 펌웨어 전환 (장비당 한 번)

```bash
cd linux_daqmxbase
./run_fwupdate.sh                      # GUI 안내 따라 Update → 끝나면 USB 뽑았다 다시 꽂기
```

### 3) 실행 — 터미널 2개

```bash
# 터미널 1 — 드라이버 (1000 Hz 수집 → 소켓 송출).  sudo 쓰지 마세요.
cd linux_daqmxbase
./run_driver.sh --device Dev1 --channels 8 --rate 1000

# 터미널 2 — 시각화 (소켓을 100 Hz 로 읽어 그래프)
cd ..            # NI6009 폴더
./run_visualizer.sh --display-rate 100 --window 5
```

소켓 경로는 양쪽 모두 `/tmp/surgitag/daq.sock` (환경변수 `SOCKDIR` 로 변경 가능).

---

## 직접 실행 (스크립트 없이)

```bash
# 드라이버: linux_daqmxbase/README.md 의 run_driver.sh 가 하는 일과 동일
# 시각화:
conda activate surgitag
python visualizer.py --socket /tmp/surgitag/daq.sock --display-rate 100 --window 5
```

| 시각화 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--socket` | `/tmp/surgitag/daq.sock` | 드라이버가 여는 소켓 경로 |
| `--display-rate` | 100 | 화면 표시 샘플레이트(Hz). 드라이버 1000 Hz 를 이 값으로 솎음 |
| `--window` | 5 | 그래프에 표시할 시간 윈도우(초) |
| `--vmin` / `--vmax` | -10 / 10 | 세로축 전압 범위(V) |

---

## 6-force 데이터셋 레코더 (`dataset_recorder.py`)

`visualize_forces.py` 가 그리는 **6가지 힘 그래프를 그대로 띄운 채로**, 각 샘플을
PC 의 **절대시간(Unix epoch, UTC 초)** 과 함께 HDF5 에피소드 파일로 기록합니다.
`MCU/dataset_recorder.py` 와 짝을 이루도록 설계되어(같은 저장 구조·같은 조작키·같은
절대시간 규약), NI6009 의 '힘' 데이터와 MCU 의 '기구학' 데이터를 나중에 **절대시간
기준으로 정렬·병합**할 수 있습니다.

저장되는 `forces` 값은 `visualize_forces.py` 가 화면에 그리는 값과 **완전히
동일**합니다 — 데이터 소스·calib(6×6 디커플링) 행렬·tare(영점)·저역통과(LPF) 처리
로직을 `visualize_forces.py` 에서 그대로 가져와 쓰기 때문입니다. 나중에 다른 calib 로
재계산할 수 있도록 채널 **원전압(raw)** 도 함께 저장합니다.

```bash
# 드라이버(터미널 1)는 위와 동일하게 먼저 실행해 둔 상태에서:
cd ..            # NI6009 폴더
./run_dataset_recorder.sh --rate 100 --window 5
#  또는 하드웨어 없이 테스트:
./run_dataset_recorder.sh --simulate --dataset test
```

* 실행 시 **데이터셋 번호**(= 저장 폴더명)를 입력받습니다. `--dataset 1` 로 미리 줄 수도 있습니다.
* 단축키:
  * **[C]** : 지금 무부하 상태를 힘 0점(tare)으로 다시 설정
  * **[R]** : 녹화 시작 / 중지 토글 (녹화 중에는 화면에 🔴 REC)
  * **[Q]** : 안전 종료 (녹화 중이면 자동 저장 후 종료)
* 파이프라인 옵션(`--rate`, `--calib`/`--no-calib`, `--cutoff`, `--quarter-cutoff`,
  `--tare-sec`, `--decimate` 등)은 `visualize_forces.py` 와 의미가 같으며, 같은 값으로
  주면 화면 표시값과 저장값이 정확히 일치합니다.

### 저장 위치 및 HDF5 구조
`collected_data/<데이터셋번호>/episode_<날짜시각>.h5` (`.gitignore` 로 버전 관리 제외).
각 데이터셋은 시간순으로 쌓이는 확장형 배열입니다.

| 데이터셋 | 형태 / 타입 | 설명 |
|---|---|---|
| `timestamp` | `()` `f8` | PC 절대시간 (Unix epoch, UTC 초). 다중 데이터셋 병합의 기준 |
| `forces` | `(6,)` `f4` | 6가지 힘 — **화면 표시값과 동일**(calib+tare+LPF 후). 컬럼 순서는 `force_keys` attr |
| `raw` | `(6,)` `f4` | 채널 원전압(V). 컬럼 순서는 `raw_keys` attr(bgx,btx,bgy,bty,grip,tor) |

#### `forces` 컬럼 순서 (6개 힘, 인덱스 0→5)
`forces[:, i]` 와 `raw[:, i]` 는 **같은 인덱스 `i` 가 같은 힘/채널**을 가리킵니다.
순서는 [`wiring.py`](wiring.py) 의 `WIRING`(= `force_keys` attr)에서 나오며 고정입니다:

| 인덱스 | `force_keys` (힘 키) | 표시 이름 | AI 채널 | 단위 | 비고 |
|---|---|---|---|---|---|
| 0 | `bending_gripper_x` | Bend grip x | A0 | N | half-bridge |
| 1 | `bending_trocar_x`  | Bend troc x | A1 | N | half-bridge |
| 2 | `bending_gripper_y` | Bend grip y | A4 | N | half-bridge |
| 3 | `bending_trocar_y`  | Bend troc y | A5 | N | half-bridge |
| 4 | `gripping`          | Gripping    | A2 | N | quarter-bridge |
| 5 | `torsion`           | Torsion     | A6 | N·m | quarter-bridge (오른쪽 축) |

`raw` 컬럼도 같은 순서의 원전압(V)이며, `raw_keys` = `[bgx, btx, bgy, bty, grip, tor]`.

주요 속성(attrs): `dataset_number`, `force_keys`, `force_units`, `raw_keys`,
`calib_mode`/`calib_path`, `rate`, `actual_rate`, `tare_enabled`, `cutoff_hz`,
`start_unix`, `start_iso_utc`, `end_unix`, `num_samples`, `timestamp_units`.

### 갑작스러운 종료에도 안전 (crash-safe)
녹화 중 **약 1초마다 HDF5 를 디스크로 flush** 합니다. 따라서 프로그램이 강제
종료되거나 전원이 꺼져도 파일은 손상 없이 열리며 **마지막 flush 시점까지의 데이터가
모두 복구**됩니다(손실 상한 ≈ 1초). 정상 종료(`[Q]`/창 닫기/`Ctrl-C`)는 `num_samples`,
`end_unix` 까지 기록하고 닫습니다. (flush 가 전혀 없으면 미완결 HDF5 파일은 truncated
되어 통째로 열리지 않습니다 — 그래서 주기적 flush 가 필수입니다.)

> 의존성: `pip install h5py` (`requirements.txt` 에 포함). GUI 표시가 가능한 환경에서 실행하세요.
