#!/bin/bash
# 팬텀 세션 — 비파괴 범위 (파단·젤 제외).
#
#   Phase 1   100 Hz · average · cutoff 5     grasp×10 · lift×20 · move×20 = 50 시행
#   Phase 2   1000 Hz · subsample · cutoff 0  slip_dry×10   (slip_gel·rupture 는 안 함)
#
# 두 Phase 는 수집 설정이 달라 한 프로세스로 못 돈다: rate/decimate 가 SocketSource
# 생성 때 고정되고(phantom_experiment.py:261) 거기서 LPF 계수·버퍼·h5 속성이 파생된다.
#
# Usage:
#   ./run_phantom_session.sh --specimen A01             # Phase 1 → (확인) → Phase 2
#   ./run_phantom_session.sh --specimen A01 -y          # 묻지 않고 이어서
#   ./run_phantom_session.sh --specimen A01 --phase 1   # Phase 1 만
#   ./run_phantom_session.sh --specimen A01 --phase 2   # slip_dry 만 (중단 후 재개)
#   ./run_phantom_session.sh --specimen A01 --phase 23  # 파괴 포함 (slip + rupture)
#   ./run_phantom_session.sh --specimen A01 --ymax 10   # 나머지 인자는 그대로 전달
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
RUN="$HERE/run_phantom_experiment.sh"

ASSUME_YES=0
PHASE=all
PASS=()
while [ $# -gt 0 ]; do
    case "$1" in
        -y|--yes) ASSUME_YES=1; shift ;;
        --phase)  PHASE="$2"; shift 2 ;;
        *)        PASS+=("$1"); shift ;;
    esac
done

bar() { echo "────────────────────────────────────────────────────────"; }

# ---- 드라이버: 없으면 띄운다 (명령 하나로 끝내기 위함) -------------------
SOCKDIR=${SOCKDIR:-/tmp/surgitag}
SOCK="$SOCKDIR/daq.sock"
DRV_NAME=surgitag-daq
DRV_DIR="$HERE/linux_daqmxbase"          # 이 호스트는 generic 커널. RT 커널이면 _realtime

ensure_driver() {
    # 이미 붙어 있으면 그대로 쓴다
    if docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$DRV_NAME"; then
        echo "[드라이버] 이미 실행 중 ($DRV_NAME)"
        return 0
    fi
    # 남의 컨테이너가 소켓을 잡고 있으면 건드리지 않는다
    if [ -S "$SOCK" ] && docker ps --format '{{.Names}}' 2>/dev/null | grep -q .; then
        echo "[드라이버] 다른 컨테이너가 떠 있다 — 그대로 쓴다:"
        docker ps --format '    {{.Names}}  {{.Status}}'
        return 0
    fi

    echo "[드라이버] 기동한다 (Dev1 · 8ch · 1000 Hz)"
    mkdir -p "$SOCKDIR"; chmod 777 "$SOCKDIR" 2>/dev/null || true
    rm -f "$SOCK"
    docker run --rm -d --name "$DRV_NAME" --privileged \
        -v /dev/bus/usb:/dev/bus/usb -v "$SOCKDIR":/run/surgitag \
        surgitag-daq:6009 --socket /run/surgitag/daq.sock \
        --device Dev1 --channels 8 --rate 1000 >/dev/null || {
            echo "[실패] 드라이버 컨테이너를 못 띄웠다."; return 1; }
    STARTED_DRIVER=1

    for _ in $(seq 1 60); do
        docker logs "$DRV_NAME" 2>&1 | grep -q "waiting for visualizer" && break
        docker logs "$DRV_NAME" 2>&1 | grep -q "incompatible firmware" && {
            echo "[실패] USB-6009 펌웨어 불일치."
            echo "       USB 를 뽑았다 꽂고 다시 실행. 그래도 같으면:"
            echo "       cd $DRV_DIR && ./run_fwupdate.sh"
            docker stop "$DRV_NAME" >/dev/null 2>&1; return 1; }
        sleep 1
    done
    if [ ! -S "$SOCK" ]; then
        echo "[실패] 소켓이 안 생겼다. 로그:"; docker logs "$DRV_NAME" 2>&1 | tail -8
        docker stop "$DRV_NAME" >/dev/null 2>&1; return 1
    fi
    echo "[드라이버] 준비됨 — 스트림 안정까지 6초 대기"
    sleep 6
}

STARTED_DRIVER=0
cleanup_driver() {
    if [ "$STARTED_DRIVER" = "1" ]; then
        echo; echo "[드라이버] 이 스크립트가 띄운 컨테이너를 정리한다"
        docker stop "$DRV_NAME" >/dev/null 2>&1 || true
    fi
}
trap cleanup_driver EXIT

ensure_driver || exit 1

run_phase1() {
    bar
    echo " Phase 1 — 조작 힘   (100 Hz · average · cutoff 5)"
    echo "   grasp×10 · lift_05/10/15/20×5 · move_L/R/F/B×5   = 50 시행"
    echo "   거상은 3초 유지. 이동은 10 mm 거상 유지한 채 왕복."
    echo "   [n]/[b] 단계  [space] 시행  [c] 영점  [u] 취소  [q] 종료"
    bar
    "$RUN" --rate 100 --decimate average --cutoff 5 "${PASS[@]}"
}

run_slip() {
    bar
    echo " Phase 2 — 미끄러짐 임계력   (1000 Hz · subsample · cutoff 0)"
    echo "   slip_dry×10 만 한다.  slip_gel·rupture 는 이번 범위가 아니다."
    echo "   파지 → 미끄러질 때까지 수직 견인. 판정은 F_tip 급락."
    echo "   ※ [n] 으로 slip_gel / rupture 까지 넘어가지 말 것"
    bar
    "$RUN" --rate 1000 --decimate subsample --cutoff 0 "${PASS[@]}"
}

run_slip_rupture() {
    bar
    echo " Phase 2~3 — 미끄러짐 + 파단   (1000 Hz · subsample · cutoff 0)"
    echo "   slip_dry×10 · slip_gel×10 · rupture×8"
    echo "   ※ 파괴 단계 포함. 종료 후 [1..4] 로 파단 부위 분류 입력 필수."
    bar
    "$RUN" --rate 1000 --decimate subsample --cutoff 0 "${PASS[@]}"
}

case "$PHASE" in
    1)  run_phase1;        exit $? ;;
    2)  run_slip;          exit $? ;;
    23) run_slip_rupture;  exit $? ;;
esac

run_phase1
rc=$?
if [ $rc -ne 0 ]; then
    echo
    echo "[중단] Phase 1 이 비정상 종료했다 (exit $rc). Phase 2 는 띄우지 않는다."
    echo "       이어서 하려면:  $0 --specimen <시편> --phase 2"
    exit $rc
fi

if [ "$ASSUME_YES" -ne 1 ]; then
    echo
    echo "Phase 1 종료. 같은 시편으로 Phase 2 (slip_dry) 를 이어서 한다."
    read -r -p "계속할까? [Y/n] " ans
    case "$ans" in
        n|N|no|NO) echo "중단. 이어서 하려면:  $0 --specimen <시편> --phase 2"; exit 0 ;;
    esac
fi

run_slip
