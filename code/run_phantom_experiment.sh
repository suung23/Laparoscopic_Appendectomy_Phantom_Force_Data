#!/bin/bash
# 팬텀 힘 스펙 측정 실험 GUI (PHANTOM_SPEC_PROTOCOL.md Phase 1~3 을 한 번에).
# visualize_forces 와 같은 드라이버 소켓·calib 행렬을 그대로 쓴다. NOT in Docker.
#
# 준비: 터미널 1 에서 드라이버가 떠 있어야 함
#   cd linux_daqmxbase_realtime && ./run_driver.sh --device Dev1 --channels 8 --rate 1000
#
# Usage:
#   ./run_phantom_experiment.sh --specimen A01                  # 조작 힘 단계 (100 Hz, 평활)
#   ./run_phantom_experiment.sh --specimen A01 --rate 1000 \
#        --decimate subsample --cutoff 0                        # 미끄러짐·파단 단계
#   ./run_phantom_experiment.sh --simulate                      # 하드웨어 없이 UI 확인
set -e

SOCKDIR=${SOCKDIR:-/tmp/surgitag}
HERE="$(cd "$(dirname "$0")" && pwd)"

if command -v conda >/dev/null 2>&1; then
    exec conda run --no-capture-output -n surgitag \
        python "$HERE/phantom_experiment.py" --socket "$SOCKDIR/daq.sock" "$@"
else
    exec python3 "$HERE/phantom_experiment.py" --socket "$SOCKDIR/daq.sock" "$@"
fi
