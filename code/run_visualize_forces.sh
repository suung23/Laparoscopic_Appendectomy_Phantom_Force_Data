#!/bin/bash
# Step 3 (6 forces, 새 wiring) -- bending gripper/trocar x·y, gripping, torsion
# (채널↔힘 매핑은 wiring.py 참고). calib.json 있으면 자동 디커플링 적용.
# 드라이버 소켓에서 읽음. NOT in Docker.
#
# Usage:  ./run_visualize_forces.sh [--rate 100] [--window 5]
#         ./run_visualize_forces.sh --simulate
set -e

SOCKDIR=${SOCKDIR:-/tmp/surgitag}
HERE="$(cd "$(dirname "$0")" && pwd)"

if command -v conda >/dev/null 2>&1; then
    exec conda run --no-capture-output -n surgitag \
        python "$HERE/visualize_forces.py" --socket "$SOCKDIR/daq.sock" "$@"
else
    exec python3 "$HERE/visualize_forces.py" --socket "$SOCKDIR/daq.sock" "$@"
fi
