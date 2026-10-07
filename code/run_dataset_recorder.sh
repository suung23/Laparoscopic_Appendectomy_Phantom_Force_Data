#!/bin/bash
# 6-force dataset recorder - records the same values as visualize_forces, with
# absolute time (Unix epoch), to collected_data/episode_<time>.h5. Auto-applies
# calib.json if present. Single device, so no dataset number is needed (never asks).
# Reads from the driver socket. NOT in Docker.
#
# Usage:  ./run_dataset_recorder.sh [--rate 100] [--window 5]
#         ./run_dataset_recorder.sh --simulate
set -e

SOCKDIR=${SOCKDIR:-/tmp/surgitag}
HERE="$(cd "$(dirname "$0")" && pwd)"

if command -v conda >/dev/null 2>&1; then
    exec conda run --no-capture-output -n surgitag \
        python "$HERE/dataset_recorder.py" --socket "$SOCKDIR/daq.sock" "$@"
else
    exec python3 "$HERE/dataset_recorder.py" --socket "$SOCKDIR/daq.sock" "$@"
fi
