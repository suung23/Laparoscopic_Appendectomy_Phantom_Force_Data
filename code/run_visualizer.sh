#!/bin/bash
# Step 3 -- start the host-native visualizer (NOT in Docker).
# Connects to the driver's Unix socket and plots the 8 channels live,
# decimating the 1000 Hz acquisition stream down to a 100 Hz display rate.
#
# Usage:  ./run_visualizer.sh [--display-rate 100] [--window 5]
set -e

SOCKDIR=${SOCKDIR:-/tmp/surgitag}
HERE="$(cd "$(dirname "$0")" && pwd)"

if command -v conda >/dev/null 2>&1; then
    exec conda run --no-capture-output -n surgitag \
        python "$HERE/visualizer.py" --socket "$SOCKDIR/daq.sock" "$@"
else
    exec python3 "$HERE/visualizer.py" --socket "$SOCKDIR/daq.sock" "$@"
fi
