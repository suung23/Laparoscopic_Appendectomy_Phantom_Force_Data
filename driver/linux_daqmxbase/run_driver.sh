#!/bin/bash
# Start the USB-6009 "driver" container (privileged).
# It loads the kernel-6.8 NI-PAL/NI-KAL modules, enumerates the device, and
# streams 8-channel samples over a Unix socket shared with the host.
#
# Run WITHOUT sudo (sudo breaks the shared socket path). If your shell isn't in
# the docker group yet, this falls back to `sg docker` automatically.
#
# Usage:  ./run_driver.sh [--device Dev1] [--channels 8] [--rate 1000] [--chunk N]
set -e

SOCKDIR=${SOCKDIR:-/tmp/surgitag}
mkdir -p "$SOCKDIR"; chmod 777 "$SOCKDIR" 2>/dev/null || true

ARGS=(run --rm -it --privileged
      -v /dev/bus/usb:/dev/bus/usb
      -v "$SOCKDIR":/run/surgitag
      surgitag-daq:6009 --socket /run/surgitag/daq.sock "$@")

if docker info >/dev/null 2>&1; then
    exec docker "${ARGS[@]}"
else
    echo "(docker group not active in this shell — using 'sg docker')"
    exec sg docker -c "docker $(printf '%q ' "${ARGS[@]}")"
fi
