#!/bin/bash
# ONE-TIME firmware switch for the USB-6009 (run WITHOUT sudo — needs your X).
#
# The USB-6009 ships with full NI-DAQmx firmware; to use it through this Linux
# (NI-DAQmx Base) driver it must carry NI-DAQmx Base firmware. This flashes it.
# Reversible (re-flash with NI-DAQmx / MAX on Windows to go back).
#
# FWUpdate is a graphical (LabVIEW) tool, so we forward the host X display.
# After it finishes:  UNPLUG and REPLUG the USB-6009, then use ./run_driver.sh
set -e

SOCKDIR=${SOCKDIR:-/tmp/surgitag}
mkdir -p "$SOCKDIR"; chmod 777 "$SOCKDIR" 2>/dev/null || true
DISP="${DISPLAY:-:0}"

# allow the container's root user to talk to the local X server
xhost +local:root >/dev/null 2>&1 || true
trap 'xhost -local:root >/dev/null 2>&1 || true' EXIT

ARGS=(run --rm -it --privileged
      -e DISPLAY="$DISP"
      -v /tmp/.X11-unix:/tmp/.X11-unix
      -v /dev/bus/usb:/dev/bus/usb
      -v "$SOCKDIR":/run/surgitag
      surgitag-daq:6009 --fwupdate)

if docker info >/dev/null 2>&1; then
    docker "${ARGS[@]}"
else
    echo "(docker group not active in this shell — using 'sg docker')"
    sg docker -c "docker $(printf '%q ' "${ARGS[@]}")"
fi
