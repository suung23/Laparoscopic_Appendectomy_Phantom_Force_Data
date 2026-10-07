#!/bin/bash
# One-shot installer for native USB-6009 acquisition on Linux.
# Safe to re-run. Works on a fresh machine (any recent Ubuntu + kernel 6.x).
#
#   ./setup.sh
#
# Steps: prereq check -> CentOS7 base image -> build kernel modules for THIS
# kernel (downloads NI ISO ~381 MB) -> assemble build context -> driver image.
# Visualizer deps are set up separately (see --with-conda).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
KREL="$(uname -r)"
WITH_CONDA=0
[ "${1:-}" = "--with-conda" ] && WITH_CONDA=1

say() { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }
err() { printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

say "1/6  Checking prerequisites"
need() { command -v "$1" >/dev/null 2>&1 || err "missing '$1' — install it first ($2)"; }
need docker      "https://docs.docker.com/engine/install/"
need gcc         "sudo apt-get install build-essential"
need make        "sudo apt-get install build-essential"
need objcopy     "sudo apt-get install binutils"
need patch       "sudo apt-get install patch"
need curl        "sudo apt-get install curl"
need bsdtar      "sudo apt-get install libarchive-tools"

# docker usable? (group may not be active in this shell -> try sg)
DOCKER="docker"
if ! docker info >/dev/null 2>&1; then
    if sg docker -c "docker info" >/dev/null 2>&1; then
        DOCKER="sg docker -c"
        say "    (using 'sg docker' — your shell's docker group isn't active yet)"
    else
        err "cannot talk to docker. Add yourself to the 'docker' group: sudo usermod -aG docker \$USER (then re-login), or run with sudo."
    fi
fi
dk() { if [ "$DOCKER" = "docker" ]; then docker "$@"; else sg docker -c "docker $*"; fi; }

[ -d "/lib/modules/$KREL/build" ] || err "kernel headers missing for $KREL — sudo apt-get install linux-headers-$KREL"

if command -v mokutil >/dev/null 2>&1; then
    if mokutil --sb-state 2>/dev/null | grep -qi enabled; then
        err "Secure Boot is ENABLED — unsigned modules cannot load. Disable it in firmware, or sign the modules."
    fi
fi
echo "    OK (kernel $KREL)"

say "2/6  Building CentOS 7 base image (daqbase:base)"
dk build -f Dockerfile.base -t daqbase:base .

say "3/6  Building kernel modules for $KREL (downloads NI ISO if needed)"
./kernel_modules/build_modules.sh

say "4/6  Assembling driver build context"
rm -rf iso_x modules
cp -r kernel_modules/work/iso iso_x
mkdir -p modules
cp kernel_modules/out/nikal.ko kernel_modules/out/nipalk.ko modules/
echo "    iso_x/ + modules/ ready"

say "5/6  Building driver image (surgitag-daq:6009)"
dk build -f Dockerfile -t surgitag-daq:6009 .

say "6/6  Host visualizer dependencies"
if [ "$WITH_CONDA" = 1 ] && command -v conda >/dev/null 2>&1; then
    conda env list | grep -q '^surgitag ' || conda create -y -n surgitag python=3.11
    conda run -n surgitag pip install -r requirements-host.txt
    echo "    conda env 'surgitag' ready"
else
    echo "    skipped. For the visualizer install: numpy, matplotlib"
    echo "    e.g.  conda create -n surgitag python=3.11 -y && conda run -n surgitag pip install -r requirements-host.txt"
    echo "    (or re-run: ./setup.sh --with-conda)"
fi

cat <<EOF

\033[1;32mDONE.\033[0m  Run it:
  (once per device)         ./run_fwupdate.sh
  Terminal 1 (driver):      ./run_driver.sh --device Dev1 --channels 8 --rate 1000
  Terminal 2 (visualizer):  ../run_visualizer.sh --window 5      # in NI6009/
EOF
