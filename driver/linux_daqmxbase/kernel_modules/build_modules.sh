#!/bin/bash
# Rebuild the kernel-6.8 ported NI-KAL / NI-PAL modules (nikal.ko, nipalk.ko)
# from the official NI-DAQmx Base 15.0 ISO. Run on the Ubuntu host (needs the
# matching kernel headers + a host gcc; NOT inside the CentOS-7 container).
#
#   sudo apt-get install -y linux-headers-$(uname -r) build-essential bsdtar
#   ./build_modules.sh
#
# Output: ./out/nikal.ko  ./out/nipalk.ko
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="${WORK:-$HERE/work}"
OUT="$HERE/out"
KREL="$(uname -r)"
KDIR="/lib/modules/$KREL/build"
ISO_URL="https://download.ni.com/support/softlib/multifunction_daq/nidaqmxbase/15.0/linux/nidaqmxbase-15.0.0.iso"
ISO="$WORK/nidaqmxbase-15.0.0.iso"

mkdir -p "$WORK" "$OUT"
# ISO/RPM 추출물은 읽기전용 권한(dr-xr-xr-x)으로 풀려, 재실행 때 rm -rf 가
# Permission denied 로 죽는다. 지우기 전에 쓰기권한을 돌려준다.
chmod -R u+w "$WORK"/iso "$WORK"/visa "$WORK"/nikal "$WORK"/nipalk 2>/dev/null || true
[ -d "$KDIR" ] || { echo "ERROR: kernel headers missing: $KDIR"; exit 1; }

echo "== fetch ISO =="
[ -f "$ISO" ] || curl -L -o "$ISO" "$ISO_URL"

echo "== extract NI-VISA runtime (has nikal/nipalk sources) =="
rm -rf "$WORK/iso" && mkdir -p "$WORK/iso"
bsdtar -C "$WORK/iso" -xf "$ISO"
rm -rf "$WORK/visa" && mkdir -p "$WORK/visa"
tar -C "$WORK/visa" -xzf "$WORK/iso/nivisa/nivisa-runtime-15.0.0f2.tar.gz"
RPMS="$WORK/visa/rpms"

echo "== unpack nikal source + nipalk x86_64 object =="
rm -rf "$WORK/nikal" && mkdir -p "$WORK/nikal"
bsdtar -C "$WORK/nikal" -xf "$RPMS"/nikali-15.0.0-f1.noarch.rpm
rm -rf "$WORK/nipalk" && mkdir -p "$WORK/nipalk"
bsdtar -C "$WORK/nipalk" -xf "$RPMS"/nipalki-15.0.0-f0.x86_64.rpm

NSRC="$WORK/nikal/usr/local/natinst/nikal/src/nikal"
CLIENT="$WORK/nikal/usr/local/natinst/nikal/src/client"
NIPALK_O="$WORK/nipalk/usr/local/natinst/nipal/src/objects64/nipalk-unversioned.o"

echo "== build nikal.ko (apply kernel-6.8 port) =="
B="$WORK/build_nikal"; rm -rf "$B"; mkdir -p "$B"
cp "$NSRC/nikal.c" "$NSRC/nikal.h" "$B/"
patch -d "$B" -p0 < "$HERE/nikal.h.kernel68.patch" 2>/dev/null || \
    patch "$B/nikal.h" < "$HERE/nikal.h.kernel68.patch"
patch -d "$B" -p0 < "$HERE/nikal.c.kernel68.patch" 2>/dev/null || \
    patch "$B/nikal.c" < "$HERE/nikal.c.kernel68.patch"
cat > "$B/Makefile" <<EOF
obj-m := nikal.o
ccflags-y += -DHAVE_UNLOCKED_IOCTL -DHAVE_COMPAT_IOCTL -DnNIKAL100_kTargetName=\\"nikal.ko\\"
EOF
make -C "$KDIR" M="$B" modules
cp "$B/nikal.ko" "$OUT/"
cp "$B/Module.symvers" "$WORK/nikal.symvers"

echo "== build nipalk.ko (link precompiled .o against nikal symbols) =="
C="$WORK/build_nipalk"; rm -rf "$C"; mkdir -p "$C"
cp "$NIPALK_O" "$C/nipalk-unversioned.o"
cp "$B/nikal.h" "$C/nikal.h"
bash "$CLIENT/processmodule.sh" --cmd_geninterface --module="$C/nipalk-unversioned.o" > "$C/nipalk-interface.c"
sed -i 's|#include <config/modversions.h>|/* removed */|' "$C/nipalk-interface.c"
printf '\nMODULE_LICENSE("GPL");\n' >> "$C/nipalk-interface.c"
bash "$CLIENT/processmodule.sh" --cmd_updatesymrename --symfile="$C/nipalk.resyms" --module="$C/nipalk-unversioned.o"
objcopy --redefine-syms="$C/nipalk.resyms" \
    --rename-section .ctors=.ni_ctors --rename-section .init_array=.ni_init_array \
    --remove-section .exportedSymbols \
    "$C/nipalk-unversioned.o" "$C/nipalk-processed.o_shipped"
cat > "$C/Makefile" <<EOF
obj-m := nipalk.o
nipalk-objs := nipalk-interface.o nipalk-processed.o
ccflags-y += -I\$(src) -DHAVE_UNLOCKED_IOCTL -DHAVE_COMPAT_IOCTL -DnNIKAL100_kTargetName=\\"nipalk.ko\\"
KBUILD_EXTRA_SYMBOLS := $WORK/nikal.symvers
EOF
: > "$C/.nipalk-processed.o.cmd"
make -C "$KDIR" M="$C" modules
cp "$C/nipalk.ko" "$OUT/"

echo "== DONE =="
ls -la "$OUT"/nikal.ko "$OUT"/nipalk.ko
