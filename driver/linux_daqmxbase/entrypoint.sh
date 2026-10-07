#!/bin/bash
# Driver-container entrypoint: bring up the kernel-6.8-ported NI-PAL/NI-KAL
# stack, enumerate the USB-6009, then run the reader that streams samples
# over a Unix socket. Must run with --privileged (loads host kernel modules).
set -e

MODS=/opt/mods
NIPAL=/usr/local/natinst/nipal
DMXB=/usr/local/natinst/nidaqmxbase
export LD_LIBRARY_PATH=$DMXB/lib:/usr/local/lib/LabVIEW-2015:/usr/local/vxipnp/linux/lib:$NIPAL/lib

echo "[entry] loading kernel modules (host kernel $(uname -r))..."
lsmod | grep -q '^nikal'  || insmod "$MODS/nikal.ko"
lsmod | grep -q '^nipalk' || insmod "$MODS/nipalk.ko"
lsmod | grep -E 'nikal|nipalk' || true

echo "[entry] creating /dev/nipalk (major of 'nikal')..."
MAJOR=$(awk '$2=="nikal"{print $1}' /proc/devices)
if [ -z "$MAJOR" ]; then echo "[entry] ERROR: 'nikal' not in /proc/devices"; exit 1; fi
rm -f /dev/nipalk
mknod /dev/nipalk c "$MAJOR" 0
chmod 0666 /dev/nipalk
echo "[entry]   /dev/nipalk -> c $MAJOR 0"

echo "[entry] NI-PAL persistent data + inf registration..."
mkdir -p /etc/natinst/nipal/inf /var/lib/natinst/nipal
echo "$NIPAL" > /etc/natinst/nipal/nipal.dir
PS=/var/lib/natinst/nipal/nipalps.bin
if [ ! -f "$PS" ]; then printf '%%' > "$PS"; chmod 600 "$PS"; fi
cat "$PS" > /dev/nipalk || true
# collect device .inf files and register them
find /usr/local/natinst -iname '*.inf' -exec cp -f {} /etc/natinst/nipal/inf/ \; 2>/dev/null || true
INFS=$(find /etc/natinst/nipal/inf -iname '*.inf' | sort)
if [ -n "$INFS" ]; then "$NIPAL/bin/readInfFiles" $INFS >/dev/null 2>&1 || true; fi

echo "[entry] starting nipalps service daemon..."
pkill -f nipalps 2>/dev/null || true
"$NIPAL/bin/nipalps" /dev/nipalk "$PS" &
sleep 1

echo "[entry] downloading USB firmware (if needed) + enumerating devices..."
"$DMXB/bin/dmxbUSBfirmware.sh" 2>/dev/null || true
"$DMXB/bin/lsdaq" || true

# One-time firmware switch: USB-6009 ships with full NI-DAQmx firmware; DAQmx
# Base needs its own firmware. Run `--fwupdate` once, then power-cycle the USB.
for a in "$@"; do
    if [ "$a" = "--fwupdate" ]; then
        echo "[entry] ===== FIRMWARE UPDATE MODE ====="
        echo "[entry] switching device to NI-DAQmx Base firmware via FWUpdate."
        echo "[entry] follow prompts; UNPLUG/REPLUG the USB-6009 afterwards."
        cd "$DMXB/bin"
        exec "$DMXB/bin/FWUpdate"
    fi
done

echo "[entry] launching reader: daq_reader $*"
exec /opt/daq_reader "$@"
