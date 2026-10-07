# NI USB-6009 on Linux (kernel 6.8) — native acquisition + host visualization

Run the NI USB-6009's 8 analog inputs **natively on this Ubuntu machine** (no
Windows, no VM) and visualize them live. NI officially supports the USB-6009 on
Linux only through the legacy **NI-DAQmx Base 15.0** driver, which targets old
distros (CentOS 7 / RHEL 6-7) and old kernels. We make it work on Ubuntu 22.04 /
kernel 6.8 with two pieces:

1. **A driver container** (CentOS 7) that holds the 32-bit NI-DAQmx Base runtime
   and reads the device, **publishing samples over a Unix socket**.
2. **A host-native visualizer** (runs in the `surgitag` conda env, *outside*
   Docker) that connects to that socket and plots — near-zero IPC latency.

The hard part was the kernel driver: NI-DAQmx Base needs the **NI-PAL/NI-KAL
kernel modules** (`/dev/nipalk`), whose 2015 source does not build on modern
kernels. We **ported NI-KAL (`nikal.c`, 8127 lines) to kernel 6.8** and relinked
the precompiled `nipalk` object against it. The kernel-abstraction design means
the proprietary `nipalk.o` only references NI-KAL symbols, so porting the open
`nikal.c` was sufficient (all 107 required symbols are re-exported unchanged).

## Architecture

```
  ┌─────────────────────────── this Ubuntu host (kernel 6.8) ───────────────────────────┐
  │                                                                                       │
  │   USB-6009 ──USB──▶  [ Docker: surgitag-daq:6009  (--privileged) ]                    │
  │                         • loads nikal.ko + nipalk.ko into the HOST kernel             │
  │                         • NI-DAQmx Base 15 (32-bit) reads ai0:7 @ rate                │
  │                         • daq_reader → writes float64 frames to                       │
  │                                  /run/surgitag/daq.sock  (Unix socket)                │
  │                                          │                                            │
  │                                          ▼  (bind-mounted to /tmp/surgitag)         │
  │   [ ../visualizer.py  in conda env `surgitag`, NOT in Docker ]  ◀── matplotlib       │
  └───────────────────────────────────────────────────────────────────────────────────┘
```

## Install — one command

```bash
# prerequisites (once)
sudo apt-get install -y docker.io build-essential binutils patch curl \
                        libarchive-tools linux-headers-$(uname -r)
sudo usermod -aG docker $USER          # docker without sudo — re-login after

# build everything: base image -> kernel modules (downloads NI ISO ~381 MB)
# -> driver image. Add --with-conda to also create the `surgitag` viz env.
./setup.sh --with-conda
```

`setup.sh` is safe to re-run and checks prerequisites (docker, kernel headers,
Secure Boot off). It builds the kernel modules **for whatever kernel you are
running**, so it works on any machine / any kernel 6.x.

<details><summary>What setup.sh does (manual equivalent)</summary>

```bash
docker build -f Dockerfile.base -t daqbase:base .   # CentOS7 + 32-bit libs
./kernel_modules/build_modules.sh                    # -> out/{nikal,nipalk}.ko
cp -r kernel_modules/work/iso iso_x                  # extracted RPMs
mkdir -p modules && cp kernel_modules/out/*.ko modules/
docker build -f Dockerfile -t surgitag-daq:6009 .    # driver image
```
</details>

> Prebuilt `kernel_modules/out/*.ko` match kernel `6.8.0-124-generic`.
> **On a different kernel, re-run `./setup.sh` (or `build_modules.sh`).**

## Run

```bash
# (ONCE per device) switch the USB-6009 to NI-DAQmx Base firmware.
# Follow the prompt, then UNPLUG/REPLUG the device. Reversible.
./run_fwupdate.sh

# Terminal 1 — the driver (privileged: loads kernel modules, reads the device)
./run_driver.sh --device Dev1 --channels 8 --rate 1000

# Terminal 2 — the host-native visualizer lives in the parent NI6009/ folder
#              and decimates the 1000 Hz stream to a 100 Hz display.
( cd .. && ./run_visualizer.sh --display-rate 100 --window 5 )
```

Both use `/tmp/surgitag/daq.sock` (override with `SOCKDIR=...`).

> **Firmware note:** the USB-6009 ships with full NI-DAQmx firmware. On Linux it
> must carry NI-DAQmx **Base** firmware — `run_fwupdate.sh` does this once. If
> `lsdaq` says *"incompatible firmware"* or the reader says *"Device identifier
> is invalid"*, you still need this step. To use the device on Windows again,
> re-flash via NI MAX.

## Files

| File | Role |
|------|------|
| `setup.sh` | one-command installer (base image + modules + driver image) |
| `run_driver.sh` / `run_fwupdate.sh` | driver + firmware launchers |
| `requirements-host.txt` | visualizer deps (numpy, matplotlib) |
| `Dockerfile.base` | CentOS 7 (vault) + 32-bit runtime libraries |
| `kernel_modules/nikal.c.kernel68.patch` | the NI-KAL → kernel-6.8 source port |
| `kernel_modules/build_modules.sh` | rebuilds `nikal.ko` + `nipalk.ko` from the NI ISO |
| `Dockerfile` | driver image: DAQmx Base userspace + modules + reader |
| `entrypoint.sh` | loads modules, brings up NI-PAL, enumerates, runs reader |
| `daq_reader.c` | 32-bit reader: 8-ch AI → Unix-socket binary stream |
| `../visualizer.py` / `../run_visualizer.sh` | host-native matplotlib client (step 3, in NI6009/) |

## Wire protocol (Unix socket)

On connect: 16-byte header `char magic[4]="NIDQ"; int32 nchan; int32 rate; int32 chunk`.
Then repeating frames of `nchan*chunk` `float64`, grouped by channel
(`[ch0 s0..s_{chunk-1}][ch1 ...] ...`). Same-host byte order.

## What was modified to reach kernel 6.8

`nikal.c` (NI-KAL) carried ~24 categories of removed/changed kernel APIs, all
fixed in the patch: `mmap_sem`→`mmap_lock`, write-protected `vm_flags`
accessors, 4-arg `get_user_pages`, `timer_setup`/`from_timer`, `proc_ops`,
generic-netlink family registration, `vm_fault_t` fault handlers, 5-level page
tables, `ktime_get_real_ts64`, `vm_mmap`/`vm_munmap`, `sg_set_page`,
`from_kuid`, `pci_bus->busn_res`, void ACPI `.remove`, GPL `MODULE_LICENSE`
(so modern GPL-only symbols link; the NI-KAL ABI is still `EXPORT_SYMBOL`, so
the proprietary `nipalk.o` still binds). Serial/UART support is compiled out
(not needed for DAQ, not in the `nipalk` ABI).

## Notes & caveats

- The driver container needs `--privileged` because it inserts modules into the
  **host** kernel; they persist until `rmmod nipalk nikal` or reboot.
- Secure Boot must be off (unsigned modules). Verified off on this host.
- USB-6009 via DAQmx Base supports hardware-timed multi-channel multi-sample AI;
  aggregate rate ≤ 48 kS/s (8 ch × rate). Lower `--chunk` = lower latency, more
  USB transactions.
- This uses NI's proprietary NI-DAQmx Base under its own license; you must
  download the ISO yourself (the build script does). The port patch is a
  derivative of NI's GPL-context kernel-abstraction source.
