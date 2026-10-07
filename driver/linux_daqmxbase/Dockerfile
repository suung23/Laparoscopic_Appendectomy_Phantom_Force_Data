# NI USB-6009 driver container for Linux (kernel 6.8) via NI-DAQmx Base 15.0.
#
# Stage layers:
#   - daqbase:base  : CentOS 7 (vault) + 32-bit runtime libs   (built separately)
#   - this image    : NI-VISA + DAQmx Base userspace (32-bit) + the kernel-6.8
#                     ported nikal.ko/nipalk.ko + the 8-ch reader binary.
#
# Build context must contain:  iso_x/  modules/{nikal.ko,nipalk.ko}  daq_reader.c  entrypoint.sh
#   docker build -f Dockerfile -t surgitag-daq:6009 .
#
# Run (privileged — loads host kernel modules; passes USB through; shares socket dir):
#   docker run --rm -it --privileged \
#       -v /dev/bus/usb:/dev/bus/usb \
#       -v /run/surgitag:/run/surgitag \
#       surgitag-daq:6009 --device Dev1 --channels 8 --rate 1000 \
#                         --socket /run/surgitag/daq.sock
FROM daqbase:base

COPY iso_x /opt/iso

# NI-VISA userspace (skip kernel rpms: nikali/nipalki-kernel/nivisak/niorbki);
# install nipalki *userspace* with --noscripts (we use our own kernel modules).
RUN cd /tmp && tar xzf /opt/iso/nivisa/nivisa-runtime-15.0.0f2.tar.gz && \
    rpm -i --nodeps --force \
        rpms/nipalerri-*.noarch.rpm rpms/niorberri-*.noarch.rpm \
        rpms/nipali-15.0.0-f0.i386.rpm rpms/niorbi-14.1.1-f0.i386.rpm \
        rpms/nimdnsresponder-14.0.0-f0.i386.rpm rpms/nivisa-15.0.0-f0.i386.rpm && \
    rpm -i --nodeps --noscripts --force rpms/nipalki-15.0.0-f0.i386.rpm && \
    rm -rf /tmp/rpms

# LabVIEW 2015 runtime engine + NI-DAQmx Base (32-bit)
RUN cd /opt/iso && rpm -i --nodeps --force \
        labview-2015-rte-15.0.0-2.i386.rpm \
        nidaqmxbase-common-15.0.0-f1.i386.rpm \
        nidaqmxbase-cinterface-15.0.0-f1.i386.rpm \
        nidaqmxbase-board-support-15.0.0-f1.i386.rpm \
        nidaqmxbase-usb-support-15.0.0-f1.i386.rpm

# 8-channel reader (32-bit, links libnidaqmxbase)
COPY daq_reader.c /opt/daq_reader.c
RUN gcc -m32 -O2 -std=gnu99 -o /opt/daq_reader /opt/daq_reader.c \
        -I/usr/local/natinst/nidaqmxbase/include \
        -L/usr/local/natinst/nidaqmxbase/lib -lnidaqmxbase

# kernel-6.8 ported modules + entrypoint
RUN mkdir -p /opt/mods
COPY modules/nikal.ko modules/nipalk.ko /opt/mods/
COPY entrypoint.sh /opt/entrypoint.sh
RUN chmod +x /opt/entrypoint.sh && rm -rf /opt/iso

ENTRYPOINT ["/opt/entrypoint.sh"]
