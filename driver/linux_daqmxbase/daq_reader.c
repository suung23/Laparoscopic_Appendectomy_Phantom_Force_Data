/*
 * daq_reader.c — NI USB-6009 8-channel analog-input reader for Linux,
 * built on NI-DAQmx Base 15.0 (32-bit). Runs inside the CentOS-7 driver
 * container. Hardware-timed acquisition; publishes raw samples over a
 * Unix-domain socket so a native host program can visualize them with
 * essentially zero IPC latency.
 *
 * Wire protocol (little-endian, host byte order — same machine both ends):
 *   On client connect, one 16-byte header:
 *       char  magic[4] = "NIDQ"
 *       int32 nchan
 *       int32 rate_hz
 *       int32 chunk           (samples per channel per frame)
 *   Then repeating frames of (nchan * chunk) float64, GroupByChannel:
 *       [ch0 s0..s(chunk-1)][ch1 s0..s(chunk-1)]...[ch(n-1) ...]
 *
 * Build (32-bit):
 *   gcc -m32 -O2 -o daq_reader daq_reader.c \
 *       -I/usr/local/natinst/nidaqmxbase/include \
 *       -L/usr/local/natinst/nidaqmxbase/lib -lnidaqmxbase
 */
#include <NIDAQmxBase.h>

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <sys/un.h>

static volatile sig_atomic_t g_stop = 0;
static void on_signal(int s) { (void)s; g_stop = 1; }

static TaskHandle g_task = 0;

static void die_daq(const char *what, int32 err)
{
    char buf[2048] = {0};
    DAQmxBaseGetExtendedErrorInfo(buf, sizeof(buf));
    fprintf(stderr, "[daq_reader] %s failed (err %d): %s\n", what, (int)err, buf);
    if (g_task) { DAQmxBaseStopTask(g_task); DAQmxBaseClearTask(g_task); }
    exit(1);
}

int main(int argc, char **argv)
{
    const char *device   = "Dev1";
    const char *sockpath = "/run/surgitag/daq.sock";
    int   nchan = 8;
    double rate = 1000.0;
    int   chunk = 0;          /* 0 -> auto (rate/50, ~20 ms / 50 fps) */
    double vmin = -10.0, vmax = 10.0;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--device")   && i+1 < argc) device   = argv[++i];
        else if (!strcmp(argv[i], "--socket")   && i+1 < argc) sockpath = argv[++i];
        else if (!strcmp(argv[i], "--channels") && i+1 < argc) nchan    = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--rate")     && i+1 < argc) rate     = atof(argv[++i]);
        else if (!strcmp(argv[i], "--chunk")    && i+1 < argc) chunk    = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--vmin")     && i+1 < argc) vmin     = atof(argv[++i]);
        else if (!strcmp(argv[i], "--vmax")     && i+1 < argc) vmax     = atof(argv[++i]);
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); return 2; }
    }
    if (chunk <= 0) { chunk = (int)(rate / 50.0); if (chunk < 1) chunk = 1; }

    signal(SIGINT,  on_signal);
    signal(SIGTERM, on_signal);
    signal(SIGPIPE, SIG_IGN);

    /* ---- DAQ task: ai0:(nchan-1), RSE, +/-10 V, hardware-timed cont. ---- */
    char chanspec[64];
    snprintf(chanspec, sizeof(chanspec), "%s/ai0:%d", device, nchan - 1);

    int32 err;
    if ((err = DAQmxBaseCreateTask("", &g_task)))
        die_daq("CreateTask", err);
    if ((err = DAQmxBaseCreateAIVoltageChan(g_task, chanspec, "",
                 DAQmx_Val_RSE, vmin, vmax, DAQmx_Val_Volts, NULL)))
        die_daq("CreateAIVoltageChan", err);
    if ((err = DAQmxBaseCfgSampClkTiming(g_task, "", rate, DAQmx_Val_Rising,
                 DAQmx_Val_ContSamps, (uInt64)(chunk * 10))))
        die_daq("CfgSampClkTiming", err);

    fprintf(stderr, "[daq_reader] %s  %dch @ %.0f Hz  chunk=%d  socket=%s\n",
            chanspec, nchan, rate, chunk, sockpath);

    /* ---- Unix-domain socket server ---- */
    int srv = socket(AF_UNIX, SOCK_STREAM, 0);
    if (srv < 0) { perror("socket"); return 1; }
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", sockpath);
    unlink(sockpath);
    if (bind(srv, (struct sockaddr*)&addr, sizeof(addr)) < 0) { perror("bind"); return 1; }
    if (listen(srv, 1) < 0) { perror("listen"); return 1; }
    chmod(sockpath, 0666);

    const int frame_samps = nchan * chunk;
    double *buf = malloc(sizeof(double) * frame_samps);
    if (!buf) { perror("malloc"); return 1; }

    while (!g_stop) {
        fprintf(stderr, "[daq_reader] waiting for visualizer to connect...\n");
        int cli = accept(srv, NULL, NULL);
        if (cli < 0) { if (errno == EINTR) break; perror("accept"); continue; }
        fprintf(stderr, "[daq_reader] visualizer connected.\n");

        /* header */
        struct { char magic[4]; int32_t nchan, rate, chunk; } hdr;
        memcpy(hdr.magic, "NIDQ", 4);
        hdr.nchan = nchan; hdr.rate = (int32_t)rate; hdr.chunk = chunk;
        if (write(cli, &hdr, sizeof(hdr)) != (ssize_t)sizeof(hdr)) { close(cli); continue; }

        if ((err = DAQmxBaseStartTask(g_task))) die_daq("StartTask", err);

        int client_ok = 1;
        while (!g_stop && client_ok) {
            int32 read = 0;
            err = DAQmxBaseReadAnalogF64(g_task, chunk, 10.0,
                      DAQmx_Val_GroupByChannel, buf, frame_samps, &read, NULL);
            if (err) { die_daq("ReadAnalogF64", err); }
            if (read <= 0) continue;

            size_t nbytes = (size_t)read * nchan * sizeof(double);
            char *p = (char*)buf; size_t left = nbytes;
            while (left) {
                ssize_t w = write(cli, p, left);
                if (w <= 0) { client_ok = 0; break; }   /* viz went away */
                p += w; left -= (size_t)w;
            }
        }
        DAQmxBaseStopTask(g_task);
        close(cli);
        fprintf(stderr, "[daq_reader] visualizer disconnected.\n");
    }

    free(buf);
    DAQmxBaseClearTask(g_task);
    unlink(sockpath);
    fprintf(stderr, "[daq_reader] stopped.\n");
    return 0;
}
