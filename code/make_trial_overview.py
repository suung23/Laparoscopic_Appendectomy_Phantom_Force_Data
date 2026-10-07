"""Overview figure for the A01 phantom session: what each trial did, when, what it
looked like, and what it measured.

    python make_trial_overview.py            # -> docs/fig_trial_overview.{pdf,png}

Frames are pulled from the full-session mp4 in data/video/ (not tracked) using the
per-frame timestamps in *_cam.h5, so they line up with the force samples exactly.
"""
from pathlib import Path
from datetime import datetime, timedelta, timezone

import cv2
import h5py
import numpy as np
import csv
from types import SimpleNamespace
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "docs" / "fig_trial_overview"
KST = timezone(timedelta(hours=9))

SESSIONS = ["161659", "163214", "164339", "165306", "170625", "171030", "171401"]
EXCLUDED_SESSION = "170625"          # slip at Phase 1 settings, not analysed
REMOVED = [                           # (session, stage, t0, t1) from h5 removed_note
    ("164339", "move_L", 1791359282.369, 1791359287.755),
    ("165306", "move_B", 1791359752.816, 1791359756.807),
    ("171401", "slip_dry", 1791360909.816, 1791360921.922),
]

# stage, row label, action, task
ROWS = [
    ("grasp",    "Grasp",       "Grasp distal third, close jaws",        "grasp"),
    ("lift_05",  "Lift 5 mm",   "Lift vertically 5 mm, hold 3 s",        "lift"),
    ("lift_10",  "Lift 10 mm",  "Lift vertically 10 mm, hold 3 s",       "lift"),
    ("lift_15",  "Lift 15 mm",  "Lift vertically 15 mm, hold 3 s",       "lift"),
    ("lift_20",  "Lift 20 mm",  "Lift vertically 20 mm, hold 3 s",       "lift"),
    ("move_L",   "Move L",      "From 10 mm lift, push left",            "move"),
    ("move_R",   "Move R",      "From 10 mm lift, push right",           "move"),
    ("move_F",   "Move A",      "From 10 mm lift, push anterior",        "move"),
    ("move_B",   "Move P",      "From 10 mm lift, push posterior",       "move"),
    ("slip_dry", "Slip",        "Pull anterior until jaws slip, 1 kHz",  "slip"),
]
TASK_COLOR = {"grasp": "#2a78d6", "lift": "#eb6834", "move": "#1baf7a", "slip": "#eda100"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#9a9893", "#e4e3df"

CROP = (0.28, 0.92, 0.06, 0.84)       # x0, x1, y0, y1 as fractions of 1920x1080
TRACE_XMAX = 18.0

mpl.rcParams.update({
    "font.family": "Arial", "font.size": 7, "axes.linewidth": 0.6,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.spines.top": False,
    "axes.spines.right": False, "pdf.fonttype": 42, "savefig.dpi": 300,
})


NUMERIC = {"trial", "start_unix", "end_unix", "duration_s", "peak_F_tip", "peak_unix",
           "mean_F_tip", "gripping_peak", "rate"}


class Trials(list):
    """List of trial records with just enough selection for this script."""
    def where(self, stage=None, used=True):
        return Trials(r for r in self if (stage is None or r.stage == stage)
                      and (used is None or r.used == used))

    def col(self, name):
        return np.array([getattr(r, name) for r in self], dtype=float)


def load():
    df, h5 = Trials(), {}
    for s in SESSIONS:
        with open(DATA / f"phantom_A01_20261007_{s}_trials.csv") as f:
            for rec in csv.DictReader(f):
                rec = {k: (float(v) if k in NUMERIC else v) for k, v in rec.items()}
                rec["trial"] = int(rec["trial"])
                df.append(SimpleNamespace(**rec, session=s, used=s != EXCLUDED_SESSION))
        with h5py.File(DATA / f"phantom_A01_20261007_{s}.h5") as f:
            fx = f["forces"][:]
            h5[s] = dict(t=f["timestamp"][:], trial=f["trial"][:],
                         ftip=np.hypot(fx[:, 0], fx[:, 2]), grip=fx[:, 4])
        with h5py.File(DATA / f"phantom_A01_20261007_{s}_cam.h5") as f:
            h5[s]["cam_t"] = f["timestamp"][:]
    return df, h5


def trace(h5, row):
    d = h5[row.session]
    m = np.abs(d["trial"]) == row.trial
    return d["t"][m] - row.start_unix, d["ftip"][m]


def representative(df):
    """Trial whose peak is closest to the stage median (same rule as video_clips/)."""
    rep = {}
    for st, *_ in ROWS:
        g = df.where(st)
        pk = g.col("peak_F_tip")
        rep[st] = g[int(np.argmin(np.abs(pk - np.median(pk))))]
    return rep


def grab_frames(h5, row, times):
    cap = cv2.VideoCapture(str(DATA / "video" / f"phantom_A01_20261007_{row.session}_cam.mp4"))
    ts = h5[row.session]["cam_t"]
    out = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(np.argmin(np.abs(ts - t))))
        ok, im = cap.read()
        h, w = im.shape[:2]
        x0, x1, y0, y1 = CROP
        im = im[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
        out.append(cv2.cvtColor(cv2.resize(im, None, fx=0.4, fy=0.4,
                                           interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
    cap.release()
    return out


def panel_letter(ax, s, x=-0.02, y=1.0):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="bottom", ha="right")


def draw_timeline(ax, df):
    order = [r[0] for r in ROWS]
    y_of = {st: i for i, st in enumerate(order)}
    t_ref = df.col("start_unix").min()
    to_min = lambda u: (u - t_ref) / 60
    for r in df:
        y = y_of[r.stage]
        task = dict((x[0], x[3]) for x in ROWS)[r.stage]
        kw = dict(color=TASK_COLOR[task]) if r.used else dict(facecolor="none", edgecolor=MUTED, lw=0.5)
        ax.barh(y, to_min(r.end_unix) - to_min(r.start_unix), left=to_min(r.start_unix),
                height=0.62, **kw)
    for _, st, t0, t1 in REMOVED:
        ax.barh(y_of[st], to_min(t1) - to_min(t0), left=to_min(t0), height=0.62,
                facecolor="none", edgecolor=MUTED, lw=0.5, hatch="////")
    # session boundaries
    for s in SESSIONS:
        x0 = to_min(min(r.start_unix for r in df if r.session == s))
        ax.axvline(x0 - 0.25, color=GRID, lw=0.6, zorder=0)
        ax.text(x0 - 0.1, -1.05, s, fontsize=5.5, color=INK2, va="bottom")
    ax.set_yticks(range(len(order)), [r[1] for r in ROWS], fontsize=6)
    ax.set_ylim(len(order) - 0.4, -1.3)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    span = to_min(df.col("end_unix").max())
    ticks = np.arange(0, span + 5, 10)
    t0 = datetime.fromtimestamp(t_ref, KST)
    ax.set_xticks(ticks, [(t0 + timedelta(minutes=float(m))).strftime("%H:%M") for m in ticks])
    ax.set_xlim(-1, span + 0.5)
    ax.set_xlabel("Time (KST, 2026-10-07)")
    # key for open / hatched
    kx = to_min(min(r.start_unix for r in df if r.session == "165306")) + 5
    ax.barh(1, 0.8, left=kx, height=0.62, facecolor="none", edgecolor=MUTED, lw=0.5)
    ax.text(kx + 1.1, 1, "excluded (100 Hz)", va="center", fontsize=5.5, color=INK2)
    ax.barh(0, 0.8, left=kx, height=0.62, facecolor="none", edgecolor=MUTED,
            lw=0.5, hatch="////")
    ax.text(kx + 1.1, 0, "removed", va="center", fontsize=5.5, color=INK2)


def draw_grid(fig, spec, df, h5, rep):
    n = len(ROWS)
    gs = GridSpecFromSubplotSpec(n + 1, 6, subplot_spec=spec, hspace=0.12, wspace=0.06,
                                 width_ratios=[0.95, 1, 1, 1, 0.12, 2.1 + 1.25],
                                 height_ratios=[0.22] + [1] * n)
    # right block split into trace + dots
    for j, head in enumerate(["Start", "Peak $F_{tip}$", "End"]):
        a = fig.add_subplot(gs[0, 1 + j]); a.axis("off")
        a.text(0.5, 0.1, head, ha="center", va="bottom", fontsize=6.5, color=INK2)
    axes_tr, axes_pk = [], []
    for i, (st, label, action, task) in enumerate(ROWS):
        r = rep[st]
        g = df.where(st)
        col = TASK_COLOR[task]
        # label
        a = fig.add_subplot(gs[i + 1, 0]); a.axis("off")
        a.add_patch(plt.Rectangle((0, 0.08), 0.035, 0.84, color=col, transform=a.transAxes))
        a.text(0.09, 0.66, label, fontsize=7, fontweight="bold", va="center")
        a.text(0.09, 0.40, f"n = {len(g)}", fontsize=6, color=INK2, va="center")
        a.text(0.09, 0.16, f"#{r.trial}  ({r.session})", fontsize=5.5, color=MUTED, va="center")
        # frames
        tx, ty = trace(h5, r)
        inside = (tx >= 0) & (tx <= r.end_unix - r.start_unix)
        t_pk = r.start_unix + tx[inside][np.argmax(ty[inside])]
        times = [r.start_unix + 0.3, t_pk, r.end_unix - 0.1]
        for j, im in enumerate(grab_frames(h5, r, times)):
            a = fig.add_subplot(gs[i + 1, 1 + j])
            a.imshow(im, aspect="auto"); a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values():
                sp.set_visible(True); sp.set_color(col if j == 1 else GRID); sp.set_linewidth(0.9 if j == 1 else 0.4)
            a.text(0.04, 0.93, f"{times[j] - r.start_unix:.1f} s", transform=a.transAxes,
                   fontsize=5, color=INK, va="top",
                   bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.75))
        # trace + peak strip
        sub = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[i + 1, 5],
                                      width_ratios=[2.1, 0.95], wspace=0.2)
        at = fig.add_subplot(sub[0]); ap = fig.add_subplot(sub[1])
        for rr in g:
            x, y = trace(h5, rr)
            if rr.trial == r.trial:
                continue
            at.plot(x, y, color=MUTED, lw=0.45, alpha=0.8)
        x, y = trace(h5, r)
        at.plot(x, y, color=col, lw=1.0)
        for t in times:
            at.axvline(t - r.start_unix, color=INK2, lw=0.4, ls=(0, (2, 2)))
        at.set_xlim(0, TRACE_XMAX); at.set_ylim(0, 7)
        at.set_yticks([0, 3, 6])
        pk = g.col("peak_F_tip")
        jit = np.linspace(-0.18, 0.18, len(pk))
        ap.scatter(pk, jit, s=9, color=col, edgecolor="white", linewidth=0.4, zorder=3)
        ap.errorbar(pk.mean(), 0, xerr=pk.std(ddof=1), fmt="|", color=INK, ms=7,
                    mew=1.0, elinewidth=0.7, capsize=0, zorder=4)
        ap.text(9.6, 0, f"{pk.mean():.2f}\n± {pk.std(ddof=1):.2f}", ha="right",
                linespacing=1.1,
                va="center", fontsize=5.5, color=INK)
        ap.set_xlim(0, 9.6); ap.set_ylim(-0.6, 0.6); ap.set_yticks([])
        ap.spines["bottom"].set_bounds(0, 7)
        ap.spines["left"].set_visible(False)
        ap.set_xticks([0, 2, 4, 6])
        for a in (at, ap):
            a.grid(axis="x", color=GRID, lw=0.4); a.set_axisbelow(True)
            if i < n - 1:
                a.tick_params(labelbottom=False)
        axes_tr.append(at); axes_pk.append(ap)
    axes_tr[-1].set_xlabel("Time from trial start (s)")
    mid = axes_tr[n // 2].get_position()
    fig.text(mid.x0 - 0.03, (axes_tr[0].get_position().y1 + axes_tr[-1].get_position().y0) / 2,
             "$F_{tip}$ (N)", rotation=90, ha="center", va="center")
    axes_pk[-1].set_xlabel("Peak $F_{tip}$ (N)")
    axes_pk[-1].set_xticks([0, 2, 4, 6])
    return axes_tr[0], axes_pk[0]


def draw_lift(ax, df):
    hs = [5, 10, 15, 20]
    m, s = [], []
    for h in hs:
        pk = df.where(f"lift_{h:02d}").col("peak_F_tip")
        ax.scatter(np.full(len(pk), h) + np.linspace(-0.6, 0.6, len(pk)), pk, s=7,
                   color=TASK_COLOR["lift"], alpha=0.45, linewidth=0)
        m.append(pk.mean()); s.append(pk.std(ddof=1))
    ax.errorbar(hs, m, yerr=s, color=INK, lw=1.0, marker="o", ms=3.5, capsize=0, elinewidth=0.7)
    ax.set_xticks(hs); ax.set_xlim(2, 23); ax.set_ylim(0, 4)
    ax.set_xlabel("Lift height (mm)"); ax.set_ylabel("Peak $F_{tip}$ (N)")


def draw_move(ax, df):
    order = [("move_R", "R"), ("move_L", "L"), ("move_B", "P"), ("move_F", "A")]
    ref = df.where("move_F").col("peak_F_tip").mean()
    for k, (st, lab) in enumerate(order):
        pk = df.where(st).col("peak_F_tip")
        ax.bar(k, pk.mean(), width=0.6, color=TASK_COLOR["move"], alpha=0.85)
        ax.errorbar(k, pk.mean(), yerr=pk.std(ddof=1), color=INK, elinewidth=0.7, capsize=0)
        ax.scatter(k + np.linspace(-0.15, 0.15, len(pk)), pk, s=5, color=INK, zorder=3, linewidth=0)
        ax.text(k, max(pk.max(), pk.mean() + pk.std(ddof=1)) + 0.3, f"{pk.mean() / ref:.2f}×",
                ha="center", fontsize=5.5, color=INK2)
    ax.set_xticks(range(4), [o[1] for o in order])
    ax.set_ylim(0, 7.5); ax.set_xlabel("Direction"); ax.set_ylabel("Peak $F_{tip}$ (N)")


def draw_slip(ax, df):
    groups = [("move_F", "Move A", "move"), ("slip_dry", "Slip A", "slip")]
    means = []
    for k, (st, lab, task) in enumerate(groups):
        pk = df.where(st).col("peak_F_tip")
        ax.scatter(np.full(len(pk), k) + np.linspace(-0.12, 0.12, len(pk)), pk, s=9,
                   color=TASK_COLOR[task], edgecolor="white", linewidth=0.4, zorder=3)
        ax.errorbar(k + 0.28, pk.mean(), yerr=pk.std(ddof=1), fmt="o", ms=3, color=INK,
                    elinewidth=0.7, capsize=0)
        means.append(pk.mean())
    ax.annotate("", xy=(1.28, means[1]), xytext=(0.28, means[0]),
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.6, shrinkA=4, shrinkB=4))
    ax.text(0.78, max(means) + 0.45, f"×{means[1] / means[0]:.2f}", ha="center",
            fontsize=6, color=INK)
    ax.set_xticks([0, 1], [g[1] for g in groups]); ax.set_xlim(-0.5, 1.6)
    ax.set_ylim(0, 3.5); ax.set_ylabel("Peak $F_{tip}$ (N)")


def stats(g):
    ms = lambda v, d=2: f"{v.mean():.{d}f} ± {v.std(ddof=1):.{d}f}"
    return [", ".join(sorted({r.session for r in g})), f"{int(g[0].rate)} Hz", str(len(g)),
            ms(g.col("duration_s"), 1), ms(g.col("peak_F_tip")), ms(g.col("mean_F_tip")),
            ms(g.col("gripping_peak"))]


def draw_table(ax, df):
    ax.axis("off")
    cols = ["Condition", "Action", "Session", "Rate", "n", "Duration (s)",
            "Peak $F_{tip}$ (N)", "Mean $F_{tip}$ (N)", "Peak gripping (N)"]
    xs = [0.0, 0.085, 0.33, 0.43, 0.488, 0.525, 0.635, 0.755, 0.872]
    rows = []
    for st, label, action, task in ROWS:
        rows.append((task, [label, action] + stats(df.where(st))))
    rows.append((None, ["Slip", "Same as Slip, 100 Hz / 5 Hz LPF"] + stats(df.where(used=False))))
    nr = len(rows) + 1
    dy = 1 / (nr + 0.4)
    y = 1 - dy * 0.6
    for x, c in zip(xs, cols):
        ax.text(x + 0.012, y, c, fontsize=6, fontweight="bold", va="center")
    ax.plot([0, 1], [1, 1], color=INK, lw=0.8, transform=ax.transAxes)
    ax.plot([0, 1], [y - dy / 2] * 2, color=INK, lw=0.5, transform=ax.transAxes)
    prev = None
    for task, vals in rows:
        y -= dy
        if task != prev and prev is not None:
            ax.plot([0, 1], [y + dy / 2] * 2, color=GRID, lw=0.5, transform=ax.transAxes)
        color = INK if task else MUTED
        if task:
            ax.add_patch(plt.Rectangle((0, y - dy * 0.32), 0.004, dy * 0.64,
                                       color=TASK_COLOR[task], transform=ax.transAxes))
        for x, v in zip(xs, vals):
            ax.text(x + 0.012, y, v, fontsize=6, va="center", color=color)
        prev = task
    ax.plot([0, 1], [y - dy / 2] * 2, color=INK, lw=0.8, transform=ax.transAxes)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)


def main():
    df, h5 = load()
    rep = representative(df)
    fig = plt.figure(figsize=(7.2, 11.4))
    gs = GridSpec(4, 1, figure=fig, height_ratios=[1.25, 6.4, 1.35, 1.75], hspace=0.28,
                  left=0.075, right=0.985, top=0.985, bottom=0.015)

    ax = fig.add_subplot(gs[0]); draw_timeline(ax, df); panel_letter(ax, "a", x=-0.065)

    a_tr, a_pk = draw_grid(fig, gs[1], df, h5, rep)
    fig.text(0.075 - 0.065 * 0.91, a_tr.get_position().y1 + 0.012, "b", fontsize=9,
             fontweight="bold", ha="right", va="bottom")

    sub = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[2], wspace=0.45, width_ratios=[1.2, 1.2, 0.9])
    for k, (fn, L) in enumerate([(draw_lift, "c"), (draw_move, "d"), (draw_slip, "e")]):
        a = fig.add_subplot(sub[k]); fn(a, df); a.grid(axis="y", color=GRID, lw=0.4)
        a.set_axisbelow(True); panel_letter(a, L, x=-0.2)

    ax = fig.add_subplot(gs[3]); draw_table(ax, df); panel_letter(ax, "f", x=-0.012, y=0.98)

    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT.with_suffix(".pdf"))
    fig.savefig(OUT.with_suffix(".png"))
    print(OUT.with_suffix(".png"))


if __name__ == "__main__":
    main()
