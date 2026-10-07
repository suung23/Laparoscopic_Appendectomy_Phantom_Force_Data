#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
새 wiring 6축 디커플링/스케일 캘리브레이션.

각 스냅샷은 '알고 있는 방향·크기'로 가한 힘(applied_dir, applied_force)과
그때의 raw 채널 6개(wiring.RAW_KEYS: bgx,btx,bgy,bty,grip,tor) 전압을 담는다.
raw 채널 → 깨끗한 6축 힘으로 보내는 affine 변환을 최소제곱으로 구한다:

        clean(6) = W^T · g(6) + b(6)

W(6x6) 의 비대각 성분이 커플링을 상쇄하고, 대각 성분이 스케일을 applied_force
단위(N, torsion 은 N·m)에 맞춘다. b 는 무부하 오프셋을 흡수한다.

half-bridge 가 하드웨어에서 이미 좌우/상하 빼기를 하므로, 예전 wiring 의
bending 라벨 자동교정 같은 보정은 필요 없다.

Run:
    conda activate surgitag
    python calibrate.py calib_snapshots_YYYYMMDD_HHMMSS.csv -o calib.json
    # 여러 파일을 이어서 쓰거나, 베이스 파일의 특정 축을 새 파일로 교체:
    python calibrate.py base.csv more_grip.csv --base-drop-dir gripping -o calib.json
"""
import argparse
import collections
import csv
import json

import numpy as np

from wiring import AXES, RAW_KEYS, ALLOWED_SIGNALS, TARGET_GROUP, SELF_ROWS_ONLY


def load_one(csv_path, drop_rows=(), drop_dirs=()):
    """한 CSV 를 읽어 (G[N,6], dirs[N], mag[N]) 반환.
    raw 채널(RAW_KEYS) 컬럼만 쓰므로, 캘리브레이션 적용 후 기록된 CSV(force 컬럼이
    이미 디커플링된 값)를 다시 넣어도 안전하다."""
    rows = list(csv.DictReader(open(csv_path)))
    drop = set(drop_rows)
    if drop:
        rows = [r for i, r in enumerate(rows) if i not in drop]
    if drop_dirs:
        rows = [r for r in rows if r["applied_dir"] not in set(drop_dirs)]
    G = np.array([[float(r[k]) for k in RAW_KEYS] for r in rows])
    dirs = [r["applied_dir"] for r in rows]
    mag = np.array([float(r["applied_force"]) for r in rows])
    return G, dirs, mag


def load(csv_paths, drop_rows=(), base_drop_dirs=()):
    """여러 CSV 를 읽어 이어 붙인다. drop_rows / base_drop_dirs 는 첫(베이스) 파일에만."""
    Gs, dirs, mags = [], [], []
    for k, path in enumerate(csv_paths):
        dr = drop_rows if k == 0 else ()
        dd = base_drop_dirs if k == 0 else ()
        g, d, m = load_one(path, drop_rows=dr, drop_dirs=dd)
        if len(g):
            Gs.append(g); dirs += d; mags.append(m)
        note = []
        if dr: note.append(f"dropped rows {list(drop_rows)}")
        if dd: note.append(f"dropped dirs {list(base_drop_dirs)}")
        print(f"  + {path}: {len(d)} rows" + (f" ({'; '.join(note)})" if note else ""))
    return np.vstack(Gs), dirs, np.concatenate(mags)


def fit(G, dirs, mag, outlier_z=2.5):
    """구조화된 affine least squares (이상치 제거 포함).

    각 힘(출력 컬럼)을 ALLOWED_SIGNALS[force] 의 raw 신호만으로 따로 푼다.
    SELF_ROWS_ONLY 축(gripping)은 자기 부하 행으로만 적합(스케일 보정).
    각 컬럼마다 |표준화잔차| > outlier_z 인 행을 반복 제거하고 재적합한다.
    반환: W(6x6), b(6), Y, P(예측), dropped(축별 제거 행수)."""
    N = len(dirs)
    # 타깃: 가한 방향과 같은 TARGET_GROUP 인 모든 힘 컬럼에 크기를 넣는다.
    Y = np.zeros((N, len(AXES)))
    for i, d in enumerate(dirs):
        if d not in AXES:
            raise SystemExit(f"unknown applied_dir '{d}' (expected one of {AXES})")
        gd = TARGET_GROUP[d]
        for j, ax in enumerate(AXES):
            if TARGET_GROUP[ax] == gd:
                Y[i, j] = mag[i]

    W = np.zeros((len(RAW_KEYS), len(AXES)))
    b = np.zeros(len(AXES))
    dropped = {}
    for j, ax in enumerate(AXES):
        cols = [RAW_KEYS.index(s) for s in ALLOWED_SIGNALS[ax]]
        if ax in SELF_ROWS_ONLY:        # 자기 부하 행으로만 (오염된 타축 행 제외)
            base = [i for i in range(N) if TARGET_GROUP[dirs[i]] == TARGET_GROUP[ax]]
        else:
            base = list(range(N))
        idx = list(base)
        Xa = np.hstack([G[idx][:, cols], np.ones((len(idx), 1))])
        coef, *_ = np.linalg.lstsq(Xa, Y[idx, j], rcond=None)
        # 초기 적합의 잔차 σ 기준으로 한 번만 이상치 제거(σ 재계산 캐스케이드 방지)
        resid = Y[idx, j] - Xa @ coef
        s = resid.std()
        if outlier_z < 1e8 and s > 1e-9:
            keep = [idx[k] for k in range(len(idx)) if abs(resid[k]) <= outlier_z * s]
            if len(keep) < len(idx) and len(keep) >= len(cols) + 2:
                idx = keep
                Xa = np.hstack([G[idx][:, cols], np.ones((len(idx), 1))])
                coef, *_ = np.linalg.lstsq(Xa, Y[idx, j], rcond=None)
        W[cols, j] = coef[:-1]
        b[j] = coef[-1]
        dropped[ax] = len(base) - len(idx)
    P = G @ W + b
    return W, b, Y, P, dropped


def report(dirs, Y, P, dropped):
    # R²/RMSE 는 '그 힘이 실제로 가해진 행'(own target group)에서 평가 = 스케일 충실도.
    print(f"\n{'axis':20s} {'R^2':>7s} {'RMSE':>9s} {'drop':>5s}   target range")
    for j, a in enumerate(AXES):
        ev = [i for i in range(len(dirs)) if TARGET_GROUP[dirs[i]] == TARGET_GROUP[a]]
        yj, pj = Y[ev, j], P[ev, j]
        res = ((yj - pj) ** 2).sum()
        tot = ((yj - yj.mean()) ** 2).sum()
        r2 = 1 - res / tot if tot > 1e-12 else float("nan")
        rmse = np.sqrt(((yj - pj) ** 2).mean())
        print(f"{a:20s} {r2:7.3f} {rmse:9.4f} {dropped.get(a,0):5d}   "
              f"{yj.min():+.3f}..{yj.max():+.3f}")

    print("\nresidual cross-coupling (mean |off-axis out| as % of on-axis out):")
    print("  ('=' = same target group, 의도된 동일값 / 커플링 아님)")
    print(f"{'applied':20s} " + " ".join(f"{a[:6]:>7s}" for a in AXES))
    for src in AXES:
        idx = [i for i, d in enumerate(dirs) if d == src]
        if not idx:
            continue
        on = abs(P[idx, AXES.index(src)]).mean()
        cells = []
        for j, a in enumerate(AXES):
            if a == src:
                cells.append("  --  ")
            elif TARGET_GROUP[a] == TARGET_GROUP[src]:
                cells.append("   =  ")     # 같은 그룹 → 같은 값이 정상
            else:
                cells.append(f"{100*abs(P[idx, j]).mean()/on:6.1f}")
        print(f"{src:20s} " + " ".join(f"{c:>7s}" for c in cells))


def main():
    p = argparse.ArgumentParser(description="6-axis strain-gauge decoupling/scale calibration")
    p.add_argument("csv", nargs="+", help="snapshot CSV(들). 여러 개면 이어 붙임")
    p.add_argument("-o", "--out", default="calib.json", help="output calibration JSON")
    p.add_argument("--drop-rows", type=int, nargs="*", default=[],
                   help="첫 CSV 에서 제외할 행 인덱스(0부터, 헤더 제외)")
    p.add_argument("--base-drop-dir", nargs="*", default=[],
                   help="첫(베이스) CSV 에서 통째로 제외할 applied_dir "
                        "(뒤 CSV 가 그 축을 새로 대체할 때)")
    p.add_argument("--outlier-z", type=float, default=2.5,
                   help="컬럼별 |표준화잔차|>이 값인 행을 이상치로 제거(반복). 0=끔")
    args = p.parse_args()

    print(f"loading {len(args.csv)} CSV file(s):")
    G, dirs, mag = load(args.csv, drop_rows=args.drop_rows,
                        base_drop_dirs=args.base_drop_dir)
    print(f"total {len(G)} snapshots")
    print("  per-axis counts:", dict(collections.Counter(dirs)))
    print("\ndecoupling structure (force <- allowed raw signals):")
    for ax in AXES:
        print(f"  {ax:20s} <- {ALLOWED_SIGNALS[ax]}")

    oz = args.outlier_z if args.outlier_z > 0 else 1e9
    W, b, Y, P, dropped = fit(G, dirs, mag, outlier_z=oz)
    report(dirs, Y, P, dropped)

    calib = {
        "axes": AXES,
        "signals": RAW_KEYS,    # W 의 입력(행) 순서 = raw 채널 순서
        "W": W.tolist(),        # (6 signals) x (6 axes)
        "b": b.tolist(),        # (6 axes)
        "source_csv": args.csv if len(args.csv) > 1 else args.csv[0],
        "n_snapshots": len(G),
        "note": "clean(6) = W.T @ signals(6) + b ; signals order = 'signals' field",
    }
    with open(args.out, "w") as f:
        json.dump(calib, f, indent=2)
    print(f"\nwrote calibration -> {args.out}")


if __name__ == "__main__":
    main()
