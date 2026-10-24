"""Loading-mode generality and well-posedness of the screened constitutive laws.

`constitutive_model_screening.py` asked one question in one loading mode: can a
model class reach the reported compression-stiffening factor? Yeoh and Ogden
passed. Passing one gate does not make a law usable -- a law tuned to stiffen in
compression can lose material stability, or lose ellipticity, somewhere else in
the deformation range the task actually visits.

This script checks the mathematics directly, from the same analytic stress
functions, still with no mesh and no solver:

  Drucker stability
      the material tangent dS/dE (6x6 Voigt, symmetric) is positive definite.
      Violation means the material can release energy under an imposed strain
      increment -- the quasi-static problem stops being a minimisation.

  Strong ellipticity (Legendre-Hadamard / rank-one convexity)
      the acoustic tensor Q(n)_ik = A_iJkL n_J n_L is positive definite for
      every unit n, where A_iJkL = dP_iJ/dF_kL. This is the condition for the
      incremental problem to stay elliptic: lose it and the governing equations
      change type, wave speeds go imaginary, and the discrete problem admits
      mesh-dependent localisation. For a solver this is not a cosmetic property
      -- it is the difference between a converging Newton step and a spurious
      shear band.

  Monotone uniaxial response
      dP11/dlambda > 0, i.e. no snap-back along the loading path.

Loading modes, each volume-resolved by its own free-surface condition:

  uniaxial      F = diag(l, s, s),  P22 = 0     traction / retraction, compression
  equibiaxial   F = diag(l, l, s),  P33 = 0     tissue flattened under a jaw
  planar        F = diag(l, 1, s),  P33 = 0     strip pulled with one side held
  simple_shear  F = I + g*e1(x)e2             sliding, volume preserving

Everything is evaluated per material point. Nothing here says whether a given
parameter is CORRECT -- no displacement-paired measurement exists to fit
against. It says only where each law remains mathematically admissible.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from constitutive_model_screening import MODELS  # noqa: E402

NU = 0.45
VOIGT = ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2))


# --------------------------------------------------------------------------
# tangent operators by central difference of the analytic Piola stress
# --------------------------------------------------------------------------
def tangent_A(piola, f: np.ndarray, h: float = 1e-6) -> np.ndarray:
    """A[i,J,k,L] = dP_iJ / dF_kL."""
    a = np.empty((3, 3, 3, 3))
    for k in range(3):
        for l in range(3):
            step = np.zeros((3, 3))
            step[k, l] = h
            a[:, :, k, l] = (piola(f + step) - piola(f - step)) / (2.0 * h)
    return a


def material_tangent_voigt(piola, f: np.ndarray, h: float = 1e-6) -> np.ndarray:
    """6x6 dS/dE in Voigt form, S = F^-1 P the second Piola-Kirchhoff stress
    and E = (F^T F - I)/2 the Green-Lagrange strain.

    Built by perturbing E directly: for each Voigt component, form C = I + 2E,
    take F = sqrt(C) (symmetric, rotation-free -- legitimate because the response
    is isotropic and S depends on F only through C), and difference S.
    """
    c0 = f.T @ f
    e0 = 0.5 * (c0 - np.eye(3))

    def s_of_e(e: np.ndarray) -> np.ndarray:
        c = np.eye(3) + 2.0 * e
        w, v = np.linalg.eigh(c)
        if (w <= 0).any():
            raise ValueError("non-positive-definite C in tangent probe")
        fs = (v * np.sqrt(w)) @ v.T
        return np.linalg.solve(fs, piola(fs))

    out = np.empty((6, 6))
    for col, (k, l) in enumerate(VOIGT):
        step = np.zeros((3, 3))
        # engineering shear: perturb both symmetric slots by h/2
        step[k, l] += h / 2.0
        step[l, k] += h / 2.0
        ds = (s_of_e(e0 + step) - s_of_e(e0 - step)) / h
        for row, (i, j) in enumerate(VOIGT):
            out[row, col] = ds[i, j]
    return 0.5 * (out + out.T)


def fibonacci_sphere(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.column_stack([np.sin(phi) * np.cos(theta),
                            np.sin(phi) * np.sin(theta), np.cos(phi)])


def min_acoustic_eigenvalue(piola, f: np.ndarray, directions: np.ndarray) -> float:
    """min over unit n of the smallest eigenvalue of Q(n). > 0 is strong
    ellipticity. Normalised by the undeformed shear modulus (= 1 by the
    screening's parametrisation) so the number is dimensionless."""
    a = tangent_A(piola, f)
    q = np.einsum("iJkL,nJ,nL->nik", a, directions, directions)
    return float(np.linalg.eigvalsh(0.5 * (q + np.transpose(q, (0, 2, 1)))).min())


# --------------------------------------------------------------------------
# loading modes
# --------------------------------------------------------------------------
def _free_surface(piola, build, idx: int) -> float:
    """Solve the free stretch from P[idx, idx] = 0, searching both branches."""
    def residual(s: float) -> float:
        return float(piola(build(s))[idx, idx])

    for far in (4.0, 0.2):
        prev_s, prev_v = None, None
        for s in np.linspace(1.0, far, 300):
            try:
                v = residual(float(s))
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                prev_s, prev_v = None, None
                continue
            if not np.isfinite(v):
                prev_s, prev_v = None, None
                continue
            if prev_v is not None and prev_v * v <= 0.0:
                return float(brentq(residual, min(prev_s, s), max(prev_s, s),
                                    xtol=1e-13))
            prev_s, prev_v = float(s), v
    raise ValueError("no free-surface root bracketed")


def deformation(piola, mode: str, level: float) -> np.ndarray:
    if mode == "uniaxial":
        s = _free_surface(piola, lambda s: np.diag([level, s, s]), 1)
        return np.diag([level, s, s])
    if mode == "equibiaxial":
        s = _free_surface(piola, lambda s: np.diag([level, level, s]), 2)
        return np.diag([level, level, s])
    if mode == "planar":
        s = _free_surface(piola, lambda s: np.diag([level, 1.0, s]), 2)
        return np.diag([level, 1.0, s])
    if mode == "simple_shear":
        f = np.eye(3)
        f[0, 1] = level
        return f
    raise KeyError(mode)


MODE_LEVELS = {
    # stretch ranges bracket what laparoscopic handling plausibly imposes:
    # 50 % compression through 80 % extension
    "uniaxial": np.round(np.arange(0.50, 1.81, 0.05), 4),
    "equibiaxial": np.round(np.arange(0.60, 1.61, 0.05), 4),
    "planar": np.round(np.arange(0.60, 1.61, 0.05), 4),
    "simple_shear": np.round(np.arange(0.0, 1.51, 0.05), 4),
}


CANDIDATES = {
    # the value the screening says reaches the control 3.0x endpoint
    "yeoh_c2_0.565": ("yeoh", (0.565, 0.0)),
    "yeoh_c2_2": ("yeoh", (2.0, 0.0)),
    "yeoh_c2_5": ("yeoh", (5.0, 0.0)),
    "yeoh_c2_10": ("yeoh", (10.0, 0.0)),
    "ogden_alpha_4.247": ("ogden", (4.247,)),
    "neo_hookean": ("mooney_rivlin", (0.0,)),
    "fixed_corotated": ("fixed_corotated", ()),
}


def examine(piola, mode: str, directions: np.ndarray) -> dict:
    levels, rows = MODE_LEVELS[mode], []
    for lv in levels:
        row = {"level": float(lv)}
        try:
            f = deformation(piola, mode, float(lv))
        except (ValueError, np.linalg.LinAlgError, FloatingPointError,
                OverflowError, ZeroDivisionError) as exc:
            row.update(solvable=False, reason=str(exc)[:60])
            rows.append(row)
            continue
        row["solvable"] = True
        row["jacobian"] = float(np.linalg.det(f))
        try:
            dr = material_tangent_voigt(piola, f)
            row["min_drucker_eigenvalue"] = float(np.linalg.eigvalsh(dr).min())
            row["min_acoustic_eigenvalue"] = min_acoustic_eigenvalue(
                piola, f, directions)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            row.update(min_drucker_eigenvalue=None,
                       min_acoustic_eigenvalue=None, reason=str(exc)[:60])
        rows.append(row)

    def first_violation(key: str):
        ref = 1.0 if mode == "simple_shear" else 1.0
        bad = [r for r in rows
               if not r["solvable"] or r.get(key) is None or r[key] <= 0.0]
        if not bad:
            return None
        # report the violation closest to the undeformed state
        return min(bad, key=lambda r: abs(r["level"] - (0.0 if mode == "simple_shear" else ref)))["level"]

    return {
        "levels_examined": [float(x) for x in levels],
        "all_solvable": all(r["solvable"] for r in rows),
        "drucker_stable_everywhere": all(
            r["solvable"] and (r.get("min_drucker_eigenvalue") or -1.0) > 0.0
            for r in rows),
        "strongly_elliptic_everywhere": all(
            r["solvable"] and (r.get("min_acoustic_eigenvalue") or -1.0) > 0.0
            for r in rows),
        "first_drucker_violation_level": first_violation("min_drucker_eigenvalue"),
        "first_ellipticity_violation_level": first_violation("min_acoustic_eigenvalue"),
        "rows": rows,
    }


def uniaxial_monotonicity(piola) -> dict:
    """dP11/dlambda along the free-lateral uniaxial path."""
    lam = np.round(np.arange(0.50, 1.81, 0.025), 4)
    pts = []
    for lv in lam:
        try:
            f = deformation(piola, "uniaxial", float(lv))
            pts.append((float(lv), float(piola(f)[0, 0])))
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            continue
    if len(pts) < 3:
        return {"evaluated": len(pts), "monotone": None}
    l = np.array([p[0] for p in pts])
    p11 = np.array([p[1] for p in pts])
    d = np.diff(p11) / np.diff(l)
    bad = np.where(d <= 0.0)[0]
    return {
        "evaluated": len(pts),
        "monotone": bool(len(bad) == 0),
        "min_dP11_dlambda": float(d.min()),
        "first_non_monotone_stretch": None if not len(bad) else float(l[bad[0]]),
    }


def tension_compression_coupling(piola, c2: float) -> dict:
    """Yeoh depends on the state only through I1bar, which rises under BOTH
    compression and extension. So the compression and tension stiffening are
    not independent: whatever reproduces 40 % compression also fixes the
    tension response at the stretch of equal distortion. This reports the
    tension stretch matched in I1bar to 40 % compression, and the shear tangent
    at both, to show how tightly the two are tied."""
    from constitutive_model_screening import shear_tangent_state

    def i1bar(f):
        j = float(np.linalg.det(f))
        return j ** (-2.0 / 3.0) * float(np.trace(f.T @ f))

    f_comp = deformation(piola, "uniaxial", 0.6)
    target = i1bar(f_comp)
    lam_t = brentq(lambda l: i1bar(deformation(piola, "uniaxial", l)) - target,
                   1.01, 2.5, xtol=1e-10)
    base, _ = shear_tangent_state(piola, 1.0)
    t_comp, _ = shear_tangent_state(piola, 0.6)
    t_tens, _ = shear_tangent_state(piola, lam_t)
    return {
        "c2": c2,
        "i1bar_at_40pct_compression": float(target),
        "tension_stretch_of_equal_i1bar": float(lam_t),
        "shear_tangent_ratio_compression": float(t_comp / base),
        "shear_tangent_ratio_tension": float(t_tens / base),
        "ratio_of_ratios": float((t_tens / base) / (t_comp / base)),
    }


AXIAL_SCAN = (0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8)


def transverse_shear_scan() -> dict:
    """Transverse shear tangent over the full uniaxial path, compression AND
    tension, normalised to the undeformed value.

    The calibration gate lives at a single point of this curve (40 % compression).
    Two laws can agree there and disagree everywhere else, so this is what
    decides whether the gate actually constrains the law in the loading mode the
    task uses -- appendix traction is tension, not compression.
    """
    from constitutive_model_screening import shear_tangent_state
    out = {}
    for label, (name, params) in CANDIDATES.items():
        piola = MODELS[name]["fn"](params, NU)
        base, _ = shear_tangent_state(piola, 1.0)
        row = {}
        for lam in AXIAL_SCAN:
            try:
                t, _ = shear_tangent_state(piola, lam)
                row[str(lam)] = float(t / base)
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                row[str(lam)] = None
        out[label] = row
    return out


def yeoh_stability_boundary() -> dict:
    """Largest c2 that keeps every admissibility criterion over the envelope."""
    rows = []
    last_ok = None
    for c2 in [0.5, 1, 2, 3, 4, 5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 10]:
        piola = MODELS["yeoh"]["fn"]((float(c2), 0.0), NU)
        dirs = fibonacci_sphere(200)
        modes = {m: examine(piola, m, dirs) for m in MODE_LEVELS}
        mono = uniaxial_monotonicity(piola)
        ok = all(v["all_solvable"] and v["drucker_stable_everywhere"]
                 and v["strongly_elliptic_everywhere"]
                 for v in modes.values()) and bool(mono["monotone"])
        rows.append({
            "c2": float(c2), "admissible": bool(ok),
            "monotone_uniaxial": bool(mono["monotone"]),
            "first_drucker_violation": {
                m: v["first_drucker_violation_level"] for m, v in modes.items()
                if v["first_drucker_violation_level"] is not None},
        })
        if ok:
            last_ok = float(c2)
    return {"scan": rows, "largest_fully_admissible_c2": last_ok}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sphere", type=int, default=400)
    args = ap.parse_args()

    directions = fibonacci_sphere(args.sphere)
    results = {}
    for label, (name, params) in CANDIDATES.items():
        piola = MODELS[name]["fn"](params, NU)
        entry = {"model": name,
                 "shape_parameters": dict(zip(MODELS[name]["shape_params"],
                                              [float(v) for v in params])),
                 "modes": {m: examine(piola, m, directions) for m in MODE_LEVELS},
                 "uniaxial_monotonicity": uniaxial_monotonicity(piola)}
        entry["admissible_in_every_mode"] = all(
            v["all_solvable"] and v["drucker_stable_everywhere"]
            and v["strongly_elliptic_everywhere"]
            for v in entry["modes"].values())
        results[label] = entry

    coupling = {}
    for label, (name, params) in CANDIDATES.items():
        if name != "yeoh":
            continue
        piola = MODELS[name]["fn"](params, NU)
        try:
            coupling[label] = tension_compression_coupling(piola, params[0])
        except (ValueError, np.linalg.LinAlgError):
            coupling[label] = {"error": "not solvable over the matched range"}

    report = {
        "status": "analytical_loading_mode_generality_and_well_posedness",
        "question": ("Does a law tuned to the compression-stiffening endpoint "
                     "stay mathematically admissible in the other loading modes "
                     "a grasping task visits?"),
        "poisson_ratio": NU,
        "criteria": {
            "drucker": "min eigenvalue of the 6x6 dS/dE material tangent > 0",
            "strong_ellipticity": ("min over unit n of the smallest eigenvalue of "
                                   "the acoustic tensor Q(n)_ik = A_iJkL n_J n_L > 0; "
                                   f"{args.sphere} Fibonacci directions"),
            "monotonicity": "dP11/dlambda > 0 along the free-lateral uniaxial path",
            "normalisation": ("every law carries unit small-strain shear modulus, "
                              "so the eigenvalues are dimensionless multiples of G0"),
        },
        "candidates": results,
        "transverse_shear_vs_axial_stretch": transverse_shear_scan(),
        "yeoh_stability_boundary": yeoh_stability_boundary(),
        "yeoh_tension_compression_coupling": coupling,
        "limitations": [
            "Material-point analysis. Admissibility here is necessary, not sufficient, for a stable discretisation: element quality, volumetric locking and time-integration stability are untouched.",
            "Purely hyperelastic. No rate dependence, relaxation or hysteresis, while the calibration endpoint came from an oscillatory viscoelastic measurement.",
            "Strong ellipticity is sampled over a finite set of directions, so a narrow violation cone could be missed between samples.",
            "Tangents come from central differences of the analytic stress, not closed-form fourth-order derivatives; eigenvalues near zero carry the differencing error.",
            "No parameter here is identified or promoted. The deformation ranges are plausible handling envelopes, not measured task trajectories.",
        ],
        "source": {
            "extends": "local_validation/constitutive_model_screening.py",
            "analysis_script_sha256": hashlib.sha256(
                Path(__file__).read_bytes()).hexdigest(),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    for label, entry in results.items():
        print(f"{label:<20} admissible_everywhere={entry['admissible_in_every_mode']}")
        for m, v in entry["modes"].items():
            print(f"    {m:<13} solvable={v['all_solvable']!s:<5} "
                  f"drucker={v['drucker_stable_everywhere']!s:<5} "
                  f"SE={v['strongly_elliptic_everywhere']!s:<5} "
                  f"first_SE_violation={v['first_ellipticity_violation_level']}")
        print(f"    monotone uniaxial: {entry['uniaxial_monotonicity']['monotone']}")


if __name__ == "__main__":
    main()
