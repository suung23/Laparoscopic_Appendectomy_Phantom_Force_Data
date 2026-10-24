"""Analytical model-class screening for appendix compression stiffening.

Extends `physx_corotated_compression_audit.py` from ONE constitutive law to four.
The question is not "which parameters fit" -- no displacement-paired measurement
exists to fit against (see research/SCOPE_SIMULATOR_EVALUATION_ALIGNMENT.md) --
but the strictly weaker one:

    can this model CLASS reach the reported compression-stiffening factor at all,
    for ANY admissible parameter choice?

A class that cannot is eliminated on paper, before any solver work. A class that
can is only a candidate; this says nothing about which parameters are correct.

Method, identical in structure to the corotated audit it extends: a single
material point, no mesh, no solver, no time integration, no contact.

  1. prescribe axial stretch 1-c, solve the free lateral stretch from P22 = 0
  2. superpose a small simple shear in the transverse plane
  3. read the current-configuration Cauchy shear tangent by central difference
  4. report tangent(c) / tangent(0)

Step 4 differs from the audit it extends, which divides by the input mu. Here
tangent(0) is MEASURED the same way as tangent(c), so the ratio is well defined
for models whose small-strain shear modulus is not an input parameter. The
corotated reproduction below checks the two agree.

The stiffening ratio is invariant to the overall stiffness scale, so every model
is normalised to unit small-strain shear modulus and only dimensionless shape
parameters are swept. That is what makes the sweep cheap: the search space is
1-2 dimensions, not the full parameter count.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

# Reported endpoints, research/appendix-mechanics-reference.json (rheometry,
# 8 mm disc, 1 Hz, 1 % shear amplitude, storage modulus at 0-40 % compression).
GROUPS = {
    "control": {"g0_pa": 1136.0, "increase_40pct_percent": 200.0},
    "purulent": {"g0_pa": 666.0, "increase_40pct_percent": 350.0},
    "gangrenous": {"g0_pa": 642.0, "increase_40pct_percent": 400.0,
                   "approximate": True},
}
COMPRESSIONS = (0.0, 0.1, 0.2, 0.3, 0.4)
TENSION_CHECKS = (1.2, 1.4)       # axial stretches used only as a stability gate


# --------------------------------------------------------------------------
# kinematic helpers
# --------------------------------------------------------------------------
def invariants(f: np.ndarray) -> tuple:
    c = f.T @ f
    i1 = float(np.trace(c))
    i2 = 0.5 * (i1 * i1 - float(np.trace(c @ c)))
    j = float(np.linalg.det(f))
    return c, i1, i2, j


def cofactor(f: np.ndarray, j: float) -> np.ndarray:
    return j * np.linalg.inv(f).T


def bulk_from_shear(g0: float, nu: float) -> float:
    return 2.0 * g0 * (1.0 + nu) / (3.0 * (1.0 - 2.0 * nu))


# --------------------------------------------------------------------------
# constitutive laws -- first Piola-Kirchhoff stress, unit small-strain shear
# --------------------------------------------------------------------------
def piola_fixed_corotated(f: np.ndarray, nu: float) -> np.ndarray:
    """PhysX eCO_ROTATIONAL. W = mu*||F-R||^2 + lambda/2*(J-1)^2.

    Verbatim the law physx_corotated_compression_audit.py audits, so this
    screening reproduces that report's number as a harness check.
    """
    mu = 1.0
    lame = 2.0 * nu * mu / (1.0 - 2.0 * nu)
    u, _, vh = np.linalg.svd(f)
    rot = u @ vh
    if np.linalg.det(rot) <= 0:
        raise ValueError("improper rotation from polar decomposition")
    j = float(np.linalg.det(f))
    return 2.0 * mu * (f - rot) + lame * (j - 1.0) * cofactor(f, j)


def _piola_from_invariant_derivatives(f, w1, w2, wj):
    """P = 2*W1*F + 2*W2*(I1*F - F*C) + WJ*cof(F), the chain rule for any
    W(I1, I2, J) on the unreduced invariants."""
    c, i1, _, j = invariants(f)
    return (2.0 * w1 * f
            + 2.0 * w2 * (i1 * f - f @ c)
            + wj * cofactor(f, j))


def piola_mooney_rivlin(f: np.ndarray, beta: float, nu: float) -> np.ndarray:
    """W = C10*(I1bar-3) + C01*(I2bar-3) + (1/D1)*(J-1)^2.

    beta = C01 / (C10 + C01) splits the two terms; G0 = 2*(C10+C01) = 1 fixes
    the scale, so beta is the entire shape freedom. beta = 0 is neo-Hookean.
    """
    c10 = 0.5 * (1.0 - beta)
    c01 = 0.5 * beta
    d1 = 2.0 / bulk_from_shear(1.0, nu)
    _, i1, i2, j = invariants(f)
    w1 = c10 * j ** (-2.0 / 3.0)
    w2 = c01 * j ** (-4.0 / 3.0)
    wj = (-2.0 / 3.0 * c10 * j ** (-5.0 / 3.0) * i1
          - 4.0 / 3.0 * c01 * j ** (-7.0 / 3.0) * i2
          + (2.0 / d1) * (j - 1.0))
    return _piola_from_invariant_derivatives(f, w1, w2, wj)


def piola_yeoh(f: np.ndarray, c2: float, c3: float, nu: float) -> np.ndarray:
    """W = sum_k Ck0*(I1bar-3)^k + (1/D1)*(J-1)^2, k = 1..3.

    c2 = C20/C10 and c3 = C30/C10 are the shape freedoms; C10 = G0/2 = 0.5.
    Designed for strain stiffening through I1 alone.
    """
    c10 = 0.5
    c20, c30 = c2 * c10, c3 * c10
    d1 = 2.0 / bulk_from_shear(1.0, nu)
    _, i1, _, j = invariants(f)
    i1bar = j ** (-2.0 / 3.0) * i1
    x = i1bar - 3.0
    dwdx = c10 + 2.0 * c20 * x + 3.0 * c30 * x * x
    w1 = dwdx * j ** (-2.0 / 3.0)
    wj = (dwdx * (-2.0 / 3.0) * j ** (-5.0 / 3.0) * i1
          + (2.0 / d1) * (j - 1.0))
    return _piola_from_invariant_derivatives(f, w1, 0.0, wj)


def piola_ogden(f: np.ndarray, alpha: float, nu: float) -> np.ndarray:
    """One-term Ogden in principal stretches, W = (mu/alpha)*sum(lbar_i^alpha-3).

    G0 = mu*alpha/2 = 1 fixes mu, leaving alpha as the single shape freedom.
    Evaluated spectrally because Ogden is not a function of I1/I2.
    """
    mu = 2.0 / alpha
    d1 = 2.0 / bulk_from_shear(1.0, nu)
    c = f.T @ f
    w, vecs = np.linalg.eigh(c)
    if (w <= 0).any():
        raise ValueError("non-positive-definite right Cauchy-Green tensor")
    lam = np.sqrt(w)
    j = float(np.linalg.det(f))
    if j <= 0:
        raise ValueError("non-positive Jacobian")
    jm = j ** (-alpha / 3.0)
    pow_sum = float(np.sum(lam ** alpha))
    # dW/dlam_i, isochoric Ogden term plus the volumetric penalty
    dw = (mu / lam) * jm * (lam ** alpha - pow_sum / 3.0) \
        + (2.0 / d1) * (j - 1.0) * j / lam
    s = (vecs * (dw / lam)) @ vecs.T          # S = sum_i (dW/dlam_i)/lam_i N_i N_i^T
    return f @ s


MODELS = {
    "fixed_corotated": {
        "energy": "mu*||F-R||_F^2 + lambda/2*(J-1)^2  (PhysX eCO_ROTATIONAL)",
        "shape_params": [],
        "fn": lambda p, nu: (lambda f: piola_fixed_corotated(f, nu)),
    },
    "mooney_rivlin": {
        "energy": "C10*(I1bar-3) + C01*(I2bar-3) + (1/D1)*(J-1)^2",
        "shape_params": ["beta = C01/(C10+C01)"],
        "fn": lambda p, nu: (lambda f: piola_mooney_rivlin(f, p[0], nu)),
    },
    "yeoh": {
        "energy": "C10*x + C20*x^2 + C30*x^3 + (1/D1)*(J-1)^2, x = I1bar-3",
        "shape_params": ["c2 = C20/C10", "c3 = C30/C10"],
        "fn": lambda p, nu: (lambda f: piola_yeoh(f, p[0], p[1], nu)),
    },
    "ogden": {
        "energy": "(mu/alpha)*sum_i(lambdabar_i^alpha - 3) + (1/D1)*(J-1)^2",
        "shape_params": ["alpha"],
        "fn": lambda p, nu: (lambda f: piola_ogden(f, p[0], nu)),
    },
}


# --------------------------------------------------------------------------
# the virtual rheometer
# --------------------------------------------------------------------------
def free_lateral_stretch(piola, axial: float) -> float:
    """Lateral stretch satisfying P22 = P33 = 0 at the prescribed axial stretch.

    The disc is compressed between plates but its rim is free, so the lateral
    direction carries no load. This is the one equation the screening solves.
    """
    if np.isclose(axial, 1.0):
        return 1.0

    def residual(s: float) -> float:
        return float(piola(np.diag([axial, s, s]))[1, 1])

    # March outward from the undeformed lateral stretch, trying the EXPANDING
    # branch first and the contracting branch second. The expected branch is a
    # bulge under compression, but strongly stiffening isochoric models can
    # instead put the root on the contracting side: they buy their way out of
    # distortion by collapsing volume, which the soft volumetric penalty at
    # moderate nu barely resists. That is a real material pathology, not a
    # bracketing failure, so the root is found either way and judged afterwards
    # by `admissible`.
    for far in (4.0, 0.2):
        prev_s, prev_v = None, None
        for s in np.linspace(1.0, far, 400):
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
    raise ValueError("no free-lateral root bracketed")


def shear_tangent(piola, axial: float, gamma: float = 1e-5) -> float:
    return shear_tangent_state(piola, axial, gamma)[0]


def shear_tangent_state(piola, axial: float, gamma: float = 1e-5) -> tuple:
    """Current-configuration Cauchy shear tangent, plus the lateral stretch the
    free-surface condition produced."""
    lateral = free_lateral_stretch(piola, axial)
    f0 = np.diag([axial, lateral, lateral])
    shear = np.zeros((3, 3))
    shear[1, 2] = 1.0
    eye = np.eye(3)

    def sigma23(sign: float) -> float:
        f = (eye + sign * gamma * shear) @ f0
        j = float(np.linalg.det(f))
        return float((piola(f) @ f.T / j)[1, 2])

    return (sigma23(1.0) - sigma23(-1.0)) / (2.0 * gamma), lateral


def stiffening_curve(piola) -> list:
    base, _ = shear_tangent_state(piola, 1.0)
    if not np.isfinite(base) or base <= 0:
        raise ValueError("non-positive uncompressed shear tangent")
    out = []
    for c in COMPRESSIONS:
        axial = 1.0 - c
        t, lateral = shear_tangent_state(piola, axial)
        out.append({"compression_fraction": float(c),
                    "axial_stretch": float(axial),
                    "free_lateral_stretch": float(lateral),
                    "volume_ratio": float(axial * lateral * lateral),
                    "shear_tangent_over_uncompressed": float(t / base),
                    "shear_tangent_positive": bool(t > 0),
                    # under axial compression a positive-Poisson solid bulges
                    "lateral_bulges": bool(lateral >= 1.0 - 1e-12)})
    return out


def admissible(piola, curve: list) -> tuple:
    """Reject parameter points that reach the target by going pathological.

    A model that stiffens in compression only by losing positive-definiteness
    somewhere else is not a usable candidate, so the screening checks the shear
    tangent stays positive across the compression range and in moderate tension.
    """
    if not all(p["shear_tangent_positive"] for p in curve):
        return False, "non_positive_shear_tangent_under_compression"
    if not all(p["lateral_bulges"] for p in curve):
        # the free-surface root sits on the contracting branch: the solid
        # shrinks in every direction under uniaxial compression
        return False, "auxetic_lateral_response_under_compression"
    for lam in TENSION_CHECKS:
        try:
            t, lateral = shear_tangent_state(piola, lam)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError,
                OverflowError, ZeroDivisionError):
            return False, f"no_free_surface_state_at_axial_stretch_{lam}"
        if t <= 0:
            return False, f"non_positive_shear_tangent_at_axial_stretch_{lam}"
        if lateral > 1.0 + 1e-12:
            return False, f"auxetic_lateral_response_at_axial_stretch_{lam}"
    return True, ""


def sweep_grid(name: str) -> list:
    """Dimensionless shape-parameter grids. Ratios are scale-invariant, so the
    overall stiffness is NOT a sweep axis -- that is the dimensional reduction
    that makes this cheap."""
    if name == "fixed_corotated":
        return [()]
    if name == "mooney_rivlin":
        return [(float(b),) for b in np.linspace(0.0, 0.98, 50)]
    if name == "yeoh":
        c2 = np.concatenate([np.linspace(-2.0, 0.0, 9)[:-1], np.linspace(0.0, 30.0, 31)])
        c3 = np.concatenate([[0.0], np.geomspace(0.01, 300.0, 30)])
        return [(float(a), float(b)) for a in c2 for b in c3]
    if name == "ogden":
        a = np.concatenate([np.linspace(-30.0, -0.5, 60), np.linspace(0.5, 30.0, 60)])
        return [(float(v),) for v in a]
    raise KeyError(name)


# Canonical one-dimensional path per model along which the target crossing is
# reported. The sweep above answers "is it reachable"; this answers the more
# useful "how extreme must the parameter be", which is what decides whether a
# reachable class is also a sensible one.
CROSSING_PATHS = {
    "mooney_rivlin": {"axis": "beta = C01/(C10+C01)", "lo": 0.0, "hi": 0.98,
                      "build": lambda v: (v,)},
    "yeoh": {"axis": "c2 = C20/C10 (c3 = 0)", "lo": 0.001, "hi": 10.0,
             "build": lambda v: (v, 0.0)},
    "ogden": {"axis": "alpha", "lo": 2.05, "hi": 20.0,
              "build": lambda v: (v,)},
}


def target_crossings(nu: float) -> dict:
    targets = {k: 1.0 + v["increase_40pct_percent"] / 100.0
               for k, v in GROUPS.items()}
    out = {}
    for name, path in CROSSING_PATHS.items():
        spec = MODELS[name]

        def factor(v: float) -> float:
            piola = spec["fn"](path["build"](v), nu)
            return stiffening_curve(piola)[-1]["shear_tangent_over_uncompressed"]

        row = {"axis": path["axis"], "searched": [path["lo"], path["hi"]],
               "crossings": {}}
        for key, tgt in targets.items():
            try:
                f_lo, f_hi = factor(path["lo"]), factor(path["hi"])
            except (ValueError, np.linalg.LinAlgError, FloatingPointError,
                    OverflowError, ZeroDivisionError):
                row["crossings"][key] = {"reached": False,
                                         "reason": "path endpoint not solvable"}
                continue
            if (f_lo - tgt) * (f_hi - tgt) > 0:
                row["crossings"][key] = {
                    "reached": False,
                    "factor_at_path_endpoints": [float(f_lo), float(f_hi)],
                    "reason": "target not bracketed anywhere on this path"}
                continue
            v = float(brentq(lambda x: factor(x) - tgt, path["lo"], path["hi"],
                             xtol=1e-9))
            piola = spec["fn"](path["build"](v), nu)
            curve = stiffening_curve(piola)
            ok, why = admissible(piola, curve)
            row["crossings"][key] = {
                "reached": True, "parameter_value": v,
                "factor_there": float(curve[-1]["shear_tangent_over_uncompressed"]),
                "admissible_there": bool(ok),
                "inadmissibility": why or None,
                "compression_curve": curve if key == "control" else None,
            }
        out[name] = row
    return out


def screen(nu: float) -> dict:
    targets = {k: 1.0 + v["increase_40pct_percent"] / 100.0 for k, v in GROUPS.items()}
    out = {}
    for name, spec in MODELS.items():
        best = None
        admitted = 0
        reasons: dict = {}
        for params in sweep_grid(name):
            piola = spec["fn"](params, nu)
            try:
                curve = stiffening_curve(piola)
            except (ValueError, np.linalg.LinAlgError, FloatingPointError,
                    OverflowError, ZeroDivisionError) as exc:
                key = f"no_curve:{exc}"[:80]
                reasons[key] = reasons.get(key, 0) + 1
                continue
            ok, why = admissible(piola, curve)
            if not ok:
                reasons[why] = reasons.get(why, 0) + 1
                continue
            admitted += 1
            factor = curve[-1]["shear_tangent_over_uncompressed"]
            if best is None or factor > best["factor_at_40pct"]:
                best = {"factor_at_40pct": float(factor),
                        "shape_parameters": dict(zip(spec["shape_params"],
                                                     [float(v) for v in params])),
                        "compression_curve": curve}
        row = {
            "energy": spec["energy"],
            "grid_points": len(sweep_grid(name)),
            "admissible_points": admitted,
            "rejections_by_reason": dict(sorted(reasons.items(),
                                                key=lambda kv: -kv[1])),
        }
        if best is None:
            row.update(max_factor_at_40pct=None,
                       reaches={k: False for k in targets},
                       note="no admissible parameter point")
        else:
            row.update(
                max_factor_at_40pct=best["factor_at_40pct"],
                best_shape_parameters=best["shape_parameters"],
                best_compression_curve=best["compression_curve"],
                reaches={k: bool(best["factor_at_40pct"] >= t)
                         for k, t in targets.items()},
            )
        out[name] = row
    return {"poisson_ratio": nu, "target_factors": targets, "models": out,
            "target_crossings": target_crossings(nu)}


def harness_check() -> dict:
    """Reproduce the corotated audit's published endpoint before trusting the
    new laws. physx-appendix-compression-stiffening-analytical.json reports
    1.9928432751493188 at 40 % compression, nu = 0.45."""
    piola = MODELS["fixed_corotated"]["fn"]((), 0.45)
    curve = stiffening_curve(piola)
    got = curve[-1]["shear_tangent_over_uncompressed"]
    published = 1.9928432751493188
    base = shear_tangent(piola, 1.0)
    return {
        "published_corotated_factor_at_40pct": published,
        "recomputed_here": float(got),
        "absolute_difference": float(abs(got - published)),
        "reproduces_published_result": bool(abs(got - published) < 1e-6),
        "measured_uncompressed_tangent_over_input_mu": float(base),
        "note": ("The audit divides by the input mu; this script measures "
                 "tangent(0) instead. The ratio above confirms they agree, "
                 "which is what makes the measured normalisation valid for "
                 "models whose shear modulus is not an input."),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--poisson", type=float, nargs="+", default=[0.45, 0.49])
    args = p.parse_args()

    check = harness_check()
    sweeps = [screen(float(nu)) for nu in args.poisson]

    report = {
        "status": "analytical_constitutive_model_class_screening",
        "question": ("Can each model class reach the reported compression-"
                     "stiffening factor for ANY admissible parameter choice? "
                     "This is reachability, not identification."),
        "method": {
            "domain": "single material point; no mesh, solver, time integration or contact",
            "compression": "prescribed axial stretch 1-c, free lateral stretch from P22 = 0",
            "probe": "small simple shear in the transverse plane, central difference of Cauchy sigma23",
            "reported_quantity": "shear tangent at c divided by shear tangent at c = 0",
            "scale_invariance": ("the stiffening ratio does not depend on overall "
                                 "stiffness, so every model is normalised to unit "
                                 "small-strain shear modulus and only dimensionless "
                                 "shape parameters are swept"),
            "admissibility_gate": ("shear tangent must stay positive across 0-40 % "
                                   "compression and at axial stretches "
                                   f"{list(TENSION_CHECKS)}"),
            "shear_probe_amplitude": 1e-5,
        },
        "harness_check": check,
        "sweeps": sweeps,
        "limitations": [
            "Reachability only. A reachable class is a candidate; no parameter is identified, promoted or validated.",
            "Single material point: says nothing about mesh quality, volumetric locking, element inversion or solver convergence.",
            "Purely hyperelastic: no rate dependence, stress relaxation or hysteresis, while the reference quantity is a storage modulus from an oscillatory viscoelastic measurement.",
            "One loading mode. Compression stiffening only; tension, bending and grasping are separate gates.",
            "Reference endpoints are small-cohort reported percentages; raw rheometer curves were not digitised.",
            "The volumetric penalty (1/D1)(J-1)^2 is one of several admissible forms; another choice shifts the near-incompressible results.",
        ],
        "source": {
            "appendix_article": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11591559/",
            "extends": "local_validation/physx_corotated_compression_audit.py",
            "analysis_script_sha256": hashlib.sha256(
                Path(__file__).read_bytes()).hexdigest(),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"harness_check": check,
                      "sweeps": [{"poisson_ratio": s["poisson_ratio"],
                                  "models": {k: {"max_factor_at_40pct": v["max_factor_at_40pct"],
                                                 "reaches": v["reaches"]}
                                             for k, v in s["models"].items()}}
                                 for s in sweeps]}, indent=2))


if __name__ == "__main__":
    main()
