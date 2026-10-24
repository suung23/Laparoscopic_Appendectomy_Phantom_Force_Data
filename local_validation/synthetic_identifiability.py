"""Can the Surgilogger + endoscopic-camera pipeline tell Yeoh from Ogden?

A twin experiment. The truth is CHOSEN, not measured, so this needs no new data:
generate traction data from a known law on a known specimen, corrupt it the way
video reconstruction would, fit both candidate laws back, and see which one wins.
Where the winner stops being the truth is the accuracy specification the
reconstruction has to meet -- known before the reconstruction is built.

Forward model -- a slender tube in traction, which is what appendix retraction
is. Reference cross-section A(X) varies along the bar, so geometry enters
non-trivially while no mesh or solver does:

    1D equilibrium under an end load F:   P11(lambda(X)) * A(X) = F
    each slice solves its own stretch:    lambda(X) = P11^-1( F / A(X) )
    the observable is the gross stretch:  Lambda = (1/L0) * integral lambda dX

P11 is the free-lateral uniaxial nominal stress of the candidate law, taken from
`constitutive_model_screening`. It is monotone for every candidate here (checked
in `constitutive_mode_generality`), so the inverse exists.

Two identification arms, because they answer different design questions:

  free    shape parameter and stiffness both fitted to traction data alone. The
          two families OVERLAP at neo-Hookean (Yeoh c2 = 0 is Ogden alpha = 2),
          so a truth near that point is intrinsically hard to call.
  gated   shape pinned by the compression-stiffening endpoint (Yeoh c2 = 0.565,
          Ogden alpha = 4.247 from the screening), stiffness fitted. The grasp
          phase compresses tissue, so a capture that logs grasp as well as
          traction can impose this.

Comparing the two says whether the next capture must record BOTH phases.

Three error axes are swept SEPARATELY, so each specification is attributable:

  per-frame noise     random error on reconstructed stretch, clean geometry
  area shape error    non-uniform bias on the reconstructed cross-section
  strain scale bias   the reconstruction misreads elongation by a factor

Every systematic error is redrawn INSIDE the Monte Carlo. A bias realised once
per cell would decide that cell deterministically and make cells incomparable --
the reported accuracy is over the population of reconstruction errors of a given
magnitude, which is the quantity a specification needs.

Units are normalised: G0 = 1, mean A = 1, L0 = 1. Absolute scale cannot matter,
and the script proves it rather than assuming it: an area scale error is exactly
degenerate with G0 (both enter only as F / (G0 * A)), so it is absorbed by the
fit. Only NON-UNIFORM area error and stretch-axis error survive. That asymmetry
is the study's main structural claim, and it is why absolute newtons -- the
headline limitation of the A01 capture -- do not threaten this particular
decision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from constitutive_model_screening import MODELS  # noqa: E402
from constitutive_mode_generality import _free_surface  # noqa: E402

NU = 0.45
RNG_SEED = 20261019

GATED = {"yeoh": 0.565, "ogden": 4.247}      # pinned by the compression endpoint
TRUTH = ("yeoh", 0.565)

# The truth value MUST lie on the grid, or the true family cannot fit itself and
# the comparison is decided by grid resolution rather than by the data.
SHAPE_GRID = {
    "yeoh": np.unique(np.concatenate([np.linspace(0.0, 6.0, 33), [0.565]])),
    "ogden": np.unique(np.concatenate([np.linspace(2.05, 20.0, 33), [4.247]])),
}
LAMBDA_GRID = np.linspace(1.0, 2.3, 140)
G0_GRID = np.geomspace(0.2, 5.0, 31)
N_SAMPLES = 40           # frames along the pull (~1.3 s ramp at 30 Hz)
N_BIAS = 24              # systematic-error realisations per cell
N_NOISE = 12             # noise draws per realisation  -> 288 decisions / cell


def build_piola(model: str, shape: float):
    return MODELS[model]["fn"]((shape, 0.0) if model == "yeoh" else (shape,), NU)


def p11_curve(model: str, shape: float) -> np.ndarray:
    """Free-lateral uniaxial nominal stress P11(lambda) at G0 = 1."""
    piola = build_piola(model, shape)
    out = np.empty_like(LAMBDA_GRID)
    for i, lam in enumerate(LAMBDA_GRID):
        if np.isclose(lam, 1.0):
            out[i] = 0.0
            continue
        s = _free_surface(piola, lambda s: np.diag([lam, s, s]), 1)
        out[i] = float(piola(np.diag([lam, s, s]))[0, 0])
    if np.any(np.diff(out) <= 0):
        raise ValueError(f"non-monotone P11 for {model} shape={shape}")
    return out


def stretch_of_stress(curve: np.ndarray, stress: np.ndarray) -> np.ndarray:
    """Invert P11 by interpolation; out-of-range stress is clamped, which can
    only make a candidate look worse, never better."""
    return np.interp(np.clip(stress, curve[0], curve[-1]), curve, LAMBDA_GRID)


def gross_stretch(curve, areas, forces) -> np.ndarray:
    return stretch_of_stress(curve, forces[:, None] / areas[None, :]).mean(axis=1)


def gross_stretch_over_g0(curve, areas, forces, g0) -> np.ndarray:
    """(n_g0, n_forces) in one vectorised pass; stiffness enters as F/(G0*A)."""
    stress = forces[None, :, None] / (g0[:, None, None] * areas[None, None, :])
    return stretch_of_stress(curve, stress).mean(axis=2)


def force_for_target(curve, areas, target: float) -> float:
    lo, hi = 0.0, 1.0
    while gross_stretch(curve, areas, np.array([hi]))[0] < target and hi < 1e4:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if gross_stretch(curve, areas, np.array([mid]))[0] < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def true_areas(n: int = 24) -> np.ndarray:
    x = (np.arange(n) + 0.5) / n
    a = 1.0 + 0.15 * np.sin(2 * np.pi * x) + 0.08 * np.sin(4 * np.pi * x)
    return a / a.mean()


def reconstructed_areas(truth, shape_err, rng) -> np.ndarray:
    """Non-uniform reconstruction bias on the cross-section. The uniform part is
    deliberately absent: it is degenerate with G0 (see `degeneracy_demo`)."""
    if shape_err <= 0.0:
        return truth.copy()
    x = (np.arange(truth.size) + 0.5) / truth.size
    bump = sum(rng.normal() * np.sin(np.pi * k * x) for k in (1, 2, 3))
    return truth * (1.0 + shape_err * bump / (np.abs(bump).max() + 1e-12))


def run_cell(curves, lam_max, strain_bias, noise, area_shape_err, arm,
             rng) -> dict:
    a_true = true_areas()
    truth_curve = curves[TRUTH]
    f_max = force_for_target(truth_curve, a_true, lam_max)
    forces = np.linspace(0.0, f_max, N_SAMPLES)
    lam_clean = gross_stretch(truth_curve, a_true, forces)

    wins = {"yeoh": 0, "ogden": 0}
    gaps = []
    for _ in range(N_BIAS):
        a_rec = reconstructed_areas(a_true, area_shape_err, rng)
        preds = {}
        for model in ("yeoh", "ogden"):
            shapes = ([GATED[model]] if arm == "gated"
                      else list(SHAPE_GRID[model]))
            preds[model] = np.concatenate([
                gross_stretch_over_g0(curves[(model, float(sh))], a_rec,
                                      forces, G0_GRID) for sh in shapes])
        for _ in range(N_NOISE):
            e = rng.uniform(-strain_bias, strain_bias) if strain_bias > 0 else 0.0
            lam_meas = (1.0 + (lam_clean - 1.0) * (1.0 + e)
                        + rng.normal(0.0, noise, N_SAMPLES))
            best = {m: float(((b - lam_meas) ** 2).sum(axis=1).min())
                    for m, b in preds.items()}
            wins[min(best, key=best.get)] += 1
            gaps.append((max(best.values()) - min(best.values()))
                        / (min(best.values()) + 1e-18))
    total = float(N_BIAS * N_NOISE)
    return {
        "lambda_max": float(lam_max), "arm": arm,
        "strain_scale_bias": float(strain_bias),
        "per_frame_noise": float(noise),
        "area_shape_error": float(area_shape_err),
        "accuracy": wins[TRUTH[0]] / total,
        "decisions": int(total),
        "median_relative_sse_gap": float(np.median(gaps)),
        "peak_force_normalised": float(f_max),
    }


def degeneracy_demo(curves) -> dict:
    a = true_areas()
    c = curves[TRUTH]
    f = np.linspace(0.0, force_for_target(c, a, 1.5), 25)
    diff = np.abs(gross_stretch(c, a, f) - gross_stretch(c, a * 1.30, f * 1.30))
    return {
        "claim": "area scale and G0 enter only as F/(G0*A), so a uniform area "
                 "error is exactly compensated by the fitted stiffness and "
                 "cannot affect which law wins",
        "max_abs_difference_in_gross_stretch": float(diff.max()),
        "exact_to_machine_precision": bool(diff.max() < 1e-12),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(RNG_SEED)

    needed = {TRUTH}
    for model in ("yeoh", "ogden"):
        needed.add((model, float(GATED[model])))
        needed.update((model, float(sh)) for sh in SHAPE_GRID[model])
    curves = {k: p11_curve(*k) for k in sorted(needed)}

    lam_sweep = [1.2, 1.3, 1.4, 1.5, 1.6, 1.8]
    axes = {
        "per_frame_noise": ([0.002, 0.005, 0.010, 0.020, 0.040],
                            lambda v: dict(strain_bias=0.0, noise=v,
                                           area_shape_err=0.0)),
        "area_shape_error": ([0.0, 0.05, 0.10, 0.20, 0.35],
                             lambda v: dict(strain_bias=0.0, noise=0.005,
                                            area_shape_err=v)),
        "strain_scale_bias": ([0.0, 0.02, 0.05, 0.10, 0.20, 0.30],
                              lambda v: dict(strain_bias=v, noise=0.005,
                                             area_shape_err=0.0)),
    }

    sweeps = {}
    for axis, (values, build) in axes.items():
        rows = []
        for arm in ("gated", "free"):
            for lam in lam_sweep:
                for v in values:
                    rows.append(run_cell(curves, lam, arm=arm, rng=rng,
                                         **build(v)))
        sweeps[axis] = rows

    report = {
        "status": "synthetic_identifiability_twin_experiment",
        "question": ("With Surgilogger force and camera-derived kinematics only, "
                     "at what reconstruction accuracy does the Yeoh-vs-Ogden "
                     "decision stop being recoverable?"),
        "truth": {"model": TRUTH[0], "shape_parameter": TRUTH[1],
                  "note": "chosen, not measured; identifies nothing about real tissue"},
        "forward_model": {
            "specimen": "slender tube in traction, non-uniform reference area",
            "equilibrium": "P11(lambda(X)) * A(X) = F on every slice",
            "observable": "gross stretch Lambda = mean of lambda(X)",
            "samples_per_pull": N_SAMPLES,
            "decisions_per_cell": N_BIAS * N_NOISE,
            "systematic_error_realisations_per_cell": N_BIAS,
            "normalisation": "G0 = 1, mean A = 1, L0 = 1",
        },
        "arms": {
            "gated": f"shape pinned (yeoh c2={GATED['yeoh']}, "
                     f"ogden alpha={GATED['ogden']}), stiffness fitted; needs the "
                     "grasp phase logged as well as traction",
            "free": "shape and stiffness both fitted to traction alone",
        },
        "chance_level": 0.5,
        "area_scale_degeneracy": degeneracy_demo(curves),
        "sweeps": sweeps,
        "limitations": [
            "The truth is chosen. Nothing here says which law real tissue follows.",
            "Reduced 1D specimen: no contact, mesh, solver, bending or torsion, so element quality and locking are untouched.",
            "Both candidates and the synthetic truth are hyperelastic; a viscoelastic truth is NOT exercised, which makes these accuracies optimistic.",
            "Injected errors are smooth and zero-mean in shape; real endoscopic reconstruction error is structured and may be worse.",
            "Force noise is folded into the stretch residual rather than modelled on the force axis.",
            "Quasi-static ramp only. The slip transient is excluded -- note 30 Hz video cannot resolve an event that needed 1 kHz force sampling.",
            "Accuracy near 0.5 means the two laws are indistinguishable, not that the method failed.",
        ],
        "source": {
            "extends": ["local_validation/constitutive_model_screening.py",
                        "local_validation/constitutive_mode_generality.py"],
            "seed": RNG_SEED,
            "analysis_script_sha256": hashlib.sha256(
                Path(__file__).read_bytes()).hexdigest(),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("area-scale degeneracy exact:",
          report["area_scale_degeneracy"]["exact_to_machine_precision"])
    for axis, (values, _) in axes.items():
        for arm in ("gated", "free"):
            print(f"\n--- {axis} | arm: {arm}   (chance = 0.50)")
            print("lam_max |" + "".join(f"{v:>8}" for v in values))
            for lam in lam_sweep:
                row = sorted([r for r in sweeps[axis]
                              if r["arm"] == arm and r["lambda_max"] == lam],
                             key=lambda r: r[axis])
                print(f"  {lam:<5} |" + "".join(f"{r['accuracy']:>8.2f}"
                                                for r in row))


if __name__ == "__main__":
    main()
