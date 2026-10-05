"""Audit Appendix-relevant compression/shear response of Newton's VBD material.

This is an analytical constitutive probe, not a tissue calibration. It mirrors
Newton 1.4's stable Neo-Hookean first Piola stress, solves lateral free-stress
expansion under axial compression, then finite-differences current Cauchy
shear stress against a small-amplitude shear in the plate plane.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq


DATA = {
    "control": {"storage_modulus_pa": 1136.0, "increase_at_40pct_percent": 200.0},
    "purulent": {"storage_modulus_pa": 666.0, "increase_at_40pct_percent": 350.0},
    "gangrenous": {"storage_modulus_pa": 642.0, "increase_at_40pct_percent": 400.0,
                   "increase_reported_as_approximately": True},
}


def lame_from_shear_storage_modulus(shear_storage_pa: float, poisson: float) -> tuple[float, float]:
    if (not np.isfinite(shear_storage_pa) or shear_storage_pa <= 0 or
            not np.isfinite(poisson) or not 0 <= poisson < .5):
        raise ValueError("Positive finite shear storage modulus and Poisson ratio in [0, .5) required")
    mu = float(shear_storage_pa)
    lame = 2. * poisson * mu / (1. - 2. * poisson)
    return mu, lame


def newton_stable_neo_hookean_piola(f: np.ndarray, mu: float, lame: float) -> np.ndarray:
    f = np.asarray(f, dtype=np.float64)
    if f.shape != (3, 3) or not np.isfinite(f).all() or np.linalg.det(f) <= 0:
        raise ValueError("A finite orientation-preserving 3x3 deformation gradient is required")
    mu_nh = mu
    lame_nh = lame + mu
    alpha = 1. + mu_nh / lame_nh
    j = float(np.linalg.det(f))
    cof = j * np.linalg.inv(f).T
    return mu_nh * f + lame_nh * (j - alpha) * cof


def cauchy_stress(f: np.ndarray, mu: float, lame: float) -> np.ndarray:
    j = float(np.linalg.det(f))
    return newton_stable_neo_hookean_piola(f, mu, lame) @ f.T / j


def free_lateral_stretch(axial_stretch: float, mu: float, lame: float) -> tuple[float, float]:
    if not np.isfinite(axial_stretch) or not 0 < axial_stretch <= 1.:
        raise ValueError("Axial stretch must be finite and in (0, 1]")
    if np.isclose(axial_stretch, 1.):
        residual = float(newton_stable_neo_hookean_piola(np.eye(3), mu, lame)[1, 1])
        return 1., residual

    def lateral_piola(stretch: float) -> float:
        return float(newton_stable_neo_hookean_piola(
            np.diag([axial_stretch, stretch, stretch]), mu, lame)[1, 1])

    lateral = brentq(lateral_piola, 1., 1.8, xtol=1e-13)
    return float(lateral), lateral_piola(lateral)


def shear_tangent_at_compression(axial_compression: float, mu: float, lame: float,
                                 shear_gamma: float = 1e-5) -> dict:
    if not np.isfinite(axial_compression) or not 0 <= axial_compression < 1.:
        raise ValueError("Axial compression must be finite and in [0, 1)")
    if not np.isfinite(shear_gamma) or shear_gamma <= 0:
        raise ValueError("Finite positive shear perturbation required")
    axial = 1. - axial_compression
    lateral, lateral_residual = free_lateral_stretch(axial, mu, lame)
    f0 = np.diag([axial, lateral, lateral])
    shear = np.zeros((3, 3), dtype=np.float64)
    shear[1, 2] = 1.
    plus = (np.eye(3) + shear_gamma * shear) @ f0
    minus = (np.eye(3) - shear_gamma * shear) @ f0
    modulus = float((cauchy_stress(plus, mu, lame)[1, 2] -
                     cauchy_stress(minus, mu, lame)[1, 2]) / (2. * shear_gamma))
    return {"axial_compression_fraction": float(axial_compression),
            "axial_stretch": float(axial), "free_lateral_stretch": lateral,
            "volume_ratio": float(np.linalg.det(f0)),
            "free_lateral_piola_residual_pa": float(lateral_residual),
            "current_cauchy_shear_tangent_pa": modulus,
            "tangent_ratio_to_uncompressed": float(modulus / mu)}


def audit() -> dict:
    poisson = .45
    groups = {}
    for name, datum in DATA.items():
        mu, lame = lame_from_shear_storage_modulus(datum["storage_modulus_pa"], poisson)
        rows = [shear_tangent_at_compression(c, mu, lame) for c in (0., .1, .2, .3, .4)]
        increase = datum["increase_at_40pct_percent"]
        measured_factor = 1. + increase / 100.
        predicted = rows[-1]["tangent_ratio_to_uncompressed"]
        groups[name] = {
            "baseline_storage_modulus_pa": datum["storage_modulus_pa"],
            "reported_storage_modulus_increase_at_40pct_percent": increase,
            "reported_40pct_modulus_factor": measured_factor,
            "newton_stable_nh_predicted_40pct_factor": predicted,
            "newton_stable_nh_predicted_40pct_modulus_pa": rows[-1]["current_cauchy_shear_tangent_pa"],
            "reported_implied_40pct_modulus_pa": mu * measured_factor,
            "relative_underprediction_of_40pct_factor": 1. - predicted / measured_factor,
            "increase_reported_as_approximately": bool(datum.get("increase_reported_as_approximately", False)),
            "compression_curve": rows,
        }
    return {
        "status": "compression_stiffening_gate_failed_for_current_newton_stable_neo_hookean_law",
        "model": "Newton 1.4 stable Neo-Hookean first Piola law evaluated analytically",
        "deformation_assumptions": {
            "axial_loading": "Prescribed axial stretch 1-c; two equal free lateral stretches are solved from zero lateral first Piola stress.",
            "oscillatory_shear": "Small-amplitude shear in the plane transverse to axial compression; current Cauchy shear stress is differentiated with respect to current simple shear.",
            "poisson_ratio": poisson,
            "baseline_parameter_mapping": "mu is set equal to the measured uncompressed G-prime mean; lambda is derived from isotropic nu. This is a scale comparison, not a fitted dynamic parameter."
        },
        "groups": groups,
        "conclusion": "The current stable Neo-Hookean law predicts only the geometric prestress factor 1/(1-c), reaching 1.667x at 40% compression. Published Appendix data report about 3x, 4.5x, and about 5x for control, purulent, and gangrenous groups. The current isotropic law therefore lacks enough compression-dependent shear response to represent those reported group means under this explicitly stated kinematic comparison.",
        "limitations": [
            "The article reports small-sample rheometer group means and normalized compression curves, not in-vivo surgical force-displacement ground truth.",
            "The numeric 40% increase is read from the article text; raw per-sample compression curves are not digitized here.",
            "The analytical audit mirrors Newton's law and does not directly audit PhysX FEM material response.",
            "No candidate material was fitted or promoted; viscoelastic loss, rate dependence, shear weakening, anisotropy, fluid flow, and tissue damage remain unmodelled."
        ]
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--newton-source", type=Path, required=True,
                   help="Installed Newton rigid_vbd_kernels.py used to bind the audited law")
    args = p.parse_args()
    report = audit()
    report["source"] = {
        "doi": "10.3390/biomedicines12112588",
        "dataset_file": "research/appendix-mechanics-reference.json",
        "newton_source_sha256": hashlib.sha256(args.newton_source.read_bytes()).hexdigest(),
        "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "figure_and_methods_reference": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11591559/"
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(1 if report["status"].startswith("compression_stiffening_gate_failed") else 0)


if __name__ == "__main__":
    main()
