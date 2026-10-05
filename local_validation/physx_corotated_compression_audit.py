"""Analytical compression/shear probe for the PhysX fixed-corotated FEM law.

The equations match the public PhysX 5.4.1 Soft Bodies manual. This script is
an independent continuum calculation; it does not start Isaac Sim or verify
the integrated runtime implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq


GROUPS = {
    "control": {"g0_pa": 1136., "increase_40pct_percent": 200.},
    "purulent": {"g0_pa": 666., "increase_40pct_percent": 350.},
    "gangrenous": {"g0_pa": 642., "increase_40pct_percent": 400., "approximate": True},
}


def lame_from_g_nu(g0_pa: float, nu: float) -> tuple[float, float]:
    if not np.isfinite(g0_pa) or g0_pa <= 0 or not np.isfinite(nu) or not 0 <= nu < .5:
        raise ValueError("Positive finite baseline modulus and Poisson ratio in [0, .5) required")
    mu = float(g0_pa)
    lame = 2. * nu * mu / (1. - 2. * nu)
    return mu, lame


def fixed_corotated_piola(f: np.ndarray, mu: float, lame: float) -> np.ndarray:
    f = np.asarray(f, dtype=np.float64)
    if f.shape != (3, 3) or not np.isfinite(f).all() or np.linalg.det(f) <= 0:
        raise ValueError("Finite orientation-preserving 3x3 deformation gradient required")
    u, _, vh = np.linalg.svd(f)
    rotation = u @ vh
    if np.linalg.det(rotation) <= 0:
        raise ValueError("Polar decomposition returned an improper rotation")
    j = float(np.linalg.det(f))
    cofactor = j * np.linalg.inv(f).T
    return 2. * mu * (f - rotation) + lame * (j - 1.) * cofactor


def cauchy_stress(f: np.ndarray, mu: float, lame: float) -> np.ndarray:
    j = float(np.linalg.det(f))
    return fixed_corotated_piola(f, mu, lame) @ f.T / j


def free_lateral_stretch(axial_stretch: float, mu: float, lame: float) -> tuple[float, float]:
    if not np.isfinite(axial_stretch) or not 0 < axial_stretch <= 1:
        raise ValueError("Axial stretch must be finite and in (0, 1]")

    def lateral_stress(stretch: float) -> float:
        return float(fixed_corotated_piola(np.diag([axial_stretch, stretch, stretch]), mu, lame)[1, 1])

    if np.isclose(axial_stretch, 1.):
        return 1., lateral_stress(1.)
    lateral = brentq(lateral_stress, 1., 1.8, xtol=1e-13)
    return float(lateral), lateral_stress(lateral)


def shear_tangent(compression: float, mu: float, lame: float, gamma: float = 1e-5) -> dict:
    if not np.isfinite(compression) or not 0 <= compression < 1 or not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("Compression in [0, 1) and positive finite shear probe required")
    axial = 1. - compression
    lateral, free_residual = free_lateral_stretch(axial, mu, lame)
    f0 = np.diag([axial, lateral, lateral])
    shear = np.zeros((3, 3), dtype=np.float64)
    shear[1, 2] = 1.
    plus = (np.eye(3) + gamma * shear) @ f0
    minus = (np.eye(3) - gamma * shear) @ f0
    tangent = float((cauchy_stress(plus, mu, lame)[1, 2] -
                     cauchy_stress(minus, mu, lame)[1, 2]) / (2. * gamma))
    return {"compression_fraction": float(compression), "axial_stretch": axial,
            "free_lateral_stretch": lateral, "volume_ratio": float(np.linalg.det(f0)),
            "free_lateral_piola_residual_pa": float(free_residual),
            "current_cauchy_shear_tangent_pa": tangent,
            "ratio_to_uncompressed": tangent / mu}


def audit() -> dict:
    nu = .45
    result = {}
    for name, group in GROUPS.items():
        mu, lame = lame_from_g_nu(group["g0_pa"], nu)
        curve = [shear_tangent(c, mu, lame) for c in (0., .1, .2, .3, .4)]
        observed_factor = 1. + group["increase_40pct_percent"] / 100.
        predicted_factor = curve[-1]["ratio_to_uncompressed"]
        result[name] = {
            "baseline_storage_modulus_pa": group["g0_pa"],
            "reported_increase_at_40pct_percent": group["increase_40pct_percent"],
            "reported_factor_at_40pct": observed_factor,
            "approximate_reported_increase": bool(group.get("approximate", False)),
            "analytical_physx_law_factor_at_40pct": predicted_factor,
            "analytical_physx_law_modulus_at_40pct_pa": curve[-1]["current_cauchy_shear_tangent_pa"],
            "reported_implied_modulus_at_40pct_pa": group["g0_pa"] * observed_factor,
            "relative_underprediction_of_factor": 1. - predicted_factor / observed_factor,
            "compression_curve": curve,
        }
    return {
        "status": "analytical_physx_constitutive_compression_stiffening_gate_failed_runtime_unverified",
        "model": "Fixed-corotated PhysX soft-body energy: mu*||F-R||_F^2 + lambda/2*(det(F)-1)^2",
        "stress": "P=2*mu*(F-R)+lambda*(J-1)*cof(F); sigma=P*F^T/J",
        "assumptions": {
            "physx_material_model": "eCO_ROTATIONAL, documented PhysX default; current scene specifies only Young's modulus and Poisson ratio",
            "axial_compression": "Prescribed axial stretch 1-c with equal free lateral stretches solved from P22=P33=0",
            "oscillatory_shear": "Small-amplitude shear in transverse plate plane; central difference of Cauchy sigma23 against current simple shear",
            "poisson_ratio": nu,
            "baseline_parameter_mapping": "mu=uncompressed appendix G-prime mean and isotropic lambda from nu; scale comparison only"
        },
        "groups": result,
        "summary": "Under this explicit free-lateral, current-configuration shear test, the documented fixed-corotated law predicts about 1.99x at 40% compression. The Appendix article reports about 3x, 4.5x and about 5x for control, purulent and gangrenous groups. Thus the analytical law underpredicts the reported group endpoint; this is not an Isaac runtime result.",
        "runtime": {"isaac_started": False, "physx_simulation_executed": False,
                    "reason": "Existing Windows Isaac Python startup was blocked before SimulationApp construction when policy blocked importing select; the same startup was not repeated."},
        "limitations": [
            "The article reports a small cohort and endpoint increase text; raw per-sample rheometer curves were not digitized.",
            "This derivation does not establish the exact compiled PhysX runtime response, contact, viscoelasticity, hysteresis, damage, or fluid effects.",
            "No material parameters were promoted or changed in the simulator."
        ]
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--scene", type=Path, required=True)
    args = p.parse_args()
    report = audit()
    report["source"] = {
        "appendix_article": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11591559/",
        "physx_constitutive_docs": "https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/SoftBodies.html",
        "physx_material_model_api": "https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/_api_build/structPxFEMSoftBodyMaterialModel.html",
        "scene_script_sha256": hashlib.sha256(args.scene.read_bytes()).hexdigest(),
        "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
