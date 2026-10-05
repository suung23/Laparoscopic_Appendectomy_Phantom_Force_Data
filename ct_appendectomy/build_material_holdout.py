"""Author a literature-scaled material sensitivity asset for RL evaluation.

This is a domain-shift fixture, not a fitted or clinically validated material.
It changes only the tissue Young's-modulus value in a USD layer over a copied
source scene contract; original training assets remain untouched.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

from pxr import Usd, UsdShade


MATERIAL_PATH = "/World/TissueMaterial"
BODY_PATH = "/World/Atlas/appendix/sim"
MODULUS_ATTR = "omniphysics:youngsModulus"
POISSON_ATTR = "omniphysics:poissonsRatio"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(source_assets, output, reference, group):
    source_assets, output, reference = map(Path, (source_assets, output, reference))
    source_assets, output, reference = map(lambda p: p.resolve(), (source_assets, output, reference))
    source_stage = source_assets / "attached.usda"
    source_config = source_assets / "forceps_config.json"
    if not source_stage.is_file() or not source_config.is_file():
        raise FileNotFoundError("Source assets must contain attached.usda and forceps_config.json")
    config = json.loads(source_config.read_text(encoding="utf-8"))
    if config.get("stage_sha256") != sha256(source_stage):
        raise ValueError("Source forceps config does not match its stage hash")

    source = json.loads(reference.read_text(encoding="utf-8"))
    measured_g, measured_sd = source["rheometry"]["storage_modulus_pa_mean_sd"][group]
    source_usd = Usd.Stage.Open(str(source_stage))
    material = source_usd.GetPrimAtPath(MATERIAL_PATH)
    body = source_usd.GetPrimAtPath(BODY_PATH)
    e_attr, nu_attr = material.GetAttribute(MODULUS_ATTR), material.GetAttribute(POISSON_ATTR)
    if not material.IsValid() or not body.IsValid() or not e_attr or not nu_attr:
        raise ValueError("Source scene is missing its documented tissue material parameters")
    bound_material, binding_rel = UsdShade.MaterialBindingAPI(body).ComputeBoundMaterial("physics")
    if not bound_material or bound_material.GetPath() != MATERIAL_PATH:
        raise ValueError(f"Appendix deformable body does not resolve to {MATERIAL_PATH} for physics")
    baseline_e, poisson = float(e_attr.Get()), float(nu_attr.Get())
    if baseline_e <= 0 or not 0 <= poisson < 0.5 or measured_g <= 0 or measured_sd < 0:
        raise ValueError("Invalid source or literature material values")
    target_e = 2.0 * (1.0 + poisson) * float(measured_g)
    target_e_sd = 2.0 * (1.0 + poisson) * float(measured_sd)

    output.mkdir(parents=True, exist_ok=True)
    variant_stage = output / "attached.usda"
    stage = Usd.Stage.CreateNew(str(variant_stage))
    stage.GetRootLayer().subLayerPaths.append(
        os.path.relpath(source_stage, output).replace("\\", "/"))
    variant_attr = stage.GetPrimAtPath(MATERIAL_PATH).GetAttribute(MODULUS_ATTR)
    variant_attr.Set(float(target_e))
    stage.GetRootLayer().Save()
    composed = Usd.Stage.Open(str(variant_stage))
    variant_body = composed.GetPrimAtPath(BODY_PATH)
    resolved_material, resolved_rel = UsdShade.MaterialBindingAPI(variant_body).ComputeBoundMaterial("physics")
    if not resolved_material or resolved_material.GetPath() != MATERIAL_PATH:
        raise ValueError("Variant stage no longer resolves the appendix physics material binding")
    if abs(float(composed.GetPrimAtPath(MATERIAL_PATH).GetAttribute(MODULUS_ATTR).Get()) - target_e) > 0.1:
        raise ValueError("Variant Young's modulus failed USD readback")
    config["stage_sha256"] = sha256(variant_stage)
    config_path = output / "forceps_config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    report = {
        "status": "asset_authored_runtime_pending",
        "evaluation_role": "material_sensitivity_domain_shift_not_patient_holdout",
        "group": group,
        "source": {"title": source["title"], "doi": source["doi"],
                   "url": source["source"], "sample": source["sample"],
                   "sample_counts": source.get("sample_counts"),
                   "measured_storage_modulus_pa_mean_sd": [measured_g, measured_sd]},
        "mapping": {
            "formula": "E = 2 * (1 + nu) * G_prime",
            "nu": poisson,
            "mapped_equivalent_youngs_modulus_pa_mean_sd": [target_e, target_e_sd],
            "interpretation": "small-strain isotropic equivalent only; G-prime is not measured Young's modulus",
        },
        "material_change": {"prim": MATERIAL_PATH, "attribute": MODULUS_ATTR,
                            "baseline_pa": baseline_e, "variant_pa": target_e,
                            "ratio_to_baseline": target_e / baseline_e},
        "physics_material_binding": {
            "body_prim": BODY_PATH,
            "resolved_material": str(resolved_material.GetPath()),
            "binding_relationship": str(resolved_rel.GetPath()),
            "source_and_variant_resolution_passed": True,
            "runtime_physx_consumption_verified": False,
        },
        "source_assets": {"stage_sha256": sha256(source_stage),
                          "config_sha256": sha256(source_config)},
        "variant_assets": {"stage_sha256": sha256(variant_stage),
                           "config_sha256": sha256(config_path)},
        "geometry_edited": False,
        "production_assets_edited": False,
        "runtime_verified": False,
        "limitations": [
            "Published storage modulus measured ex vivo at 1 Hz and 1% shear, not in-vivo grasp force",
            "Isotropic conversion ignores loss modulus, rate dependence, anisotropy, disease heterogeneity and compression stiffening",
            "This asset is a sensitivity condition, not validated ground truth or a patient-level held-out case",
        ],
    }
    (output / "material-variant.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-assets", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path,
                        default=Path(__file__).resolve().parents[1] / "research/appendix-mechanics-reference.json")
    parser.add_argument("--group", choices=("control", "purulent", "gangrenous"), default="control")
    args = parser.parse_args()
    print(json.dumps(build(args.source_assets, args.out, args.reference, args.group), indent=2))
