"""Compare the current FEM elastic parameter with published appendix rheology.

This derives a small-strain, isotropic-equivalent scale only. It does not fit
or validate the nonlinear, frequency-dependent PhysX tissue model.
"""
import argparse
import ast
import json
from pathlib import Path


def runtime_material(path):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    matches = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else ""
        if name != "add_deformable_material":
            continue
        values = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords
                  if kw.arg in {"youngs_modulus", "poissons_ratio"}}
        if set(values) == {"youngs_modulus", "poissons_ratio"}:
            matches.append(values)
    if len(matches) != 1:
        raise ValueError(f"Expected one authoritative FEM material call, found {len(matches)}")
    youngs, poisson = matches[0]["youngs_modulus"], matches[0]["poissons_ratio"]
    if not (youngs > 0 and 0 <= poisson < 0.5):
        raise ValueError("Invalid isotropic elastic parameters")
    return float(youngs), float(poisson)


def build(root):
    root = Path(root).resolve()
    source_path = root / "research/appendix-mechanics-reference.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    runtime_path = root / "ct_appendectomy/run_scene.py"
    youngs, poisson = runtime_material(runtime_path)
    baseline_shear = youngs / (2 * (1 + poisson))
    measured = source["rheometry"]["storage_modulus_pa_mean_sd"]
    groups = {}
    for state, (mean, sd) in measured.items():
        equivalent_e = 2 * (1 + poisson) * mean
        groups[state] = {
            "measured_storage_shear_modulus_pa_mean_sd": [mean, sd],
            "isotropic_small_strain_equivalent_youngs_modulus_pa_mean_sd": [
                round(equivalent_e, 2), round(2 * (1 + poisson) * sd, 2)],
            "current_baseline_equivalent_shear_over_measured_mean": round(baseline_shear / mean, 3),
        }
    return {
        "status": "literature_prior_comparison_only",
        "source": {"title": source["title"], "doi": source["doi"], "url": source["source"],
                   "sample": source["sample"], "sample_counts": source.get("sample_counts")},
        "test_context": source["rheometry"],
        "current_runtime_material": {
            "file": str(runtime_path.relative_to(root)).replace("\\", "/"),
            "youngs_modulus_pa": youngs, "poissons_ratio": poisson,
            "implied_linear_isotropic_shear_modulus_pa": round(baseline_shear, 2),
        },
        "group_comparisons": groups,
        "assumptions": [
            "G-prime is mapped to E-prime using E=2(1+nu)G only as a small-strain isotropic scale comparison",
            "This mapping ignores loss modulus, frequency dependence, anisotropy, freezing effects, and compression nonlinearity",
            "Human ex-vivo biopsy summaries are priors, not in-vivo force-displacement ground truth",
        ],
        "not_validated": ["PhysX FEM constitutive response", "force-displacement response", "stress relaxation",
                          "contact and slip", "tissue damage", "bimanual task performance"],
        "production_material_changed": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
