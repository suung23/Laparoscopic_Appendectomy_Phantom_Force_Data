"""Verify a local Appendix300 texture USD composition without Isaac rendering."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom, UsdShade


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _same_array(left, right):
    if left is None or right is None:
        return left is None and right is None
    return np.array_equal(np.asarray(left), np.asarray(right))


def validate(parent_layer, candidate_layer, reference_dir):
    parent_layer, candidate_layer = Path(parent_layer).resolve(), Path(candidate_layer).resolve()
    reference_dir = Path(reference_dir).resolve()
    parent_report = json.loads(parent_layer.with_name("appearance.json").read_text(encoding="utf-8"))
    candidate_report = json.loads(candidate_layer.with_name("appearance.json").read_text(encoding="utf-8"))
    provenance = json.loads((reference_dir / "provenance.json").read_text(encoding="utf-8"))

    source_name = Path(candidate_report["source_video"]).name
    source_record = next((item for item in provenance["files"]
                          if item["local_name"] == source_name), None)
    if source_record is None or source_record["sha256"] != candidate_report["source_video_sha256"]:
        raise ValueError("Candidate source video hash does not match local Appendix300 provenance")
    if candidate_report.get("license") != provenance.get("license") or provenance.get("license") != "CC BY 4.0":
        raise ValueError("Dataset license differs from the local provenance manifest")
    if candidate_report.get("static_usd_layer", {}).get("parent_layer_sha256") != _sha256(parent_layer):
        raise ValueError("Candidate does not identify the supplied reviewed UV parent layer")
    if candidate_report.get("static_usd_layer", {}).get("sha256") != _sha256(candidate_layer):
        raise ValueError("Candidate USD layer hash differs from its local report")

    parent_text = parent_layer.read_text(encoding="utf-8")
    expected = parent_text.replace("@appendix_appendix300_rgb.png@",
                                  "@appendix_appendix300_temporal_rgb.png@")
    expected = expected.replace("@mesoappendix_appendix300_rgb.png@",
                                 "@mesoappendix_appendix300_temporal_rgb.png@")
    if expected == parent_text or candidate_layer.read_text(encoding="utf-8") != expected:
        raise ValueError("Candidate layer contains changes beyond the two reviewed RGB asset paths")

    baseline = Usd.Stage.Open(str(parent_layer))
    candidate = Usd.Stage.Open(str(candidate_layer))
    if baseline is None or candidate is None:
        raise ValueError("USD parser could not open parent or candidate composition")

    def meshes(stage):
        return {str(prim.GetPath()): prim for prim in stage.Traverse()
                if prim.IsA(UsdGeom.Mesh)}

    parent_meshes, candidate_meshes = meshes(baseline), meshes(candidate)
    if parent_meshes.keys() != candidate_meshes.keys():
        raise ValueError("Mesh path sets differ between parent and candidate")
    mesh_count = 0
    for path, parent_prim in parent_meshes.items():
        parent_mesh, candidate_mesh = UsdGeom.Mesh(parent_prim), UsdGeom.Mesh(candidate_meshes[path])
        attributes = [
            ("points", parent_mesh.GetPointsAttr(), candidate_mesh.GetPointsAttr()),
            ("face_counts", parent_mesh.GetFaceVertexCountsAttr(), candidate_mesh.GetFaceVertexCountsAttr()),
            ("face_indices", parent_mesh.GetFaceVertexIndicesAttr(), candidate_mesh.GetFaceVertexIndicesAttr()),
            ("normals", parent_mesh.GetNormalsAttr(), candidate_mesh.GetNormalsAttr()),
        ]
        for name, parent_attr, candidate_attr in attributes:
            if not _same_array(parent_attr.Get(), candidate_attr.Get()):
                raise ValueError(f"Mesh {path} changed {name}")
        parent_uv = UsdGeom.PrimvarsAPI(parent_prim).GetPrimvar("st")
        candidate_uv = UsdGeom.PrimvarsAPI(candidate_meshes[path]).GetPrimvar("st")
        if parent_uv.IsDefined() != candidate_uv.IsDefined():
            raise ValueError(f"Mesh {path} changed UV definition")
        if parent_uv.IsDefined() and not _same_array(parent_uv.Get(), candidate_uv.Get()):
            raise ValueError(f"Mesh {path} changed UV values")
        mesh_count += 1

    bound_materials = {}
    for mesh_name in ("appendix", "mesoappendix"):
        path = f"/World/Atlas/{mesh_name}/render"
        prim = candidate.GetPrimAtPath(path)
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if not material:
            raise ValueError(f"No material is bound to {path}")
        shader = UsdShade.Shader(candidate.GetPrimAtPath(str(material.GetPath()) + "/Color"))
        asset = shader.GetInput("file").Get()
        texture_path = candidate_layer.parent / asset.path
        expected_asset = f"{mesh_name}_appendix300_temporal_rgb.png"
        if asset.path != expected_asset or not texture_path.is_file():
            raise ValueError(f"Texture reference for {path} is missing or unexpected")
        reported_hash = candidate_report["rois_and_textures"][mesh_name]["texture_sha256"]
        texture_hash = _sha256(texture_path)
        if texture_hash != reported_hash:
            raise ValueError(f"Texture hash for {path} differs from candidate report")
        bound_materials[path] = {"material": str(material.GetPath()),
                                 "texture": asset.path, "texture_sha256": texture_hash}

    return {
        "status": "usd_structure_verified_render_pending",
        "source_video": source_record["name"],
        "source_video_sha256": source_record["sha256"],
        "license": provenance["license"],
        "parent_layer_sha256": _sha256(parent_layer),
        "candidate_layer_sha256": _sha256(candidate_layer),
        "parent_layer_only_two_texture_paths_changed": True,
        "parent_sublayers_resolved": list(candidate.GetRootLayer().subLayerPaths),
        "mesh_count": mesh_count,
        "mesh_geometry_normals_and_uvs_unchanged": True,
        "bound_materials": bound_materials,
        "isaac_runtime_load_verified": False,
        "endoscope_render_verified": False,
        "clinical_appearance_gate_passed": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-layer", type=Path, required=True)
    parser.add_argument("--candidate-layer", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = validate(args.parent_layer, args.candidate_layer, args.reference)
    encoded = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.write_text(encoded, encoding="utf-8")
    print(encoded)
