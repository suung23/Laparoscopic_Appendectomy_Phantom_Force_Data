# Rosota simulator — local engineering baseline

2026-10-04. Internal working copy; no remote push or publication performed.

## Reproduced on this PC

The source snapshot is pinned to `9135fe9eb8549d011124c8c57b7c86edeaaba735` of the private `Rosota-Research/Rosota-Simulator` repository. This is a **partial, hash-verified snapshot, not a Git clone**. `SOURCE_MANIFEST.json` records each original file. Upstream files are unmodified; local additions include `ct_appendectomy/`, `local_validation/`, launchers and `LOCAL_*` files.

Newton 1.4.0 / Warp 1.15.0 runs on Windows, Python 3.12.14, RTX 4090. The six reconstructed rabbit organs contain 11,962 nodes and 43,956 tetrahedra. The 120-frame, two-second tissue-only rest test passed finite-state, fixed-node drift, final speed, and final tetrahedron orientation checks. Exact measurements and solver settings are in `validation/tissue-baseline.json`; initial/final geometry is retained in the adjacent NPZ.

The test excludes instruments and rendering. It disables self-contact and retains the upstream mass floor, pins, and artificial appendix shelf. Material parameters are engineering estimates, not identified human tissue properties. Passing this test establishes a runnable numerical baseline, not clinical fidelity or surgical safety. Timing includes kernel loading; it is not a steady-state performance benchmark.

## Reproduce

Run `./validate-local.ps1` in PowerShell. The launcher uses the task's existing virtual environment by default; `-Python` can select another interpreter. `LOCAL_ENVIRONMENT.txt` records installed package versions, and `LOCAL_DEPENDENCY_LICENSES.json` records installed metadata for the major dependencies. These files are an inventory, not a completed license review.

The validator verifies upstream Git blob hashes before running. Deliberate upstream edits should be reviewed as a patch against the pinned baseline rather than silently refreshing its hashes. No runtime download occurs in the validator. CUDA kernel cache stays under this working copy.

## Remaining integration gates

1. Authenticate the local GitHub CLI to obtain a complete private checkout and large binary assets. The connector supplied source, tissue meshes, and the ASCII phantom; the full surgical background and robot assets are not yet available locally. Keep this verified baseline when obtaining the full checkout.
2. Install and reproduce the authoritative Isaac Sim 5.1 path on the RTX workstation, or document any necessary Windows port. Isaac Sim 5.1 is installed in a separate Python 3.11 environment. The CT/atlas scene renders with D3D12 on RTX 4090; PhysX FEM cooks 1,388 appendix nodes and passes prescribed 10 mm retraction. See ct_appendectomy/README.md and validation/ct-isaac-rl.json. This is not a throughput benchmark.
3. Reproduce dual-arm RCM, grasp/lift, camera calibration, and action/observation semantics; test reset, action limits, invalid input, episode termination and privileged observation separation before exposing a Gymnasium training API.
4. Implement and validate actual topology change, stump/ligation state and relevant vessel/tissue damage. Upstream `isaac/vla_rollout.py` explicitly reports `appendix_severed=False`: its cut-stage success is a gesture score, never completion of physical appendectomy.
5. Develop the human appendectomy anatomy and endoscopic appearance pipeline. The existing rabbit RGB-D reconstruction and the separate public human CT viewer are distinct data sources; the new human CT/atlas path implements physical appendix retraction, not complete appendectomy.

H100 access remains unverified. No server contents or training performance are asserted. No external experiment submission, cloud policy upload, repository push, or publication was performed.

## Asset provenance and ownership

Private upstream reconstruction assets remain private and retain their existing ownership and terms. Retrieval through an authorized account does not establish permission to redistribute or relicense them. Do not apply an open-source license to this repository snapshot by inference. Public human CT/atlas provenance remains in the separate `appendectomy_lab` manifests; the CT builder explicitly imports public geometry, records hashes, and labels the appendix as an unvalidated atlas graft.

Before product release, record acquisition consent/rights for reconstruction data, upstream/submodule licenses, CAD redistribution terms, model weights, dataset terms and notices. No personal identifiers, SSH credentials, or signed download links belong in manifests or reports.

## Contact extension

The CT path now includes a base-pinned, collision-driven blunt-tip contact probe, separate from pre-attached retraction. See `ct_appendectomy/README.md`. All-organ collision remains experimental: the provisional atlas registration creates initial overlaps and fails stability checks. Default contact mode disables those CT colliders while testing the physical tool/appendix interaction. This does not implement grasping, coupled mesentery or cutting.
