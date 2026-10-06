"""PPO training/evaluation for Rosota bimanual trocar-constrained tasks.

Tasks include waypoint reach, unassisted bimanual patch traction and an
endoscope-RGB traction policy. None is a complete appendectomy.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def usd_dependency_hashes(path):
    """Hash local files referenced by text USD layers without recording paths."""
    root=Path(path).resolve()
    pending=[root];visited=set();hashes=set()
    while pending:
        layer=pending.pop().resolve()
        if layer in visited:continue
        visited.add(layer)
        if not layer.is_file():raise FileNotFoundError("Missing local USD asset: "+layer.name)
        try:text=layer.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("Appearance contract requires text USD so texture dependencies can be hashed") from error
        for reference in sorted(set(re.findall(r"@([^@]+)@",text))):
            if "://" in reference:
                raise ValueError("Remote USD dependencies are not allowed in a local appearance contract")
            dependency=Path(reference)
            if not dependency.is_absolute():dependency=(layer.parent/dependency).resolve()
            if not dependency.is_file():
                raise FileNotFoundError("Missing USD dependency: "+Path(reference).name)
            digest=file_sha256(dependency)
            hashes.add(digest)
            if dependency.suffix.lower() in {".usd",".usda",".usdc"}:
                pending.append(dependency)
    return sorted(hashes)


def asset_contract(assets, fr3_urdf, task="waypoint", appearance=None,
                   visibility_shaping_weight=.1):
    task_dimensions={"waypoint":("RosotaCTEnv/waypoint-reach-v1","privileged_state_v1",53,
                                  "mlp_privileged_v1"),
                     "unassisted-traction":("RosotaCTEnv/unassisted-bimanual-traction-v1",
                                              "privileged_traction_state_v1",65,"mlp_privileged_v1"),
                     "vision-traction":("RosotaCTEnv/endoscope-rgb-traction-visibility-v2",
                                         "endoscope_rgb_proprioception_v1",45,
                                         "cnn_endoscope_rgb_proprio_v1")}
    if task not in task_dimensions:raise ValueError("Unknown RL task contract: "+str(task))
    if not np.isfinite(visibility_shaping_weight) or visibility_shaping_weight < 0.:
        raise ValueError("Visibility shaping weight must be finite and nonnegative")
    environment,observation_contract,observation_dim,policy_architecture=task_dimensions[task]
    assets = Path(assets).resolve()
    scene, config = assets / "attached.usda", assets / "bimanual_config.json"
    if not scene.is_file() or not config.is_file():
        raise FileNotFoundError("Asset directory must contain attached.usda and bimanual_config.json")
    return {
        "environment": environment,
        "task": task,
        "action_contract": "bimanual_trocar_scope_v1",
        "observation_contract": observation_contract,
        "observation_dim": observation_dim,
        "action_dim": 14,
        "policy_architecture": policy_architecture,
        "visibility_shaping_weight": (float(visibility_shaping_weight)
                                       if task=="vision-traction" else None),
        "rgb_observation_shape": [90,160,3] if task=="vision-traction" else None,
        "appearance_sha256": file_sha256(appearance) if appearance else None,
        "appearance_dependency_sha256": usd_dependency_hashes(appearance) if appearance else [],
        "scene_sha256": file_sha256(scene),
        "config_sha256": file_sha256(config),
        "fr3_urdf_sha256": file_sha256(fr3_urdf),
        "adapter_sha256": file_sha256(Path(__file__).with_name("rosota_rl_adapter.py")),
        "semantic_visibility_sha256": file_sha256(Path(__file__).with_name("semantic_visibility.py")),
        "controller_sha256": file_sha256(Path(__file__).with_name("rosota_controller.py")),
        "ppo_sha256": file_sha256(Path(__file__).with_name("ppo_agent.py")),
        "runner_sha256": file_sha256(__file__),
        "policy_observes_rgb": task=="vision-traction",
        "tissue_force_calibrated": False,
        "patient_level_holdout_verified": False,
        "appendectomy_completed": False,
    }


def train_ppo(env, model, optimizer, *, total_steps, seed, rollout_steps=256):
    """Collect actual adapter episodes and apply PPO updates; simulator injected by caller."""
    import torch
    from ppo_agent import advantages, prepare_observation, stack_observations, update
    from physics_substep_audit import EpisodeSubstepHealth

    torch.manual_seed(seed)
    np.random.seed(seed)
    obs, _ = env.reset(seed=seed)
    trajectories = {k: [] for k in ("obs", "raw", "log_prob", "value", "reward",
                                    "next_value", "terminated", "ended")}
    episodes, updates = [], []
    episode_return = 0.0
    episode_length = 0
    episode_min_distance = float("inf")
    episode_substep_health = EpisodeSubstepHealth()
    episode_index = 0

    for step_index in range(total_steps):
        with torch.no_grad():
            state = prepare_observation(obs)
            action, raw, log_prob, value, _ = model.evaluate(state)
            action_np = action[0].cpu().numpy()
            raw_np = raw[0].cpu().numpy()
            logp = float(log_prob[0].cpu())
            v = float(value[0].cpu())
        next_obs, reward, terminated, truncated, info = env.step(action_np)
        episode_substep_health.update(info.get("physics_substep_health"))
        ended = bool(terminated or truncated)
        with torch.no_grad():
            next_state = prepare_observation(next_obs)
            next_value = 0.0 if terminated else float(model.value(next_state)[0].cpu())
        trajectories["obs"].append({key:value.copy() for key,value in obs.items()}
                                    if isinstance(obs,dict) else obs.copy())
        trajectories["raw"].append(raw_np)
        trajectories["log_prob"].append(logp)
        trajectories["value"].append(v)
        trajectories["reward"].append(float(reward))
        trajectories["next_value"].append(next_value)
        trajectories["terminated"].append(bool(terminated))
        trajectories["ended"].append(ended)
        episode_return += float(reward)
        episode_length += 1
        distance=np.asarray(info.get("task_distance_m",info.get("waypoint_distance_m")),float)
        if distance.size==0 or not np.isfinite(distance).all():raise ValueError("RL task distance is missing or invalid")
        episode_min_distance=min(episode_min_distance,float(np.linalg.norm(distance)))
        failure = info.get("failure")
        obs = next_obs

        if ended:
            episodes.append({
                "episode": episode_index, "seed": seed + episode_index,
                "return": episode_return, "length": episode_length,
                "task_success": bool(info.get("task_success",info.get("waypoint_is_success",False))),
                "waypoint_success": info.get("waypoint_is_success"),
                "minimum_bimanual_task_distance_m": episode_min_distance,
                "failure": failure or "",
                "physics_substep_health": episode_substep_health.as_dict(),
            })
            episode_index += 1
            obs, _ = env.reset(seed=seed + episode_index)
            episode_return = 0.0
            episode_length = 0
            episode_min_distance = float("inf")
            episode_substep_health = EpisodeSubstepHealth()

        if len(trajectories["obs"]) >= rollout_steps or step_index + 1 == total_steps:
            adv, returns = advantages(trajectories["reward"], trajectories["value"],
                                      trajectories["next_value"], trajectories["terminated"],
                                      trajectories["ended"])
            batch = {
                "obs": stack_observations(trajectories["obs"]),
                "raw": np.asarray(trajectories["raw"], np.float32),
                "log_prob": np.asarray(trajectories["log_prob"], np.float32),
                "advantage": adv,
                "return": returns,
            }
            updates.append({"environment_steps": step_index + 1, "batch_size": len(batch["obs"]),
                            **update(model, optimizer, batch)})
            trajectories = {key: [] for key in trajectories}
    return {"environment_steps": total_steps, "episodes": episodes, "updates": updates,
            "unclosed_episode_at_rollout_end": bool(episode_length),
            "unclosed_episode_physics_substep_health": episode_substep_health.as_dict()
            if episode_length else None}


def _points_hash(env):
    core = getattr(env, "env", None)
    if core is None or not hasattr(core, "points"):
        return None
    return hashlib.sha256(np.asarray(core.points()).tobytes()).hexdigest()


def evaluate_policy(env, model, policy, *, seeds):
    import torch
    from physics_substep_audit import EpisodeSubstepHealth
    from ppo_agent import prepare_observation
    results = []
    if policy == "random":
        env.action_space.seed(seeds[0] if seeds else 0)
    for episode, seed in enumerate(seeds):
        obs, _ = env.reset(seed=seed)
        start_hash = _points_hash(env)
        total_reward, length = 0.0, 0
        min_distance = float("inf")
        episode_substep_health = EpisodeSubstepHealth()
        info = {}
        while True:
            if policy == "scripted":
                action = (env.scripted_task_action() if hasattr(env,"scripted_task_action")
                          else env.scripted_waypoint_action())
            elif policy == "random":
                action = env.action_space.sample()
            elif policy == "ppo":
                with torch.no_grad():
                    state = prepare_observation(obs)
                    action = model.evaluate(state, deterministic=True)[0][0].cpu().numpy()
            else:
                raise ValueError("Unknown evaluation policy: " + policy)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_substep_health.update(info.get("physics_substep_health"))
            total_reward += float(reward)
            length += 1
            distance=np.asarray(info.get("task_distance_m",info.get("waypoint_distance_m")),float)
            if distance.size==0 or not np.isfinite(distance).all():raise ValueError("RL task distance is missing or invalid")
            min_distance=min(min_distance,float(np.linalg.norm(distance)))
            if terminated or truncated:
                break
        results.append({
            "episode": episode, "seed": seed, "policy": policy,
            "return": total_reward, "length": length,
            "task_success": bool(info.get("task_success",info.get("waypoint_is_success",False))),
            "waypoint_success": info.get("waypoint_is_success"),
            "failure": info.get("failure") or "",
            "minimum_bimanual_task_distance_m": min_distance,
            "initial_tissue_state_sha256": start_hash,
            "physics_substep_health": episode_substep_health.as_dict(),
            "appendectomy_completed": False,
        })
    return results


def summarize(rows):
    physics = [row.get("physics_substep_health") or {} for row in rows]
    sampled_ratios = [audit["minimum_signed_volume_ratio"] for audit in physics
                      if audit.get("minimum_signed_volume_ratio") is not None]
    return {
        "episodes": len(rows),
        "task_success_rate": float(np.mean([r.get("task_success",False) for r in rows])) if rows else None,
        "waypoint_success_rate": (float(np.mean([r["waypoint_success"] for r in rows]))
                                  if rows and all(r.get("waypoint_success") is not None for r in rows) else None),
        "mean_return": float(np.mean([r["return"] for r in rows])) if rows else None,
        "mean_steps": float(np.mean([r["length"] for r in rows])) if rows else None,
        "mean_minimum_bimanual_task_distance_m": float(np.mean(
            [r["minimum_bimanual_task_distance_m"] for r in rows])) if rows else None,
        "failures": {name: sum(r["failure"] == name for r in rows)
                     for name in sorted({r["failure"] for r in rows if r["failure"]})},
        "physics_substep_safety": {
            "sampling_complete": bool(rows) and all(audit.get("sampling_complete") is True
                                                      for audit in physics),
            "physics_ticks": sum(int(audit.get("physics_ticks", 0)) for audit in physics),
            "ticks_with_inversion": sum(int(audit.get("ticks_with_inversion", 0))
                                         for audit in physics),
            "max_inverted_tetrahedra_in_one_tick": max(
                (int(audit.get("max_inverted_tetrahedra_in_one_tick", 0)) for audit in physics),
                default=0),
            "minimum_signed_volume_ratio": min(sampled_ratios) if sampled_ratios else None,
            "all_sampled_ticks_inversion_free": bool(rows) and all(
                audit.get("all_sampled_ticks_inversion_free") is True for audit in physics),
        },
    }


def make_env(app, assets, fr3_urdf, out, *, max_steps=300, appearance=None,
             task="waypoint", visibility_shaping_weight=.1):
    from rosota_ct_env import RosotaCTEnv
    from rosota_rl_adapter import (RosotaRLAdapter, RosotaTractionRLAdapter,
                                  RosotaVisionTractionRLAdapter)
    if task not in {"waypoint", "unassisted-traction", "vision-traction"}:
        raise ValueError("Unknown RL task: " + str(task))
    core = RosotaCTEnv(app, assets, fr3_urdf, out, max_steps=max_steps,
                       grasp_mode="none", workspace="appendix_patches",
                       settle_policy="adaptive", appearance=appearance)
    if task=="vision-traction":return RosotaVisionTractionRLAdapter(
        core,visibility_shaping_weight=visibility_shaping_weight)
    return RosotaTractionRLAdapter(core) if task == "unassisted-traction" else RosotaRLAdapter(core)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    work = ROOT.parent.parent / "work"
    parser.add_argument("--assets", type=Path, default=ROOT / "out/ct_appendectomy/multitask_assets",
                        help="Training scene directory with attached.usda and bimanual_config.json")
    parser.add_argument("--eval-assets", type=Path,
                        help="Separate evaluation scene; does not imply patient-level holdout")
    parser.add_argument("--fr3-urdf", type=Path, default=work / "isaac-venv/Lib/site-packages/isaacsim/exts/isaacsim.robot_motion.motion_generation/motion_policy_configs/FR3/fr3.urdf")
    parser.add_argument("--appearance", type=Path, help="Optional appearance layer for the CT environment")
    parser.add_argument("--task", choices=("waypoint", "unassisted-traction", "vision-traction"), default="waypoint",
                        help="Vision task uses endoscope RGB and proprioception; reward still reads target geometry")
    parser.add_argument("--visibility-shaping-weight", type=float, default=.1,
                        help="Potential reward weight for appendix image fraction on vision-traction; use 0 for ablation")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-steps", type=int, default=0)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path, help="Load/resume a version-matched PPO checkpoint")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    if args.train_steps < 0 or args.rollout_steps < 64 or args.eval_episodes < 1:
        parser.error("train-steps must be nonnegative, rollout-steps at least 64, eval-episodes positive")
    if not np.isfinite(args.visibility_shaping_weight) or args.visibility_shaping_weight < 0.:
        parser.error("visibility-shaping-weight must be finite and nonnegative")
    if args.train_steps == 0 and args.checkpoint is None:
        parser.error("Evaluation-only mode requires --checkpoint")
    if not args.fr3_urdf.is_file():
        parser.error("FR3 URDF is missing: " + str(args.fr3_urdf))
    training_contract = asset_contract(args.assets, args.fr3_urdf, args.task,args.appearance,
                                       args.visibility_shaping_weight)
    evaluation_assets = args.eval_assets or args.assets
    evaluation_contract = asset_contract(evaluation_assets, args.fr3_urdf, args.task,args.appearance,
                                         args.visibility_shaping_weight)
    if args.checkpoint and not args.checkpoint.is_file():
        parser.error("Checkpoint not found: " + str(args.checkpoint))

    import torch
    from ppo_agent import ActorCritic, VisionActorCritic, load_checkpoint, save_checkpoint
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = (VisionActorCritic(proprioception=training_contract["observation_dim"],actions=14)
             if args.task=="vision-traction" else
             ActorCritic(observations=training_contract["observation_dim"],actions=14))
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)
    checkpoint_metadata = None
    if args.checkpoint:
        checkpoint_metadata = load_checkpoint(args.checkpoint, model, optimizer, training_contract)

    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "starting",
        "task": ("two-arm trocar-constrained waypoint reach; scope RCM actions exposed but not rewarded"
                 if args.task == "waypoint" else
                 ("endoscope RGB appendix-patch traction with simulator-only appendix visibility reward shaping; engineering benchmark only"
                  if args.task == "vision-traction" else
                  "unassisted bimanual appendix-patch contact and upward traction; engineering benchmark only")),
        "policy_input": ("endoscope RGB and robot/jaw/contact proprioception; target patch geometry is reward-only"
                         if args.task=="vision-traction" else "privileged proprioceptive state"),
        "task_is_appendectomy": False,
        "appendix_grasp_task": args.task != "waypoint",
        "traction_task": args.task != "waypoint",
        "traction_task_live_runtime_verified": False,
        "traction_thresholds": {"patch_proximity_m": 0.005, "target_lift_m": 0.005,
                                "hold_steps": 10} if args.task != "waypoint" else None,
        "shaft_organ_collision_enabled": False,
        "policy_observes_rgb": args.task=="vision-traction",
        "tissue_force_calibrated": False,
        "patient_level_holdout_verified": False,
        "paired_reset_state_verified": False,
        "engine": "Isaac Sim / PhysX runtime pending",
        "training_contract": training_contract,
        "evaluation_assets": str(evaluation_assets.resolve()),
        "evaluation_contract": evaluation_contract,
        "evaluation_scene_sha256": evaluation_contract["scene_sha256"],
        "training_steps_requested": args.train_steps,
        "checkpoint_in": str(args.checkpoint.resolve()) if args.checkpoint else None,
        "checkpoint_metadata": checkpoint_metadata,
    }
    report_path = args.out / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    app = None
    try:
        for name in ("WARP_CACHE_PATH", "OPTIX_CACHE_PATH"):
            os.environ.setdefault(name, str(ROOT / ".cache" / name.lower()))
        os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
        from isaacsim import SimulationApp
        runtime = ROOT / ".cache/kit"
        token_names = ("shared_documents", "app_documents", "omni_documents", "omni_config",
                       "omni_logs", "cache", "omni_cache", "omni_global_cache")
        tokens = {name: runtime / name for name in token_names}
        for path in tokens.values():
            path.mkdir(parents=True, exist_ok=True)
        app = SimulationApp({"headless": args.headless, "width": 1280, "height": 720,
                             "renderer": "RaytracedLighting", "sync_loads": True,
                             "hide_ui": args.headless, "extra_args": [
                                 "--/app/vulkan=false", "--/app/d3d12=true",
                                 "--/telemetry/enableNVDF=false", "--/telemetry/enableSentry=false",
                                 "--/telemetry/useOpenEndpoint=false",
                                 "--/privacy/extraDiagnosticDataOptIn=false",
                                 "--/crashreporter/enabled=false"] + [
                                 f"--/app/tokens/{k}={v.as_posix()}" for k, v in tokens.items()]})
    except Exception as exc:
        report.update(status="runtime_start_failed", error=repr(exc))
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if app is not None:
            app.close()
        raise
    env = None
    try:
        env = make_env(app, args.assets, args.fr3_urdf, args.out / "training",
                       max_steps=300, appearance=args.appearance, task=args.task,
                       visibility_shaping_weight=args.visibility_shaping_weight)
        training = None
        if args.train_steps:
            training = train_ppo(env, model, optimizer, total_steps=args.train_steps,
                                 seed=args.seed, rollout_steps=args.rollout_steps)
            report["training"] = training
            metadata = dict(training_contract, trained_steps=args.train_steps,
                            algorithm="local PPO", task_description=report["task"],
                            task_is_appendectomy=False,
                            policy_observes_rgb=args.task=="vision-traction")
            save_checkpoint(args.out / "policy.pt", model, optimizer, metadata)
        env.close()
        env = make_env(app, evaluation_assets, args.fr3_urdf, args.out / "evaluation",
                       max_steps=300, appearance=args.appearance, task=args.task,
                       visibility_shaping_weight=args.visibility_shaping_weight)
        seeds = [args.seed + 100_000 + i for i in range(args.eval_episodes)]
        policies = ["scripted", "random"]
        if args.train_steps or args.checkpoint:
            policies.append("ppo")
        evaluations = {}
        for policy in policies:
            evaluations[policy] = evaluate_policy(env, model, policy, seeds=seeds)
        report["evaluation"] = {name: {"summary": summarize(rows), "episodes": rows}
                                for name, rows in evaluations.items()}
        report["paired_start_state_match_by_seed"] = {
            str(seed): len({evaluations[name][i]["initial_tissue_state_sha256"] for name in policies}) == 1
            for i, seed in enumerate(seeds)
            if all(evaluations[name][i]["initial_tissue_state_sha256"] is not None for name in policies)
        }
        report["paired_reset_state_verified"] = bool(report["paired_start_state_match_by_seed"]) and all(
            report["paired_start_state_match_by_seed"].values())
        completed_status={"waypoint":"completed_waypoint_benchmark",
                          "unassisted-traction":"completed_traction_benchmark",
                          "vision-traction":"completed_vision_traction_benchmark"}[args.task]
        report["status"] = completed_status if all(
            not row["failure"] for rows in evaluations.values() for row in rows) else "completed_with_failures"
        report["rl_validation_gate"] = ("not achieved: single local task, privileged state, uncalibrated contact proxy, no patient-level holdout or clinical validation")
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    except Exception as exc:
        report.update(status="failed", error=repr(exc))
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        raise
    finally:
        if env is not None:
            env.close()
        if app is not None:
            app.close()


if __name__ == "__main__":
    main()
