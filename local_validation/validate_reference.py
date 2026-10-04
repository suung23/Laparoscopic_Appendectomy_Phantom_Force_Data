"""Verify local pinned reference provenance and completed rollout evidence."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def finite_json(value):
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finite_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def verify_blobs(manifest):
    bad = []
    files = manifest.get('files', manifest.get('verified_files', []))
    if not files:
        raise ValueError('Empty provenance manifest')
    for item in files:
        p = ROOT / item['path']
        if not p.exists():
            bad.append(item['path'])
            continue
        b = p.read_bytes()
        sha = hashlib.sha1(b'blob ' + str(len(b)).encode() + b'\0' + b).hexdigest()
        if sha != item['sha']:
            bad.append(item['path'])
    return dict(count=len(files), mismatches=bad, passed=not bad)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rollout', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    report = dict(source=verify_blobs(json.loads((ROOT/'SOURCE_MANIFEST.json').read_text())),
                  assets=verify_blobs(json.loads((ROOT/'validation/rosota-reference-assets.json').read_text())),
                  ct_baseline_parity=False, material_calibrated=False,
                  physical_severing=False, learned_policy=False)
    report['scenes'] = {s: (ROOT/'out'/s).exists() for s in ('scene_soft.usd', 'surgical_cell.usd')}
    summary = a.rollout/'summary.json'
    report['completed'] = summary.exists()
    calibration = ROOT/'out/vla_force_calib.json'
    if calibration.exists():
        model = json.loads(calibration.read_text())
        report['force_model_fit'] = {hand: {k: values.get(k) for k in ('rms_f', 'rms_t', 'c')}
                                     for hand, values in model.items() if isinstance(values, dict)}
    report['camera_frames'] = {tag: len(list((a.rollout/'frames/ep000').glob(tag+'_*.png')))
                               for tag in ('endoscope', 'd405', 'd405depth', 'd405right')}
    if summary.exists():
        data = json.loads(summary.read_text())
        cfg = data['config']
        report['correct_video_fps'] = cfg['control_hz']/cfg['chunk']/cfg['image_every']
        report['playback_video'] = 'endoscope_realtime.mp4'
        report['episodes'] = data['episodes']
        report['aggregate'] = data['aggregate']
        report['saved_channels'] = {}
        with np.load(a.rollout/'episode_000.npz', allow_pickle=False) as episode:
            for k in episode.files:
                report['saved_channels'][k] = list(episode[k].shape)
            required = ('t','action','left_joint_pos','right_joint_pos',
                        'left_tip_pose_rig','right_tip_pose_rig','left_rcm4','right_rcm4')
            report['core_channels_finite'] = all(k in episode.files
                and np.isfinite(episode[k]).all() for k in required)
            report['joint_excursion_rad'] = {hand: float(np.max(np.abs(
                episode[hand+'_joint_pos']-episode[hand+'_joint_pos'][0])))
                for hand in ('left','right')}
            report['rgb_nonblank'] = {cam: bool(cam+'_rgb' in episode.files
                and np.std(episode[cam+'_rgb'][0])>1) for cam in ('endoscope','d405')}
            depth = episode['d405_depth'] if 'd405_depth' in episode.files else np.zeros(1)
            report['depth_valid_fraction'] = float(np.mean(np.isfinite(depth)&(depth>0)))
    report['reference_runtime_verified'] = bool(report['completed'] and report['source']['passed']
        and report['assets']['passed'] and all(report['scenes'].values())
        and all(n > 0 for n in report['camera_frames'].values())
        and report.get('core_channels_finite') and all(report.get('rgb_nonblank', {}).values())
        and report.get('depth_valid_fraction',0)>0
        and all(x>1e-4 for x in report.get('joint_excursion_rad',{}).values()))
    report['limits'] = [
        'Native reference is rabbit anatomy; CT extension is a separate scene.',
        'Original grasp is attachment-assisted; cutting is a gesture, never tissue division.',
        'Runtime verification does not imply task success or calibrated tissue mechanics.',
        'Force model residuals must be assessed before using its output as contact-force ground truth.',
        'Original default video fps ignores chunk length; use control_hz/chunk/image_every.'
    ]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(finite_json(report), indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('source','assets','completed','camera_frames','reference_runtime_verified')}))


if __name__ == '__main__':
    main()
