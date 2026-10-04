"""Audit downloaded clinical data without assuming controller inputs are tool poses."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


def audit(root, out, ffmpeg):
    import pyarrow.parquet as pq
    root, out = Path(root), Path(out)
    manifest = json.loads((root / 'provenance.json').read_text())
    if manifest['subset']!='Surgical/cmr_surgical/cholecystectomy':
        raise ValueError('This audit uses CMR-specific controller semantics; choose the matching dataset adapter')
    for record in manifest['files']:
        if hashlib.sha256((root / record['path']).read_bytes()).hexdigest() != record['sha256']:
            raise ValueError('Reference hash mismatch')
    info = json.loads((root / 'meta/info.json').read_text())
    modality = json.loads((root / 'meta/modality.json').read_text())
    table = pq.read_table(next(root.glob('data/**/*.parquet')))
    time = np.asarray(table['timestamp'].to_pylist(), float).reshape(-1)
    indices = np.asarray(table['frame_index'].to_pylist()).reshape(-1)
    action = np.asarray(table['action'].to_pylist(), float)
    state = np.asarray(table['observation.state'].to_pylist(), float)
    dt = np.diff(time)
    report = {'schema_version': 1, 'repository': manifest['repository'],
              'revision': manifest['revision'], 'episode': manifest['episode'],
              'license': manifest['license'], 'rows': len(time), 'fps': info['fps'],
              'timestamp_span_s': float(time[-1]-time[0]),
              'monotonic_timestamps': bool(np.all(dt > 0)),
              'contiguous_frame_indices': bool(np.all(np.diff(indices) == 1)),
              'maximum_clock_grid_error_s': float(np.max(np.abs(time-time[0]-(indices-indices[0])/info['fps']))),
              'action_shape': list(action.shape), 'state_shape': list(state.shape),
              'finite_telemetry': bool(np.isfinite(action).all() and np.isfinite(state).all()),
              'kinematics_semantics': 'Versius hand-controller input; NOT measured distal robot pose',
              'force_ground_truth_available': False,
              'physical_video_telemetry_alignment_verified': False,
              'material_identification_supported': False,
              'anatomy': 'cholecystectomy, not appendectomy', 'channels': {},
              'schema_warnings': []}
    for name, values in [('action', action), ('observation.state', state)]:
        names = info['features'][name].get('names', [])
        if names and len(names) != values.shape[1]:
            report['schema_warnings'].append(f'{name}: {len(names)} names for {values.shape[1]} values; use modality slices, not positional name inference')
    for hand in ('left', 'right'):
        field = modality['action'][hand + '_pose']
        values = action[:, field['start']:field['end']]
        norms = np.linalg.norm(values[:, 3:7], axis=1)
        report['channels'][hand] = {'quaternion_norm_range': [float(norms.min()), float(norms.max())],
            'controller_position_range_m': np.ptp(values[:, :3], axis=0).tolist(),
            'zero_quaternion_rows': int(np.sum(norms < 1e-8))}
    out.mkdir(parents=True, exist_ok=True)
    frames = out / 'frames'
    frames.mkdir(exist_ok=True)
    if any(frames.glob('*.png')):
        raise ValueError('Use a fresh audit output directory')
    stride = int(round(info['fps'] * 3))
    subprocess.run([str(ffmpeg), '-hide_banner', '-loglevel', 'error', '-i',
        str(next(root.glob('videos/**/*.mp4'))), '-vf',
        rf'select=not(mod(n\,{stride})),scale=640:-1', '-vsync', 'vfr', '-frames:v', '12',
        str(frames / '%03d.png')], check=True)
    images = sorted(frames.glob('*.png'))
    if not images:
        raise RuntimeError('Video decode produced no frames')
    sheet = Image.new('RGB', (1280, 390 * ((len(images)+1)//2)), '#101820')
    draw = ImageDraw.Draw(sheet)
    for n, path in enumerate(images):
        im = Image.open(path).convert('RGB')
        x, y = (n % 2)*640, (n//2)*390
        sheet.paste(im, (x, y+30))
        draw.text((x+8,y+7), f'Open-H / CMR | frame {n*stride} | {n*stride/info["fps"]:.1f}s', fill='white')
    sheet.save(out / 'contact_sheet.jpg', quality=92)
    report['decoded_reference_frames'] = len(images)
    report['source_frame_stride'] = stride
    report['status'] = 'audited reference; not simulator validation'
    (out/'audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--ffmpeg', type=Path, default=Path('.cache/bin/ffmpeg.exe'))
    a = p.parse_args()
    audit(a.source, a.out, a.ffmpeg)
