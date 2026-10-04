"""Validate the Hamlyn adapter without treating colorized depth as geometry."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import pyarrow.parquet as pq


def audit(source,out,ffmpeg):
    source,out=Path(source),Path(out)
    provenance=json.loads((source/'provenance.json').read_text())
    if provenance['subset']!='Surgical/hamlyn/tissue_retraction':raise ValueError('Wrong dataset adapter')
    for f in provenance['files']:
        if hashlib.sha256((source/f['path']).read_bytes()).hexdigest()!=f['sha256']:
            raise ValueError('Source file hash mismatch')
    info=json.loads((source/'meta/info.json').read_text())
    table=pq.read_table(next(source.glob('data/**/*.parquet')))
    time=np.asarray(table['timestamp'].to_pylist(),float)
    frames=np.asarray(table['frame_index'].to_pylist(),int)
    if time.ndim!=1 or len(time)<2 or not np.isfinite(time).all():raise ValueError('Invalid timestamp sequence')
    report=dict(repository=provenance['repository'],revision=provenance['revision'],
        subset=provenance['subset'],episode=provenance['episode'],license=provenance['license'],
        rows=len(time),fps=info['fps'],timestamp_span_s=float(time[-1]-time[0]),
        clock_grid_error_s=float(np.max(np.abs(time-time[0]-(frames-frames[0])/info['fps']))),
        monotonic_timestamps=bool(np.all(np.diff(time)>0)),
        contiguous_frame_indices=bool(np.all(np.diff(frames)==1)),
        physical_video_telemetry_alignment_verified=False,arms={},videos={})
    for hand in ('left','right'):
        state=np.asarray(table[f'observation.state.{hand}_arm_cartesian'].to_pylist(),float)
        if state.shape!=(len(time),8) or not np.isfinite(state).all():raise ValueError('Invalid Cartesian state')
        norm=np.linalg.norm(state[:,3:7],axis=1)
        calibration=np.asarray(table[f'observation.meta.{hand}_arm_tpv_cali_mtx'].to_pylist(),float)
        if calibration.shape!=(len(time),7) or not np.isfinite(calibration).all():raise ValueError('Invalid calibration vector')
        report['arms'][hand]=dict(position_range_m=np.ptp(state[:,:3],axis=0).tolist(),
            quaternion_norm_range=[float(norm.min()),float(norm.max())],
            gripper_range=[float(state[:,7].min()),float(state[:,7].max())],
            calibration_vector_shape=list(calibration.shape),
            calibration_max_change=float(np.max(np.abs(calibration-calibration[0]))),
            reference_frame='each PSM base; not a shared world frame',
            camera_extrinsic_direction_verified=False)
    for f in provenance['files']:
        if not f['path'].endswith('.mp4'):continue
        camera=Path(f['path']).parent.name
        run=subprocess.run([str(ffmpeg),'-hide_banner','-loglevel','error','-i',str(source/f['path']),
            '-progress','pipe:1','-f','null','-'],capture_output=True,text=True,check=True)
        counts=[int(line.split('=',1)[1]) for line in run.stdout.splitlines() if line.startswith('frame=')]
        if not counts:raise RuntimeError('No decoded frame count')
        report['videos'][camera]=dict(decoded_frames=counts[-1],matches_telemetry=counts[-1]==len(time),
            declared_shape=info['features'][camera]['shape'])
    report.update(metric_depth_available=False,
        depth_rejection='Depth is a colorized lossy RGB H264 stream. No metric scale/intrinsics verified; do not unproject it.',
        force_ground_truth_available=False,retargeting_ready=False,
        retargeting_blockers=['PSM-base to common-camera transform direction must be verified',
            'Rosota tool-tip offsets and tool-specific jaw normalization must be calibrated'],
        anatomy='ex-vivo porcine tissue on bench; not a human abdominal cavity')
    report['status']='passed integrity audit' if (report['monotonic_timestamps'] and
        report['contiguous_frame_indices'] and report['clock_grid_error_s']<1e-4 and
        all(np.max(np.abs(np.asarray(v['quaternion_norm_range'])-1))<1e-3 and
            v['gripper_range'][0]>=-.001 and v['gripper_range'][1]<=1.001 for v in report['arms'].values()) and
        all(v['matches_telemetry'] for v in report['videos'].values())) else 'failed integrity audit'
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--ffmpeg',type=Path,default=Path('.cache/bin/ffmpeg.exe'))
    a=p.parse_args();audit(a.source,a.out,a.ffmpeg)
