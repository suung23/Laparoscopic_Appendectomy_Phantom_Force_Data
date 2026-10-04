"""Encode and fully decode-check a passed exploration recording."""
import argparse
import json
from pathlib import Path
from PIL import Image
import imageio_ffmpeg as ff


def encode(folder,allow_failed=False):
    report=json.loads((folder/'report.json').read_text())
    if report['status']!='passed' and not (allow_failed and report['status']=='failed' and 'frames' in report):
        raise ValueError('Failed evaluation requires explicit --allow-failed')
    frames=[Image.open(p).convert('RGB') for p in sorted((folder/'frames').glob('*.png'))]
    if len(frames)!=report['frames']:raise ValueError('Recording frame count mismatch')
    writer=ff.write_frames(str(folder/'exploration.mp4'),frames[0].size,fps=report['fps'],
        codec='libx264',macro_block_size=2,output_params=['-movflags','+faststart'])
    writer.send(None)
    for frame in frames:writer.send(frame.tobytes())
    writer.close()
    frames[0].save(folder/'exploration.gif',save_all=True,append_images=frames[1:],
        duration=round(1000/report['fps']),loop=0)
    reader=ff.read_frames(str(folder/'exploration.mp4'));metadata=next(reader)
    count=sum(1 for _ in reader)
    if count!=len(frames):raise RuntimeError('Encoded video frame loss')
    (folder/'video_validation.json').write_text(json.dumps({'status':'passed','decoded_frames':count,
        'size':metadata['size'],'fps':metadata['fps'],'nominal_sampling_fps':report['fps'],
        'simulated_seconds':report.get('simulated_seconds',report['steps']/report.get('control_hz',60)),
        'policy':report['policy'],'learned_policy':report['learned_policy'],
        'evaluation_status':report['status']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path)
    p.add_argument('--allow-failed',action='store_true')
    a=p.parse_args();encode(a.folder,a.allow_failed)
