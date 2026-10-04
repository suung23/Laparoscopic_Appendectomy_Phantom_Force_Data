"""Encode only a passed, actually rendered bimanual acceptance run."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import imageio_ffmpeg


def encode(run):
    report=json.loads((run/'report.json').read_text())
    if report['status']!='passed':raise ValueError('The physical acceptance run did not pass')
    demo=report['bimanual_demo'];count=demo['recorded_frames']
    if not demo['camera_unchanged'] or not demo['released']:raise ValueError('Incomplete demonstration')
    video=run/'bimanual_grasp.mp4'
    writer=imageio_ffmpeg.write_frames(str(video),(1280,720),fps=7.5,codec='libx264',pix_fmt_in='rgb24',output_params=['-movflags','+faststart'])
    writer.send(None);preview=[]
    try:
        for i in range(count):
            frame=Image.open(run/'bimanual_frames'/f'{i:04d}.png').convert('RGB')
            writer.send(np.asarray(frame));preview.append(frame.resize((800,450)))
    finally:writer.close()
    preview[0].save(run/'bimanual_grasp.gif',save_all=True,append_images=preview[1:],duration=133,loop=0)
    decoder=imageio_ffmpeg.read_frames(str(video),pix_fmt='rgb24');metadata=next(decoder)
    decoded=sum(1 for _ in decoder)
    if decoded!=count:raise RuntimeError('Encoded frame mismatch')
    print(json.dumps({'decoded_frames':decoded,'fps':metadata['fps'],'video':str(video)}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);encode(p.parse_args().run)
