"""Encode actual recorded RTX views, with an explicit non-RL caption."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import imageio_ffmpeg


def encode(run):
    report=json.loads((run/'report.json').read_text())
    if report['status']!='passed':raise ValueError('Refusing to present failed validation as a verified scan')
    count=report['endoscope_scan']['frames']
    files=[run/'scan_frames'/f'{i:04d}.png' for i in range(count)]
    target=run/'endoscope_scan.mp4'
    writer=imageio_ffmpeg.write_frames(str(target),(1280,752),fps=15,codec='libx264',pix_fmt_in='rgb24',
        output_params=['-movflags','+faststart'])
    writer.send(None)
    previews=[]
    try:
        for file in files:
            frame=Image.new('RGB',(1280,752),(16,19,23))
            frame.paste(Image.open(file).convert('RGB'),(0,32))
            ImageDraw.Draw(frame).text((12,9),'RTX ENDOSCOPE EXPLORATION | Rosota CAD | gravity-settled appendix | camera scan, not RL policy',(240,240,240))
            writer.send(np.asarray(frame))
            previews.append(frame.resize((800,470)))
    finally:writer.close()
    previews[0].save(run/'endoscope_scan.gif',save_all=True,append_images=previews[1:],duration=67,loop=0)
    decoder=imageio_ffmpeg.read_frames(str(target),pix_fmt='rgb24')
    metadata=next(decoder);decoded=sum(1 for _ in decoder)
    if decoded!=count:raise RuntimeError('Video frame count mismatch')
    print(json.dumps({'frames':decoded,'size':metadata['size'],'video':str(target)}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);encode(p.parse_args().run)
