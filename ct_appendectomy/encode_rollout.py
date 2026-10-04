"""Encode actual recorded RTX frames; labels distinguish PPO from a baseline."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import imageio_ffmpeg


def load_run(directory):
    report=json.loads((directory/'report.json').read_text(encoding='utf-8'))
    if report['status']!='passed':raise ValueError('Recording did not finish')
    frames=directory/report['recordings'][0]
    recording=json.loads((frames/'recording.json').read_text(encoding='utf-8'))
    if recording['capture_every_actions']!=2:raise ValueError('Comparison expects a 15 FPS capture')
    return report,frames,recording


def encode(left,right,output):
    output.mkdir(parents=True,exist_ok=True)
    runs=[load_run(left),load_run(right)]
    if [run[2]['policy'] for run in runs]!=['checkpoint','scripted']:
        raise ValueError('Expected a learned-policy recording and a scripted baseline')
    font_path=Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts/segoeui.ttf'
    font=ImageFont.truetype(str(font_path),17)
    small=ImageFont.truetype(str(font_path),14)
    length=max(len(run[2]['frames']) for run in runs)
    video=output/'rl_rollout_comparison.mp4'
    writer=imageio_ffmpeg.write_frames(str(video),(1280,448),fps=15,codec='libx264',
        pix_fmt_in='rgb24',pix_fmt_out='yuv420p',quality=8,
        output_params=['-movflags','+faststart'])
    writer.send(None)
    gif_frames=[]
    try:
        for i in range(length+30):
            canvas=Image.new('RGB',(1280,448),(15,20,29))
            draw=ImageDraw.Draw(canvas)
            for column,(report,directory,recording) in enumerate(runs):
                index=min(i,len(recording['frames'])-1)
                item=recording['frames'][index]
                raw=Image.open(directory/item['file']).convert('RGB')
                if raw.size!=(640,360):raise ValueError('Unexpected RTX frame size')
                x=column*640
                canvas.paste(raw,(x,88))
                title=(f"Learned PPO | {report['trained_steps_before']:,} RL steps | checkpoint" if column==0 else
                       'Scripted baseline | NOT a learned policy')
                draw.text((x+12,7),title,font=font,fill=(238,241,245))
                finished=i>=len(recording['frames'])-1
                status=('SUCCESS' if recording['result']['success'] else 'TIME LIMIT - NO SUCCESS') if finished else 'RUNNING'
                color=(107,230,159) if status=='SUCCESS' else ((255,156,137) if finished else (151,190,245))
                draw.text((x+12,34),f"{status} | t={item['simulation_time_s']:.2f}s | lift={item['lift_m']*1000:.1f} mm",font=font,fill=color)
                draw.text((x+12,61),'Actual Isaac/PhysX frames | contact-lift task | final frame held after episode',font=small,fill=(178,187,202))
            draw.line((640,0,640,448),fill=(225,230,235),width=2)
            writer.send(np.asarray(canvas))
            if i%2==0:gif_frames.append(canvas.resize((960,336),Image.Resampling.LANCZOS))
    finally:writer.close()
    gif=output/'rl_rollout_comparison.gif'
    gif_frames[0].save(gif,save_all=True,append_images=gif_frames[1:],duration=133,loop=0,optimize=False)
    manifest={'source':'actual Isaac Sim RTX frames, freshly rolled out from saved policies',
        'video_fps':15,'video_frames':length+30,'duration_s':(length+30)/15,
        'initialization_note':'Consult the associated training report; replay is not evidence of learning from scratch.',
        'full_appendectomy':False,'end_frames_held':True,
        'runs':[{'policy':r[2]['policy'],'frame_count':len(r[2]['frames']),'result':r[2]['result'],
                 'recording_sha256':hashlib.sha256((r[1]/'recording.json').read_bytes()).hexdigest()} for r in runs],
        'video_sha256':hashlib.sha256(video.read_bytes()).hexdigest()}
    (output/'video_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ppo',type=Path,required=True)
    parser.add_argument('--scripted',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    encode(args.ppo,args.scripted,args.out)
