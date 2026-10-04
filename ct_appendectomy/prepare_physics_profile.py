"""Create a separate higher-precision profile without modifying source assets."""
import argparse
import hashlib
import json
from pathlib import Path
from pxr import Usd,Sdf


def prepare(source,output,hz,iterations):
    if source.resolve()==output.resolve():raise ValueError('Use a separate output directory')
    output.mkdir(parents=True,exist_ok=True)
    config=json.loads((source/'bimanual_config.json').read_text())
    original=source/'attached.usda'
    if hashlib.sha256(original.read_bytes()).hexdigest()!=config['stage_sha256']:raise ValueError('Source hash mismatch')
    stage=Usd.Stage.Open(str(original.resolve()))
    prim=stage.GetPrimAtPath('/World/Atlas/appendix')
    prim.CreateAttribute('physxDeformableBody:solverPositionIterationCount',Sdf.ValueTypeNames.UInt).Set(iterations)
    destination=output/'attached.usda'
    stage.GetRootLayer().Export(str(destination.resolve()))
    config.update(stage_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),physics_dt=1/hz,
        precision_profile={'physics_hz':hz,'solver_position_iterations':iterations,
            'source_stage_sha256':hashlib.sha256(original.read_bytes()).hexdigest()})
    (output/'bimanual_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--hz',type=int,choices=(480,960,1920),default=960)
    p.add_argument('--iterations',type=int,default=128)
    a=p.parse_args()
    if not 1<=a.iterations<=255:p.error('Iterations must be 1..255')
    prepare(a.source,a.out,a.hz,a.iterations)
