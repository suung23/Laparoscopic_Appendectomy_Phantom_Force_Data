"""Run pinned upstream entry points unchanged, with a Windows/D3D12 boot adapter."""
import argparse
import os
from pathlib import Path
import runpy
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task',choices=['build-tissue','build-cell','rollout'])
    a,rest=parser.parse_known_args()
    scripts={'build-tissue':'build_deformable_usd.py','build-cell':'build_surgical_cell.py','rollout':'vla_rollout.py'}
    os.chdir(ROOT)
    os.environ.setdefault('WARP_CACHE_PATH',str(ROOT/'.cache/isaac-warp'))
    os.environ.setdefault('OPTIX_CACHE_PATH',str(ROOT/'.cache/optix'))
    import isaacsim
    original=isaacsim.SimulationApp
    def boot(config=None,*args,**kwargs):
        config=dict(config or {})
        runtime=ROOT/'.cache/kit'
        tokens={k:runtime/k for k in ('shared_documents','app_documents','omni_documents','omni_config','omni_logs','cache','omni_cache','omni_global_cache')}
        for p in tokens.values():p.mkdir(parents=True,exist_ok=True)
        extra=['--/app/vulkan=false','--/app/d3d12=true','--/telemetry/enableNVDF=false',
            '--/telemetry/enableSentry=false','--/telemetry/useOpenEndpoint=false',
            '--/privacy/extraDiagnosticDataOptIn=false','--/crashreporter/enabled=false']
        config['extra_args']=list(config.get('extra_args',[]))+extra+[f'--/app/tokens/{k}={v.as_posix()}' for k,v in tokens.items()]
        app=original(config,*args,**kwargs)
        # Observe throughput without changing the controller, dt or scene.
        # Long first-run calibration otherwise emits no progress at all.
        from isaacsim.core.api import SimulationContext
        step=SimulationContext.step
        ticks=0;started=time.perf_counter()
        def observed_step(context,*step_args,**step_kwargs):
            nonlocal ticks
            result=step(context,*step_args,**step_kwargs)
            ticks+=1
            if ticks%1000==0:
                wall=time.perf_counter()-started
                print(f'[runtime observer] {ticks} physics steps, {wall:.1f}s wall, {ticks/wall:.1f} steps/s',flush=True)
            return result
        SimulationContext.step=observed_step
        return app
    isaacsim.SimulationApp=boot
    target=ROOT/'isaac'/scripts[a.task]
    sys.path.insert(0,str(ROOT/'isaac'))
    sys.argv=[str(target)]+rest
    runpy.run_path(str(target),run_name='__main__')


if __name__=='__main__':main()
