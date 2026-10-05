"""Check the GPU runtime and Warp/PyTorch bridge required by Isaac articulations."""
import argparse
import json
from pathlib import Path
import importlib.metadata as metadata
from packaging.requirements import Requirement
import torch
import warp as wp
wp.config.kernel_cache_dir=str(Path(__file__).resolve().parents[1]/'.cache/warp-runtime-check')

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
r=dict(status='failed',torch_version=torch.__version__,torch_cuda=torch.version.cuda,warp_version=wp.__version__)
try:
    requirement=next(Requirement(x) for x in metadata.requires('isaacsim-core') if Requirement(x).name=='torch')
    r['isaac_torch_requirement']=str(requirement)
    if torch.__version__ not in requirement.specifier:raise RuntimeError('Isaac torch version mismatch')
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    wp.init()
    values=torch.arange(16,device='cuda',dtype=torch.float32)
    if float((values*values).sum())!=1240:raise RuntimeError('GPU arithmetic mismatch')
    view=wp.from_torch(values);roundtrip=wp.to_torch(view)
    if roundtrip.data_ptr()!=values.data_ptr():raise RuntimeError('Warp bridge unexpectedly copied tensor')
    roundtrip[0]=42
    if values[0].item()!=42:raise RuntimeError('Warp bridge lost shared data')
    torch.cuda.synchronize()
    r.update(status='passed',device=torch.cuda.get_device_name(0),cuda_arithmetic=True,warp_torch_shared_buffer=True,
        scope='GPU runtime and array interoperability only; no mechanics or task validation')
except Exception as exc:
    r['error']=str(exc)
    raise
finally:
    a.out.write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8')
print(json.dumps(r,indent=2))
