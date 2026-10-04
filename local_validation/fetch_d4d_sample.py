"""Bounded streaming sample from the official D4D archive, without unpacking it all."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
import requests

API='https://opara.zih.tu-dresden.de/server/api/core/bitstreams/'
ARCHIVE='96efe821-1088-4cdf-a07d-7d4a72f9f87a'


class BudgetReader:
    def __init__(self,raw,limit):self.raw,self.limit,self.read_bytes=raw,limit,0
    def read(self,n=-1):
        if self.read_bytes>=self.limit:raise ValueError('Compressed download budget reached')
        n=min(n if n>=0 else self.limit, self.limit-self.read_bytes)
        value=self.raw.read(n)
        self.read_bytes+=len(value)
        return value


def fetch(out, budget_mb=128):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    metadata=requests.get(API+ARCHIVE,timeout=30);metadata.raise_for_status()
    meta=metadata.json()
    records, scanned, used=[],[],0
    status='stream ended'
    with requests.get(API+ARCHIVE+'/content',stream=True,timeout=(30,60)) as response:
        response.raise_for_status()
        reader=BudgetReader(response.raw,budget_mb*1024*1024)
        try:
            with tarfile.open(fileobj=reader,mode='r|gz') as archive:
                for member in archive:
                    if not member.isfile():continue
                    p=PurePosixPath(member.name)
                    if p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe archive path')
                    if len(scanned)<40:scanned.append({'path':str(p),'bytes':member.size})
                    selected=(p.suffix.lower() in ('.ply','.yaml','.json','.txt') or
                              (p.suffix.lower()=='.png' and sum(x['path'].endswith('.png') for x in records)<4))
                    if not selected or member.size>64*1024*1024:continue
                    if used+member.size>128*1024*1024:
                        status='Extracted byte budget reached';break
                    data=archive.extractfile(member).read()
                    target=out.joinpath(*p.parts)
                    if not target.resolve().is_relative_to(out.resolve()):raise ValueError('Unsafe output path')
                    target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
                    used+=len(data)
                    records.append({'path':str(p),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
                    print(f'Sampled {p.name}: {len(data)} bytes',flush=True)
                    if sum(x['path'].endswith('.ply') for x in records)>=2:
                        status='Two pointcloud files sampled; correspondence and calibration not yet verified';break
        except ValueError as error:
            if str(error)!='Compressed download budget reached':raise
            status=str(error)
    report={'doi':'10.25532/OPARA-1033','license':'CC-BY-4.0',
            'archive_id':ARCHIVE,'archive_name':meta['name'],
            'archive_checksum_advertised':meta.get('checkSum'),
            'complete_archive_checksum_verified':False,
            'compressed_bytes_read':reader.read_bytes,'files':records,
            'first_scanned_members':scanned,'status':status,
            'scope':'Partial development sample; not material force ground truth or validated registration'}
    (out/'sample_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(status,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--budget-mb',type=int,default=128)
    a=p.parse_args()
    if not 1<=a.budget_mb<=1024:p.error('Budget must be 1..1024 MiB')
    fetch(a.out,a.budget_mb)
