"""Fetch a bounded, revision-pinned clinical reference; never upload local data."""
import argparse
import hashlib
import json
from pathlib import Path
import requests

REPO = 'nvidia/PhysicalAI-Robotics-Open-H-Embodiment'
SUBSET = 'Surgical/cmr_surgical/cholecystectomy'
BASE = 'https://huggingface.co'


def fetch(output, episode=0, max_bytes=150_000_000):
    if not 0 <= episode < 4792:
        raise ValueError('Episode outside documented subset')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    response = session.get(f'{BASE}/api/datasets/{REPO}', timeout=30)
    response.raise_for_status()
    revision = response.json()['sha']
    chunk = episode // 1000
    paths = ['README.md', f'{SUBSET}/README.md', f'{SUBSET}/meta/info.json',
             f'{SUBSET}/meta/modality.json',
             f'{SUBSET}/data/chunk-{chunk:03d}/episode_{episode:06d}.parquet',
             f'{SUBSET}/videos/chunk-{chunk:03d}/observation.images.endoscope/episode_{episode:06d}.mp4']
    records, total = [], 0
    for remote in paths:
        relative = remote.removeprefix(SUBSET + '/') if remote != 'README.md' else 'DATASET_CARD.md'
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + '.partial')
        digest, size = hashlib.sha256(), 0
        try:
            with session.get(f'{BASE}/datasets/{REPO}/resolve/{revision}/{remote}',
                             stream=True, timeout=(30, 60)) as stream:
                stream.raise_for_status()
                with temporary.open('wb') as handle:
                    for block in stream.iter_content(1024 * 1024):
                        total += len(block)
                        size += len(block)
                        if total > max_bytes:
                            raise ValueError('Reference download byte budget exceeded')
                        handle.write(block)
                        digest.update(block)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        records.append({'path': relative, 'repository_path': remote,
                        'sha256': digest.hexdigest(), 'bytes': size})
        print(f'Downloaded {relative}: {size} bytes', flush=True)
    manifest = {'schema_version': 1, 'repository': REPO, 'revision': revision,
                'subset': SUBSET, 'episode': episode, 'license': 'CC-BY-4.0',
                'attribution': 'CMR Surgical / Open-H-Embodiment, hosted by NVIDIA',
                'source': f'{BASE}/datasets/{REPO}', 'files': records,
                'use': 'Clinical cholecystectomy appearance and motion reference; not appendix anatomy or material ground truth',
                'split': 'development-reference; exclude this surgery from held-out evaluation'}
    (output / 'provenance.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--episode', type=int, default=0)
    args = parser.parse_args()
    fetch(args.out, args.episode)
