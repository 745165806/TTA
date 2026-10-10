"""Extract only fixed LA/DF development IDs into new three-view LL shards."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parents[1]
sys.path.insert(0, str(WORKTREE / 'experiments/representation_tta'))
from extract_ll import PROBE, build_model, safe_audio  # noqa: E402
from author_training import load_audio  # noqa: E402
from baseline_bridge import _views  # noqa: E402


def run(domain, shard, gpu, run_id, limit=None, shards=4, batch_audio=4):
    if domain not in ('LA21', 'DF21') or not 0 <= shard < shards:
        raise ValueError('unsupported domain/shard')
    config = json.loads((HERE / 'config.json').read_text())
    metadata = config['domains'][domain]
    manifest = WORKTREE / metadata['runner']
    with manifest.open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) != 4096 or len({row['sample_id'] for row in rows}) != 4096:
        raise ValueError('fixed subset coverage mismatch')
    if any(set(row) != {'sample_id', 'sample_index', 'audio_relpath', 'root_key'} for row in rows):
        raise ValueError('runner field contract mismatch')
    if limit is not None:
        rows = rows[:limit]
    selected = [(index, row) for index, row in enumerate(rows) if index % shards == shard]
    output = HERE / 'results' / run_id / 'll_cache' / domain / f'shard_{shard:02d}'
    output.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available() or torch.cuda.device_count() <= gpu:
        raise RuntimeError('visible CUDA GPU required')
    device = torch.device(f'cuda:{gpu}')
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(2)
    start = time.monotonic()
    model, bundle = build_model(device)
    root = Path(metadata['audio_root'])
    ids, chunks = [], []
    for offset in range(0, len(selected), batch_audio):
        entries = selected[offset:offset + batch_audio]
        audio = [torch.from_numpy(load_audio(safe_audio(root, row['audio_relpath']))) for _, row in entries]
        views = torch.stack([_views(waveform, row['sample_index'], PROBE)
                             for waveform, (_, row) in zip(audio, entries)]).reshape(-1, 64600).to(device)
        with torch.inference_mode():
            ll = model.LL(model.ssl_model.extract_feat(views))
        if tuple(ll.shape) != (len(entries) * 3, 201, 128) or not torch.isfinite(ll).all():
            raise ValueError('LL shape/finite check failed')
        array = ll.reshape(len(entries), 3, 201, 128).cpu().numpy().astype(np.float32, copy=False)
        chunk = output / f'chunk_{offset // batch_audio:05d}.npy'
        with chunk.open('xb') as handle:
            np.save(handle, array, allow_pickle=False)
        these_ids = [row['sample_id'] for _, row in entries]
        ids.extend(these_ids)
        chunks.append({'array_ref': chunk.name, 'ids': these_ids, 'count': len(these_ids)})
        if len(chunks) % 25 == 0:
            print(json.dumps({'domain': domain, 'shard': shard, 'done': len(ids),
                              'total': len(selected), 'seconds': time.monotonic()-start}), flush=True)
    index = {'status': 'READY', 'run_id': run_id, 'domain': domain,
             'shard': shard, 'shards': shards, 'count': len(ids),
             'sample_ids': ids, 'chunks': chunks, 'shape_per_sample': [3, 201, 128],
             'dtype': 'float32', 'manifest_ref': str(manifest), 'view_recipe': PROBE,
             'source_bundle_ref': config['source_bundle'], 'source_run_id': bundle['source_run_id'],
             'gpu': gpu, 'elapsed_seconds': time.monotonic()-start,
             'peak_gpu_bytes': torch.cuda.max_memory_allocated(device)}
    with (output / 'index.json').open('x') as handle:
        json.dump(index, handle, indent=2)
        handle.write('\n')
    print(json.dumps({'status': 'PASS', 'domain': domain, 'shard': shard,
                      'count': len(ids), 'seconds': index['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', choices=('LA21', 'DF21'), required=True)
    parser.add_argument('--shard', type=int, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--shards', type=int, default=4)
    parser.add_argument('--batch-audio', type=int, default=4)
    args = parser.parse_args()
    run(args.domain, args.shard, args.gpu, args.run_id, args.limit, args.shards, args.batch_audio)
