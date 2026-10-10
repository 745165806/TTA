"""GPU extraction of actual pre-backend SSL-AASIST LL sequences, three audio views."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT = Path('/media/dell/data/fakeAudioDection/TTA')
sys.path.insert(0, str(PROJECT / 'workers/compat'))
sys.path.insert(0, str(PROJECT / 'workers'))
from author_training import build_author_model, load_audio
from baseline_bridge import _views

BASE = PROJECT / '.worktrees/exp-representation-tta/experiments/representation_tta'
WAVEFAKE = PROJECT / '.worktrees/exp-capacity-audit/experiments/capacity_audit/results/wavefake_capacity_cache_20260927a'
PROBE = {'num_views': 3, 'seed': 13, 'noise_snr_db': 30.0, 'fir_side_gain': 0.05}
SOURCES = {
    'fit': (PROJECT / 'data/manifests_v2/asv2019_la/inference/fit.jsonl',
            Path('/media/dell/data/fakedata/asvspoof2019/LA'), 25380),
    'select': (PROJECT / 'data/manifests_v2/asv2019_la/inference/select.jsonl',
               Path('/media/dell/data/fakedata/asvspoof2019/LA'), 11723),
    'itw_target10': (PROJECT / 'experiments/target10_selection/manifests/inwild_target10_select.json',
                     Path('/media/dell/data/fakedata/release_in_the_wild'), 3178),
    'wavefake_dev': (WAVEFAKE / 'manifests/worker_select.jsonl', WAVEFAKE / 'derived_audio', 4096),
}


def rows_for(domain):
    manifest, root, expected = SOURCES[domain]
    if manifest.suffix == '.jsonl':
        with manifest.open() as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
    else:
        document = json.loads(manifest.read_text())
        rows = document['records']
        if domain == 'itw_target10' and document['count'] != expected:
            raise ValueError('ITW assignment count mismatch')
    if len(rows) != expected or len({row['sample_id'] for row in rows}) != expected:
        raise ValueError(f'{domain} selected ID coverage mismatch')
    if any('label' in row or 'canonical_label' in row for row in rows):
        raise ValueError(f'{domain} extraction manifest contains labels')
    if any(type(row.get('sample_index')) is not int or row['sample_index'] < 0 for row in rows):
        raise ValueError('missing stable sample_index')
    role = {'fit':'fit','select':'select','itw_target10':'select','wavefake_dev':'select'}[domain]
    if any(row['split_role'] != role for row in rows):
        raise ValueError(f'{domain} role mismatch')
    return rows, root, manifest


def safe_audio(root, relative):
    root = root.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('audio path missing or escapes approved root')
    return path


def build_model(device):
    finalized = json.loads((PROJECT / 'outputs/source/ssl-aasist-full-asvspoof2019train-random/finalized.json').read_text())
    bundle = json.loads((PROJECT / 'outputs_v2/ssl_aasist/frozen/bundle.json').read_text())
    job = {'source_job': {'model_id': bundle['model_id'], 'initialization': bundle['initialization']},
           'execution': {'architecture': finalized['architecture']}}
    adapter, _ = build_author_model(job, device)
    state = torch.load(PROJECT / 'outputs_v2/ssl_aasist/frozen/detector_state.pt',
                       map_location='cpu', weights_only=True)
    adapter.model.load_state_dict(state['model_state'], strict=True)
    model = adapter.model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, bundle


def run(domain, shard, shards, gpu, run_id, limit, batch_audio):
    if not torch.cuda.is_available() or torch.cuda.device_count() <= gpu:
        raise RuntimeError('visible CUDA device required for SSL waveform extraction')
    if shards < 1 or not 0 <= shard < shards or batch_audio < 1:
        raise ValueError('invalid shard/batch parameters')
    device = torch.device(f'cuda:{gpu}')
    torch.cuda.set_device(device)
    # Match the production frozen extractor's numerical mode exactly.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows, root, manifest = rows_for(domain)
    if limit is not None:
        if limit < 1:
            raise ValueError('bad smoke limit')
        rows = rows[:limit]
    selected = [(i, row) for i, row in enumerate(rows) if i % shards == shard]
    output = BASE / 'results' / run_id / 'll_cache' / domain / f'shard_{shard:02d}'
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    model, bundle = build_model(device)
    ids, chunks = [], []
    for offset in range(0, len(selected), batch_audio):
        entries = selected[offset:offset + batch_audio]
        view_batch = []
        for _, row in entries:
            waveform = torch.from_numpy(load_audio(safe_audio(root, row['audio_relpath'])))
            view_batch.append(_views(waveform, row['sample_index'], PROBE))
        waveforms = torch.stack(view_batch).reshape(-1, 64600).to(device)
        with torch.inference_mode():
            ll = model.LL(model.ssl_model.extract_feat(waveforms))
        if tuple(ll.shape) != (len(entries) * 3, 201, 128) or not torch.isfinite(ll).all():
            raise ValueError(f'earlier-layer tensor shape/value mismatch: {tuple(ll.shape)}')
        array = ll.reshape(len(entries), 3, 201, 128).cpu().numpy().astype(np.float32, copy=False)
        chunk = output / f'chunk_{offset // batch_audio:05d}.npy'
        with chunk.open('xb') as stream:
            np.save(stream, array, allow_pickle=False)
        batch_ids = [row['sample_id'] for _, row in entries]
        ids.extend(batch_ids)
        chunks.append({'array_ref': chunk.name, 'ids': batch_ids, 'count': len(batch_ids)})
        if len(chunks) % 25 == 0:
            print(json.dumps({'domain': domain, 'shard': shard, 'done': len(ids),
                              'total': len(selected), 'elapsed_seconds': time.monotonic()-started}), flush=True)
    if len(ids) != len(selected) or len(set(ids)) != len(ids):
        raise AssertionError('shard ID count/uniqueness failure')
    index = {'status':'READY','run_id':run_id,'domain':domain,'shard':shard,'shards':shards,
             'count':len(ids),'shape_per_sample':[3,201,128],'dtype':'float32',
             'manifest_ref':str(manifest),'source_bundle_ref':str(PROJECT / 'outputs_v2/ssl_aasist/frozen/bundle.json'),
             'source_run_id':bundle['source_run_id'],'view_recipe':PROBE,'sample_ids':ids,
             'chunks':chunks,'gpu':gpu,'elapsed_seconds':time.monotonic()-started,
             'peak_gpu_bytes':torch.cuda.max_memory_allocated(device)}
    with (output / 'index.json').open('x') as stream:
        json.dump(index, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'status':'PASS','domain':domain,'shard':shard,'count':len(ids),
                      'elapsed_seconds':index['elapsed_seconds'],'peak_gpu_bytes':index['peak_gpu_bytes']}),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', choices=tuple(SOURCES), required=True)
    parser.add_argument('--shard', type=int, required=True)
    parser.add_argument('--shards', type=int, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--batch-audio', type=int, default=4)
    args = parser.parse_args()
    run(args.domain, args.shard, args.shards, args.gpu, args.run_id, args.limit, args.batch_audio)
