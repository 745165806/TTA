"""First-prediction online stream runner; never imports evaluator labels."""
import argparse
import itertools
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch

from cache_reader import SelectedLL
from methods import OnlineMethod

HERE = Path(__file__).resolve().parent


def stream_batches(path, size, limit=None):
    with path.open() as handle:
        lines = (json.loads(line) for line in handle if line.strip())
        if limit is not None:
            lines = itertools.islice(lines, limit)
        while True:
            batch = list(itertools.islice(lines, size))
            if not batch:
                return
            if any(set(row) != {'sample_id', 'domain', 'segment'} for row in batch):
                raise ValueError('stream record contract mismatch')
            yield batch


def run(args):
    config = json.loads(Path(args.config).read_text())
    if args.method not in config['methods'] or args.stream not in config['streams'] or args.order_seed not in config['order_seeds']:
        raise ValueError('unlocked method/stream/order')
    if args.run_id != config['run_id']:
        raise ValueError('run ID must match locked config')
    if not torch.cuda.is_available() or torch.cuda.device_count() <= args.gpu:
        raise RuntimeError('visible authorized CUDA GPU required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)
    tag = f'{args.method}_{args.stream}_order{args.order_seed}'
    if args.limit is not None and not args.smoke_id:
        raise ValueError('smoke run requires a unique --smoke-id')
    output = (HERE / 'results' / args.run_id / 'smoke' / args.smoke_id / tag) if args.limit is not None else (HERE / 'results' / args.run_id / 'runs' / tag / args.attempt_id)
    output.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, str(Path(__file__).resolve()), '--config', str(args.config),
               '--method', args.method, '--stream', args.stream, '--order-seed', str(args.order_seed),
               '--gpu', str(args.gpu), '--run-id', args.run_id]
    if args.limit is not None:
        command += ['--limit', str(args.limit), '--smoke-id', args.smoke_id]
    else:
        command += ['--attempt-id', args.attempt_id]
    (output / 'command.json').write_text(json.dumps({'argv': command, 'cwd': os.getcwd()}, indent=2) + '\n')
    (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    (output / 'status.json').write_text(json.dumps({'status': 'RUNNING', 'count': 0}) + '\n')
    start = time.monotonic()
    count = 0
    ids_seen = set()
    cuda_seconds = 0.0
    try:
        cache = SelectedLL(config, args.run_id, config['streams'][args.stream])
        source_state = None
        if args.method == 'eata':
            source_ref = HERE / 'results' / args.run_id / 'source_only' / config['source_setup_id'] / 'eata.pt'
            source_state = torch.load(source_ref, map_location='cpu', weights_only=True)
        method = OnlineMethod(args.method, device, config, source_state)
        stream = Path(config['stream_manifest_root']) / f'{args.stream}_order{args.order_seed}.jsonl'
        with (output / 'scores.jsonl').open('x') as scores:
            for number, rows in enumerate(stream_batches(stream, config['batch_size'], args.limit)):
                ids = [row['sample_id'] for row in rows]
                if any(sid in ids_seen for sid in ids) or len(set(ids)) != len(ids):
                    raise ValueError('duplicate stream ID')
                ids_seen.update(ids)
                all_views = cache.batch(ids, all_views=True).to(device) if args.method == 'rotta' else None
                features = all_views[:, 0] if all_views is not None else cache.batch(ids).to(device)
                torch.cuda.synchronize(device)
                t0 = time.monotonic()
                first_scores, first_logits = method.predict(features)
                torch.cuda.synchronize(device)
                prediction_done = time.monotonic()
                if not torch.isfinite(first_scores).all():
                    raise ValueError('nonfinite first online score')
                # Durably persist this batch before any adaptation on its features.
                for row, value in zip(rows, first_scores.tolist()):
                    scores.write(json.dumps({'sample_id': row['sample_id'], 'score': float(value),
                                             'position': count, 'segment': row['segment'],
                                             'batch': number}) + '\n')
                    count += 1
                scores.flush()
                os.fsync(scores.fileno())
                adaptation_start = time.monotonic()
                method.adapt(all_views if all_views is not None else features, first_logits)
                torch.cuda.synchronize(device)
                cuda_seconds += prediction_done - t0 + time.monotonic() - adaptation_start
                if (number + 1) % 25 == 0:
                    (output / 'status.json').write_text(json.dumps({'status': 'RUNNING', 'count': count,
                        'batches': number + 1, **method.stats()}) + '\n')
                    print(json.dumps({'tag': tag, 'count': count, 'batches': number + 1}), flush=True)
        expected = args.limit if args.limit is not None else sum(config['domains'][domain]['count']
                                                            for domain in config['streams'][args.stream])
        if count != expected or len(ids_seen) != expected:
            raise ValueError(f'stream exact coverage failed: {count}/{expected}')
        result = {'status': 'SMOKE_PASS' if args.limit is not None else 'COMPLETE',
                  'count': count, 'expected': expected, 'elapsed_seconds': time.monotonic()-start,
                  'backend_cuda_seconds': cuda_seconds, 'ms_per_audio_backend': cuda_seconds*1000/count,
                  'peak_gpu_mb': torch.cuda.max_memory_allocated(device)/1048576,
                  'gpu': args.gpu, 'method': args.method, 'stream': args.stream,
                  'order_seed': args.order_seed, **method.stats()}
        (output / 'status.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
    except Exception as error:
        failure = {'status': 'FAILED', 'count': count, 'error': repr(error),
                   'traceback': traceback.format_exc(), 'method': args.method,
                   'stream': args.stream, 'order_seed': args.order_seed}
        (output / 'status.json').write_text(json.dumps(failure, indent=2) + '\n')
        print(json.dumps(failure), flush=True)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=HERE / 'config.json')
    parser.add_argument('--method', required=True)
    parser.add_argument('--stream', required=True)
    parser.add_argument('--order-seed', type=int, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--smoke-id')
    parser.add_argument('--attempt-id', default='attempt01')
    run(parser.parse_args())
