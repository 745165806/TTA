"""Launch one independent GPU worker per stable-ID shard, preserving each log."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
DOMAINS = ('fit', 'select', 'itw_target10', 'wavefake_dev')


def run(run_id, shards=4):
    if shards != 4:
        raise ValueError('first run is fixed to four GPUs and four shards')
    output = HERE / 'results' / run_id
    output.mkdir(parents=True, exist_ok=False)
    (output / 'logs').mkdir()
    configuration = {'run_id': run_id, 'python': sys.executable,
                     'domains': DOMAINS, 'shards': shards, 'gpu_per_shard': list(range(shards)),
                     'batch_audio': 4, 'purpose': 'actual pre-backend 201x128 waveform SSL features'}
    (output / 'extraction_config.json').write_text(json.dumps(configuration, indent=2) + '\n')
    started = time.monotonic()
    failures = []
    for domain in DOMAINS:
        processes = []
        for shard in range(shards):
            command = [sys.executable, str(HERE / 'extract_ll.py'), '--domain', domain,
                       '--shard', str(shard), '--shards', str(shards), '--gpu', str(shard),
                       '--run-id', run_id, '--batch-audio', '4']
            log_ref = output / 'logs' / f'{domain}_shard{shard}.log'
            stream = log_ref.open('x')
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                       cwd=HERE.parents[1])
            processes.append((shard, process, stream, command, log_ref))
        for shard, process, stream, command, log_ref in processes:
            exit_code = process.wait()
            stream.close()
            item = {'domain': domain, 'shard': shard, 'exit_code': exit_code,
                    'command': command, 'log_ref': str(log_ref)}
            print(json.dumps(item), flush=True)
            if exit_code:
                failures.append(item)
        if failures:
            break
    summary = {'status': 'FAIL' if failures else 'PASS', 'run_id': run_id,
               'elapsed_seconds': time.monotonic()-started, 'failures': failures}
    (output / 'extraction_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)
    if failures:
        raise RuntimeError('one or more SSL extraction shards failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    run(args.run_id)
