"""Four-GPU selected-only LA/DF cache extraction with durable shard logs."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main(run_id):
    logdir = HERE / 'results' / run_id / 'extract_logs'
    logdir.mkdir(parents=True, exist_ok=False)
    jobs = []
    for domain in ('LA21', 'DF21'):
        wave = []
        for gpu in range(4):
            command = [sys.executable, str(HERE / 'extract_ll.py'), '--domain', domain,
                       '--shard', str(gpu), '--gpu', str(gpu), '--run-id', run_id]
            log = (logdir / f'{domain}_shard{gpu}.log').open('x')
            process = subprocess.Popen(command, cwd=HERE.parents[1], env={**os.environ,
                                       'PYTHONPATH': 'src:.', 'OMP_NUM_THREADS': '2'},
                                       stdout=log, stderr=subprocess.STDOUT)
            wave.append((process, log, domain, gpu, command))
        for process, log, domain, gpu, command in wave:
            code = process.wait()
            log.close()
            jobs.append({'domain': domain, 'gpu': gpu, 'exit_code': code,
                         'command': command, 'log': str(logdir / f'{domain}_shard{gpu}.log')})
        if any(job['exit_code'] for job in jobs):
            break
    (logdir / 'status.json').write_text(json.dumps(jobs, indent=2) + '\n')
    if len(jobs) != 8 or any(job['exit_code'] for job in jobs):
        raise SystemExit(1)
    print(json.dumps({'status': 'PASS', 'shards': len(jobs), 'run_id': run_id}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    main(args.run_id)
