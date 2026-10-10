"""Four-GPU method queues with a hard order2026 stage barrier."""
import argparse
import concurrent.futures
import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parents[1]
ASSIGNMENTS = {0: ('tent',), 1: ('eata',), 2: ('rotta',),
               3: ('frozen', 'bn_only', 'lame')}


def complete(root, method, stream, seed):
    directory = root / 'runs' / f'{method}_{stream}_order{seed}'
    if not directory.exists():
        return False
    return any((attempt / 'status.json').exists() and
               json.loads((attempt / 'status.json').read_text()).get('status') == 'COMPLETE'
               for attempt in directory.iterdir() if attempt.is_dir())


def next_attempt(root, method, stream, seed):
    directory = root / 'runs' / f'{method}_{stream}_order{seed}'
    for number in range(1, 100):
        attempt = f'attempt{number:02d}'
        if not (directory / attempt).exists():
            return attempt
    raise RuntimeError('too many attempts')


def worker(gpu, methods, seeds, config, root, logdir, events):
    records = []
    for seed in seeds:
        for stream in config['streams']:
            for method in methods:
                if complete(root, method, stream, seed):
                    records.append({'method': method, 'stream': stream, 'seed': seed,
                                    'status': 'SKIPPED_COMPLETE'})
                    continue
                attempt = next_attempt(root, method, stream, seed)
                tag = f'{method}_{stream}_order{seed}_{attempt}'
                command = [sys.executable, str(HERE / 'run.py'), '--config', str(HERE / 'config.json'),
                           '--method', method, '--stream', stream, '--order-seed', str(seed),
                           '--gpu', str(gpu), '--run-id', config['run_id'], '--attempt-id', attempt]
                logpath = logdir / f'{tag}.log'
                with logpath.open('x') as log:
                    result = subprocess.run(command, cwd=WORKTREE, stdout=log, stderr=subprocess.STDOUT,
                                            env={**os.environ, 'PYTHONPATH': 'src:.',
                                                 'OMP_NUM_THREADS': '2', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'})
                record = {'method': method, 'stream': stream, 'seed': seed, 'gpu': gpu,
                          'attempt': attempt, 'exit_code': result.returncode,
                          'status': 'COMPLETE' if complete(root, method, stream, seed) else 'FAILED',
                          'command': command, 'log': str(logpath)}
                records.append(record)
                with events.open('a') as output:
                    output.write(json.dumps(record) + '\n')
                print(json.dumps(record), flush=True)
    return records


def dynamic_worker(gpu, tasks, config, root, logdir, events, write_lock):
    records = []
    while True:
        try:
            method, stream, seed = tasks.get_nowait()
        except queue.Empty:
            return records
        if complete(root, method, stream, seed):
            records.append({'method': method, 'stream': stream, 'seed': seed,
                            'status': 'SKIPPED_COMPLETE'})
            tasks.task_done()
            continue
        attempt = next_attempt(root, method, stream, seed)
        tag = f'{method}_{stream}_order{seed}_{attempt}'
        command = [sys.executable, str(HERE / 'run.py'), '--config', str(HERE / 'config.json'),
                   '--method', method, '--stream', stream, '--order-seed', str(seed),
                   '--gpu', str(gpu), '--run-id', config['run_id'], '--attempt-id', attempt]
        logpath = logdir / f'{tag}.log'
        with logpath.open('x') as log:
            result = subprocess.run(command, cwd=WORKTREE, stdout=log, stderr=subprocess.STDOUT,
                                    env={**os.environ, 'PYTHONPATH': 'src:.',
                                         'OMP_NUM_THREADS': '2', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'})
        record = {'method': method, 'stream': stream, 'seed': seed, 'gpu': gpu,
                  'attempt': attempt, 'exit_code': result.returncode,
                  'status': 'COMPLETE' if complete(root, method, stream, seed) else 'FAILED',
                  'command': command, 'log': str(logpath)}
        records.append(record)
        with write_lock:
            with events.open('a') as output:
                output.write(json.dumps(record) + '\n')
            print(json.dumps(record), flush=True)
        tasks.task_done()


def main(phase):
    config = json.loads((HERE / 'config.json').read_text())
    root = HERE / 'results' / config['run_id']
    logdir = root / 'run_logs'
    logdir.mkdir(parents=True, exist_ok=True)
    events = root / 'launch_events.jsonl'
    if phase == 'order2026':
        seeds = [2026]
    elif phase == 'later':
        missing = [(method, stream) for method in config['methods'] for stream in config['streams']
                   if not complete(root, method, stream, 2026)]
        if missing:
            raise RuntimeError(f'order2026 stage barrier: {len(missing)} runs not COMPLETE')
        seeds = [2027, 2028, 2029, 2030]
    else:
        raise ValueError(phase)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        if phase == 'later':
            tasks = queue.Queue()
            for seed in seeds:
                for stream in config['streams']:
                    for method in config['methods']:
                        tasks.put((method, stream, seed))
            lock = threading.Lock()
            futures = [pool.submit(dynamic_worker, gpu, tasks, config, root, logdir, events, lock)
                       for gpu in range(4)]
        else:
            futures = [pool.submit(worker, gpu, methods, seeds, config, root, logdir, events)
                       for gpu, methods in ASSIGNMENTS.items()]
        records = [record for future in futures for record in future.result()]
    summary = {'phase': phase, 'tasks': len(records),
               'complete': sum(record['status'] in ('COMPLETE', 'SKIPPED_COMPLETE') for record in records),
               'failed': [record for record in records if record['status'] == 'FAILED']}
    (root / f'launch_{phase}.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({key: value if key != 'failed' else len(value) for key, value in summary.items()}), flush=True)
    if summary['failed']:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('order2026', 'later'), required=True)
    args = parser.parse_args()
    main(args.phase)
