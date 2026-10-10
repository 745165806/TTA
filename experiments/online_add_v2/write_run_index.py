"""Write the 180-row human-readable run status index from durable attempts."""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    config = json.loads((HERE / 'config.json').read_text())
    root = HERE / 'results' / config['run_id']
    events_path = root / 'launch_events.jsonl'
    events = {}
    if events_path.exists():
        with events_path.open() as handle:
            for line in handle:
                row = json.loads(line)
                events[(row['method'], row['stream'], row['seed'], row['attempt'])] = row
    rows = []
    for order in config['order_seeds']:
        for stream in config['streams']:
            for method in config['methods']:
                directory = root / 'runs' / f'{method}_{stream}_order{order}'
                attempts = sorted(path for path in directory.iterdir() if path.is_dir()) if directory.exists() else []
                chosen = None
                for path in attempts:
                    status_path = path / 'status.json'
                    if status_path.exists():
                        status = json.loads(status_path.read_text())
                        if status['status'] == 'COMPLETE':
                            chosen = (path, status)
                            break
                        chosen = (path, status)
                if chosen is None:
                    rows.append((method, stream, order, 'PENDING', '', '', ''))
                    continue
                path, status = chosen
                event = events.get((method, stream, order, path.name), {})
                state = status['status']
                if state not in ('RUNNING', 'COMPLETE', 'FAILED', 'BLOCKED'):
                    raise ValueError(f'unexpected run status: {state}')
                rows.append((method, stream, order, state, str(path),
                             event.get('exit_code', ''), status.get('error', '')))
    if len(rows) != 180:
        raise AssertionError('run index must have 180 rows')
    with (root / 'run_index.csv').open('w', newline='') as handle:
        writer = csv.writer(handle, lineterminator='\n')
        writer.writerow(('method', 'stream', 'order', 'status', 'result_dir', 'exit_code', 'reason'))
        writer.writerows(rows)
    totals = {state: sum(row[3] == state for row in rows)
              for state in ('PENDING', 'RUNNING', 'COMPLETE', 'FAILED', 'BLOCKED')}
    print(json.dumps(totals))


if __name__ == '__main__':
    main()
