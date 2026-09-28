"""Lock all stream orders using only already fixed, label-free domain manifests."""
import argparse
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parents[1]


def rows(path):
    path = Path(path)
    if not path.is_absolute():
        path = WORKTREE / path
    if path.suffix == '.json':
        data = json.loads(path.read_text())
        return data['records']
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def build(config_path, output):
    config = json.loads(Path(config_path).read_text())
    if output.exists():
        raise FileExistsError(output)
    domains = {}
    for name, meta in config['domains'].items():
        data = rows(meta['runner'])
        ids = [item['sample_id'] for item in data]
        if len(data) != meta['count'] or len(set(ids)) != len(ids):
            raise ValueError(f'{name} coverage/uniqueness mismatch')
        if any('label' in item or 'canonical_label' in item for item in data):
            raise ValueError(f'{name} runner manifest exposes a label')
        domains[name] = sorted(ids)
    output.mkdir(parents=True, exist_ok=False)
    index = []
    for seed in config['order_seeds']:
        for stream_index, (stream_id, segments) in enumerate(config['streams'].items()):
            shuffled = []
            for segment_index, name in enumerate(segments):
                ids = list(domains[name])
                # Stable distinct integer generators for every stream segment.
                random.Random(seed * 1000 + stream_index * 10 + segment_index).shuffle(ids)
                shuffled.append(ids)
            ordered = [{'sample_id': sid, 'domain': name, 'segment': number}
                       for number, name in enumerate(segments)
                       for sid in shuffled[number]]
            expected = sum(config['domains'][name]['count'] for name in segments)
            if len(ordered) != expected or len({item['sample_id'] for item in ordered}) != expected:
                raise ValueError('stream coverage/uniqueness mismatch')
            target = output / f'{stream_id}_order{seed}.jsonl'
            with target.open('x') as handle:
                for item in ordered:
                    handle.write(json.dumps(item) + '\n')
            index.append({'stream': stream_id, 'order_seed': seed, 'count': expected,
                          'segments': segments, 'manifest': target.name})
    (output / 'index.json').write_text(json.dumps(index, indent=2) + '\n')
    print(json.dumps({'status': 'READY', 'streams': len(index)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=HERE / 'config.json')
    parser.add_argument('--output', type=Path, default=HERE / 'streams_locked')
    args = parser.parse_args()
    build(args.config, args.output)
