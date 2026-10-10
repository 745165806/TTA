"""Fix the two ASVspoof 2021 development cohorts from official CM keys once."""
import argparse
import json
import random
import tarfile
from pathlib import Path

PROJECT = Path('/media/dell/data/fakeAudioDection/TTA')
HERE = Path(__file__).resolve().parent
KEYS = Path('/media/dell/data/fakedata/asvspoof2021')


def records(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def official_lines(domain):
    if domain == 'la':
        return (KEYS / 'LA-keys-full/keys/LA/CM/trial_metadata.txt').read_text().splitlines()
    with tarfile.open(KEYS / 'DF-keys-full.tar.gz', 'r:gz') as archive:
        entry = archive.extractfile('keys/DF/CM/trial_metadata.txt')
        if entry is None:
            raise ValueError('DF official CM key absent')
        return entry.read().decode().splitlines()


def build(domain, output, seed=2026):
    if output.exists():
        raise FileExistsError(f'fixed subset already exists: {output}')
    source = PROJECT / f'data/manifests_v2/asv2021_{domain}_eval/inference/target_test.jsonl'
    raw = records(source)
    by_id = {row['sample_id']: row for row in raw}
    if len(by_id) != len(raw):
        raise ValueError('duplicate source IDs')
    official = {}
    for line in official_lines(domain):
        fields = line.split()
        if len(fields) < 8 or fields[1] in official or fields[5] not in ('bonafide', 'spoof'):
            raise ValueError('malformed/duplicate official CM key')
        if fields[7] == 'eval':
            official[fields[1]] = 0 if fields[5] == 'bonafide' else 1
    if not set(by_id).issubset(official):
        raise ValueError('source inference manifest lacks official eval labels')
    groups = {label: sorted(sid for sid in by_id if official[sid] == label) for label in (0, 1)}
    if any(len(value) < 2048 for value in groups.values()):
        raise RuntimeError('LA_SUBSET_CONSTRUCTION_BLOCKED' if domain == 'la' else 'DF_SUBSET_CONSTRUCTION_BLOCKED')
    selected = set()
    for label in (0, 1):
        selected.update(random.Random(seed).sample(groups[label], 2048))
    rows = [by_id[sid] for sid in sorted(selected)]
    if len(rows) != 4096 or len({row['sample_id'] for row in rows}) != 4096:
        raise AssertionError('subset count/uniqueness mismatch')
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'runner.jsonl').open('x') as stream:
        for row in rows:
            if set(row) != {'sample_id', 'sample_index', 'audio_relpath', 'root_key', 'split_role', 'schema_version'}:
                raise ValueError('unexpected source inference columns')
            stream.write(json.dumps({'sample_id': row['sample_id'], 'sample_index': row['sample_index'],
                                     'audio_relpath': row['audio_relpath'], 'root_key': row['root_key']}) + '\n')
    with (output / 'private_eval_labels.jsonl').open('x') as stream:
        for row in rows:
            stream.write(json.dumps({'sample_id': row['sample_id'],
                                     'canonical_label': official[row['sample_id']]}) + '\n')
    (output / 'subset.json').write_text(json.dumps({'domain': domain, 'role': 'development_benchmark',
        'source_inference': str(source), 'official_key': 'LA-keys-full/keys/LA/CM/trial_metadata.txt' if domain == 'la' else 'DF-keys-full.tar.gz:keys/DF/CM/trial_metadata.txt',
        'seed': seed, 'count': 4096, 'bonafide': 2048, 'spoof': 2048,
        'runner_ref': 'runner.jsonl', 'evaluator_only_ref': 'private_eval_labels.jsonl'}, indent=2) + '\n')
    print(json.dumps({'domain': domain, 'status': 'READY', 'count': len(rows), 'labels': [2048, 2048]}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', choices=('la', 'df'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.domain, args.output)
