"""One-time evaluator-only sidecars for the existing ITW/WaveFake selections."""
import json
from pathlib import Path

import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
PROJECT = Path('/media/dell/data/fakeAudioDection/TTA')
WAVEFAKE = PROJECT / '.worktrees/exp-capacity-audit/experiments/capacity_audit/results/wavefake_capacity_cache_20260927a'


def write_rows(path, rows):
    with path.open('x') as handle:
        for row in rows:
            handle.write(json.dumps(row) + '\n')


def build():
    output = HERE / 'private_eval_labels_v2'
    output.mkdir(parents=True, exist_ok=False)
    audit = json.loads((PROJECT / 'experiments/target10_selection/manifests/inwild_target10.json').read_text())
    itw = [{'sample_id': row['sample_id'], 'canonical_label': row['label'],
            'group_id': row['sample_id']} for row in audit['records']]
    if len(itw) != 3178 or len({row['sample_id'] for row in itw}) != 3178 or any(row['canonical_label'] not in (0,1) for row in itw):
        raise ValueError('ITW selected label coverage mismatch')
    write_rows(output / 'itw.jsonl', itw)
    selection = json.loads((WAVEFAKE / 'manifests/wavefake_capacity_select.json').read_text())
    parquet_rows = {}
    wave = []
    for row in selection['records']:
        ref = row['parquet_ref']
        if ref not in parquet_rows:
            table = pq.read_table(ref, columns=['audio_id', 'real_or_fake'])
            parquet_rows[ref] = (table.column('audio_id').to_pylist(),
                                 table.column('real_or_fake').to_pylist())
        audio_ids, labels = parquet_rows[ref]
        index = row['parquet_row_index']
        native = {'R': 0, **{f'WF{i}': 1 for i in range(1, 8)}}
        if audio_ids[index] != row['audio_id'] or labels[index] not in native:
            raise ValueError('WaveFake parquet row/label mismatch')
        wave.append({'sample_id': row['sample_id'], 'canonical_label': native[labels[index]],
                     'group_id': row['audio_id']})
    if len(wave) != 4096 or len({row['sample_id'] for row in wave}) != 4096 or len({row['group_id'] for row in wave}) != 2048:
        raise ValueError('WaveFake selected label/group coverage mismatch')
    if sum(row['canonical_label'] for row in wave) != 2048:
        raise ValueError('WaveFake class composition mismatch')
    write_rows(output / 'wavefake.jsonl', wave)
    print(json.dumps({'status': 'READY', 'ITW': len(itw), 'WaveFake': len(wave),
                      'WaveFake_content_groups': 2048}))


if __name__ == '__main__':
    build()
