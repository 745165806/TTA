"""Strict reader for fixed-ID pre-backend feature shards."""
import json
from pathlib import Path

import numpy as np
import torch

from extract_ll import BASE, PROBE, rows_for


def load_ll(domain, run_id, expected_shards=4):
    expected_rows, _, manifest = rows_for(domain)
    ids = [row['sample_id'] for row in expected_rows]
    index_of = {sid: i for i, sid in enumerate(ids)}
    if len(index_of) != len(ids):
        raise ValueError('duplicate assignment IDs')
    result = np.empty((len(ids), 3, 201, 128), dtype=np.float32)
    seen = set()
    cache_root = BASE / 'results' / run_id / 'll_cache' / domain
    for shard in range(expected_shards):
        directory = cache_root / f'shard_{shard:02d}'
        index = json.loads((directory / 'index.json').read_text())
        expected = [sid for i, sid in enumerate(ids) if i % expected_shards == shard]
        if (index['status'] != 'READY' or index['domain'] != domain or index['run_id'] != run_id or
                index['shard'] != shard or index['shards'] != expected_shards or
                index['count'] != len(expected) or index['sample_ids'] != expected or
                index['manifest_ref'] != str(manifest) or index['view_recipe'] != PROBE or
                index['shape_per_sample'] != [3, 201, 128] or index['dtype'] != 'float32'):
            raise ValueError(f'{domain} shard {shard} provenance/coverage mismatch')
        chunk_ids = []
        for item in index['chunks']:
            array = np.load(directory / item['array_ref'], allow_pickle=False)
            these_ids = item['ids']
            if array.shape != (item['count'], 3, 201, 128) or array.dtype != np.float32 or not np.isfinite(array).all():
                raise ValueError('malformed/nonfinite earlier-layer chunk')
            if len(these_ids) != len(array):
                raise ValueError('chunk ID/sample count mismatch')
            chunk_ids.extend(these_ids)
            for i, sid in enumerate(these_ids):
                if sid in seen or sid not in index_of:
                    raise ValueError('duplicate/unexpected earlier-layer sample ID')
                result[index_of[sid]] = array[i]
                seen.add(sid)
        if chunk_ids != expected:
            raise ValueError('shard chunk order differs from fixed assignment')
    if seen != set(ids):
        raise ValueError('earlier-layer cache does not cover fixed assignment')
    return ids, torch.from_numpy(result)
