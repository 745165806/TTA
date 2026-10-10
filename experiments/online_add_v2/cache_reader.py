"""Read selected three-view LL shards by opaque ID, without label access."""
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parents[1]


def manifest_rows(ref):
    path = Path(ref)
    if not path.is_absolute():
        path = WORKTREE / path
    if path.suffix == '.json':
        return json.loads(path.read_text())['records']
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


class SelectedLL:
    def __init__(self, config, run_id, domains):
        self.mapping = {}
        self.open_arrays = OrderedDict()
        for domain in domains:
            meta = config['domains'][domain]
            rows = manifest_rows(meta['runner'])
            ids = [row['sample_id'] for row in rows]
            if len(ids) != meta['count'] or len(set(ids)) != len(ids):
                raise ValueError(f'{domain}: fixed manifest coverage mismatch')
            root = (HERE / 'results' / run_id / 'll_cache') if domain in ('LA21', 'DF21') else Path(config['existing_ll_cache'])
            cache_name = domain if domain in ('LA21', 'DF21') else meta['cache_name']
            expected_shards = 4
            covered = set()
            for shard in range(expected_shards):
                directory = root / cache_name / f'shard_{shard:02d}'
                index = json.loads((directory / 'index.json').read_text())
                expected = [sid for i, sid in enumerate(ids) if i % expected_shards == shard]
                if (index['status'] != 'READY' or index['count'] != len(expected) or
                    index['sample_ids'] != expected or index['shards'] != expected_shards or
                    index['shard'] != shard or index['shape_per_sample'] != [3, 201, 128] or
                    index['dtype'] != 'float32' or index['view_recipe'] != config['view_recipe']):
                    raise ValueError(f'{domain} shard {shard} metadata mismatch')
                if domain in ('LA21', 'DF21') and index['run_id'] != run_id:
                    raise ValueError('new cache run ID mismatch')
                chunk_ids = []
                for chunk in index['chunks']:
                    path = directory / chunk['array_ref']
                    array = np.load(path, mmap_mode='r', allow_pickle=False)
                    if array.dtype != np.float32 or array.shape != (chunk['count'], 3, 201, 128):
                        raise ValueError('LL chunk shape/dtype mismatch')
                    for position, sid in enumerate(chunk['ids']):
                        if sid in covered:
                            raise ValueError('duplicate LL ID')
                        self.mapping[sid] = (path, position)
                        covered.add(sid)
                    chunk_ids.extend(chunk['ids'])
                if chunk_ids != expected:
                    raise ValueError('LL chunk coverage mismatch')
            if covered != set(ids):
                raise ValueError('LL domain coverage mismatch')

    def batch(self, ids, all_views=False):
        values = []
        for sid in ids:
            path, position = self.mapping[sid]
            if path not in self.open_arrays:
                self.open_arrays[path] = np.load(path, mmap_mode='r', allow_pickle=False)
            self.open_arrays.move_to_end(path)
            if len(self.open_arrays) > 96:
                self.open_arrays.popitem(last=False)
            item = self.open_arrays[path][position]
            if item.shape != (3, 201, 128) or not np.isfinite(item).all():
                raise ValueError('nonfinite/malformed LL sample')
            values.append(item.copy() if all_views else item[0].copy())
        return torch.from_numpy(np.stack(values))
