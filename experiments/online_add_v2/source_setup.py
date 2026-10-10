"""Source fit Fisher and source select calibration; never reads target labels."""
import argparse
import json
import random
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from methods import OnlineMethod
from eptta.evaluation.metrics import binary_metrics

HERE = Path(__file__).resolve().parent
PROJECT = Path('/media/dell/data/fakeAudioDection/TTA')
SOURCE_CACHE = Path('/media/dell/data/fakeAudioDection/TTA/.worktrees/exp-representation-tta/experiments/representation_tta/results/rep_ll_20260928_seed13/ll_cache')


class SourceLL:
    def __init__(self, role):
        if role not in ('fit', 'select'):
            raise ValueError('source role required')
        path = PROJECT / f'data/manifests_v2/asv2019_la/inference/{role}.jsonl'
        self.rows = [json.loads(line) for line in path.open() if line.strip()]
        ids = [row['sample_id'] for row in self.rows]
        if len(ids) != len(set(ids)):
            raise ValueError('source manifest duplicate ID')
        labels_path = PROJECT / f'data/manifests_v2/asv2019_la/labels/{role}.jsonl'
        labels = [json.loads(line) for line in labels_path.open() if line.strip()]
        self.labels = {row['sample_id']: row['canonical_label'] for row in labels}
        if set(ids) != set(self.labels):
            raise ValueError('source label coverage mismatch')
        self.mapping = {}
        for shard in range(4):
            directory = SOURCE_CACHE / role / f'shard_{shard:02d}'
            index = json.loads((directory / 'index.json').read_text())
            expected = [sid for i, sid in enumerate(ids) if i % 4 == shard]
            if index['sample_ids'] != expected or index['status'] != 'READY':
                raise ValueError('source LL cache coverage mismatch')
            for chunk in index['chunks']:
                location = directory / chunk['array_ref']
                for i, sid in enumerate(chunk['ids']):
                    self.mapping[sid] = (location, i)
        if set(self.mapping) != set(ids):
            raise ValueError('source LL chunk coverage mismatch')
        self.open_arrays = OrderedDict()

    def batch(self, ids, view):
        result = []
        for sid in ids:
            path, index = self.mapping[sid]
            if path not in self.open_arrays:
                self.open_arrays[path] = np.load(path, allow_pickle=False, mmap_mode='r')
            self.open_arrays.move_to_end(path)
            if len(self.open_arrays) > 96:
                self.open_arrays.popitem(last=False)
            arr = self.open_arrays[path][index, view]
            if arr.shape != (201, 128) or not np.isfinite(arr).all():
                raise ValueError('source LL sample malformed')
            result.append(arr.copy())
        return torch.from_numpy(np.stack(result))


def fixed_ids(source, per_class, seed):
    grouped = {label: sorted(sid for sid, y in source.labels.items() if y == label) for label in (0, 1)}
    if any(len(grouped[label]) < per_class for label in (0, 1)):
        raise ValueError('source role lacks requested class sample')
    selected = [sid for label in (0, 1) for sid in random.Random(seed).sample(grouped[label], per_class)]
    random.Random(seed + 37).shuffle(selected)
    return selected


def build(config, run_id, device):
    output = HERE / 'results' / run_id / 'source_only' / config['source_setup_id']
    output.mkdir(parents=True, exist_ok=False)
    fit = SourceLL('fit')
    select = SourceLL('select')
    fit_ids = fixed_ids(fit, 1000, 2026)
    select_ids = fixed_ids(select, 512, 2026)
    (output / 'source_ids.json').write_text(json.dumps({'fit_fisher': fit_ids,
                                                        'select_tuning': select_ids}, indent=2) + '\n')
    method = OnlineMethod('tent', device, config)
    fisher = {name: torch.zeros_like(parameter) for name, parameter in method.model.named_parameters()
              if parameter.requires_grad}
    for start in range(0, len(fit_ids), 16):
        ids = fit_ids[start:start+16]
        x = fit.batch(ids, 0).to(device)
        native_y = torch.tensor([1-fit.labels[sid] for sid in ids], device=device)
        method.optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(method.model(x), native_y)
        loss.backward()
        for name, parameter in method.model.named_parameters():
            if name in fisher:
                if parameter.grad is not None:
                    fisher[name] += parameter.grad.detach().square() * len(ids)
    fisher = {name: value.detach().cpu() / len(fit_ids) for name, value in fisher.items()}
    calibration = OnlineMethod('bn_only', device, config)
    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(select_ids), 16):
            ids = select_ids[start:start+16]
            x = select.batch(ids, 0).to(device)
            probabilities.append(calibration.model(x).softmax(1).cpu())
    probabilities = torch.cat(probabilities)
    running = probabilities.mean(0, keepdim=True)
    similarities = F.cosine_similarity(probabilities, running, dim=1)
    diversity_threshold = float(torch.quantile(similarities, 0.95).item())
    if not 0 < diversity_threshold < 1:
        raise ValueError('invalid source-select diversity calibration')
    state = {'fisher': fisher, 'diversity_threshold': diversity_threshold,
             'fit_count': len(fit_ids), 'select_count': len(select_ids),
             'source_role': 'fit_and_select', 'fisher_formula': 'mean squared batch CE gradient on source fit',
             'diversity_policy': 'source-select 95th percentile cosine to clean mean probability'}
    with (output / 'eata.pt').open('xb') as handle:
        torch.save(state, handle)
    (output / 'eata_summary.json').write_text(json.dumps({k: v for k, v in state.items() if k != 'fisher'}, indent=2) + '\n')
    print(json.dumps({'phase': 'eata_source_state', 'status': 'PASS',
                      'fit_count': len(fit_ids), 'diversity_threshold': diversity_threshold}), flush=True)
    del method, calibration
    tune = []
    for name in ('tent', 'eata'):
        for factor in (1/3, 1, 3):
            for view, label in enumerate(('clean', 'noise', 'fir')):
                trial_config = dict(config)
                trial_config[f'{name}_lr'] = config[f'{name}_author_lr'] * factor
                adapter = OnlineMethod(name, device, trial_config, state if name == 'eata' else None)
                scores = []
                for start in range(0, len(select_ids), 16):
                    ids = select_ids[start:start+16]
                    x = select.batch(ids, view).to(device)
                    score, first_logits = adapter.predict(x)
                    scores.extend(score.tolist())
                    adapter.adapt(x, first_logits)
                y = [select.labels[sid] for sid in select_ids]
                metrics = binary_metrics(scores, y, config['source_threshold'])
                tune.append({'method': name, 'factor': factor, 'view': label,
                             'eer': metrics['eer'], 'auc': metrics['auroc'],
                             'updates': adapter.updates, 'adapted_samples': adapter.adapted_samples})
                print(json.dumps(tune[-1]), flush=True)
                del adapter
    (output / 'lr_trials.json').write_text(json.dumps(tune, indent=2) + '\n')
    choices = {}
    for name in ('tent', 'eata'):
        candidates = []
        for factor in (1/3, 1, 3):
            rows = [row for row in tune if row['method'] == name and row['factor'] == factor]
            candidates.append((sum(row['eer'] for row in rows)/3,
                               -sum(row['auc'] for row in rows)/3,
                               abs(factor-1), factor))
        winner = min(candidates)[3]
        choices[name] = {'author_lr': config[f'{name}_author_lr'], 'selected_factor': winner,
                         'selected_lr': config[f'{name}_author_lr'] * winner,
                         'criterion': 'source-select mean EER, then AUC, then author default'}
    (output / 'lr_selection.json').write_text(json.dumps(choices, indent=2) + '\n')
    print(json.dumps({'phase': 'source_select', 'status': 'PASS', 'choices': choices}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    config = json.loads((HERE / 'config.json').read_text())
    if args.run_id != config['run_id'] or not torch.cuda.is_available():
        raise RuntimeError('locked run/GPU required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)
    build(config, args.run_id, device)
