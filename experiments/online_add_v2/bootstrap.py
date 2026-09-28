"""Stationary 1,000-resample paired uncertainty for realized online paths."""
import argparse
import json
from pathlib import Path

import numpy as np

from summarize import labels_for, realized_run
from eptta.evaluation.metrics import binary_metrics

HERE = Path(__file__).resolve().parent


def fast_auc_eer(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    order = np.argsort(-scores, kind='stable')
    ranked_scores = scores[order]
    ranked_labels = labels[order]
    last = np.r_[np.flatnonzero(np.diff(ranked_scores)), len(scores)-1]
    tp = np.cumsum(ranked_labels)[last]
    fp = last + 1 - tp
    far = np.r_[0., fp / negatives]
    tpr = np.r_[0., tp / positives]
    frr = 1. - tpr
    auc = float(np.trapezoid(tpr, far)) if hasattr(np, 'trapezoid') else float(np.trapz(tpr, far))
    difference = far - frr
    crossing = int(np.flatnonzero(difference >= 0)[0])
    if crossing == 0:
        eer = float(far[0])
    else:
        d0, d1 = difference[crossing-1], difference[crossing]
        fraction = abs(d0)/(abs(d0)+abs(d1)) if d0 != d1 else 0.
        eer = float(far[crossing-1] + fraction*(far[crossing]-far[crossing-1]))
    return auc, eer


def main(root):
    config = json.loads((HERE / 'config.json').read_text())
    labels = labels_for(config)
    result = {'status': 'INCOMPLETE', 'replicates': 1000,
              'interpretation': 'conditional uncertainty of the realized online trajectories',
              'comparisons': {}}
    for stream_index in range(1, 5):
        stream = f'S{stream_index}'
        domain = config['streams'][stream][0]
        for seed in config['order_seeds']:
            runs = {method: realized_run(config, root, labels, method, stream, seed)
                    for method in config['methods']}
            if any(run is None for run in runs.values()):
                continue
            ids = [row['sample_id'] for row in runs['frozen']['stream']]
            y = np.array([labels[domain][sid]['canonical_label'] for sid in ids], dtype=np.int8)
            x = {method: np.array([row['score'] for row in run['scores']], dtype=np.float64)
                 for method, run in runs.items()}
            for method in config['methods']:
                if [row['sample_id'] for row in runs[method]['stream']] != ids:
                    raise ValueError('paired bootstrap ID/order mismatch')
            # Check the accelerated evaluator against the project metric on the actual trajectory.
            reference = binary_metrics(x['frozen'].tolist(), y.astype(int).tolist(), config['source_threshold'])
            fast = fast_auc_eer(x['frozen'], y)
            if max(abs(reference['auroc']-fast[0]), abs(reference['eer']-fast[1])) > 1e-10:
                raise ValueError('fast bootstrap metric differs from project metric')
            groups = {}
            for i, sid in enumerate(ids):
                group = labels[domain][sid].get('group_id', sid)
                groups.setdefault(group, []).append(i)
            group_arrays = [np.asarray(groups[key], dtype=np.int64) for key in sorted(groups)]
            rng = np.random.default_rng(2026*1000 + stream_index*10 + seed)
            comparisons = {method: {'delta_auc': [], 'eer_gain': []}
                           for method in config['methods'] if method != 'frozen'}
            valid = 0
            for _ in range(1000):
                chosen = rng.integers(0, len(group_arrays), size=len(group_arrays))
                indices = np.concatenate([group_arrays[index] for index in chosen])
                sampled_y = y[indices]
                frozen = fast_auc_eer(x['frozen'][indices], sampled_y)
                if frozen is None:
                    continue
                valid += 1
                for method in comparisons:
                    adapted = fast_auc_eer(x[method][indices], sampled_y)
                    comparisons[method]['delta_auc'].append(adapted[0]-frozen[0])
                    comparisons[method]['eer_gain'].append(frozen[1]-adapted[1])
            if valid == 0:
                raise ValueError('no valid bootstrap replicate')
            for method, values in comparisons.items():
                result['comparisons'][f'{method}_{stream}_order{seed}'] = {
                    'domain': domain, 'group_unit': 'content_pair' if domain == 'WaveFake' else 'audio_id',
                    'valid_replicates': valid,
                    'delta_auc_ci95': np.quantile(values['delta_auc'], [.025, .975]).tolist(),
                    'eer_gain_ci95': np.quantile(values['eer_gain'], [.025, .975]).tolist()}
            print(json.dumps({'stream': stream, 'seed': seed, 'valid_replicates': valid}), flush=True)
    result['status'] = 'COMPLETE' if len(result['comparisons']) == 4*5*5 else 'INCOMPLETE'
    with (root / 'bootstrap.json').open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps({'status': result['status'], 'comparisons': len(result['comparisons'])}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    main(args.root)
