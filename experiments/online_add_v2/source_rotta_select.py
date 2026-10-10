"""Choose RoTTA LR from only the locked source-select pseudo-target streams."""
import argparse
import json
from pathlib import Path

import torch

from methods import OnlineMethod
from source_setup import SourceLL
from eptta.evaluation.metrics import binary_metrics

HERE = Path(__file__).resolve().parent


def main(run_id, gpu):
    config = json.loads((HERE / 'config.json').read_text())
    if run_id != config['run_id'] or not torch.cuda.is_available():
        raise RuntimeError('locked run/GPU required')
    output = HERE / 'results' / run_id / 'source_only' / 'rotta_select_20260928a'
    output.mkdir(parents=True, exist_ok=False)
    source = SourceLL('select')
    ids = json.loads((HERE / 'results' / run_id / 'source_only' / config['source_setup_id'] / 'source_ids.json').read_text())['select_tuning']
    y = [source.labels[sid] for sid in ids]
    device = torch.device(f'cuda:{gpu}')
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    trials = []
    for factor in (1/3, 1, 3):
        for view, label in enumerate(('clean', 'noise', 'fir')):
            trial_config = dict(config)
            trial_config['rotta_lr'] = config['rotta_author_lr'] * factor
            method = OnlineMethod('rotta', device, trial_config)
            scores = []
            for start in range(0, len(ids), 16):
                batch_ids = ids[start:start+16]
                all_views = torch.stack([source.batch(batch_ids, index) for index in (view, (view+1)%3, (view+2)%3)], dim=1).to(device)
                first, logits = method.predict(all_views[:, 0])
                scores.extend(first.tolist())
                method.adapt(all_views, logits)
            metrics = binary_metrics(scores, y, config['source_threshold'])
            result = {'factor': factor, 'view': label, 'eer': metrics['eer'],
                      'auc': metrics['auroc'], 'updates': method.updates,
                      'memory_size': method.memory_size}
            trials.append(result)
            print(json.dumps(result), flush=True)
    candidates = []
    for factor in (1/3, 1, 3):
        rows = [row for row in trials if row['factor'] == factor]
        candidates.append((sum(row['eer'] for row in rows)/3,
                           -sum(row['auc'] for row in rows)/3,
                           abs(factor-1), factor))
    factor = min(candidates)[3]
    (output / 'trials.json').write_text(json.dumps(trials, indent=2) + '\n')
    selection = {'author_lr': config['rotta_author_lr'], 'selected_factor': factor,
                 'selected_lr': config['rotta_author_lr'] * factor,
                 'criterion': 'source-select mean EER, then AUC, then author default'}
    (output / 'selection.json').write_text(json.dumps(selection, indent=2) + '\n')
    print(json.dumps({'status': 'PASS', 'selection': selection}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    main(args.run_id, args.gpu)
