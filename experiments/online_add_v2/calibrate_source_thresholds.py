"""Calibrate each fixed score form on source-select only, with fresh online state."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from methods import OnlineMethod
from source_setup import SourceLL
from eptta.evaluation.metrics import binary_metrics

HERE = Path(__file__).resolve().parent


def choose_threshold(scores, labels):
    """Choose the source-select operating point nearest FPR=FNR, without target input."""
    x = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int8)
    if x.ndim != 1 or y.shape != x.shape or not np.isfinite(x).all() or set(y.tolist()) != {0, 1}:
        raise ValueError('invalid source-select score/label vector')
    candidates = np.r_[np.nextafter(x.min(), -np.inf), np.unique(x)]
    best = None
    for threshold in candidates:
        prediction = x > threshold
        fpr = float(prediction[y == 0].mean())
        fnr = float((~prediction[y == 1]).mean())
        choice = (abs(fpr-fnr), (fpr+fnr)/2, float(threshold))
        if best is None or choice < best[0]:
            best = (choice, fpr, fnr)
    return best[0][2], best[1], best[2]


def main(gpu, calibration_id):
    config = json.loads((HERE / 'config.json').read_text())
    output = HERE / 'results' / config['run_id'] / 'source_only' / calibration_id
    output.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available() or torch.cuda.device_count() <= gpu:
        raise RuntimeError('authorized GPU required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    device = torch.device(f'cuda:{gpu}')
    torch.cuda.set_device(device)
    source = SourceLL('select')
    fixed = json.loads((HERE / 'results' / config['run_id'] / 'source_only' /
                        config['source_setup_id'] / 'source_ids.json').read_text())
    ids = fixed['select_tuning']
    if len(ids) != 1024 or len(set(ids)) != 1024 or not set(ids).issubset(source.labels):
        raise ValueError('source-select calibration IDs mismatch')
    labels = [source.labels[sid] for sid in ids]
    state_path = HERE / 'results' / config['run_id'] / 'source_only' / config['source_setup_id'] / 'eata.pt'
    eata_state = torch.load(state_path, map_location='cpu', weights_only=True)
    summary = {'status': 'RUNNING', 'calibration_id': calibration_id, 'source_role': 'select',
               'source_ids_ref': str(HERE / 'results' / config['run_id'] / 'source_only' /
                                     config['source_setup_id'] / 'source_ids.json'),
               'count': len(ids), 'class_counts': {'bonafide': labels.count(0), 'spoof': labels.count(1)},
               'policy': 'fresh method stream on fixed source-select clean B16; minimize |FPR-FNR|, then mean error, then threshold',
               'note': 'calibration added after v2.1 execution package was supplied; no target scores or labels enter this program',
               'methods': {}}
    (output / 'calibration.json').write_text(json.dumps(summary, indent=2) + '\n')
    with (output / 'source_scores.jsonl').open('x') as handle:
        for name in config['methods']:
            adapter = OnlineMethod(name, device, config, eata_state if name == 'eata' else None)
            scores = []
            for start in range(0, len(ids), config['batch_size']):
                batch_ids = ids[start:start+config['batch_size']]
                clean = source.batch(batch_ids, 0).to(device)
                all_views = (torch.stack([source.batch(batch_ids, view) for view in range(3)], dim=1).to(device)
                             if name == 'rotta' else None)
                first, logits = adapter.predict(clean)
                scores.extend(first.tolist())
                adapter.adapt(all_views if all_views is not None else clean, logits)
            threshold, fpr, fnr = choose_threshold(scores, labels)
            metrics = binary_metrics(scores, labels, threshold)
            summary['methods'][name] = {'threshold': threshold, 'source_select_fpr': fpr,
                                        'source_select_fnr': fnr, 'source_select_auc': metrics['auroc'],
                                        'source_select_eer': metrics['eer'],
                                        'score_form': 'post-LAME spoof/bonafide log-odds' if name == 'lame'
                                                      else 'native spoof-minus-bonafide logits'}
            for sid, label, score in zip(ids, labels, scores):
                handle.write(json.dumps({'method': name, 'sample_id': sid,
                                         'canonical_source_label': label, 'score': score}) + '\n')
            print(json.dumps({'method': name, 'threshold': threshold,
                              'source_select_eer': metrics['eer']}), flush=True)
            del adapter
    summary['status'] = 'COMPLETE'
    (output / 'calibration.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({'status': 'COMPLETE', 'methods': len(summary['methods'])}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--calibration-id', required=True)
    args = parser.parse_args()
    main(args.gpu, args.calibration_id)
