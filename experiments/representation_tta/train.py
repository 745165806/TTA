"""Real source fit training of an earlier-layer adapter and equal-budget static arm."""
import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from cache import load_ll
from model import RepresentationDetector, adapter_parameter_count, load_frozen_backend
from source_data import source_metadata
from experiments.overnight_tta.residual import supervised_contrastive
from eptta.evaluation.metrics import binary_metrics

HERE = Path(__file__).parent


def balanced_batches(labels, seed, epoch, batch_size):
    half = batch_size // 2
    bona = [i for i, y in enumerate(labels) if y == 0]
    spoof = [i for i, y in enumerate(labels) if y == 1]
    if len(bona) < half or len(spoof) < half or batch_size % 2:
        raise ValueError('source fit cannot form balanced two-class batches')
    rng = random.Random(seed * 100003 + epoch)
    rng.shuffle(spoof)
    steps = math.ceil(len(spoof) / half)
    if len(spoof) < steps * half:
        spoof.extend(spoof[:steps * half - len(spoof)])
    for step in range(steps):
        ids = spoof[step*half:(step+1)*half] + rng.sample(bona, half)
        rng.shuffle(ids)
        yield ids


@torch.inference_mode()
def validate(detector, views, labels, batch_size, device):
    detector.eval()
    result = {}
    for view, name in enumerate(('clean', 'noise', 'fir')):
        scores = []
        for start in range(0, len(views), batch_size):
            batch = views[start:start+batch_size, view].to(device)
            scores.extend(detector(batch).cpu().tolist())
        metric = binary_metrics(scores, labels, 0.0)
        result[f'{name}_auc'] = metric['auroc']
        result[f'{name}_eer'] = metric['eer']
    return result


def run(config_ref, run_id, arm, seed, gpu):
    config = json.loads(Path(config_ref).read_text())
    if arm not in ('static_bce', 'task_supcon'):
        raise ValueError('unrecognized source training arm')
    if not torch.cuda.is_available() or torch.cuda.device_count() <= gpu:
        raise RuntimeError('earlier-layer source training requires a visible GPU')
    device = torch.device(f'cuda:{gpu}')
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    output = HERE / 'results' / run_id / arm
    output.mkdir(parents=True, exist_ok=False)
    (output / 'config.json').write_text(json.dumps({**config, 'run_id': run_id, 'arm': arm,
                                                     'seed': seed, 'gpu': gpu}, indent=2) + '\n')
    started = time.monotonic()
    fit_ids, fit = load_ll('fit', config['cache_run_id'])
    select_ids, select = load_ll('select', config['cache_run_id'])
    fit_source = source_metadata(config, 'fit', fit_ids)
    select_source = source_metadata(config, 'select', select_ids)
    fit_y=[row['label'] for row in fit_source]
    select_y=[row['label'] for row in select_source]
    if {row['audio_id'] for row in fit_source} & {row['audio_id'] for row in select_source}:
        raise ValueError('source fit/select original audio overlap')
    del fit_source, select_source
    backend, _ = load_frozen_backend(device)
    detector = RepresentationDetector(backend).to(device)
    if adapter_parameter_count(detector.adapter) != 16576:
        raise AssertionError('unexpected earlier-layer adapter parameter count')
    optimizer = torch.optim.Adam(detector.adapter.parameters(), lr=config['source_learning_rate'])
    numerical_failures = steps = presentations = 0
    seen = set()
    history = []
    best_key = (float('inf'), float('inf'), float('inf'))
    for epoch in range(1, config['source_epochs']+1):
        detector.train()
        losses, bces, contrast_losses = [], [], []
        for indices in balanced_batches(fit_y, seed, epoch, config['source_batch_size']):
            seen.update(indices)
            labels = torch.tensor([fit_y[i] for i in indices], dtype=torch.float32, device=device)
            x = fit[indices, :2].to(device, non_blocking=True)
            z0, z1 = x[:, 0], x[:, 1]
            optimizer.zero_grad(set_to_none=True)
            embedded = detector.embed(torch.cat((z0, z1)))
            logits = detector.backend(embedded)
            scores = logits[:, 0] - logits[:, 1]
            bce = F.binary_cross_entropy_with_logits(scores, labels.repeat(2))
            if arm == 'task_supcon':
                h0, h1 = embedded[:len(indices)].mean(1), embedded[len(indices):].mean(1)
                contrast = supervised_contrastive(h0, h1, labels.long(), config['supcon_temperature'])
                loss = bce + config['supcon_weight'] * contrast
            else:
                contrast = bce.detach() * 0
                loss = bce
            if not all(torch.isfinite(item).item() for item in (bce, contrast, loss)):
                numerical_failures += 1
                raise FloatingPointError('nonfinite source adapter objective')
            loss.backward()
            gradients = [p.grad for p in detector.adapter.parameters() if p.grad is not None]
            if not gradients or not all(torch.isfinite(g).all().item() for g in gradients):
                numerical_failures += 1
                raise FloatingPointError('nonfinite/missing earlier-layer adapter gradient')
            optimizer.step()
            steps += 1
            presentations += len(indices)
            losses.append(float(loss.detach()))
            bces.append(float(bce.detach()))
            contrast_losses.append(float(contrast.detach()))
        metric = validate(detector, select, select_y, config['source_batch_size'], device)
        row = {'epoch':epoch,'optimizer_steps':steps,'presentations':presentations,
               'unique_fit_ids_seen':len(seen),'loss':float(np.mean(losses)),
               'bce':float(np.mean(bces)),'supcon':float(np.mean(contrast_losses)),
               'elapsed_seconds':time.monotonic()-started,**metric}
        history.append(row)
        checkpoint = output / f'adapter_epoch_{epoch:04d}.pt'
        with checkpoint.open('xb') as stream:
            torch.save({'adapter':{k:v.detach().cpu() for k,v in detector.adapter.state_dict().items()},
                        'arm':arm,'seed':seed,'epoch':epoch,'source_bundle_ref':config['bundle_ref'],
                        'source_cache_run_id':config['cache_run_id']},stream)
        key = (row['clean_eer'],(row['noise_eer']+row['fir_eer'])/2,-row['clean_auc'])
        if key < best_key:
            best_key = key
            (output / 'selection.json').write_text(json.dumps({'epoch':epoch,'checkpoint_ref':str(checkpoint),
                                                              'source_select_key':key,'optimizer_steps':steps},indent=2)+'\n')
        print(json.dumps({'arm':arm,**row}),flush=True)
    with (output/'training_curve.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(history[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(history)
    report={'status':'PASS','arm':arm,'seed':seed,'gpu':gpu,'optimizer_steps':steps,
            'presentations':presentations,'unique_fit_ids_seen':len(seen),'fit_count':len(fit_ids),
            'select_count':len(select_ids),'adapter_parameters':adapter_parameter_count(detector.adapter),
            'numerical_failures':numerical_failures,'elapsed_seconds':time.monotonic()-started,
            'peak_gpu_bytes':torch.cuda.max_memory_allocated(device)}
    (output/'source_training.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--arm',required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--gpu',type=int,required=True)
    args=parser.parse_args()
    run(args.config,args.run_id,args.arm,args.seed,args.gpu)
