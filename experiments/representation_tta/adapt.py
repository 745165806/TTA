"""Unlabeled batch-transductive update of the pre-backend sequence adapter only."""
import copy
import math
import random
import time

import numpy as np
import torch

from model import RepresentationDetector


@torch.inference_mode()
def source_prototypes(source_model, fit_views, source_rows, batch_size, device):
    if len(source_rows) != len(fit_views):
        raise ValueError('fit prototype row mismatch')
    families = [f'A{i:02d}' for i in range(1, 7)]
    names = ['-'] + families
    sums = {name: torch.zeros(128, device=device) for name in names}
    seconds = {name: torch.zeros(128, device=device) for name in names}
    counts = {name: 0 for name in names}
    logit_sum = logit_square = 0.0
    source_model.eval()
    for start in range(0, len(fit_views), batch_size):
        embedded = source_model.embed(fit_views[start:start+batch_size, 0].to(device))
        pooled = embedded.mean(1)
        logits = source_model.backend(embedded)
        score = logits[:,0]-logits[:,1]
        logit_sum += float(score.sum())
        logit_square += float(score.square().sum())
        for offset, h in enumerate(pooled):
            name = source_rows[start+offset]['attack']
            if name not in sums:
                raise ValueError('unexpected source attack class')
            sums[name] += h
            seconds[name] += h.square()
            counts[name] += 1
    if any(value == 0 for value in counts.values()):
        raise ValueError('empty source prototype family')
    bona = sums['-'] / counts['-']
    spoof = torch.stack([sums[name]/counts[name] for name in families])
    variance = torch.stack([seconds[name]/counts[name]-(sums[name]/counts[name]).square()
                            for name in names]).mean().clamp_min(1e-6)
    logit_count = sum(counts.values())
    logit_scale = max(logit_square/logit_count-(logit_sum/logit_count)**2,1.0)
    return {'bonafide':bona,'spoof':spoof,'scale':float(variance),
            'logit_scale':logit_scale,'family_counts':counts,'source_only':True}


def adapt_unlabeled(source_model, views, prototypes, config, seed, device):
    """No labels, IDs, paths, target class proportions, or attack codes enter here."""
    if views.ndim != 4 or views.shape[1:] != (3,201,128) or not torch.isfinite(views).all():
        raise ValueError('target adapter requires finite unlabeled [N,3,201,128]')
    source_model.eval()
    adapter = copy.deepcopy(source_model.adapter)
    model = RepresentationDetector(source_model.backend, adapter).to(device)
    model.train()
    base = [p.detach().clone() for p in source_model.adapter.parameters()]
    params = list(model.adapter.parameters())
    norm = max(sum(float(p.square().sum()) for p in base),1e-6)
    optimizer = torch.optim.Adam(params,lr=config['target_learning_rate'])
    rng = random.Random(seed)
    updates = numeric_failures = selected_bona = selected_spoof = 0
    skipped_bona = skipped_spoof = 0
    started = time.monotonic()
    for _ in range(config['target_epochs']):
        indices = list(range(len(views)))
        rng.shuffle(indices)
        for start in range(0,len(indices),config['target_batch_size']):
            part = indices[start:start+config['target_batch_size']]
            if len(part) < 2:
                continue
            batch = views[part].to(device)
            with torch.no_grad():
                teacher = source_model(batch[:,0]).sigmoid()
                bona_mask = teacher <= 1-config['confidence']
                spoof_mask = teacher >= config['confidence']
            optimizer.zero_grad(set_to_none=True)
            flattened = batch.reshape(-1,201,128)
            embedded = model.embed(flattened)
            logits = model.backend(embedded)
            score = (logits[:,0]-logits[:,1]).reshape(len(part),3)
            pooled = embedded.reshape(len(part),3,201,128)[:,0].mean(1)
            terms=[]
            if bona_mask.any():
                terms.append((pooled[bona_mask]-prototypes['bonafide']).square().mean()/prototypes['scale'])
                selected_bona += int(bona_mask.sum())
            else:
                skipped_bona += 1
            if spoof_mask.any():
                h=pooled[spoof_mask]
                nearest=torch.cdist(h,prototypes['spoof']).argmin(1)
                terms.append((h-prototypes['spoof'][nearest]).square().mean()/prototypes['scale'])
                selected_spoof += int(spoof_mask.sum())
            else:
                skipped_spoof += 1
            anchor=torch.stack(terms).mean() if terms else pooled.sum()*0
            view=((score[:,1]-score[:,0]).square().mean()+
                  (score[:,2]-score[:,0]).square().mean())/(2*prototypes['logit_scale'])
            move=sum((p-p0).square().sum() for p,p0 in zip(params,base))/norm
            loss=(config['anchor_weight']*anchor + config['view_weight']*view +
                  config['move_weight']*move)
            if not torch.isfinite(loss):
                numeric_failures += 1
                raise FloatingPointError('nonfinite unlabeled target loss')
            loss.backward()
            if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in params):
                numeric_failures += 1
                raise FloatingPointError('nonfinite target adapter gradient')
            torch.nn.utils.clip_grad_norm_(params,1.0)
            optimizer.step()
            updates += 1
    model.eval()
    movement=math.sqrt(sum(float((p.detach()-p0).square().sum()) for p,p0 in zip(params,base)))
    info={'updates':updates,'numerical_failures':numeric_failures,
          'selected_bonafide_presentations':selected_bona,
          'selected_spoof_presentations':selected_spoof,
          'skipped_bonafide_batches':skipped_bona,'skipped_spoof_batches':skipped_spoof,
          'adapter_parameter_delta_norm':movement,'adapt_seconds':time.monotonic()-started,
          'peak_gpu_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0}
    return model,info
