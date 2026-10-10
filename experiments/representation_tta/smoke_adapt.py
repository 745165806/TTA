"""Bounded engineering smoke of label-free representation adaptation."""
import json
from pathlib import Path

import numpy as np
import torch

from adapt import adapt_unlabeled
from model import RepresentationDetector, load_frozen_backend

HERE=Path(__file__).parent


def main():
    if not torch.cuda.is_available():raise RuntimeError('GPU smoke requires CUDA')
    cache=HERE/'results/ll_smoke_20260928b/ll_cache/fit/shard_00'
    index=json.loads((cache/'index.json').read_text())
    views=torch.from_numpy(np.concatenate([np.load(cache/item['array_ref'],allow_pickle=False)
                                            for item in index['chunks']]))
    device=torch.device('cuda:0')
    backend,_=load_frozen_backend(device)
    source=RepresentationDetector(backend).to(device).eval()
    with torch.no_grad():
        pooled=source.embed(views[:,0].to(device)).mean(1)
    # Synthetic class assignment is engineering-only; no target or source labels used.
    prototypes={'bonafide':pooled[:8].mean(0).detach(),
                'spoof':pooled[8:].reshape(2,4,128).mean(1).detach(),
                'scale':float(pooled.var(dim=0,unbiased=False).mean().clamp_min(1e-6)),
                'logit_scale':1.0}
    config={'target_epochs':2,'target_batch_size':8,'target_learning_rate':1e-4,
            'confidence':0.9,'anchor_weight':0.1,'view_weight':0.05,'move_weight':0.01}
    adapted,info=adapt_unlabeled(source,views,prototypes,config,13,device)
    with torch.inference_mode():
        off=source(views[:,0].to(device))
        on=adapted(views[:,0].to(device))
    if not torch.isfinite(on).all() or info['updates']!=4 or info['numerical_failures']:
        raise AssertionError('unlabeled earlier-layer update smoke failed')
    report={'status':'PASS','type':'engineering_synthetic_prototypes_only','count':len(views),
            'updates':info['updates'],'adapter_parameter_delta_norm':info['adapter_parameter_delta_norm'],
            'max_score_change':float((on-off).abs().max()),'peak_gpu_bytes':info['peak_gpu_bytes'],
            'no_labels_in_adapt_signature':True}
    with (HERE/'smoke_adapt.json').open('x') as stream:
        json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
