"""One real-source-batch BCE+SupCon gradient smoke, not full source training."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from model import RepresentationDetector, load_frozen_backend
from experiments.overnight_tta.residual import supervised_contrastive

HERE=Path(__file__).parent
PROJECT=Path('/media/dell/data/fakeAudioDection/TTA')


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required for bounded source objective smoke')
    cache=HERE/'results/ll_smoke_20260928b/ll_cache/fit/shard_00'
    index=json.loads((cache/'index.json').read_text())
    features=np.concatenate([np.load(cache/item['array_ref'],allow_pickle=False)
                             for item in index['chunks']])
    ids=index['sample_ids']
    with (PROJECT/'data/manifests_v2/asv2019_la/labels/fit.jsonl').open() as stream:
        source_labels={row['sample_id']:row['canonical_label'] for row in
                       (json.loads(line) for line in stream if line.strip())}
    if set(ids)-set(source_labels):
        raise ValueError('source smoke ID missing canonical label')
    device=torch.device('cuda:0')
    torch.manual_seed(13)
    backend,_=load_frozen_backend(device)
    model=RepresentationDetector(backend).to(device).train()
    views=torch.from_numpy(features[:,:2]).to(device)
    y=torch.tensor([source_labels[sid] for sid in ids],device=device,dtype=torch.float32)
    before={name:p.detach().clone() for name,p in model.adapter.named_parameters()}
    optimizer=torch.optim.Adam(model.adapter.parameters(),lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    embedded=model.embed(torch.cat((views[:,0],views[:,1])))
    logits=model.backend(embedded)
    scores=logits[:,0]-logits[:,1]
    bce=F.binary_cross_entropy_with_logits(scores,y.repeat(2))
    pooled=embedded.mean(1)
    supcon=supervised_contrastive(pooled[:len(y)],pooled[len(y):],y.long(),0.1)
    loss=bce+0.05*supcon
    if not torch.isfinite(loss):
        raise FloatingPointError('nonfinite source objective')
    loss.backward()
    norms=[p.grad.norm() for p in model.adapter.parameters() if p.grad is not None]
    if not norms or not all(torch.isfinite(value).item() for value in norms):
        raise FloatingPointError('missing/nonfinite source objective gradient')
    optimizer.step()
    movement=float(torch.stack([(p.detach()-before[name]).norm()
                                 for name,p in model.adapter.named_parameters()]).norm())
    if movement<=0 or not np.isfinite(movement):
        raise AssertionError('source optimizer step did not update earlier-layer adapter')
    report={'status':'PASS','scope':'one_real_source_batch_engineering_only',
            'count':len(ids),'class_counts':{'bonafide':int((y==0).sum()),
                                            'spoof':int((y==1).sum())},
            'bce':float(bce.detach()),'supcon':float(supcon.detach()),
            'total_loss':float(loss.detach()),
            'gradient_norm':float(torch.stack(norms).norm()),
            'adapter_parameter_delta_norm':movement,
            'gpu_peak_bytes':torch.cuda.max_memory_allocated(device)}
    with (HERE/'smoke_source_train.json').open('x') as stream:
        json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
