"""Small GPU parity/gradient check; not a method training run."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from eptta.cache.reader import FeatureCache
from model import RepresentationDetector, adapter_parameter_count, load_frozen_backend

PROJECT = Path('/media/dell/data/fakeAudioDection/TTA')
HERE = Path(__file__).parent


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('GPU smoke requires a visible CUDA device')
    cache = HERE / 'results/ll_smoke_20260928b/ll_cache/fit/shard_00'
    index = json.loads((cache / 'index.json').read_text())
    arrays = [np.load(cache / entry['array_ref'], allow_pickle=False) for entry in index['chunks']]
    ll = torch.from_numpy(np.concatenate(arrays)).to('cuda:0')
    backend, _ = load_frozen_backend(torch.device('cuda:0'))
    detector = RepresentationDetector(backend).to('cuda:0')
    detector.eval()
    with torch.inference_mode():
        disabled = backend(ll[:, 0])
        off = detector(ll[:, 0])
    if not torch.equal(off, disabled[:, 0] - disabled[:, 1]):
        raise AssertionError('zero-initialized adapter does not recover source backend')
    source = FeatureCache(PROJECT / 'outputs_v2/ssl_aasist/cache-fit').load_by_id()
    head = torch.load(PROJECT / 'outputs_v2/ssl_aasist/frozen/linear_head.pt',
                      map_location='cpu', weights_only=True)
    expected = np.array([source[sid][0] @ head['w'].numpy() + head['b'].item()
                         for sid in index['sample_ids']])
    discrepancy = float(np.max(np.abs(off.cpu().numpy() - expected)))
    if discrepancy > 1e-3:
        raise AssertionError(f'pre-backend cache does not reproduce Frozen scores: {discrepancy}')
    labels = torch.tensor([i % 2 for i in range(len(ll))], device='cuda:0', dtype=torch.float32)
    loss = F.binary_cross_entropy_with_logits(detector(ll[:, 0]), labels)
    loss.backward()
    gradients = [p.grad for p in detector.adapter.parameters() if p.grad is not None]
    grad_norm = float(torch.stack([g.norm() for g in gradients]).norm())
    if not np.isfinite(grad_norm) or grad_norm <= 0:
        raise AssertionError('earlier-layer adapter gradient is zero or nonfinite')
    report = {'status':'PASS', 'samples':len(ll),'shape':list(ll.shape),
              'adapter_parameters':adapter_parameter_count(detector.adapter),
              'disabled_frozen_score_parity':'exact',
              'max_abs_vs_existing_frozen_cache':discrepancy,
              'adapter_gradient_norm':grad_norm,
              'gpu_peak_bytes':torch.cuda.max_memory_allocated()}
    with (HERE / 'smoke_adapter.json').open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
