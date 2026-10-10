"""Inspect and verify the actual earlier-layer SSL-AASIST replay boundary."""
import json
import sys
import time
from pathlib import Path

import torch
from torch import nn

PROJECT = Path('/media/dell/data/fakeAudioDection/TTA')
sys.path.insert(0, str(PROJECT / 'workers/compat'))
from author_training import build_author_model, load_audio


class PassthroughSSL(nn.Module):
    def extract_feat(self, value):
        return value


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('GPU is required for waveform/SSL model probe')
    device = torch.device('cuda:0')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = json.loads((PROJECT / 'outputs/source/ssl-aasist-full-asvspoof2019train-random/finalized.json').read_text())
    bundle = json.loads((PROJECT / 'outputs_v2/ssl_aasist/frozen/bundle.json').read_text())
    job = {'source_job': {'model_id': bundle['model_id'], 'initialization': bundle['initialization']},
           'execution': {'architecture': source['architecture']}}
    started = time.monotonic()
    adapter, patch = build_author_model(job, device)
    state = torch.load(PROJECT / 'outputs_v2/ssl_aasist/frozen/detector_state.pt',
                       map_location='cpu', weights_only=True)
    adapter.model.load_state_dict(state['model_state'], strict=True)
    model = adapter.model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    waveform = torch.from_numpy(load_audio(
        '/media/dell/data/fakedata/asvspoof2019/LA/ASVspoof2019_LA_train/flac/LA_T_1000137.flac')).unsqueeze(0).to(device)
    with torch.inference_mode():
        ssl = model.ssl_model.extract_feat(waveform)
        ll = model.LL(ssl)
        logits = model(waveform)
        model.ssl_model = PassthroughSSL()
        model.LL = nn.Identity()
        replay = model(ll)
    max_error = float((logits-replay).abs().max())
    if max_error > 1e-5:
        raise AssertionError(f'backend replay differs from waveform forward: {max_error}')
    report = {'status': 'PASS', 'device': str(device), 'ssl_shape': list(ssl.shape),
              'll_shape': list(ll.shape), 'logits_shape': list(logits.shape),
              'boundary': 'after model.LL, before max_pool2d/RawNet2 encoder',
              'backend_replay_max_abs': max_error, 'elapsed_seconds': time.monotonic()-started,
              'gpu_peak_bytes': torch.cuda.max_memory_allocated(),
              'generic_ssl_patch': patch, 'task_checkpoint': str(PROJECT / 'outputs_v2/ssl_aasist/frozen/detector_state.pt')}
    destination = Path(__file__).parent / 'probe_model.json'
    with destination.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
