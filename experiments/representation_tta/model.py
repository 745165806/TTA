"""A task-trained adapter on the 201×128 sequence before AASIST pooling."""
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

PROJECT = Path('/media/dell/data/fakeAudioDection/TTA')
sys.path.insert(0, str(PROJECT / 'workers/compat'))
from author_training import build_author_model


class PassthroughSSL(nn.Module):
    def extract_feat(self, ll_sequence):
        return ll_sequence


class BottleneckAdapter(nn.Module):
    def __init__(self, width=128, bottleneck=64):
        super().__init__()
        self.down = nn.Linear(width, bottleneck)
        self.up = nn.Linear(bottleneck, width)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, sequence):
        if sequence.ndim != 3 or sequence.shape[1:] != (201, 128):
            raise ValueError('adapter expects actual pre-backend [N,201,128] sequence')
        return sequence + self.up(F.gelu(self.down(sequence)))


class RepresentationDetector(nn.Module):
    def __init__(self, frozen_backend, adapter=None):
        super().__init__()
        self.backend = frozen_backend
        self.adapter = adapter if adapter is not None else BottleneckAdapter()
        self.backend.eval()
        for parameter in self.backend.parameters():
            parameter.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self.backend.eval()
        return self

    def embed(self, ll_sequence):
        return self.adapter(ll_sequence)

    def forward(self, ll_sequence):
        logits = self.backend(self.embed(ll_sequence))
        # The task-trained native class map is spoof=0, bonafide=1.
        return logits[:, 0] - logits[:, 1]


def load_frozen_backend(device):
    """Load project task-trained weights, then retain the exact original backend forward."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    finalized = json.loads((PROJECT / 'outputs/source/ssl-aasist-full-asvspoof2019train-random/finalized.json').read_text())
    bundle = json.loads((PROJECT / 'outputs_v2/ssl_aasist/frozen/bundle.json').read_text())
    if bundle['task_weight_origin'] != 'trained_in_project' or bundle['class_index_map'] != {'spoof': 0, 'bonafide': 1}:
        raise ValueError('unrecognized task-trained source detector or native class mapping')
    job = {'source_job': {'model_id': bundle['model_id'], 'initialization': bundle['initialization']},
           'execution': {'architecture': finalized['architecture']}}
    author, _patch = build_author_model(job, device)
    state = torch.load(PROJECT / 'outputs_v2/ssl_aasist/frozen/detector_state.pt',
                       map_location='cpu', weights_only=True)
    if state['class_index_map'] != bundle['class_index_map']:
        raise ValueError('frozen source checkpoint class mapping mismatch')
    author.model.load_state_dict(state['model_state'], strict=True)
    backend = author.model.eval()
    backend.ssl_model = PassthroughSSL()
    backend.LL = nn.Identity()
    for parameter in backend.parameters():
        parameter.requires_grad_(False)
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return backend, bundle


def adapter_parameter_count(adapter):
    return sum(parameter.numel() for parameter in adapter.parameters())
