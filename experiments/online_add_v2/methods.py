"""Online baseline state. Each object is new for one independent stream."""
import math
import torch
from torch import nn
from torch.nn import functional as F

from experiments.representation_tta.model import load_frozen_backend
from eptta.baselines.ports.common import configure_normalization_adaptation
from experiments.online_add_v2.rotta import RoTTAState


def entropy(logits):
    return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)


def lame_probabilities(logits, features, k):
    """Author kNN graph and Laplacian fixed-point optimization on current batch."""
    if len(logits) < 2:
        return logits.softmax(-1)
    features = F.normalize(features, p=2, dim=-1)
    distances = torch.cdist(features, features)
    neighbors = distances.topk(min(k + 1, len(logits)), largest=False).indices[:, 1:]
    kernel = torch.zeros(len(logits), len(logits), device=logits.device)
    kernel.scatter_(1, neighbors, 1.0)
    kernel = (kernel + kernel.T) / 2
    probs = logits.softmax(-1)
    unary = -torch.log(probs + 1e-10)
    old_energy = float('inf')
    output = probs
    for iteration in range(100):
        pairwise = kernel @ output
        output = (-unary + pairwise).softmax(-1)
        energy = (unary * output - pairwise * output + output * output.clamp_min(1e-20).log()).sum().item()
        if iteration > 1 and abs(energy - old_energy) <= 1e-8 * abs(old_energy):
            break
        old_energy = energy
    return output


class OnlineMethod:
    def __init__(self, name, device, config, source_state=None):
        self.name = name
        self.device = device
        self.config = config
        self.model, bundle = load_frozen_backend(device)
        if bundle['class_index_map'] != {'spoof': 0, 'bonafide': 1}:
            raise ValueError('native class mapping mismatch')
        self.model.eval()
        self.updates = 0
        self.adapted_samples = 0
        self.seen = 0
        self.memory_size = 0
        self.trainable_names = []
        self.fisher = None
        self.model_probs = None
        self.diversity_threshold = None
        if name in ('bn_only', 'tent', 'eata'):
            configure_normalization_adaptation(self.model)
            self.trainable_names = sorted(key for key, parameter in self.model.named_parameters()
                                          if parameter.requires_grad)
            if not self.trainable_names:
                raise RuntimeError('INCOMPATIBLE: backend has no trainable BN affine')
            if name == 'bn_only':
                for parameter in self.model.parameters():
                    parameter.requires_grad_(False)
                self.trainable_names = []
            else:
                lr = config[f'{name}_lr']
                self.optimizer = torch.optim.SGD((p for p in self.model.parameters() if p.requires_grad),
                                                 lr=lr, momentum=0.9)
        if name == 'eata':
            if source_state is None or 'fisher' not in source_state or 'diversity_threshold' not in source_state:
                raise RuntimeError('EATA source Fisher/diversity summary is required')
            self.fisher = {key: tensor.to(device) for key, tensor in source_state['fisher'].items()}
            self.anchor = {key: parameter.detach().clone() for key, parameter in self.model.named_parameters()
                           if key in self.fisher}
            self.diversity_threshold = float(source_state['diversity_threshold'])
            if set(self.fisher) != set(self.trainable_names):
                raise ValueError('EATA Fisher parameter coverage mismatch')
        if name == 'lame':
            self.features = None
            def save_features(module, args):
                self.features = args[0].detach()
            self.hook = self.model.out_layer.register_forward_pre_hook(save_features)
        if name not in ('frozen', 'bn_only', 'tent', 'eata', 'lame', 'rotta'):
            raise ValueError(name)
        if name == 'rotta':
            self.rotta = RoTTAState(self.model, config, device)
            self.trainable_names = self.rotta.trainable_names

    def predict(self, sequence):
        with torch.no_grad():
            logits = self.rotta.predict(sequence) if self.name == 'rotta' else self.model(sequence)
            if logits.shape != (len(sequence), 2) or not torch.isfinite(logits).all():
                raise ValueError('nonfinite/wrong-shape native logits')
            if self.name == 'lame':
                if self.features is None or self.features.shape[0] != len(sequence):
                    raise ValueError('LAME penultimate feature missing')
                probabilities = lame_probabilities(logits, self.features,
                                                   min(self.config['lame_knn'], len(sequence)-1))
                score = probabilities[:, 0].clamp_min(1e-10).log() - probabilities[:, 1].clamp_min(1e-10).log()
            else:
                score = logits[:, 0] - logits[:, 1]
        return score.detach().cpu(), logits.detach()

    def adapt(self, sequence, predicted_logits):
        self.seen += len(sequence)
        if self.name == 'rotta':
            self.rotta.adapt(sequence, predicted_logits)
            self.updates = self.rotta.updates
            self.adapted_samples = self.rotta.adapted_samples
            self.memory_size = self.rotta.memory.count()
            if self.memory_size > self.config['rotta_memory_limit']:
                raise ValueError('RoTTA memory cap exceeded')
            return
        if self.name not in ('tent', 'eata'):
            return
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(sequence)
        entropies = entropy(logits)
        if self.name == 'tent':
            loss = entropies.mean()
            selected = len(sequence)
        else:
            reliable = entropies < self.config['eata_entropy_threshold']
            selected_mask = reliable.clone()
            if self.model_probs is not None and reliable.any():
                cosine = F.cosine_similarity(logits.softmax(1), self.model_probs.unsqueeze(0), dim=1)
                selected_mask &= cosine < self.diversity_threshold
            selected = int(selected_mask.sum().item())
            if selected:
                selected_entropy = entropies[selected_mask]
                coefficient = torch.exp(self.config['eata_entropy_threshold'] - selected_entropy.detach())
                loss = (selected_entropy * coefficient).mean()
                regularization = sum((self.fisher[key] * (parameter - self.anchor[key]).square()).sum()
                                     for key, parameter in self.model.named_parameters() if key in self.fisher)
                loss = loss + 2000.0 * regularization
                current = logits[selected_mask].softmax(1).detach().mean(0)
                self.model_probs = current if self.model_probs is None else 0.9*self.model_probs + 0.1*current
        if selected:
            if not torch.isfinite(loss):
                raise ValueError('nonfinite adaptation loss')
            loss.backward()
            self.optimizer.step()
            self.updates += 1
            self.adapted_samples += selected

    def stats(self):
        return {'updates': self.updates, 'adapted_samples': self.adapted_samples,
                'seen': self.seen, 'memory_size': self.memory_size,
                'trainable_names': self.trainable_names}
