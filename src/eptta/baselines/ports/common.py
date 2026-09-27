"""Shared helpers for audited audio TTA ports (TENT/SAR/MEMO).

These operate on the full SSL-AASIST model (waveform path), never on the EP
feature cache.  ``run_method`` (the feature-cache path) must not reach here.
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as functional


def backend_batch_norm_modules(model):
    """BatchNorm1d/2d modules in the task-trained backend (excludes XLS-R frontend)."""
    result = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            if not name.startswith("ssl_model"):
                result.append((name, module))
    return result


def configure_normalization_adaptation(model):
    """TENT-style normalization-scope configuration.

    Whole model eval(); selected backend BatchNorm modules use batch statistics
    with trainable affine; every other parameter requires_grad=False.
    """
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    for _name, module in backend_batch_norm_modules(model):
        module.train()
        module.track_running_stats = False
        module.running_mean = None
        module.running_var = None
        for param in module.parameters():
            param.requires_grad = True
    return model


def configure_full_model(model):
    """Full-model (MEMO) scope: eval mode, every parameter trainable."""
    model.eval()
    for param in model.parameters():
        param.requires_grad = True
    return model


def frozen_parameters(model):
    """Snapshot parameters only (BN running stats are nulled by configuration)."""
    return {name: p.detach().clone() for name, p in model.named_parameters()}


def restore_parameters(model, snapshot):
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in snapshot:
                p.copy_(snapshot[name])


def updated_parameter_names(model):
    return sorted(name for name, p in model.named_parameters() if p.requires_grad)


def frozen_state_dict(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def restore_state_dict(model, state):
    model.load_state_dict(state)


def prediction_entropy(logits):
    """Softmax entropy per row (batch, C) -> (batch,)."""
    p = functional.softmax(logits, dim=-1)
    return -(p * p.clamp_min(1e-12).log()).sum(dim=-1)


def marginal_entropy(logits_list):
    """Entropy of the AVERAGED softmax probability (MEMO marginal entropy).

    ``logits_list`` is a list/tensor of per-augmentation logits with shape
    (n_aug, C).  This is NOT the mean of per-view entropies.
    """
    logits = torch.stack(list(logits_list), dim=0) if not torch.is_tensor(logits_list) \
        else logits_list
    p = functional.softmax(logits, dim=-1)
    marginal = p.mean(dim=0)
    return -(marginal * marginal.clamp_min(1e-12).log()).sum()
