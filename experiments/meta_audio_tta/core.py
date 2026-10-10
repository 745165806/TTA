"""MABN-inspired, single-record BYOL adaptation for project SSL-AASIST.

The source meta objective is either cross-sample or same-sample.  Fast weights
are functional tensors, so an episode cannot change its starting checkpoint.
"""
from __future__ import annotations

import copy
from collections import OrderedDict

import torch
from torch import nn
from torch.func import functional_call
from torch.nn import functional as F


class Projection(nn.Module):
    def __init__(self, input_dim=160, hidden_dim=256, output_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.fc2(F.relu(self.bn(self.fc1(x))))


class BYOLSystem(nn.Module):
    """Online anti-spoofing detector and stop-gradient EMA target branch."""

    def __init__(self, detector, target_detector, class_index_map, ema_decay=0.996):
        super().__init__()
        if class_index_map != {"spoof": 0, "bonafide": 1}:
            raise ValueError("unexpected native SSL-AASIST class mapping")
        if not 0 <= ema_decay < 1:
            raise ValueError("EMA decay must be in [0,1)")
        if detector is target_detector:
            raise ValueError("BYOL target detector must be independently constructed")
        self.online_model = detector
        self.target_model = target_detector
        self.target_model.load_state_dict(detector.state_dict(), strict=True)
        self.projector = Projection()
        self.predictor = Projection(input_dim=128, output_dim=128)
        self.target_projector = copy.deepcopy(self.projector)
        self.ema_decay = float(ema_decay)
        self._online_feature = None
        self._target_feature = None
        self.online_model.out_layer.register_forward_pre_hook(self._capture_online)
        self.target_model.out_layer.register_forward_pre_hook(self._capture_target)
        for module in (self.target_model, self.target_projector):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def _capture_online(self, _module, inputs):
        self._online_feature = inputs[0]

    def _capture_target(self, _module, inputs):
        self._target_feature = inputs[0]

    def source_train_mode(self):
        self.online_model.train()
        self.projector.train()
        self.predictor.train()
        self.target_model.eval()
        self.target_projector.eval()

    def adaptation_mode(self):
        self.eval()
        self.target_model.eval()
        self.target_projector.eval()
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                if module.running_mean is None or module.running_var is None:
                    raise ValueError("MABN requires intact source BN statistics")

    def _online(self, waveform):
        self._online_feature = None
        logits = self.online_model(waveform)
        if self._online_feature is None:
            raise RuntimeError("online head feature hook did not fire")
        return logits, self._online_feature

    def _target(self, waveform):
        self._target_feature = None
        with torch.no_grad():
            self.target_model(waveform)
            if self._target_feature is None:
                raise RuntimeError("target head feature hook did not fire")
            return self.target_projector(self._target_feature)

    def byol_loss(self, views):
        if views.ndim != 2 or views.shape[0] != 2:
            raise ValueError("BYOL needs two views of exactly one audio")
        _logits, features = self._online(views)
        predictions = F.normalize(self.predictor(self.projector(features)), dim=-1)
        targets = F.normalize(self._target(views), dim=-1).detach()
        loss = 2.0 - 2.0 * (predictions * targets.flip(0)).sum(dim=-1)
        return loss.mean()

    def forward(self, waveform, mode="logits"):
        if mode == "logits":
            return self._online(waveform)[0]
        if mode == "byol":
            return self.byol_loss(waveform)
        raise ValueError("unknown BYOLSystem mode")

    @torch.no_grad()
    def update_ema(self):
        """Call exactly once after an online optimizer step, never inside an episode."""
        for target, online in zip(self.target_model.parameters(), self.online_model.parameters()):
            target.lerp_(online, 1.0 - self.ema_decay)
        for target, online in zip(self.target_projector.parameters(), self.projector.parameters()):
            target.lerp_(online, 1.0 - self.ema_decay)
        # Target BN always uses source statistics copied after source training.
        for target, online in zip(self.target_model.buffers(), self.online_model.buffers()):
            target.copy_(online)
        for target, online in zip(self.target_projector.buffers(), self.projector.buffers()):
            target.copy_(online)


def selected_bn_affine_names(system: BYOLSystem):
    """Only task-trained backend BN and auxiliary projector/predictor BN affine."""
    names = []
    for module_name, module in system.named_modules():
        if not isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            continue
        if not (module_name.startswith("online_model.") or
                module_name.startswith("projector.") or
                module_name.startswith("predictor.")):
            continue
        if module_name.startswith("online_model.ssl_model."):
            continue
        # In the pinned author Residual_block.forward, encoder[1:6].0.bn1
        # feeds a local `out` that is then overwritten by conv1(x).  These
        # tensors exist in state_dict but cannot affect detector predictions.
        if module_name in {f"online_model.encoder.{i}.0.bn1" for i in range(1, 6)}:
            continue
        for local in ("weight", "bias"):
            if getattr(module, local) is not None:
                names.append(module_name + "." + local)
    backbone = [name for name in names if name.startswith("online_model.")]
    if not backbone:
        raise ValueError("no task-trained backbone BN affine parameters")
    return tuple(sorted(names))


def configure_meta(system: BYOLSystem):
    names = set(selected_bn_affine_names(system))
    for name, parameter in system.named_parameters():
        parameter.requires_grad_(name in names)
    system.adaptation_mode()
    return tuple(sorted(names))


def fast_update(system, support_views, inner_lr, *, create_graph):
    if inner_lr <= 0:
        raise ValueError("inner_lr must be positive")
    system.adaptation_mode()
    names = selected_bn_affine_names(system)
    params = OrderedDict(system.named_parameters())
    loss = system(support_views, mode="byol")
    grads = torch.autograd.grad(loss, [params[name] for name in names],
                                create_graph=create_graph, allow_unused=True)
    unused = [name for name, grad in zip(names, grads) if grad is None]
    if unused:
        raise RuntimeError("BYOL has no gradient for selected BN affine: " + ", ".join(unused))
    fast = OrderedDict(params)
    for name, grad in zip(names, grads):
        fast[name] = params[name] - inner_lr * grad
    return fast, loss, dict(zip(names, grads))


def score(system, waveform, fast=None):
    if waveform.ndim != 2 or waveform.shape[0] != 1:
        raise ValueError("score expects one original waveform")
    logits = system(waveform, mode="logits") if fast is None else functional_call(
        system, fast, (waveform,), {"mode": "logits"}, strict=False)
    if logits.shape != (1, 2) or not torch.isfinite(logits).all():
        raise FloatingPointError("invalid native detector logits")
    return logits[0, 0] - logits[0, 1]


def meta_objective(system, support_views, query_views, canonical_label, inner_lr,
                   auxiliary_weight=0.1):
    """Second-order outer objective; caller chooses independent B or support A."""
    if canonical_label not in (0, 1):
        raise ValueError("query label must be canonical 0/1")
    fast, support_loss, gradients = fast_update(
        system, support_views, inner_lr, create_graph=True)
    native_label = 1 - canonical_label
    query_logits = functional_call(system, fast, (query_views[:1],),
                                   {"mode": "logits"}, strict=False)
    ce = F.cross_entropy(query_logits, torch.tensor([native_label], device=query_logits.device))
    query_ssl = functional_call(system, fast, (query_views,),
                                {"mode": "byol"}, strict=False)
    outer = ce + auxiliary_weight * query_ssl
    return outer, {"support_byol": support_loss, "query_ce": ce,
                   "query_byol": query_ssl, "fast": fast, "inner_gradients": gradients}


def episodic_scores(system, original, views, inner_lr):
    """Stateless K=0/K=1 pair; no EMA or optimizer state can cross records."""
    system.adaptation_mode()
    before = score(system, original)
    fast, loss, grads = fast_update(system, views, inner_lr, create_graph=False)
    after = score(system, original, fast)
    return before.detach(), after.detach(), loss.detach(), grads
