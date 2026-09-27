"""Audio-native parameter and normalization mapping for episodic TTA.

The mapping deliberately preserves normal inference semantics.  In particular,
BatchNorm never switches to per-test-sample statistics: running buffers remain
active and immutable while selected affine parameters may receive gradients.
"""
from __future__ import annotations

import torch
import torch.nn as nn


NORM_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)
SCOPE_A = "backend_norm_affine_v1"
SCOPE_B = "backend_norm_plus_graph_modulation_v1"
GRAPH_TENSOR_TOKENS = ("att_weight", "pos_S", "master1", "master2")


def component_for_name(name: str) -> str:
    if name == "ssl_model" or name.startswith("ssl_model."):
        return "frontend"
    if name == "out_layer" or name.startswith("out_layer."):
        return "head"
    return "backend"


def _norm_affine_names(model):
    names = set()
    for module_name, module in model.named_modules():
        if component_for_name(module_name) != "backend" or not isinstance(module, NORM_TYPES):
            continue
        for local_name, parameter in module.named_parameters(recurse=False):
            if local_name in ("weight", "bias") and parameter is not None:
                names.add("%s.%s" % (module_name, local_name) if module_name else local_name)
    return names


def preregistered_parameter_names(model, scope_id=SCOPE_A):
    """Return an audited allow-list; never infer a scope from target behavior."""
    if scope_id not in (SCOPE_A, SCOPE_B):
        raise ValueError("unknown audio-native parameter scope: %s" % scope_id)
    names = _norm_affine_names(model)
    if scope_id == SCOPE_B:
        for name, _parameter in model.named_parameters():
            if component_for_name(name) == "backend" and any(
                    token in name.split(".")[-1] for token in GRAPH_TENSOR_TOKENS):
                names.add(name)
    if not names:
        raise ValueError("audio-native scope selects no parameters")
    if any(component_for_name(name) != "backend" or name.startswith("out_layer") for name in names):
        raise AssertionError("audio-native allow-list escaped the task-trained backend")
    return sorted(names)


def configure_audio_native(model, scope_id=SCOPE_A):
    """Freeze the model, retain eval semantics, and enable the selected allow-list."""
    model.eval()
    selected = set(preregistered_parameter_names(model, scope_id))
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name in selected
    # Explicitly enforce source-statistics behavior even if a caller previously
    # configured a reference TENT port.
    for _name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            module.eval()
            if not module.track_running_stats or module.running_mean is None or module.running_var is None:
                raise ValueError("audio-native BN requires intact source running statistics")
    return model


def configure_full_safeaug(model):
    """MEMO full-model scope while retaining source-statistics inference semantics."""
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = True
    for _name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            module.eval()
            if not module.track_running_stats or module.running_mean is None or module.running_var is None:
                raise ValueError("MEMO-FullSafeAug requires intact source BN statistics")
    return model


def snapshot_episode_state(model):
    """Clone parameters and buffers so an episode can be reset exactly."""
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def reset_episode_state(model, state):
    model.load_state_dict(state, strict=True)
    return model


def assert_bn_buffers_unchanged(model, before):
    current = model.state_dict()
    changed = []
    for name, value in before.items():
        if name.endswith(("running_mean", "running_var", "num_batches_tracked")):
            if not torch.equal(value, current[name]):
                changed.append(name)
    if changed:
        raise RuntimeError("audio-native adaptation mutated BN source statistics: %s" % changed)


def selected_parameters(model):
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def selected_parameter_names(model):
    return sorted(name for name, parameter in model.named_parameters() if parameter.requires_grad)


def audit_candidates(model):
    """Machine-readable inventory of normalization and explicit modulation candidates."""
    rows = []
    norm_classes = NORM_TYPES
    for module_name, module in model.named_modules():
        if not isinstance(module, norm_classes):
            continue
        params = [("%s.%s" % (module_name, local) if module_name else local, p)
                  for local, p in module.named_parameters(recurse=False)]
        component = component_for_name(module_name)
        rows.append({
            "module_name": module_name,
            "module_type": type(module).__name__,
            "parameter_names": [name for name, _p in params],
            "parameter_count": sum(p.numel() for _name, p in params),
            "component": component,
            "weight_origin": "generic-pretrained" if component == "frontend" else "task-trained",
            "scope_a_selected": any(name in _norm_affine_names(model) for name, _p in params),
        })
    for name, parameter in model.named_parameters():
        if component_for_name(name) == "backend" and any(
                token in name.split(".")[-1] for token in GRAPH_TENSOR_TOKENS):
            rows.append({
                "module_name": name,
                "module_type": "StandaloneParameter",
                "parameter_names": [name],
                "parameter_count": parameter.numel(),
                "component": "backend",
                "weight_origin": "task-trained",
                "scope_a_selected": False,
                "scope_b_selected": True,
            })
    return rows
