"""Source diagnostic helpers shared by the archived meta-audio experiments.

These helpers load project-trained checkpoints and inspect labeled source
diagnostics. They do not perform target adaptation or read target labels.
"""

import numpy as np
import torch

from experiments.meta_audio_tta.core import BYOLSystem, configure_meta
from experiments.meta_audio_tta.stage1_source import _construction
from workers.compat.author_training import build_author_model


def load_checkpoint(source_config, checkpoint, device):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    online, _ = build_author_model(_construction(source_config), device)
    online.model.load_state_dict(state["model_state"], strict=True)
    if state.get("byol_state") is None:
        if state.get("mode") != "ce":
            raise ValueError("unrecognized CE-only checkpoint")
        online.model.eval()
        return online.model, None
    target, _ = build_author_model(_construction(source_config), device)
    system = BYOLSystem(
        online.model,
        target.model,
        source_config["architecture"]["class_index_map"],
        ema_decay=source_config["ema_decay"],
    ).to(device)
    saved = state["byol_state"]
    for key, module in (
        ("projector", system.projector),
        ("predictor", system.predictor),
        ("target_model", system.target_model),
        ("target_projector", system.target_projector),
    ):
        module.load_state_dict(saved[key], strict=True)
    configure_meta(system)
    return system.online_model, system


def pair_flips(before, after, labels):
    before = np.asarray(before, dtype=np.float64)
    after = np.asarray(after, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    fake0, real0 = before[labels == 1], before[labels == 0]
    fake1, real1 = after[labels == 1], after[labels == 0]
    helpful = harmful = ties = 0
    for start in range(0, len(fake0), 128):
        first = np.sign(fake0[start:start + 128, None] - real0[None, :])
        second = np.sign(fake1[start:start + 128, None] - real1[None, :])
        helpful += int(np.count_nonzero((first < 0) & (second > 0)))
        harmful += int(np.count_nonzero((first > 0) & (second < 0)))
        ties += int(np.count_nonzero((first != second) & ((first == 0) | (second == 0))))
    return {
        "total_pairs": len(fake0) * len(real0),
        "incorrect_to_correct": helpful,
        "correct_to_incorrect": harmful,
        "tie_status_changed": ties,
    }
