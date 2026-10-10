"""Generate one linear head from an unlabeled domain descriptor."""
import torch
from torch import nn


class DomainHeadNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(320, 128), nn.ReLU(), nn.Linear(128, 161))
        # Start at the audited source classifier, then learn a correction.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, support_z):
        if support_z.ndim != 2 or support_z.shape[1] != 160 or len(support_z) < 2:
            raise ValueError("unlabeled support must be [N,160], N>=2")
        if not torch.isfinite(support_z).all():
            raise ValueError("nonfinite support")
        descriptor = torch.cat((support_z.mean(0), support_z.std(0, unbiased=False)))
        correction = self.net(descriptor)
        return correction[:160], correction[160]


def target_head(net, support_z, source_w, source_b):
    """No labels, optimizer or backward path is part of target generation."""
    delta_w, delta_b = net(support_z)
    return source_w + delta_w, source_b + delta_b, delta_w, delta_b
