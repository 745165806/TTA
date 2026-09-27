"""Source-trained residual 160D feature map and supervised two-view contrastive loss."""
import torch
from torch import nn
from torch.nn import functional as F


class ResidualDetector(nn.Module):
    def __init__(self, source_w, source_b):
        super().__init__()
        self.w1 = nn.Linear(160, 32)
        self.w2 = nn.Linear(32, 160)
        nn.init.zeros_(self.w2.weight)
        nn.init.zeros_(self.w2.bias)
        self.head = nn.Linear(160, 1)
        with torch.no_grad():
            self.head.weight.copy_(source_w.reshape(1, 160))
            self.head.bias.copy_(source_b.reshape(1))

    def embed(self, z):
        return z + self.w2(F.gelu(self.w1(z)))

    def forward(self, z):
        return self.head(self.embed(z)).squeeze(-1)


def supervised_contrastive(h0, h1, labels, temperature):
    h = F.normalize(torch.cat((h0, h1)), dim=1)
    y = labels.repeat(2)
    logits = h @ h.T / temperature
    same = y[:, None].eq(y[None, :])
    eye = torch.eye(len(h), device=h.device, dtype=torch.bool)
    positives = same & ~eye
    if not positives.any(dim=1).all():
        raise ValueError("contrastive batch lacks a positive")
    logits = logits.masked_fill(eye, float("-inf"))
    logp = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    return -(logp.masked_fill(~positives, 0).sum(1) / positives.sum(1)).mean()
