"""Two-phase Sharpness-Aware Minimization (SAM) for the SAR port.

This is a real first-step / second-step SAM, not gradient clipping.
"""
import torch


class SAM:
    def __init__(self, params, base_optimizer, rho=0.05):
        self.params = list(params)
        self.base_optimizer = base_optimizer(self.params)
        self.rho = rho
        self.state = {}

    def _grad_norm(self):
        norms = []
        for p in self.params:
            if p.grad is None:
                continue
            norms.append(torch.linalg.vector_norm(p.grad))
        if not norms:
            return torch.tensor(0.0, device=self.params[0].device)
        return torch.linalg.vector_norm(torch.stack(norms))

    def first_step(self):
        norm = self._grad_norm()
        scale = self.rho / (norm + 1e-12)
        self._ew = []
        with torch.no_grad():
            for p in self.params:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                p.add_(e_w)
                self._ew.append(e_w)

    def second_step(self):
        with torch.no_grad():
            for p in self.params:
                if p.grad is None:
                    continue
                if self._ew:
                    p.sub_(self._ew.pop(0))
        self.base_optimizer.step()
        self.base_optimizer.zero_grad()

    def zero_grad(self):
        self.base_optimizer.zero_grad()
