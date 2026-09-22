"""Two-phase Sharpness-Aware Minimization (SAM) with parameter-bound state.

The perturbation is stored per parameter (``self.state[p]``), never in a
pop-from-list order that can drift when the first and second backward passes
have different grad sets.
"""
import torch


class SAM:
    def __init__(self, params, base_optimizer, rho=0.05):
        self.params = list(params)
        self.base_optimizer = base_optimizer(self.params)
        self.rho = rho
        self.state = {}
        self.sam_grad_norm_first = None
        self.sam_perturb_norm = None
        self.sam_grad_norm_second = None

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
        self.sam_grad_norm_first = float(norm.item())
        scale = self.rho / (norm + 1e-12)
        perturb_sq = 0.0
        with torch.no_grad():
            for p in self.params:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                p.add_(e_w)
                self.state[p] = e_w
                perturb_sq += float((e_w * e_w).sum().item())
        self.sam_perturb_norm = perturb_sq ** 0.5

    def second_step(self):
        # Restore every perturbed parameter from parameter-bound state, independent
        # of whether the second backward pass produced a gradient for it.
        with torch.no_grad():
            for p, e_w in list(self.state.items()):
                p.sub_(e_w)
            self.state.clear()
        self.sam_grad_norm_second = float(self._grad_norm().item())
        self.base_optimizer.step()
        self.base_optimizer.zero_grad()

    def zero_grad(self):
        self.base_optimizer.zero_grad()
