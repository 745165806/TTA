"""Audio LL port of RoTTA: robust BN, CSTU memory, EMA teacher and temporal loss."""
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F


class RobustBN(nn.Module):
    def __init__(self, source, alpha):
        super().__init__()
        if not isinstance(source, (nn.BatchNorm1d, nn.BatchNorm2d)) or source.running_mean is None:
            raise ValueError('RoTTA requires source BN running statistics')
        self.register_buffer('source_mean', source.running_mean.detach().clone())
        self.register_buffer('source_var', source.running_var.detach().clone())
        self.weight = nn.Parameter(source.weight.detach().clone())
        self.bias = nn.Parameter(source.bias.detach().clone())
        self.alpha = alpha
        self.eps = source.eps
        self.dim = 1 if isinstance(source, nn.BatchNorm1d) else 2

    def forward(self, x):
        reduce = (0,) if x.ndim == 2 else (0,) + tuple(range(2, x.ndim))
        if self.training:
            var, mean = torch.var_mean(x, dim=reduce, unbiased=False)
            mixed_mean = (1-self.alpha)*self.source_mean + self.alpha*mean
            mixed_var = (1-self.alpha)*self.source_var + self.alpha*var
            self.source_mean = mixed_mean.detach()
            self.source_var = mixed_var.detach()
        else:
            mixed_mean, mixed_var = self.source_mean, self.source_var
        shape = (1, -1) + (1,)*(x.ndim-2)
        return (x - mixed_mean.view(shape)) / torch.sqrt(mixed_var.view(shape) + self.eps) * self.weight.view(shape) + self.bias.view(shape)


def wrap_robust_bn(model, alpha):
    names = [name for name, module in model.named_modules()
             if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d))]
    if not names:
        raise RuntimeError('INCOMPATIBLE: RoTTA backend has no BN')
    model.requires_grad_(False)
    for name in names:
        parent_name, _, child_name = name.rpartition('.')
        parent = model.get_submodule(parent_name) if parent_name else model
        old = model.get_submodule(name)
        wrapper = RobustBN(old, alpha)
        setattr(parent, child_name, wrapper)
    model.eval()
    trainable = sorted(name for name, parameter in model.named_parameters() if parameter.requires_grad)
    if not trainable or any(not name.endswith(('weight', 'bias')) or '.source_' in name for name in trainable):
        raise ValueError('RoTTA normalization scope mismatch')
    return trainable


class CSTUMemory:
    def __init__(self, capacity=64, num_class=2):
        self.capacity = capacity
        self.num_class = num_class
        self.data = [[] for _ in range(num_class)]

    def count(self):
        return sum(len(items) for items in self.data)

    def heuristic(self, age, uncertainty):
        return 1 / (1 + math.exp(-age / self.capacity)) + uncertainty / math.log(self.num_class)

    def add(self, views, predicted_class, uncertainty):
        for group in self.data:
            for item in group:
                item[2] += 1
        score_new = self.heuristic(0, uncertainty)
        current = self.data[predicted_class]
        if len(current) < self.capacity / self.num_class and self.count() < self.capacity:
            current.append([views.detach().cpu().clone(), uncertainty, 0])
            return
        if len(current) >= self.capacity / self.num_class:
            candidate_classes = [predicted_class]
        else:
            largest = max(len(items) for items in self.data)
            candidate_classes = [index for index, items in enumerate(self.data) if len(items) == largest]
        worst = None
        for cls in candidate_classes:
            for index, item in enumerate(self.data[cls]):
                score = self.heuristic(item[2], item[1])
                if worst is None or score >= worst[0]:
                    worst = (score, cls, index)
        if worst is None or worst[0] <= score_new:
            return
        self.data[worst[1]].pop(worst[2])
        self.data[predicted_class].append([views.detach().cpu().clone(), uncertainty, 0])

    def records(self):
        return [item for group in self.data for item in group]


class RoTTAState:
    def __init__(self, model, config, device):
        self.student = model
        self.device = device
        self.trainable_names = wrap_robust_bn(model, 0.05)
        self.teacher = copy.deepcopy(model).eval()
        self.teacher.requires_grad_(False)
        self.optimizer = torch.optim.Adam((parameter for parameter in model.parameters() if parameter.requires_grad),
                                          lr=config['rotta_lr'])
        self.memory = CSTUMemory(capacity=config['rotta_memory_limit'])
        self.frequency = config['rotta_memory_limit']
        self.nu = 0.001
        self.seen = 0
        self.updates = 0
        self.adapted_samples = 0

    def predict(self, clean):
        self.teacher.eval()
        with torch.no_grad():
            return self.teacher(clean)

    def adapt(self, views, first_logits):
        if views.ndim != 4 or views.shape[1:] != (3, 201, 128):
            raise ValueError('RoTTA requires three mapped audio LL views')
        prob = first_logits.softmax(1)
        classes = prob.argmax(1).tolist()
        uncertainties = (-(prob * prob.clamp_min(1e-10).log()).sum(1)).tolist()
        for i in range(len(views)):
            self.memory.add(views[i], classes[i], uncertainties[i])
            self.seen += 1
            if self.seen % self.frequency == 0:
                self.update()

    def update(self):
        records = self.memory.records()
        if not records:
            return
        views = torch.stack([item[0] for item in records]).to(self.device)
        ages = torch.tensor([item[2] for item in records], device=self.device, dtype=torch.float32)
        clean = views[:, 0]
        # Views 1 and 2 originate from waveform noise/FIR before the frozen SSL frontend.
        stronger = views[torch.arange(len(views), device=self.device), 1 + (torch.arange(len(views), device=self.device) % 2)]
        self.student.train()
        self.teacher.eval()
        with torch.no_grad():
            teacher_logits = self.teacher(clean)
        student_logits = self.student(stronger)
        target = teacher_logits.softmax(1)
        cross_entropy = -(target * student_logits.log_softmax(1)).sum(1)
        weights = torch.exp(-ages) / (1 + torch.exp(-ages))
        loss = (cross_entropy * weights).mean()
        if not torch.isfinite(loss):
            raise ValueError('RoTTA nonfinite loss')
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            for teacher_parameter, student_parameter in zip(self.teacher.parameters(), self.student.parameters()):
                teacher_parameter.mul_(1-self.nu).add_(student_parameter, alpha=self.nu)
            # RoTTA's robust BN statistics also belong to teacher state.
            for teacher_buffer, student_buffer in zip(self.teacher.buffers(), self.student.buffers()):
                if teacher_buffer.shape == student_buffer.shape:
                    teacher_buffer.copy_(student_buffer)
        self.student.eval()
        self.updates += 1
        self.adapted_samples += len(records)
