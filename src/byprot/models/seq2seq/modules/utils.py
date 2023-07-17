import copy
from functools import partial

import torch
import torch.functional as F
import torch.nn as nn


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class ResNorm(nn.Module):
    def __init__(
        self,
        net,
        dim,
        alpha=1.0,
        dropout=0.0,
        normalize_before=True,
        norm=None,
        fn=None
    ):
        super().__init__()

        self.net = net
        self.fn = partial(
            fn or (lambda net, *args, **kwargs: net(*args, **kwargs)), self.net
        )
        self.normalize_before = normalize_before
        self.norm = norm or nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.alpha = alpha

        self.reset_parameters()

    def reset_parameters(self):
        self.net.reset_parameters()

    def forward(self, x, *args, **kwargs):
        identity = x

        if self.normalize_before:
            x = self.norm(x)

        x = self.fn(x, *args, **kwargs)
        x = [x] if not isinstance(x, tuple) else list(x)

        x[0] = self.alpha * identity + self.dropout(x[0])

        if not self.normalize_before:
            x[0] = self.norm(x[0])

        return x[0] if len(x) == 1 else tuple(x)

    def extra_repr(self):
        lines = f"(normalize_before): {self.normalize_before}"
        return lines


def apply_weight_norm(net: nn.Module):
    for name, module in net.named_modules():
        if isinstance(module, nn.Linear):
            nn.utils.weight_norm(module, name='weight')


class RepeatedSequential(nn.Sequential):
    def forward(self, x, *args, **kwds):
        for m in self:
            x = m(x, *args, **kwds)
        return x
