"""Gradient-safe forms of expressions whose unused or singular lanes would put NaN into
autograd. Values are exactly those of the plain expressions (plan section 6: masked-out
branches get safe inputs); only the gradient changes, from 0*inf = NaN to 0.
"""
from __future__ import annotations

import torch


def nonsingular(bad: torch.Tensor, fn, *args):
    """fn(*args) for an elementwise fn, with gradient 0 on the lanes in `bad`.

    On `bad` lanes (masked out, or where fn is singular: a zero denominator, log(0), a
    root of 0) the value comes from detached inputs; the other lanes are computed from
    inputs whose `bad` lanes are replaced by 1. Arguments that are not tensors of bad's
    shape are passed through unchanged.
    """
    with torch.no_grad():
        raw = fn(*[a.detach() if isinstance(a, torch.Tensor) else a for a in args])
    safe = [torch.where(bad, torch.ones_like(a), a) if isinstance(a, torch.Tensor) and a.shape == bad.shape
            else a for a in args]
    return torch.where(bad, raw, fn(*safe))


def sqrt0(x: torch.Tensor) -> torch.Tensor:
    """torch.sqrt(x), with gradient 0 instead of 0*inf = NaN where x == 0.

    x == 0 returns x itself (detached: gradient 0), so -0 stays -0; NaN and negative
    inputs behave as torch.sqrt.
    """
    zero = x == 0
    return torch.where(zero, x.detach(), torch.sqrt(torch.where(zero, torch.ones_like(x), x)))
