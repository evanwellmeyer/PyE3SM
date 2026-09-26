"""Small indexing helpers for per-column level indices (ZM's jt, jb, jd, ...)."""
from __future__ import annotations

import torch


def gather(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """x[i, idx[i]] for x (ncol, n) and integer idx (ncol,)."""
    return torch.gather(x, -1, idx.unsqueeze(-1)).squeeze(-1)


def gather_cols(cols: list, idx: torch.Tensor) -> torch.Tensor:
    """gather() on a list of per-level column tensors."""
    return gather(torch.stack(cols, dim=-1), idx)


def set_at(x: torch.Tensor, idx: torch.Tensor, val: torch.Tensor, mask=None) -> torch.Tensor:
    """x with x[i, idx[i]] = val[i] (where mask[i], if given)."""
    lev = torch.arange(x.shape[-1], device=x.device)
    sel = lev == idx.unsqueeze(-1)
    if mask is not None:
        sel = sel & mask.unsqueeze(-1)
    return torch.where(sel, val.unsqueeze(-1).expand_as(x), x)


def set_cols(cols: list, idx: torch.Tensor, val: torch.Tensor, mask=None) -> list:
    """set_at() on a list of per-level column tensors (a new list)."""
    out = []
    for k, c in enumerate(cols):
        sel = idx == k
        if mask is not None:
            sel = sel & mask
        out.append(torch.where(sel, val, c))
    return out


def seq_sum(x: torch.Tensor) -> torch.Tensor:
    """Sum over the last axis in index order, starting from 0 (a Kokkos Serial
    reduction). torch.cumsum is a sequential loop on CPU."""
    if x.shape[-1] == 0:
        return torch.zeros(x.shape[:-1], dtype=x.dtype, device=x.device)
    return torch.cumsum(x, dim=-1)[..., -1]


def levels(n: int, device) -> torch.Tensor:
    return torch.arange(n, device=device)
