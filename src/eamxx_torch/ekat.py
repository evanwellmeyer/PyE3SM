"""Scalar helpers from EKAT, with the exact semantics of the CPU reference build.

Source: externals/ekat/src/kokkos/ekat_math_utils.hpp (namespace ekat::impl) and
externals/ekat/src/pack/ekat_pack.hpp (ekat_pack_gen_bin_fn_all(min/max, impl::min/max)).

On CPU builds ekat::impl::min/max are std::min/std::max:
    std::max(a, b) = (a < b) ? b : a
    std::min(a, b) = (b < a) ? b : a
They differ from torch.maximum/torch.minimum only when an argument is NaN
(torch propagates NaN; std::max returns `a` whenever a comparison with NaN is
false). The GPU build of EKAT uses `a > b ? a : b`, which differs again for NaN.
We reproduce the CPU form, because that is the reference we compare against.
"""
from __future__ import annotations

import torch


def _pair(a, b):
    ref = a if isinstance(a, torch.Tensor) else b
    a = torch.as_tensor(a, dtype=ref.dtype, device=ref.device)
    b = torch.as_tensor(b, dtype=ref.dtype, device=ref.device)
    return a, b


def max_(a: torch.Tensor | float, b: torch.Tensor | float) -> torch.Tensor:
    """std::max(a, b) elementwise: (a < b) ? b : a. Either argument may be a scalar."""
    a, b = _pair(a, b)
    return torch.where(a < b, b, a)


def min_(a: torch.Tensor | float, b: torch.Tensor | float) -> torch.Tensor:
    """std::min(a, b) elementwise: (b < a) ? b : a. Either argument may be a scalar."""
    a, b = _pair(a, b)
    return torch.where(b < a, b, a)


def rdiv(a: float, t: torch.Tensor) -> torch.Tensor:
    """a / t for a Python scalar a, correctly rounded.

    PyTorch evaluates `scalar / tensor` as reciprocal(tensor) * scalar, which
    differs from IEEE division in the last bit for ~25% of inputs (measured).
    Dividing a 0-dim tensor by a tensor uses true division. Port code must use
    this helper instead of `scalar / tensor` (tests/conftest.py enforces it).
    """
    return torch.as_tensor(a, dtype=t.dtype, device=t.device) / t
