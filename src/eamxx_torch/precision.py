"""Run the port in a chosen floating precision and device.

Data loaded once (RRTMGP coefficients, ZM's saturation table, SPA's climatology, host
geometry) is kept in float64 on the CPU as read, and cast to the state's precision and
device where a scheme uses it. PyTorch promotes float32 (op) float64 to float64 silently,
so a float64 constant or table left in a float32 run would make parts of it double
(reference/scripts/dtype_check.py finds such leaks).

P3's tables are the exception: EAMxx's single-precision build reads different files for
its non-ice tables (*.dat4, computed in float), so they are loaded in the working
precision (p3.load_p3_tables(dtype=...)), not cast.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import torch

_CACHE: dict = {}


def cast(obj, dtype: torch.dtype, device=None):
    """obj with every floating tensor converted to dtype (and every tensor moved to device).

    Works through dataclasses, dicts, lists and tuples; other values are returned as they
    are. Results are cached per (object, dtype, device), so repeated calls are cheap.
    """
    device = torch.device(device) if device is not None else None
    key = (id(obj), dtype, device)
    hit = _CACHE.get(key)
    if hit is not None and hit[0] is obj:
        return hit[1]
    out = _cast(obj, dtype, device)
    _CACHE[key] = (obj, out)
    return out


def _cast(obj, dtype, device):
    if isinstance(obj, torch.Tensor):
        if not obj.is_floating_point():
            return obj.to(device) if device is not None else obj
        if obj.device.type == "mps" and device is not None:   # MPS has no float64: move first
            return obj.to(device).to(dtype)
        t = obj.to(dtype)                                     # else precision first (to MPS)
        return t.to(device) if device is not None else t
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        changes = {f.name: _cast(getattr(obj, f.name), dtype, device) for f in dataclasses.fields(obj)
                   if f.init}
        return dataclasses.replace(obj, **changes)
    if isinstance(obj, dict):
        return {k: _cast(v, dtype, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)) and not isinstance(obj, str):
        return type(obj)(_cast(v, dtype, device) for v in obj)
    return obj


def like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """x in ref's dtype and device (numpy arrays and floats too)."""
    if isinstance(x, np.ndarray) or not isinstance(x, torch.Tensor):
        return torch.as_tensor(x, dtype=ref.dtype, device=ref.device)
    return x.to(dtype=ref.dtype, device=ref.device)


def to_precision(obj, dtype: torch.dtype, device=None):
    """cast without the cache, for values that change every call (a state dict)."""
    return _cast(obj, dtype, torch.device(device) if device is not None else None)


def double_device(device) -> str:
    """device if it computes in float64, else "cpu" (MPS has no float64)."""
    return "cpu" if torch.device(device).type == "mps" else str(device)
