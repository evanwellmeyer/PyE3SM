"""Helpers to compare port output with saved EAMxx reference data.

Fixture format (written by reference/scripts/extract_fixtures.py):
  one .npz per process call, with arrays "pre/<field>" and "post/<field>"
  (float64, logical shapes, first dim = column) and "meta" (a JSON string with
  proc, case, step, subcycle, rank, dt, times, and the per-field units/dims).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class Call:
    """One saved reference call of an EAMxx process."""

    path: Path
    meta: dict
    pre: dict[str, np.ndarray]
    post: dict[str, np.ndarray]

    def pre_t(self, name: str, dtype=torch.float64, device="cpu") -> torch.Tensor:
        return torch.as_tensor(self.pre[name], dtype=dtype, device=device)


def load_call(path: str | Path) -> Call:
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        pre = {k[4:]: z[k] for k in z.files if k.startswith("pre/")}
        post = {k[5:]: z[k] for k in z.files if k.startswith("post/")}
    return Call(path, meta, pre, post)


def list_calls(directory: str | Path) -> list[Path]:
    return sorted(Path(directory).glob("*.npz"))


@dataclass
class FieldDiff:
    name: str
    max_abs: float
    max_rel: float
    n_mismatch: int
    first_mismatch: tuple | None  # index of first element over tolerance

    def ok(self) -> bool:
        return self.n_mismatch == 0


def compare(
    name: str,
    port: torch.Tensor | np.ndarray,
    ref: np.ndarray,
    rtol: float = 0.0,
    atol: float = 0.0,
) -> FieldDiff:
    """Elementwise |port - ref| <= atol + rtol*|ref|. NaNs must match position.

    rtol = atol = 0 means bitwise equality of values: NaN matches NaN, and
    0.0 does not match -0.0.
    """
    p = port.detach().cpu().numpy() if isinstance(port, torch.Tensor) else np.asarray(port)
    r = np.asarray(ref)
    if p.shape != r.shape:
        raise ValueError(f"{name}: shape {p.shape} != reference {r.shape}")
    same = (p == r) | (np.isnan(p) & np.isnan(r))  # equal values, equal infs, or both NaN
    with np.errstate(invalid="ignore"):
        diff = np.where(same, 0.0, np.abs(p - r))
    diff = np.where(np.isnan(diff), np.inf, diff)  # NaN in only one, or inf vs other inf
    with np.errstate(invalid="ignore"):
        tol = atol + np.where(np.isfinite(r), rtol * np.abs(r), 0.0)
    bad = ~(diff <= tol)
    if rtol == 0.0 and atol == 0.0:
        # Bitwise mode: also require the same sign of zero (0.0 vs -0.0).
        bad |= (p == 0) & (r == 0) & (np.signbit(p) != np.signbit(r))
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(r != 0, diff / np.abs(r), np.where(diff == 0, 0.0, np.inf))
    first = tuple(int(i) for i in np.argwhere(bad)[0]) if bad.any() else None
    return FieldDiff(
        name=name,
        max_abs=float(diff.max()) if diff.size else 0.0,
        max_rel=float(rel.max()) if rel.size else 0.0,
        n_mismatch=int(bad.sum()),
        first_mismatch=first,
    )
