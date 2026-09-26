"""Field property checks that can repair values.

Translation of, at E3SM 80fe0a71:
  components/eamxx/src/share/property_checks/field_within_interval_check.cpp
      FieldWithinIntervalCheck constructor, check_impl, repair_impl
  components/eamxx/src/share/atm_process/atmosphere_process.cpp
      run_property_check (Pass / Repairable -> repair / Fail)

EAMxx runs a check over all columns of an MPI rank. Here the scope is the
tensor passed in (all columns of the call). Repair clamps every value, so the
scope matters only for values that clamping changes while in range: -0.0
becomes +0.0 when lb == 0, and NaN becomes ub (std::min(ub, NaN) == ub).

A failed, unrepairable check raises in EAMxx. Here it is reported through the
returned status, because raising would need .item() (no GPU sync, no graph
break). Callers decide what to do with it.
"""
from __future__ import annotations

import sys

import torch

from .ekat import max_, min_

PASS, REPAIRED, FAIL = 0, 1, 2
S_MAX = sys.float_info.max  # FieldWithinIntervalCheck::s_max


def _minmax(f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Kokkos MinMax reduction: NaN never compares less/greater, so it is skipped."""
    big = torch.finfo(f.dtype).max
    fmin = torch.where(torch.isnan(f), torch.full_like(f, big), f).amin()
    fmax = torch.where(torch.isnan(f), torch.full_like(f, -big), f).amax()
    return fmin, fmax


def within_interval(
    f: torch.Tensor,
    lb: float,
    ub: float,
    can_repair: bool,
    lb_repairable: float = -S_MAX,
    ub_repairable: float = S_MAX,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (field after the check, status in {PASS, REPAIRED, FAIL}).

    Repairable bounds only apply when can_repair (constructor logic).
    """
    lb_rep, ub_rep = (lb_repairable, ub_repairable) if can_repair else (lb, ub)
    fmin, fmax = _minmax(f)
    passed = (fmin >= lb) & (fmax <= ub)
    failed = ~passed & ((fmin < lb_rep) | (fmax > ub_rep))
    repairable = ~passed & ~failed
    if can_repair:
        # repair_impl: ref = min(ub, ref); ref = max(lb, ref)
        repaired = max_(torch.full_like(f, lb), min_(torch.full_like(f, ub), f))
        f = torch.where(repairable, repaired, f)
        status = torch.where(passed, PASS, torch.where(failed, FAIL, REPAIRED))
    else:
        status = torch.where(passed, PASS, FAIL)
    return f, status.to(torch.int8)
