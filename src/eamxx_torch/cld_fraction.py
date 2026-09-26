"""Cloud fraction (EAMxx process ``cld_fraction``).

Line-by-line translation of, at E3SM 80fe0a71:
  components/eamxx/src/physics/cld_fraction/cld_fraction_main_impl.hpp
      CldFractionFunctions::main, calc_icefrac, calc_totalfrac
  components/eamxx/src/physics/cld_fraction/eamxx_cld_fraction_process_interface.cpp
      CldFraction::create_requests (parameter defaults), run_impl

Inputs are (..., nlev) tensors; any leading shape (ncol, or ncol x nmember)
is allowed. Level order does not matter here (the scheme is pointwise).

Not translated: the Python-module backend in run_impl (an alternative EAMxx
path used only when `py_module_name` is set; off in every configuration), and
the post-condition interval checks [0, 1] (they only raise; they never change
values). `check_output` below reproduces those checks for debugging.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .ekat import max_

# CldFraction::create_requests defaults (m_params.get<double>(..., default)).
ICE_CLOUD_THRESHOLD = 1e-12
ICE_CLOUD_FOR_ANALYSIS_THRESHOLD = 1e-5


@dataclass(frozen=True)
class CldFractionParams:
    ice_cloud_threshold: float = ICE_CLOUD_THRESHOLD
    ice_cloud_for_analysis_threshold: float = ICE_CLOUD_FOR_ANALYSIS_THRESHOLD


def calc_icefrac(qi: torch.Tensor, threshold: float) -> torch.Tensor:
    """CldFractionFunctions::calc_icefrac.

    ice_cld_frac = 0; ice_cld_frac.set(qi > threshold, 1.0)
    """
    icecld = qi > threshold
    zero = torch.zeros_like(qi)
    return torch.where(icecld, torch.ones_like(qi), zero)


def calc_totalfrac(liq_cld_frac: torch.Tensor, ice_cld_frac: torch.Tensor) -> torch.Tensor:
    """CldFractionFunctions::calc_totalfrac: tot = max(ice, liq) (std::max semantics)."""
    return max_(ice_cld_frac, liq_cld_frac)


def cld_fraction(
    qi: torch.Tensor,
    cldfrac_liq: torch.Tensor,
    params: CldFractionParams = CldFractionParams(),
) -> dict[str, torch.Tensor]:
    """CldFraction::run_impl -> CldFractionFunctions::main.

    Returns the four computed fields of the EAMxx process, keyed by their
    EAMxx field names.
    """
    # main(): same call order as the source.
    ice = calc_icefrac(qi, params.ice_cloud_threshold)
    ice_4out = calc_icefrac(qi, params.ice_cloud_for_analysis_threshold)
    tot = calc_totalfrac(cldfrac_liq, ice)
    tot_4out = calc_totalfrac(cldfrac_liq, ice_4out)
    return {
        "cldfrac_ice": ice,
        "cldfrac_tot": tot,
        "cldfrac_ice_for_analysis": ice_4out,
        "cldfrac_tot_for_analysis": tot_4out,
    }


def check_output(out: dict[str, torch.Tensor]) -> torch.Tensor:
    """Post-condition checks of CldFraction::initialize_impl: every output in [0, 1].

    Returns a boolean tensor (True = all checks pass). No .item(), so it is safe
    under torch.compile; callers decide whether to raise.
    """
    ok = torch.ones((), dtype=torch.bool, device=next(iter(out.values())).device)
    for v in out.values():
        ok = ok & ((v >= 0.0) & (v <= 1.0)).all()
    return ok
