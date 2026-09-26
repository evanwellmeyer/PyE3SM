"""Saturation vapor pressure and mixing ratio.

Translation of components/eamxx/src/share/physics/physics_saturation_impl.hpp
at E3SM 80fe0a71: MurphyKoop_svp, polysvp1, qv_sat_dry, qv_sat_wet.

check_temperature aborts EAMxx when T <= 0 or NaN. Here, `check_temperature`
returns a boolean tensor instead (no .item(), no graph break); callers may
assert on it outside compiled regions.
"""
from __future__ import annotations

import torch

from . import constants as C
from . import mathfn
from .ekat import max_, rdiv

POLYSVP1, MURPHYKOOP = 0, 1

_IC = (9.550426, 5723.265, 3.53068, 0.00728332)
_LQ = (54.842763, 6763.22, 4.210, 0.000367, 0.0415, 218.8, 53.878, 1331.22, 9.44523, 0.014025)


def check_temperature(t_atm: torch.Tensor) -> torch.Tensor:
    """True where the EAMxx check would pass (T > 0 and not NaN)."""
    return (t_atm > 0) & ~torch.isnan(t_atm)


def MurphyKoop_svp(t_atm: torch.Tensor, ice: bool) -> torch.Tensor:
    tmelt = C.Tmelt
    result = torch.zeros_like(t_atm)
    ice_mask = (t_atm < tmelt) & torch.tensor(bool(ice), device=t_atm.device)
    if ice:
        ice_result = mathfn.exp(_IC[0] - rdiv(_IC[1], t_atm) + (_IC[2] * mathfn.log(t_atm)) - (_IC[3] * t_atm))
        result = torch.where(ice_mask, ice_result, result)
    logt = mathfn.log(t_atm)
    lq = _LQ
    liq_result = mathfn.exp(
        lq[0] - rdiv(lq[1], t_atm) - (lq[2] * logt) + (lq[3] * t_atm)
        + (mathfn.tanh(lq[4] * (t_atm - lq[5]))
           * (lq[6] - rdiv(lq[7], t_atm) - (lq[8] * logt) + lq[9] * t_atm)))
    return torch.where(~ice_mask, liq_result, result)


_AI = (6.11147274, 0.503160820, 0.188439774e-1, 0.420895665e-3, 0.615021634e-5,
       0.602588177e-7, 0.385852041e-9, 0.146898966e-11, 0.252751365e-14)
_AL = (6.11239921, 0.443987641, 0.142986287e-1, 0.264847430e-3, 0.302950461e-5,
       0.206739458e-7, 0.640689451e-10, -0.952447341e-13, -0.976195544e-15)


def polysvp1(t: torch.Tensor, ice: bool) -> torch.Tensor:
    dt = max_(t - 273.15, -80.0)
    ice_mask = (t < C.Tmelt) & torch.tensor(bool(ice), device=t.device)

    def poly(a):
        return (a[0] + dt * (a[1] + dt * (a[2] + dt * (a[3] + dt * (a[4] + dt * (a[5] + dt * (a[6] + dt * (a[7] + a[8] * dt)))))))) * 100

    result = torch.zeros_like(t)
    if ice:
        result = torch.where(ice_mask, poly(_AI), result)
    return torch.where(~ice_mask, poly(_AL), result)


def qv_sat_dry(t_atm, p_atm_dry, ice: bool, func_idx: int = MURPHYKOOP):
    if func_idx == POLYSVP1:
        e_pres = polysvp1(t_atm, ice)
    elif func_idx == MURPHYKOOP:
        e_pres = MurphyKoop_svp(t_atm, ice)
    else:
        raise ValueError("Invalid func_idx supplied to qv_sat_dry.")
    return C.ep_2 * e_pres / max_(p_atm_dry, 1.e-3)


def qv_sat_wet(t_atm, p_atm_dry, ice: bool, dp_wet, dp_dry, func_idx: int = MURPHYKOOP):
    qsatdry = qv_sat_dry(t_atm, p_atm_dry, ice, func_idx)
    return qsatdry * dp_dry / dp_wet
