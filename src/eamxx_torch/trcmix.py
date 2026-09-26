"""Prescribed trace-gas mass mixing ratios.

Translation of components/eamxx/src/share/physics/eamxx_trcmix.cpp (trcmix),
E3SM 80fe0a71. clang turns std::pow(std::cos(clat_r), 2) into a product, so the
port multiplies.
"""
from __future__ import annotations

import torch

from . import constants as C
from . import mathfn


def trcmix(name: str, clat: torch.Tensor, pmid: torch.Tensor, co2vmr: float, n2ovmr: float,
           ch4vmr: float, f11vmr: float, f12vmr: float) -> torch.Tensor:
    """Mass mixing ratio of gas `name` (ncol, nlev). clat in degrees, (ncol,)."""
    mw = C.GAS_MOL_WEIGHT
    rmwn2o = mw["n2o"] / C.MWdry
    rmwch4 = mw["ch4"] / C.MWdry
    rmwf11 = mw["cfc11"] / C.MWdry
    rmwf12 = mw["cfc12"] / C.MWdry
    rmwco2 = mw["co2"] / C.MWdry
    const_map = {
        "ch4": (rmwch4 * ch4vmr, 0.2353, 0., 0.2353, 0.0225489),
        "n2o": (rmwn2o * n2ovmr, 0.3478, 0.00116, 0.4000, 0.013333),
        "cfc11": (rmwf11 * f11vmr, 0.7273, 0.00606, 1., 0.013333),
        "cfc12": (rmwf12 * f12vmr, 0.4000, 0.00222, 0.5, 0.024444),
    }
    if name in ("o2", "co2"):
        val = C.o2mmr if name == "o2" else rmwco2 * co2vmr
        return torch.full_like(pmid, val)
    if name not in const_map:
        raise ValueError("trcmix: Unhandled gas_name: " + name)
    trop_mmr, s1b, s1f, s2b, s2f = const_map[name]
    clat_r = (clat * C.Pi / 180.0)[:, None]
    dlat = torch.abs(57.2958 * clat_r)
    scale = torch.where(dlat <= 45.0, s1b + s1f * dlat, s2b + s2f * (dlat - 45))
    c = mathfn.cos(clat_r)
    ptrop = 250.0e2 - 150.0e2 * (c * c)
    pratio = pmid / ptrop
    below = pmid >= ptrop
    strat = trop_mmr * mathfn.pow(torch.where(below, torch.ones_like(pratio), pratio),
                                  scale.expand_as(pmid))
    return torch.where(below, torch.full_like(pmid, trop_mmr), strat)
