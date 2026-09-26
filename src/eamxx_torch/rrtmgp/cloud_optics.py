"""RRTMGP cloud optics from lookup tables.

Translation of CloudOpticsK::cloud_optics, compute_all_from_table and combine
(extensions/cloud_optics/mo_cloud_optics.h, Kokkos path), and of EAMxx's
get_cloud_optics_sw/lw and limit_to_bounds_k (eamxx_rrtmgp_interface.hpp),
E3SM 80fe0a71. EAMxx loads the lookup-table coefficients, so the Pade path is
not used and not ported.
"""
from __future__ import annotations

import torch

from ..ekat import max_, min_
from .data import CloudOpticsData

ICE_ROUGHNESS = 2             # get_cloud_optics_*: cloud_optics.set_ice_roughness(2)


def limit_to_bounds(x, lower: float, upper: float):
    """std::min(std::max(x, lower), upper)."""
    return min_(max_(x, lower), upper)


def compute_all_from_table(mask, lwp, re, nsteps, step_size, offset, tau_table, ssa_table,
                           asy_table):
    """Returns tau, taussa, taussag (ncol, nlay, nbnd); zero where mask is false."""
    x = (re - offset) / step_size
    index = (torch.fmin(torch.floor(x) + 1, torch.tensor(nsteps - 1., dtype=re.dtype,
                                                         device=re.device)) - 1).to(torch.int64)
    fint = (x - index.to(re.dtype))[..., None]
    lw = lwp[..., None]

    def interp(tab):
        t0 = tab[index]
        return t0 + fint * (tab[index + 1] - t0)

    t = lw * interp(tau_table)
    ts = t * interp(ssa_table)
    tsg = ts * interp(asy_table)
    m = mask[..., None]
    zero = torch.zeros((), dtype=re.dtype, device=re.device)
    return torch.where(m, t, zero), torch.where(m, ts, zero), torch.where(m, tsg, zero)


def cloud_optics(co: CloudOpticsData, clwp, ciwp, reliq, reice, two_stream: bool):
    """CloudOpticsK::cloud_optics with the effective radii already limited to the tables.

    Returns (tau, ssa, g) for two_stream, else tau alone (absorption optical depth).
    """
    liqmsk = clwp > 0.
    icemsk = ciwp > 0.
    dt = clwp.dtype
    ltau, ltaussa, ltaussag = compute_all_from_table(
        liqmsk, clwp, reliq, co.liq_nsteps, co.liq_step_size, co.radliq_lwr,
        co.lut_extliq.to(dt), co.lut_ssaliq.to(dt), co.lut_asyliq.to(dt))
    ir = ICE_ROUGHNESS - 1
    itau, itaussa, itaussag = compute_all_from_table(
        icemsk, ciwp, reice, co.ice_nsteps, co.ice_step_size, co.radice_lwr,
        co.lut_extice[:, :, ir].to(dt), co.lut_ssaice[:, :, ir].to(dt),
        co.lut_asyice[:, :, ir].to(dt))
    if not two_stream:
        return (ltau - ltaussa) + (itau - itaussa)
    tau = ltau + itau
    taussa = ltaussa + itaussa
    eps = torch.tensor(torch.finfo(dt).eps, dtype=dt, device=tau.device)   # epsilon(tau): the working type's
    g = (ltaussag + itaussag) / torch.fmax(eps, taussa)
    ssa = taussa / torch.fmax(eps, tau)
    return tau, ssa, g


def get_cloud_optics(co: CloudOpticsData, lwp, iwp, rel, rei, two_stream: bool):
    """get_cloud_optics_sw (two_stream) / get_cloud_optics_lw."""
    rel_limited = limit_to_bounds(rel, co.radliq_lwr, co.radliq_upr)
    rei_limited = limit_to_bounds(rei, co.radice_lwr, co.radice_upr)
    return cloud_optics(co, lwp, iwp, rel_limited, rei_limited, two_stream)
