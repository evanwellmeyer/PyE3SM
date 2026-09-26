"""RTE: optical-property arithmetic, the SW two-stream and LW no-scattering solvers,
and the spectral flux reductions.

Translation of the Kokkos paths of, E3SM 80fe0a71 (rte-rrtmgp 2a887acc):
  - rte/kernels/mo_optical_props_kernels.h   delta_scale_2str_kernel,
        increment_1scalar_by_1scalar, inc_1scalar_by_1scalar_bybnd,
        increment_2stream_by_2stream, inc_2stream_by_2stream_bybnd
  - rte/mo_rte_sw.h, rte/mo_rte_lw.h          rte_sw, rte_lw
  - rte/kernels/mo_rte_solver_kernels.h      sw_two_stream, sw_source_2str, adding,
        sw_solver_2stream, apply_BC, lw_source_noscat_stencil, lw_transport_noscat,
        lw_solver_noscat, lw_solver_noscat_GaussQuad
  - rte/kernels/mo_fluxes_broadband_kernels.h, extensions/fluxes_byband  sum_broadband,
        sum_byband

Arrays are (ncol, nlay, ngpt) and (ncol, nlay+1, ngpt), top of the model at
index 0 (top_at_1). Level recurrences are loops over layers, as in the source.
Sums over g-points are sequential, in g-point order (torch.cumsum is sequential
on CPU).
"""
from __future__ import annotations

import math

import torch

from .. import mathfn
from ..ekat import rdiv

# The kernels use std::numeric_limits<RealT> (min, epsilon) of the working type; in
# float64 these are the double values.
def _tiny(x: torch.Tensor) -> float:
    return torch.finfo(x.dtype).tiny


def _eps(x: torch.Tensor) -> float:
    return torch.finfo(x.dtype).eps


def _fmax(x: torch.Tensor, c: float) -> torch.Tensor:
    return torch.fmax(x, torch.tensor(c, dtype=x.dtype, device=x.device))


# ---------------------------------------------------------------- optical properties
def delta_scale_2str(tau, ssa, g):
    """delta_scale_2str_kernel with forward = g**2."""
    eps = 3 * _tiny(tau)
    m = tau > eps
    f = g * g
    wf = ssa * f
    one = torch.ones((), dtype=tau.dtype, device=tau.device)
    tau_n = (1. - wf) * tau
    ssa_n = (ssa - wf) / torch.where(m, 1.0 - wf, one)
    g_n = (g - f) / torch.where(m, 1.0 - f, one)
    return torch.where(m, tau_n, tau), torch.where(m, ssa_n, ssa), torch.where(m, g_n, g)


def increment_2stream_by_2stream(tau1, ssa1, g1, tau2, ssa2, g2):
    """(tau1, ssa1, g1) incremented by (tau2, ssa2, g2), same g-points. Returns the new 1."""
    eps = 3 * _tiny(tau1)
    tau12 = tau1 + tau2
    tauscat12 = tau1 * ssa1 + tau2 * ssa2
    g_n = (tau1 * ssa1 * g1 + tau2 * ssa2 * g2) / _fmax(tauscat12, eps)
    ssa_n = tauscat12 / _fmax(tau12, eps)
    return tau12, ssa_n, g_n


def inc_2stream_by_2stream_bybnd(tau1, ssa1, g1, tau2, ssa2, g2, gpt2band):
    """As above with (tau2, ssa2, g2) by band; each g-point takes its band's value."""
    gb = gpt2band.to(tau1.device)
    return increment_2stream_by_2stream(tau1, ssa1, g1, tau2[..., gb], ssa2[..., gb], g2[..., gb])


def increment_1scalar_by_1scalar(tau1, tau2):
    return tau1 + tau2


def inc_1scalar_by_1scalar_bybnd(tau1, tau2, gpt2band):
    return tau1 + tau2[..., gpt2band.to(tau1.device)]


# ---------------------------------------------------------------- reductions
def seq_sum(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """bb = 0; for i: bb += x[i]  (sequential; cumsum is sequential on CPU)."""
    return torch.cumsum(x, dim=dim).select(dim, -1)


def sum_broadband(gpt_flux):
    return seq_sum(gpt_flux, -1)


def sum_byband(gpt_flux, band2gpt):
    return torch.stack([seq_sum(gpt_flux[..., g0:g1 + 1], -1) for g0, g1 in band2gpt], dim=-1)


# ---------------------------------------------------------------- shortwave
def sw_two_stream(mu0, tau, w0, g):
    """Returns Rdif, Tdif, Rdir, Tdir, Tnoscat (ncol, nlay, ngpt)."""
    mu = mu0[:, None, None]
    mu0_inv = rdiv(1., mu0)[:, None, None]
    gamma1 = (8. - w0 * (5. + 3. * g)) * .25
    gamma2 = 3. * (w0 * (1. - g)) * .25
    gamma3 = (2. - 3. * mu * g) * .25
    gamma4 = 1. - gamma3
    alpha1 = gamma1 * gamma4 + gamma2 * gamma3
    alpha2 = gamma1 * gamma3 + gamma2 * gamma4
    k = torch.sqrt(_fmax((gamma1 - gamma2) * (gamma1 + gamma2), 1.e-12))
    exp_minusktau = mathfn.exp(-tau * k)
    exp_minus2ktau = exp_minusktau * exp_minusktau
    RT_term = rdiv(1., k * (1. + exp_minus2ktau) + gamma1 * (1. - exp_minus2ktau))
    Rdif = RT_term * gamma2 * (1. - exp_minus2ktau)
    Tdif = RT_term * 2. * k * exp_minusktau
    Tnoscat = mathfn.exp(-tau * mu0_inv)
    k_mu = k * mu
    k_gamma3 = k * gamma3
    k_gamma4 = k * gamma4
    d = 1. - k_mu * k_mu
    RT_term = w0 * RT_term / torch.where(torch.abs(d) >= _eps(d), d, torch.full_like(d, _eps(d)))
    Rdir = RT_term * ((1. - k_mu) * (alpha2 + k_gamma3)
                      - (1. + k_mu) * (alpha2 - k_gamma3) * exp_minus2ktau
                      - 2.0 * (k_gamma3 - alpha2 * k_mu) * exp_minusktau * Tnoscat)
    Tdir = -RT_term * ((1. + k_mu) * (alpha1 + k_gamma4) * Tnoscat
                       - (1. - k_mu) * (alpha1 - k_gamma4) * exp_minus2ktau * Tnoscat
                       - 2.0 * (k_gamma4 + alpha1 * k_mu) * exp_minusktau)
    return Rdif, Tdif, Rdir, Tdir, Tnoscat


def adding(albedo_sfc, rdif, tdif, src_dn, src_up, src_sfc, flux_dn_top):
    """adding() for top_at_1. flux_dn_top (ncol, ngpt) is the incident diffuse flux.

    Returns flux_up, flux_dn (ncol, nlay+1, ngpt), diffuse only.
    """
    nlay = rdif.shape[1]
    albedo = [None] * (nlay + 1)
    src = [None] * (nlay + 1)
    denom = [None] * nlay
    albedo[nlay] = albedo_sfc
    src[nlay] = src_sfc
    for ilev in range(nlay - 1, -1, -1):
        rd, td = rdif[:, ilev], tdif[:, ilev]
        denom[ilev] = rdiv(1., 1. - rd * albedo[ilev + 1])
        albedo[ilev] = rd + td * td * albedo[ilev + 1] * denom[ilev]
        src[ilev] = src_up[:, ilev] + td * denom[ilev] * (src[ilev + 1]
                                                         + albedo[ilev + 1] * src_dn[:, ilev])
    flux_dn = [flux_dn_top]
    flux_up = [flux_dn_top * albedo[0] + src[0]]
    for ilev in range(1, nlay + 1):
        fd = (tdif[:, ilev - 1] * flux_dn[-1] + rdif[:, ilev - 1] * src[ilev]
              + src_dn[:, ilev - 1]) * denom[ilev - 1]
        flux_dn.append(fd)
        flux_up.append(fd * albedo[ilev] + src[ilev])
    return torch.stack(flux_up, dim=1), torch.stack(flux_dn, dim=1)


def rte_sw(tau, ssa, g, mu0, inc_flux, sfc_alb_dir_gpt, sfc_alb_dif_gpt):
    """rte_sw + sw_solver_2stream on all columns given (callers mask night columns).

    inc_flux (ncol, ngpt); surface albedos already expanded to g-points (ncol, ngpt).
    Returns g-point fluxes up, dn (total), dir: (ncol, nlay+1, ngpt).
    """
    nlay = tau.shape[1]
    Rdif, Tdif, Rdir, Tdir, Tnoscat = sw_two_stream(mu0, tau, ssa, g)
    # apply_BC(inc_flux, mu0) for the direct beam; zero incident diffuse flux
    flux_dir = [inc_flux * mu0[:, None]]
    source_up, source_dn = [], []
    for ilev in range(nlay):
        source_up.append(Rdir[:, ilev] * flux_dir[ilev])
        source_dn.append(Tdir[:, ilev] * flux_dir[ilev])
        flux_dir.append(Tnoscat[:, ilev] * flux_dir[ilev])
    source_sfc = flux_dir[nlay] * sfc_alb_dir_gpt
    flux_dir = torch.stack(flux_dir, dim=1)
    source_up = torch.stack(source_up, dim=1)
    source_dn = torch.stack(source_dn, dim=1)
    zero_top = torch.zeros_like(inc_flux)
    flux_up, flux_dn = adding(sfc_alb_dif_gpt, Rdif, Tdif, source_dn, source_up, source_sfc,
                              zero_top)
    flux_dn = flux_dn + flux_dir
    return flux_up, flux_dn, flux_dir


# ---------------------------------------------------------------- longwave
GAUSS_DS_1 = 1.66     # gauss_Ds(0, 0): first-order quadrature, Clough et al. 1992
GAUSS_WTS_1 = 0.5     # gauss_wts(0, 0)


def lw_solver_noscat(D, weight, tau, lay_source, lev_source_inc, lev_source_dec, sfc_emis,
                     sfc_src, flux_dn_top):
    """lw_solver_noscat for top_at_1. D is the angle secant (scalar here, as EAMxx uses it).

    Returns radiances converted to fluxes: flux_up, flux_dn (ncol, nlay+1, ngpt).
    """
    nlay = tau.shape[1]
    tau_thresh = math.sqrt(_eps(tau))
    pi = math.pi
    lev_source_up = lev_source_dec
    lev_source_dn = lev_source_inc
    radn_dn_top = flux_dn_top / (2. * pi * weight)
    sfc_albedo = 1. - sfc_emis
    source_sfc = sfc_emis * sfc_src
    tau_loc = tau * D
    trans = mathfn.exp(-tau_loc)
    # lw_source_noscat_stencil
    big = tau_loc > tau_thresh
    fact = torch.where(big, (1. - trans) / torch.where(big, tau_loc, torch.ones_like(tau_loc))
                       - trans,
                       tau_loc * (0.5 - 1. / 3. * tau_loc))
    source_dn = (1. - trans) * lev_source_dn + 2. * fact * (lay_source - lev_source_dn)
    source_up = (1. - trans) * lev_source_up + 2. * fact * (lay_source - lev_source_up)
    # lw_transport_noscat
    radn_dn = [radn_dn_top]
    for ilev in range(1, nlay + 1):
        radn_dn.append(trans[:, ilev - 1] * radn_dn[-1] + source_dn[:, ilev - 1])
    radn_up = [None] * (nlay + 1)
    radn_up[nlay] = radn_dn[nlay] * sfc_albedo + source_sfc
    for ilev in range(nlay - 1, -1, -1):
        radn_up[ilev] = trans[:, ilev] * radn_up[ilev + 1] + source_up[:, ilev]
    radn_dn = torch.stack(radn_dn, dim=1)
    radn_up = torch.stack(radn_up, dim=1)
    c = 2. * pi * weight
    return c * radn_up, c * radn_dn


def rte_lw(tau, sources, sfc_emis_gpt):
    """rte_lw with one quadrature angle and zero incident flux (as EAMxx calls it)."""
    ncol, _, ngpt = tau.shape
    flux_dn_top = torch.zeros((ncol, ngpt), dtype=tau.dtype, device=tau.device)
    return lw_solver_noscat(GAUSS_DS_1, GAUSS_WTS_1, tau, sources["lay_source"],
                            sources["lev_source_inc"], sources["lev_source_dec"], sfc_emis_gpt,
                            sources["sfc_source"], flux_dn_top)
