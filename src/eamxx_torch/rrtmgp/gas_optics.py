"""RRTMGP gas optics: optical depths, Rayleigh scattering, Planck sources.

Translation of the Kokkos paths of, E3SM 80fe0a71 (rte-rrtmgp 2a887acc):
  - rrtmgp/mo_gas_optics_rrtmgp.h           GasOpticsRRTMGPK::gas_optics (both forms),
                                            compute_gas_taus, source, get_col_dry,
                                            combine_and_reorder
  - rrtmgp/kernels/mo_gas_optics_kernels.h  interpolation, compute_tau_absorption,
                                            gas_optical_depths_major/minor,
                                            compute_tau_rayleigh, combine_and_reorder_2str,
                                            compute_Planck_source, interpolate1D

Layout. The reference computes tau(igpt,ilay,icol) and reorders to
(icol,ilay,igpt); the port computes (ncol, nlay, ngpt) directly. Every value is
computed by the same expression. The vertical axis runs from the model top
(top_at_1 = true; see deviations in docs/plan.md).
"""
from __future__ import annotations

import torch

from .. import mathfn
from ..ekat import rdiv
from .data import GasOpticsData, MinorSet

TINY = 2.2250738585072014e-308     # std::numeric_limits<double>::min()
INT_MAX = 2147483647
INT_MIN = -2147483648

# rrtmgp_constants<double> defaults (EAMxx never calls init_constants)
GRAV = 9.80665
M_DRY = 0.028964
M_H2O = 0.018016
AVOGAD = 6.02214076e23


def _trunc_int(x: torch.Tensor) -> torch.Tensor:
    """static_cast<int>(double): truncation toward zero."""
    return x.to(torch.int64)


def get_col_dry(vmr_h2o: torch.Tensor, plev: torch.Tensor) -> torch.Tensor:
    """get_col_dry without latitude: dry-air column amounts [molec/cm^2], (ncol, nlay)."""
    delta_plev = (plev[:, :-1] - plev[:, 1:]).abs()
    fact = rdiv(1., 1. + vmr_h2o)
    m_air = (M_DRY + M_H2O * vmr_h2o) * fact
    return 10. * delta_plev * AVOGAD * fact / (1000. * m_air * 100. * GRAV)


def _take(table: torch.Tensor, *idx: torch.Tensor) -> torch.Tensor:
    """table[i0, i1, ...] with broadcastable integer index tensors."""
    flat = idx[0]
    for d, i in zip(table.shape[1:], idx[1:]):
        flat = flat * d + i
    return table.reshape(-1)[flat]


def interpolation(k: GasOpticsData, play, tlay, col_gas):
    """interpolation(): indices and weights for the absorption-coefficient tables.

    Returns jtemp, jpress (ncol, nlay) int64; tropo (ncol, nlay) bool;
    col_mix (ncol, nlay, nflav, 2[itemp]); jeta (ncol, nlay, nflav, 2[itemp]);
    fminor (ncol, nlay, nflav, 2[ieta], 2[itemp]);
    fmajor (ncol, nlay, nflav, 2[ieta], 2[ipress], 2[itemp]).
    """
    ntemp, npres, neta = k.ntemp, k.npres, k.neta
    jtemp = _trunc_int((tlay - (k.temp_ref_min - k.temp_ref_delta)) / k.temp_ref_delta)
    jtemp = torch.clamp(jtemp, 1, ntemp - 1) - 1
    ftemp = (tlay - k.temp_ref.to(tlay)[jtemp]) / k.temp_ref_delta

    logp = mathfn.log(play)
    locpress = 1. + (logp - float(k.press_ref_log[0])) / k.press_ref_log_delta
    jpress = torch.clamp(_trunc_int(locpress), 1, npres - 1)
    fpress = locpress - jpress.to(play.dtype)
    tropo = logp > k.press_ref_trop_log

    itropo = torch.where(tropo, 0, 1)                                  # merge(0,1,tropo)
    g1 = k.flavor[0].to(play.device)                                   # (nflav,)
    g2 = k.flavor[1].to(play.device)
    nflav = g1.shape[0]
    ncol, nlay = play.shape
    col_g1 = col_gas[..., g1 + 1]                                      # (ncol, nlay, nflav)
    col_g2 = col_gas[..., g2 + 1]
    vmr_ref = k.vmr_ref.to(play)
    col_mix, jeta, fminor, fmajor = [], [], [], []
    for itemp in (0, 1):
        it = (itropo[..., None], None, (jtemp + itemp)[..., None])
        ratio = _take(vmr_ref, it[0], g1 + 1, it[2]) / _take(vmr_ref, it[0], g2 + 1, it[2])
        cm = col_g1 + ratio * col_g2
        ok = cm > 2. * TINY
        eta = torch.where(ok, col_g1 / torch.where(ok, cm, 1.), 0.5)
        loceta = eta * (neta - 1.0)
        je = torch.clamp(_trunc_int(loceta) + 1, max=neta - 1) - 1
        feta = torch.fmod(loceta, 1.0)
        ftemp_term = (2.0 - (itemp + 1)) + (2.0 * (itemp + 1) - 3.0) * ftemp[..., None]
        fm0 = (1. - feta) * ftemp_term
        fm1 = feta * ftemp_term
        col_mix.append(cm)
        jeta.append(je)
        fminor.append(torch.stack([fm0, fm1], dim=-1))                 # [ieta]
        omf = 1. - fpress[..., None]
        fp = fpress[..., None]
        fmajor.append(torch.stack([torch.stack([omf * fm0, fp * fm0], dim=-1),     # ieta=0: [ipress]
                                   torch.stack([omf * fm1, fp * fm1], dim=-1)],    # ieta=1
                                  dim=-2))                              # (..., ieta, ipress)
    col_mix = torch.stack(col_mix, dim=-1)
    jeta = torch.stack(jeta, dim=-1)
    fminor = torch.stack(fminor, dim=-1)                               # (..., ieta, itemp)
    fmajor = torch.stack(fmajor, dim=-1)                               # (..., ieta, ipress, itemp)
    return jtemp, jpress, tropo, col_mix, jeta, fminor, fmajor


def _flav_gather(x: torch.Tensor, iflav: torch.Tensor) -> torch.Tensor:
    """x (ncol, nlay, nflav, *rest) at flavor iflav (ncol, nlay, ngpt) -> (ncol, nlay, ngpt, *rest)."""
    ncol, nlay, nflav = x.shape[:3]
    rest = x.shape[3:]
    xf = x.reshape(ncol, nlay, nflav, -1)
    idx = iflav[..., None].expand(-1, -1, -1, xf.shape[-1])
    return torch.gather(xf, 2, idx).reshape(ncol, nlay, iflav.shape[-1], *rest)


def _gpt_flavor(k: GasOpticsData, itropo: torch.Tensor) -> torch.Tensor:
    """gpoint_flavor(itropo, igpt) for every (icol, ilay, igpt)."""
    gf = k.gpoint_flavor.to(itropo.device)
    return gf[itropo[..., None], torch.arange(k.ngpt, device=itropo.device)]


def gas_optical_depths_major(k, tropo, jtemp, jpress, col_mix, jeta, fmajor):
    itropo = torch.where(tropo, 0, 1)
    iflav = _gpt_flavor(k, itropo)                                     # (ncol, nlay, ngpt)
    cm = _flav_gather(col_mix, iflav)                                  # (..., ngpt, 2)
    je = _flav_gather(jeta, iflav)                                     # (..., ngpt, 2)
    fm = _flav_gather(fmajor, iflav)                                   # (..., ngpt, 2, 2, 2)
    jp = (jpress + itropo)[..., None]
    jt = jtemp[..., None]
    g = torch.arange(k.ngpt, device=tropo.device)
    kmajor = k.kmajor.to(cm)

    def kk(ieta, ipress, itemp):
        return _take(kmajor, g, je[..., itemp] + ieta, jp - 1 + ipress, jt + itemp)

    def f(ieta, ipress, itemp):
        return fm[..., ieta, ipress, itemp]

    return (cm[..., 0] * (f(0, 0, 0) * kk(0, 0, 0) + f(1, 0, 0) * kk(1, 0, 0)
                          + f(0, 1, 0) * kk(0, 1, 0) + f(1, 1, 0) * kk(1, 1, 0))
            + cm[..., 1] * (f(0, 0, 1) * kk(0, 0, 1) + f(1, 0, 1) * kk(1, 0, 1)
                            + f(0, 1, 1) * kk(0, 1, 1) + f(1, 1, 1) * kk(1, 1, 1)))


def gas_optical_depths_minor(k, ms: MinorSet, idx_tropo, idx_h2o, play, tlay, col_gas, fminor,
                             jeta, jtemp, layer_mask, tau):
    """Adds each minor contribution to tau in the reference order (imnr, then igpt)."""
    PaTohPa = 0.01
    ngpt = tau.shape[-1]
    mycol_gas_h2o = col_gas[..., idx_h2o + 1]
    mycol_gas_0 = col_gas[..., 0]
    kminor = ms.kminor.to(tau)
    jt = jtemp[..., None]
    for imnr in range(len(ms.idx_minor)):
        scaling = col_gas[..., ms.idx_minor[imnr] + 1]
        if ms.scales_with_density[imnr]:
            scaling = scaling * (PaTohPa * play / tlay)
            if ms.idx_minor_scaling[imnr] > -1:
                mycol_gas_imnr = col_gas[..., ms.idx_minor_scaling[imnr] + 1]
                vmr_fact = rdiv(1., mycol_gas_0)
                dry_fact = rdiv(1., 1. + mycol_gas_h2o * vmr_fact)
                if ms.scale_by_complement[imnr]:
                    scaling = scaling * (1. - mycol_gas_imnr * vmr_fact * dry_fact)
                else:
                    scaling = scaling * mycol_gas_imnr * vmr_fact * dry_fact
        gptS, gptE = ms.limits_gpt[imnr]
        gs = torch.arange(gptS, gptE + 1, device=tau.device)
        flv = k.gpoint_flavor[idx_tropo, gptS:gptE + 1].to(tau.device)
        fm = fminor[:, :, flv]                                         # (ncol, nlay, G, 2, 2)
        je = jeta[:, :, flv]                                           # (ncol, nlay, G, 2)
        loc = ms.kminor_start[imnr] + (gs - gptS)

        def kk(ieta, itemp):
            return _take(kminor, loc, je[..., itemp] + ieta, jt + itemp)

        kminor_loc = (fm[..., 0, 0] * kk(0, 0) + fm[..., 1, 0] * kk(1, 0)
                      + fm[..., 0, 1] * kk(0, 1) + fm[..., 1, 1] * kk(1, 1))
        tau_minor = torch.where(layer_mask[..., None], kminor_loc * scaling[..., None], 0.)
        # tau(igpt) += tau_minor on [gptS, gptE]; adding exact zeros elsewhere
        tau = tau + torch.nn.functional.pad(tau_minor, (gptS, ngpt - 1 - gptE))
    return tau


def _layer_limits_mask(play, tropo):
    """compute_tau_absorption, top_at_1: layers that get lower / upper minor absorption."""
    nlay = play.shape[-1]
    ilay = torch.arange(nlay, device=play.device)
    inf = torch.tensor(float("inf"), dtype=play.dtype, device=play.device)
    # itropo_lower = [first min-pressure layer of the lower atmosphere, nlay-1]
    lo = torch.argmin(torch.where(tropo, play, inf), dim=-1)
    lo = torch.where(tropo.any(dim=-1), lo, INT_MAX)
    lower = (ilay >= lo[:, None]) & (ilay <= nlay - 1)
    # itropo_upper = [0, first max-pressure layer of the upper atmosphere]
    hi = torch.argmax(torch.where(~tropo, play, -inf), dim=-1)
    hi = torch.where((~tropo).any(dim=-1), hi, INT_MIN)
    upper = (ilay >= 0) & (ilay <= hi[:, None])
    return lower, upper


def compute_tau_rayleigh(k, tropo, jtemp, jeta, fminor, col_dry, col_gas, idx_h2o):
    itropo = torch.where(tropo, 0, 1)
    iflav = _gpt_flavor(k, itropo)
    je = _flav_gather(jeta, iflav)
    fm = _flav_gather(fminor, iflav)
    jt = jtemp[..., None]
    it = itropo[..., None]
    g = torch.arange(k.ngpt, device=tropo.device)
    krayl = k.krayl.to(fm)

    def kk(ieta, itemp):
        return _take(krayl, g, je[..., itemp] + ieta, jt + itemp, it)

    kr = (fm[..., 0, 0] * kk(0, 0) + fm[..., 1, 0] * kk(1, 0)
          + fm[..., 0, 1] * kk(0, 1) + fm[..., 1, 1] * kk(1, 1))
    return kr * (col_gas[..., idx_h2o + 1] + col_dry)[..., None]


def compute_gas_taus(k: GasOpticsData, play, plev, tlay, vmr: dict):
    """compute_gas_taus: absorption optical depth, Rayleigh optical depth (or None), and the
    interpolation coefficients the Planck source needs.

    vmr: gas name -> (ncol, nlay) volume mixing ratio (the GasConcs object).
    """
    ncol, nlay = play.shape
    lower_names = {n.lower(): n for n in vmr}
    for i, name in enumerate(k.gas_names):
        if k.is_key[i] and name.lower() not in lower_names:
            raise ValueError("gas required by k-distribution is not present in the GasConcs object")
    zeros = torch.zeros_like(play)
    vmr_k = [vmr[lower_names[g.lower()]] if g.lower() in lower_names else zeros
             for g in k.gas_names]
    idx_h2o = [g.lower() for g in k.gas_names].index("h2o")
    col_dry = get_col_dry(vmr_k[idx_h2o], plev)
    col_gas = torch.stack([col_dry] + [v * col_dry for v in vmr_k], dim=-1)

    jtemp, jpress, tropo, col_mix, jeta, fminor, fmajor = interpolation(k, play, tlay, col_gas)
    lower, upper = _layer_limits_mask(play, tropo)
    tau = torch.zeros((ncol, nlay, k.ngpt), dtype=play.dtype, device=play.device)
    tau = tau + gas_optical_depths_major(k, tropo, jtemp, jpress, col_mix, jeta, fmajor)
    tau = gas_optical_depths_minor(k, k.lower, 0, idx_h2o, play, tlay, col_gas, fminor, jeta,
                                   jtemp, lower, tau)
    tau = gas_optical_depths_minor(k, k.upper, 1, idx_h2o, play, tlay, col_gas, fminor, jeta,
                                   jtemp, upper, tau)
    tau_rayleigh = None
    if k.krayl is not None:
        tau_rayleigh = compute_tau_rayleigh(k, tropo, jtemp, jeta, fminor, col_dry, col_gas,
                                            idx_h2o)
    interp = dict(jtemp=jtemp, jpress=jpress, tropo=tropo, jeta=jeta, fmajor=fmajor)
    return tau, tau_rayleigh, interp


def combine_and_reorder_2str(tau_abs, tau_rayleigh):
    """Returns tau, ssa, g of the 2-stream gas optics."""
    t = tau_abs + tau_rayleigh
    ok = t > 2. * TINY
    ssa = torch.where(ok, tau_rayleigh / torch.where(ok, t, 1.), 0.)
    return t, ssa, torch.zeros_like(t)


def gas_optics_sw(k: GasOpticsData, play, plev, tlay, vmr: dict):
    """gas_optics() for external sources: 2-stream optics and TOA source (ncol, ngpt)."""
    tau, tau_ray, _ = compute_gas_taus(k, play, plev, tlay, vmr)
    if tau_ray is not None:
        tau, ssa, g = combine_and_reorder_2str(tau, tau_ray)
    else:
        ssa, g = torch.zeros_like(tau), torch.zeros_like(tau)
    toa_src = k.solar_src.to(play)[None, :].expand(play.shape[0], -1)
    return (tau, ssa, g), toa_src


def interpolate1D(val, offset, delta, table):
    """interpolate1D: val (...) -> (..., tab_d2), linear in the first table dimension."""
    tab_d1 = table.shape[0]
    val0 = (val - offset) / delta
    frac = val0 - _trunc_int(val0).to(val.dtype)
    index = torch.clamp(_trunc_int(val0) + 1, 1, tab_d1 - 1) - 1
    t0 = table[index]
    t1 = table[index + 1]
    return t0 + frac[..., None] * (t1 - t0)


def compute_Planck_source(k: GasOpticsData, tlay, tlev, tsfc, interp):
    """compute_Planck_source with sfc_lay = nlay-1 (top_at_1).

    Returns sfc_src (ncol, ngpt), lay_src, lev_src_inc, lev_src_dec (ncol, nlay, ngpt).
    """
    tropo, jtemp, jpress, jeta, fmajor = (interp[n] for n in ("tropo", "jtemp", "jpress", "jeta",
                                                                 "fmajor"))
    nlay = tlay.shape[-1]
    itropo = torch.where(tropo, 0, 1)
    iflav = _gpt_flavor(k, itropo)
    je = _flav_gather(jeta, iflav)
    fm = _flav_gather(fmajor, iflav)
    jp = (jpress + itropo)[..., None]
    jt = jtemp[..., None]
    g = torch.arange(k.ngpt, device=tlay.device)
    pf = k.planck_frac.to(tlay)

    def kk(ieta, ipress, itemp):
        return _take(pf, g, je[..., itemp] + ieta, jp - 1 + ipress, jt + itemp)

    def f(ieta, ipress, itemp):
        return fm[..., ieta, ipress, itemp]

    pfrac = ((f(0, 0, 0) * kk(0, 0, 0) + f(1, 0, 0) * kk(1, 0, 0)
              + f(0, 1, 0) * kk(0, 1, 0) + f(1, 1, 0) * kk(1, 1, 0))
             + (f(0, 0, 1) * kk(0, 0, 1) + f(1, 0, 1) * kk(1, 0, 1)
                + f(0, 1, 1) * kk(0, 1, 1) + f(1, 1, 1) * kk(1, 1, 1)))

    totplnk = k.totplnk.to(tlay)
    gb = k.gpt2band.to(tlay.device)
    planck_sfc = interpolate1D(tsfc, k.temp_ref_min, k.totplnk_delta, totplnk)     # (ncol, nbnd)
    sfc_src = pfrac[:, nlay - 1, :] * planck_sfc[:, gb]
    planck_lay = interpolate1D(tlay, k.temp_ref_min, k.totplnk_delta, totplnk)     # (ncol, nlay, nbnd)
    lay_src = pfrac * planck_lay[..., gb]
    planck_lev = interpolate1D(tlev, k.temp_ref_min, k.totplnk_delta, totplnk)     # (ncol, nlay+1, nbnd)
    lev_src_dec = pfrac * planck_lev[:, :-1][..., gb]
    lev_src_inc = pfrac * planck_lev[:, 1:][..., gb]
    return sfc_src, lay_src, lev_src_inc, lev_src_dec


def gas_optics_lw(k: GasOpticsData, play, plev, tlay, tsfc, vmr: dict, tlev):
    """gas_optics() for internal sources, with tlev provided (as EAMxx calls it).

    Returns tau (ncol, nlay, ngpt) and the sources dict.
    """
    tau, _, interp = compute_gas_taus(k, play, plev, tlay, vmr)
    sfc_src, lay_src, lev_inc, lev_dec = compute_Planck_source(k, tlay, tlev, tsfc, interp)
    return tau, dict(sfc_source=sfc_src, lay_source=lay_src, lev_source_inc=lev_inc,
                     lev_source_dec=lev_dec)
