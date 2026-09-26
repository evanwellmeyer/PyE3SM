"""ZM rain evaporation and snow, MCSP, and convective momentum transport.

Translation of, at E3SM 80fe0a71 (components/eamxx/src/physics/zm/impl/):
  zm_conv_evap_impl.hpp                zm_conv_evap (zm_microp off; PERGRO off)
  zm_conv_mcsp_calculate_shear_impl.hpp zm_conv_mcsp_calculate_shear
  zm_conv_mcsp_tend_impl.hpp           zm_conv_mcsp_tend
  zm_transport_momentum_impl.hpp       zm_transport_momentum

Tensors are (ncol, pver) (flux profiles: pver+1), level 0 at the top. Pressures in
Pa for evaporation and MCSP; the momentum transport works in mb (ZM's p_del).
"""
from __future__ import annotations

import torch

from .. import constants as PC
from ..gradsafe import sqrt0
from .. import mathfn
from ..ekat import max_, min_, rdiv
from .constants import ZMC, ZMRuntime
from .functions import qsat
from .util import gather, seq_sum


def zm_conv_evap(rt: ZMRuntime, estbl, time_step: float, p_mid, p_del, t_mid, q_mid, prdprec,
                 cldfrc, prec):
    """Evaporation of convective rain and snow, snow production and melting, top down.

    Returns dict(tend_s, tend_q, tend_s_snwprd, tend_s_snwevmlt, prec, snow, ntprprd,
    ntsnprd, flxprec, flxsnow).
    """
    if not rt.old_snow:
        raise NotImplementedError("old_snow = false (EAMxx default is true)")
    pver = t_mid.shape[-1]
    g = PC.gravit
    tmax_fsnow = PC.Tmelt
    tmin_fsnow = PC.Tmelt - 5
    prec = prec * 1000
    z = torch.zeros_like(prec)
    flxprec = [z]
    flxsnow = [z]
    evpvint = z
    tend_s, tend_q, snwprd, snwevmlt, ntprprd, ntsnprd = ([] for _ in range(6))
    _es, qs_all = qsat(t_mid, p_mid, estbl)
    for k in range(pver):
        t = t_mid[:, k]
        qk = q_mid[:, k]
        dp = p_del[:, k]
        qs_k = qs_all[:, k]
        fsnow = torch.where(t > tmax_fsnow, z, torch.where(
            t < tmin_fsnow, torch.ones_like(z), (tmax_fsnow - t) / (tmax_fsnow - tmin_fsnow)))
        warm = t > PC.Tmelt
        flxsntm = torch.where(warm, z, flxsnow[k])
        snowmlt = torch.where(warm, flxsnow[k] * g / dp, z)

        evplimit = max_(1 - qk / qs_k, 0.0)
        # sqrt0: at flxprec = 0 the min below picks the flux limit, so the root's infinite
        # slope is never used; gradient 0 there instead of 0*inf = NaN (gradsafe)
        evpprec = rt.ke * (1 - cldfrc[:, k]) * evplimit * sqrt0(flxprec[k])
        evplimit = max_(0.0, (qs_k - qk) / time_step)
        evplimit = min_(evplimit, flxprec[k] * g / dp)
        evplimit = min_(evplimit, (prec - evpvint) * g / dp)
        evpprec = min_(evplimit, evpprec)

        has = flxprec[k] > 0
        fp_safe = torch.where(has, flxprec[k], torch.ones_like(z))
        work1 = min_(max_(0.0, flxsntm / fp_safe), 1.0)
        evpsnow = torch.where(has, evpprec * work1, z)

        evpvint = evpvint + evpprec * dp / g
        ntp = prdprec[:, k] - evpprec
        work1 = torch.where(has, min_(max_(0.0, flxsnow[k] / fp_safe), 1.0), z)
        work2 = max_(fsnow, work1)
        work2 = torch.where(snowmlt > 0, z, work2)
        nts = prdprec[:, k] * work2 - evpsnow - snowmlt
        snwprd.append(prdprec[:, k] * work2 * PC.LatIce)
        snwevmlt.append(-(evpsnow + snowmlt) * PC.LatIce)

        fp = flxprec[k] + ntp * dp / g
        fs = flxsnow[k] + nts * dp / g
        flxprec.append(max_(fp, 0.0))
        flxsnow.append(max_(fs, 0.0))
        tend_s.append(-evpprec * PC.LatVap + nts * PC.LatIce)
        tend_q.append(evpprec)
        ntprprd.append(ntp)
        ntsnprd.append(nts)

    st = lambda x: torch.stack(x, -1)   # noqa: E731
    return dict(tend_s=st(tend_s), tend_q=st(tend_q), tend_s_snwprd=st(snwprd),
                tend_s_snwevmlt=st(snwevmlt), prec=flxprec[pver] / 1000,
                snow=flxsnow[pver] / 1000, ntprprd=st(ntprprd), ntsnprd=st(ntsnprd),
                flxprec=st(flxprec), flxsnow=st(flxsnow))


def zm_conv_mcsp_calculate_shear(state_pmid, state_u):
    """Low-level shear: wind at MCSP_storm_speed_pref minus the lowest-level wind."""
    pver = state_pmid.shape[-1]
    dev = state_pmid.device
    pref = ZMC.MCSP_storm_speed_pref
    lev = torch.arange(pver - 1, device=dev).unsqueeze(0)
    cross = (state_pmid[:, :-1] < pref) & (state_pmid[:, 1:] >= pref)
    imax = torch.iinfo(torch.int64).max
    k_below = torch.where(cross, lev.expand_as(cross), torch.full_like(cross, imax,
                                                                       dtype=torch.int64)).amin(-1)
    kb = torch.clamp(k_below, max=pver - 2)
    pb, pb1 = gather(state_pmid, kb), gather(state_pmid, kb + 1)
    ub, ub1 = gather(state_u, kb), gather(state_u, kb + 1)
    dpu = pref - pb
    dpl = pb1 - pref
    den = dpl + dpu
    interp = (ub * dpl + ub1 * dpu) / torch.where(den != 0, den, torch.ones_like(den))
    top_below = state_pmid[:, 0] >= pref
    bot_above = state_pmid[:, pver - 1] < pref
    storm_u = torch.where(top_below, state_u[:, 0], torch.where(bot_above, state_u[:, pver - 1],
                                                                 interp))
    return torch.where(state_pmid[:, pver - 1] > pref, storm_u - state_u[:, pver - 1],
                       torch.full_like(storm_u, -999.0))


def zm_conv_mcsp_tend(rt: ZMRuntime, ztodt: float, jctop, state_pmid, state_pint, state_pdel,
                      state_u, state_v, ptend_zm_s, ptend_zm_q):
    """Mesoscale coherent structure parameterization (Moncrieff et al. 2017).

    Returns dict(ptend_s, ptend_q, ptend_u, ptend_v, mcsp_ds_out, mcsp_dq_out, mcsp_du_out,
    mcsp_dv_out, mcsp_freq, mcsp_shear, zm_depth); the ptend_* are the MCSP tendencies
    (the caller's temporary tendencies start at zero).
    """
    pver = state_pmid.shape[-1]
    dev = state_pmid.device
    zero = torch.zeros_like(state_pmid)
    zero1 = torch.zeros_like(state_pmid[:, 0])
    lev = torch.arange(pver, device=dev).unsqueeze(0)
    do_t, do_q = rt.mcsp_t_coeff > 0, rt.mcsp_q_coeff > 0
    do_u, do_v = rt.mcsp_u_coeff > 0, rt.mcsp_v_coeff > 0

    mcsp_shear = zm_conv_mcsp_calculate_shear(state_pmid, state_u)

    # mass-weighted column averages of the ZM tendencies from jctop down
    has_top = jctop != pver - 1
    rng = lev >= jctop.unsqueeze(-1)
    s_sum = seq_sum(torch.where(rng, ptend_zm_s * state_pdel, zero))
    q_sum = seq_sum(torch.where(rng, ptend_zm_q * state_pdel, zero))
    pdel_sum = seq_sum(torch.where(rng, state_pdel, zero))
    pdel_safe = torch.where(has_top, pdel_sum, torch.ones_like(pdel_sum))
    zm_avg_tend_s = torch.where(has_top, s_sum / pdel_safe, zero1)
    zm_avg_tend_q = torch.where(has_top, q_sum / pdel_safe, zero1)
    pint_bot = state_pint[:, pver]
    p_top = gather(state_pmid, jctop)
    zm_depth = torch.where(has_top, pint_bot - p_top, zero1)

    on = (zm_depth >= ZMC.MCSP_conv_depth_min) & (zm_avg_tend_s > 0) & \
        (torch.abs(mcsp_shear) >= ZMC.MCSP_shear_min) & (torch.abs(mcsp_shear) < ZMC.MCSP_shear_max)
    on2 = on.unsqueeze(-1) & rng
    pdepth_mid = pint_bot.unsqueeze(-1) - state_pmid
    pdepth_total = (pint_bot - p_top).unsqueeze(-1)
    pdepth_total = torch.where(on.unsqueeze(-1), pdepth_total, torch.ones_like(pdepth_total))
    frac = pdepth_mid / pdepth_total
    tend_s = tend_q = tend_u = tend_v = zero
    if do_t:
        tend_s = torch.where(on2, -1 * rt.mcsp_t_coeff * mathfn.sin(2 * PC.Pi * frac), zero)
    if do_q:
        tend_q = torch.where(on2, -1 * rt.mcsp_q_coeff * mathfn.sin(2 * PC.Pi * frac), zero)
    if do_u:
        tend_u = torch.where(on2, rt.mcsp_u_coeff * mathfn.cos(PC.Pi * frac), zero)
    if do_v:
        tend_v = torch.where(on2, rt.mcsp_v_coeff * mathfn.cos(PC.Pi * frac), zero)
    if do_t:
        tend_s = torch.where(on2, zm_avg_tend_s.unsqueeze(-1) * tend_s, tend_s)
    if do_q:
        tend_q = torch.where(on2, zm_avg_tend_q.unsqueeze(-1) * tend_q, tend_q)
    ps = pdel_safe.unsqueeze(-1)
    avg_s = seq_sum(torch.where(on2, tend_s * state_pdel / ps, zero)) if do_t else zero1
    avg_q = seq_sum(torch.where(on2, tend_q * state_pdel / ps, zero)) if do_q else zero1
    if do_u or do_v:
        tk = (2 * tend_u * ztodt * state_u + tend_u * tend_u * ztodt * ztodt
              + 2 * tend_v * ztodt * state_v + tend_v * tend_v * ztodt * ztodt) / 2 / ztodt
        avg_k = seq_sum(torch.where(on2, tk * state_pdel / ps, zero))
    else:
        avg_k = zero1

    ds_out = torch.where(rng, tend_s - avg_s.unsqueeze(-1), zero)
    dq_out = torch.where(rng, tend_q - avg_q.unsqueeze(-1), zero)
    du_out = torch.where(rng, tend_u, zero)
    dv_out = torch.where(rng, tend_v, zero)
    if do_u or do_v:
        ds_out = torch.where(rng, ds_out - avg_k.unsqueeze(-1), ds_out)
    p_s = zero + ds_out if do_t else zero        # ptend_s(k) = ptend_s(k) + mcsp_ds_out(k), from 0
    p_q = dq_out if do_q else zero
    p_u = du_out if do_u else zero
    p_v = dv_out if do_v else zero
    if do_t:
        ds_out = torch.where(rng, ds_out / PC.Cpair, ds_out)
    any_t = ((torch.abs(tend_s) > 0) | (torch.abs(tend_q) > 0) | (torch.abs(tend_u) > 0)
             | (torch.abs(tend_v) > 0)) & rng
    mcsp_freq = torch.where(any_t.any(-1), torch.ones_like(zero1), zero1)
    return dict(ptend_s=p_s, ptend_q=p_q, ptend_u=p_u, ptend_v=p_v, mcsp_ds_out=ds_out,
                mcsp_dq_out=dq_out, mcsp_du_out=du_out, mcsp_dv_out=dv_out, mcsp_freq=mcsp_freq,
                mcsp_shear=mcsp_shear, zm_depth=zm_depth)


def zm_transport_momentum(dt: float, wind_mid, mu, md, du, eu, ed, dp, jt, mx, ktm, kbm):
    """Convective momentum transport with pressure-gradient terms (Gregory et al. 1997) and
    the kinetic-energy dissipation heating (Boville and Bretherton 2003).

    wind_mid: (ncol, nwind, pver); the fluxes and dp: (ncol, pver) in mb units; ktm, kbm:
    0-dim tensors. Returns (wind_tend (ncol, nwind, pver), seten (ncol, pver)).
    """
    ncol, nwind, pver = wind_mid.shape
    dev = wind_mid.device
    lev = torch.arange(pver, device=dev).view(1, 1, pver)
    w = wind_mid
    mu3, md3, du3, eu3, ed3, dp3 = (x.unsqueeze(1) for x in (mu, md, du, eu, ed, dp))

    # interface winds (arithmetic mean) and provisional up/downdraft values
    w_km1 = torch.cat([w[..., :1], w[..., :-1]], -1)
    wind_int = ZMC.half * (w + w_km1)

    # pressure-gradient terms
    mud = torch.zeros_like(w)
    mdd = torch.zeros_like(w)
    k_in = slice(1, pver - 1)
    mud_in = (mu3[..., 1:-1] * (w[..., 1:-1] - w[..., :-2]) / dp3[..., :-2]
              + mu3[..., 2:] * (w[..., 2:] - w[..., 1:-1]) / dp3[..., 1:-1])
    mdd_in = (md3[..., 1:-1] * (w[..., 1:-1] - w[..., :-2]) / dp3[..., :-2]
              + md3[..., 2:] * (w[..., 2:] - w[..., 1:-1]) / dp3[..., 1:-1])
    kb = pver - 1
    mud_b = mu3[..., kb] * (w[..., kb] - w[..., kb - 1]) / dp3[..., kb - 1]
    mdd_b = md3[..., kb] * (w[..., kb] - w[..., kb - 1]) / dp3[..., kb - 1]
    pgu = torch.cat([torch.zeros_like(w[..., :1]), -ZMC.momcu * ZMC.half * mud_in,
                     (-ZMC.momcu * mud_b).unsqueeze(-1)], -1)
    pgd = torch.cat([torch.zeros_like(w[..., :1]), -ZMC.momcd * ZMC.half * mdd_in,
                     (-ZMC.momcd * mdd_b).unsqueeze(-1)], -1)
    del k_in, mud, mdd

    # in-cloud winds: updraft from the bottom up, downdraft from the top down
    wu = list(wind_int.unbind(-1))
    wd = list(wind_int.unbind(-1))
    kk = pver - 1
    mupdudp = mu3[..., kk] + du3[..., kk] * dp3[..., kk]
    ok = mupdudp > ZMC.mbsth
    val = (eu3[..., kk] * w[..., kk] * dp3[..., kk] + pgu[..., kk] * dp3[..., kk]) / \
        torch.where(ok, mupdudp, torch.ones_like(mupdudp))
    wu[kk] = torch.where(ok, val, wu[kk])
    ok = md3[..., 1] < -ZMC.mbsth
    val = (-ed3[..., 0] * w[..., 0] * dp3[..., 0] - pgd[..., 0] * dp3[..., 0]) / \
        torch.where(ok, md3[..., 1], torch.ones_like(md3[..., 1]))
    wd[1] = torch.where(ok, val, wd[1])
    for kk in range(pver - 2, -1, -1):
        mupdudp = mu3[..., kk] + du3[..., kk] * dp3[..., kk]
        ok = mupdudp > ZMC.mbsth
        val = (mu3[..., kk + 1] * wu[kk + 1] + eu3[..., kk] * w[..., kk] * dp3[..., kk]
               + pgu[..., kk] * dp3[..., kk]) / torch.where(ok, mupdudp, torch.ones_like(mupdudp))
        wu[kk] = torch.where(ok, val, wu[kk])
    for k in range(2, pver):
        ok = md3[..., k] < -ZMC.mbsth
        val = (md3[..., k - 1] * wd[k - 1] - ed3[..., k - 1] * w[..., k - 1] * dp3[..., k - 1]
               - pgd[..., k - 1] * dp3[..., k - 1]) / torch.where(ok, md3[..., k],
                                                                   torch.ones_like(md3[..., k]))
        wd[k] = torch.where(ok, val, wd[k])
    wu = torch.stack(wu, -1)
    wd = torch.stack(wd, -1)

    # momentum tendency
    kp1 = torch.cat([torch.arange(1, pver, device=dev), torch.tensor([pver - 1], device=dev)])
    t_all = (mu3[..., kp1] * (wu[..., kp1] - wind_int[..., kp1])
             - mu3 * (wu - wind_int)
             + md3[..., kp1] * (wd[..., kp1] - wind_int[..., kp1])
             - md3 * (wd - wind_int)) / dp3
    zero = torch.zeros_like(w)
    wind_tend = torch.where(lev >= ktm, t_all, zero)
    t_mx = (-mu3 * (wu - wind_int) - md3 * (wd - wind_int)) * rdiv(1.0, dp3)
    at_mx = (lev == mx.view(ncol, 1, 1)) & (lev >= kbm)
    wind_tend = torch.where(at_mx, t_mx, wind_tend)

    # momentum flux and the winds at the end of the step
    mflux = torch.where(lev >= ktm, -mu3 * (wu - wind_int) - md3 * (wd - wind_int), zero)
    mflux = torch.cat([mflux, torch.zeros_like(mflux[..., :1])], -1)     # mflux(pver) = 0
    windf = torch.where(lev >= ktm, w - (mflux[..., 1:] - mflux[..., :-1]) * dt / dp3, zero)

    # kinetic energy dissipation as dry static energy tendency
    w0 = wind_mid
    km1 = torch.cat([torch.zeros(1, dtype=torch.int64, device=dev),
                     torch.arange(0, pver - 1, device=dev)])
    top = (w0 + w0[..., km1]) / 2
    bot = (w0[..., kp1] + w0) / 2
    fket = top[:, 0] * mflux[:, 0, :pver] + top[:, 1] * mflux[:, 1, :pver]
    fkeb = bot[:, 0] * mflux[:, 0, 1:] + bot[:, 1] * mflux[:, 1, 1:]
    ketend_cons = (fket - fkeb) / dp
    ketend = ((windf[:, 0] * windf[:, 0] + windf[:, 1] * windf[:, 1])
              - (w0[:, 0] * w0[:, 0] + w0[:, 1] * w0[:, 1])) * ZMC.half / dt
    seten = torch.where(lev[0] >= ktm, ketend_cons - ketend, torch.zeros_like(dp))
    return wind_tend, seten
