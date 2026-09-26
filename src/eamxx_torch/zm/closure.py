"""ZM closure (cloud-base mass flux) and the resulting large-scale tendencies.

Translation of, at E3SM 80fe0a71 (components/eamxx/src/physics/zm/impl/):
  zm_closure_impl.hpp           zm_closure (revised quasi-equilibrium closure of Zhang 2002)
  zm_calc_output_tend_impl.hpp  zm_calc_output_tend

Tensors are (ncol, pver), level 0 at the top; lcl, lel, jt, mx are int64 (ncol,).
Pressures in mb.
"""
from __future__ import annotations

import torch

from .. import constants as PC
from .. import mathfn
from ..ekat import max_, rdiv
from .constants import ZMC, ZMRuntime
from .util import gather, seq_sum


def _shift_up(x):
    """x(k+1) at level k (0 at the bottom, never used there)."""
    return torch.cat([x[:, 1:], torch.zeros_like(x[:, :1])], -1)


def zm_closure(rt: ZMRuntime, pver: int, msg: int, cape_threshold_in: float, lcl, lel, jt, mx,
               dsubcld, z_mid, z_int, p_mid, p_del, t_mid, s_mid, q_mid, qs, ql, s_int, q_int,
               t_pcl_lcl, t_pcl, q_pcl_sat, s_upd, q_upd, mflx_net, detr_up, mflx_up, mflx_dn,
               q_dnd, s_dnd, cape):
    """Cloud-base mass flux. Returns cld_base_mass_flux (ncol,)."""
    latvap, cpair, grav, Rair, epsilo = PC.LatVap, PC.Cpair, PC.gravit, PC.Rair, PC.ep_2
    dev = p_mid.device
    lev = torch.arange(pver, device=dev).unsqueeze(0)
    g = lambda x: gather(x, mx)   # noqa: E731

    # sub-cloud tendencies of virtual temperature and humidity
    p_mx, q_mx, t_mx = g(p_mid), g(q_mid), g(t_mid)
    eb = p_mx * q_mx / (epsilo + q_mx)
    inv_dsub = rdiv(1.0, dsubcld)
    dtbdt = inv_dsub * (g(mflx_up) * (g(s_int) - g(s_upd)) + g(mflx_dn) * (g(s_int) - g(s_dnd)))
    dqbdt = inv_dsub * (g(mflx_up) * (g(q_int) - g(q_upd)) + g(mflx_dn) * (g(q_int) - g(q_dnd)))
    debdt = epsilo * p_mx / ((epsilo + q_mx) * (epsilo + q_mx)) * dqbdt
    log_arg = ZMC.lcl_coeff_b * mathfn.log(t_mx) - mathfn.log(eb) - ZMC.lcl_coeff_c
    dtldt = -ZMC.lcl_coeff_a * (rdiv(ZMC.lcl_coeff_b, t_mx) * dtbdt - debdt / eb) / (log_arg * log_arg)

    # free-tropospheric tendencies dtmdt, dqmdt (levels msg .. pver-2)
    mu1, md1, su1, sd1, si1 = (_shift_up(x) for x in (mflx_up, mflx_dn, s_upd, s_dnd, s_int))
    qu1, qd1, qi1, ql1, mn1 = (_shift_up(x) for x in (q_upd, q_dnd, q_int, ql, mflx_net))
    inv_dp = rdiv(1.0, p_del)
    dt_top = inv_dp * (mu1 * (su1 - si1 - latvap / cpair * ql1) + md1 * (sd1 - si1))
    dq_top = inv_dp * (mu1 * (qu1 - qi1 + ql1) + md1 * (qd1 - qi1))
    lq = ZMC.beta * ql + (1 - ZMC.beta) * ql1
    dt_in = (mflx_net * (s_int - s_mid) + mn1 * (s_mid - si1)) / p_del \
        - latvap / cpair * detr_up * lq
    cl = cpair / latvap
    dq_in = (mu1 * (qu1 - qi1 + cl * (su1 - s_mid))
             - mflx_up * (q_upd - q_int + cl * (s_upd - s_mid))
             + md1 * (qd1 - qi1 + cl * (sd1 - s_mid))
             - mflx_dn * (q_dnd - q_int + cl * (s_dnd - s_mid))) / p_del \
        + detr_up * lq
    rng = (lev >= msg) & (lev < pver - 1)
    jt_ = jt.unsqueeze(-1)
    mx_ = mx.unsqueeze(-1)
    top = rng & (lev == jt_)
    inner = rng & (lev > jt_) & (lev < mx_)
    zero = torch.zeros_like(p_mid)
    dtmdt = torch.where(inner, dt_in, torch.where(top, dt_top, zero))
    dqmdt = torch.where(inner, dq_in, torch.where(top, dq_top, zero))

    # integrand of the CAPE change
    rc = Rair / cpair
    pw = mathfn.pow(rdiv(1000.0, p_mid), rc)
    zq_mx = (1 + ZMC.zvir * q_mx).unsqueeze(-1)
    a_mx = (dtbdt / t_mx).unsqueeze(-1)
    b_mx = (rdiv(ZMC.zvir, 1 + ZMC.zvir * q_mx) * dqbdt).unsqueeze(-1)
    thetavm = t_mid * pw * (1 + ZMC.zvir * q_mid)
    env_term = dtmdt / t_mid
    env_q = rdiv(ZMC.zvir, 1 + ZMC.zvir * q_mid) * dqmdt
    # below the LCL (parcel unsaturated)
    thetavp_b = t_pcl * pw * zq_mx
    dboy_b = (a_mx + b_mx - env_term - env_q) * grav * thetavp_b / thetavm
    # between the LCL and the cloud top (saturated parcel)
    q_mx_ = q_mx.unsqueeze(-1)
    tl = t_pcl_lcl.unsqueeze(-1)
    zv1 = 1 + ZMC.zvir
    thetavp_a = t_pcl * pw * (1 + zv1 * q_pcl_sat - q_mx_)
    dqsdtp = q_pcl_sat * (1 + q_pcl_sat / epsilo) * epsilo * latvap / (Rair * t_pcl * t_pcl)
    lc = latvap / cpair
    dtpdt = t_pcl / (1 + lc * (dqsdtp - q_pcl_sat / t_pcl)) * (
        a_mx + lc * (dqbdt.unsqueeze(-1) / tl - q_mx_ / (tl * tl) * dtldt.unsqueeze(-1)))
    dboy_a = ((dtpdt / t_pcl + rdiv(1.0, 1 + zv1 * q_pcl_sat - q_mx_) * (zv1 * dqsdtp * dtpdt
                                                                         - dqbdt.unsqueeze(-1)))
              - (env_term + env_q)) * grav * thetavp_a / thetavm
    below = (lev >= msg) & (lev > lcl.unsqueeze(-1)) & (lev < mx_)
    above = (lev >= msg) & (lev >= lel.unsqueeze(-1)) & (lev <= lcl.unsqueeze(-1))
    dboydt = torch.where(above, dboy_a, torch.where(below, dboy_b, zero))

    # vertical integral from lel to mx-1, in level order
    dzi = z_int[:, :pver] - z_int[:, 1:]
    rng = (lev >= lel.unsqueeze(-1)) & (lev < mx_)
    dadt = seq_sum(torch.where(rng, dboydt * dzi, zero))

    dltaa = -1 * (cape - cape_threshold_in)
    dadt_safe = torch.where(dadt != 0, dadt, torch.ones_like(dadt))
    cbmf = torch.where(dadt != 0, max_(dltaa / rt.tau / dadt_safe, 0.0), torch.zeros_like(dadt))
    return cbmf


def zm_calc_output_tend(pver: int, msg: int, jt, mx, ktm, kbm, dsubcld, p_del, s_int,
                        q_int, s_upd, q_upd, mflx_up, detr_up, mflx_dn, s_dnd, q_dnd, ql, evp, cu):
    """Tendencies of dry static energy, humidity and detrained liquid. ktm, kbm: 0-dim
    int64 tensors. Returns (dsdt, dqdt, dl)."""
    dev = p_del.device
    lev = torch.arange(pver, device=dev).unsqueeze(0)
    zero = torch.zeros_like(p_del)
    mu1, md1, su1, sd1, si1 = (_shift_up(x) for x in (mflx_up, mflx_dn, s_upd, s_dnd, s_int))
    qu1, qd1, qi1, ql1 = (_shift_up(x) for x in (q_upd, q_dnd, q_int, ql))
    emc = -cu + evp
    ds = -PC.LatVap / PC.Cpair * emc + (
        + mu1 * (su1 - si1) - mflx_up * (s_upd - s_int)
        + md1 * (sd1 - si1) - mflx_dn * (s_dnd - s_int)) / p_del
    dq = emc + (
        + mu1 * (qu1 - qi1) - mflx_up * (q_upd - q_int)
        + md1 * (qd1 - qi1) - mflx_dn * (q_dnd - q_int)) / p_del
    dlv = detr_up * ql1
    rng = (lev >= ktm) & (lev < pver - 1)
    dsdt = torch.where(rng, ds, zero)
    dqdt = torch.where(rng, dq, zero)
    dl = torch.where(rng, dlv, zero)

    # at and below cloud base
    inv_dsub = rdiv(1.0, dsubcld)
    ds_mx = inv_dsub * (-gather(mflx_up, mx) * (gather(s_upd, mx) - gather(s_int, mx))
                        - gather(mflx_dn, mx) * (gather(s_dnd, mx) - gather(s_int, mx)))
    dq_mx = inv_dsub * (-gather(mflx_up, mx) * (gather(q_upd, mx) - gather(q_int, mx))
                        - gather(mflx_dn, mx) * (gather(q_dnd, mx) - gather(q_int, mx)))
    s_cols = list(dsdt.unbind(-1))
    q_cols = list(dqdt.unbind(-1))
    for k in range(pver):             # k = kbm .. pver-1 (kbm <= mx: nothing happens above kbm)
        at = (mx == k) & (k >= kbm)
        below = (k > mx) & (k >= kbm)
        s_prev = s_cols[k - 1] if k > 0 else s_cols[k]
        q_prev = q_cols[k - 1] if k > 0 else q_cols[k]
        s_cols[k] = torch.where(at, ds_mx, torch.where(below, s_prev, s_cols[k]))
        q_cols[k] = torch.where(at, dq_mx, torch.where(below, q_prev, q_cols[k]))
    return torch.stack(s_cols, -1), torch.stack(q_cols, -1), dl
