"""ZM updraft and downdraft properties.

Translation of, at E3SM 80fe0a71 (components/eamxx/src/physics/zm/impl/):
  zm_cloud_properties_impl.hpp              zm_cloud_properties
  zm_calc_fractional_entrainment_impl.hpp   zm_calc_fractional_entrainment
  zm_downdraft_properties_impl.hpp          zm_downdraft_properties

Tensors are (ncol, pver) (pflx: pver+1), level 0 at the top; jb, jt, lel, ... are
int64 (ncol,). Serial level loops of the reference are Python loops over levels
that advance every column together under masks. Pressures in mb.
"""
from __future__ import annotations

import torch

from .. import constants as PC
from .. import mathfn
from ..ekat import max_, min_, rdiv
from .constants import ZMC, ZMRuntime
from .functions import qsat_hPa
from .util import gather, gather_cols, seq_sum


def _lev(pver, dev):
    return torch.arange(pver, device=dev).unsqueeze(0)


def zm_calc_fractional_entrainment(pver: int, msg: int, jb, jt, j0, z_mid, z_int, dz, h_env,
                                   h_env_sat):
    """Fractional entrainment lambda(z) from the Taylor series of ZM95 eq (A6).
    Returns (j0, h_env_min, lambda, lambda_max)."""
    ncol = z_mid.shape[0]
    dev = z_mid.device
    zero = torch.zeros(ncol, dtype=z_mid.dtype, device=dev)
    lev = _lev(pver, dev)
    jb_ = jb.unsqueeze(-1)
    jt_ = jt.unsqueeze(-1)

    # minimum environmental MSE between j0 and jb (Kokkos Min reduction, then min with 1e6)
    in_j = (lev >= msg) & (lev >= j0.unsqueeze(-1)) & (lev <= jb_)
    big = torch.finfo(h_env.dtype).max
    h_min_new = torch.where(in_j, h_env, torch.full_like(h_env, big)).amin(-1)
    h_env_min = min_(torch.full_like(h_min_new, 1.e6), h_min_new)

    h_jb = gather(h_env, jb)
    k1, i2, i3, i4 = ([zero] * pver for _ in range(4))
    for k in range(pver - 2, msg - 1, -1):
        m = (k < jb) & (k >= jt)
        k1k = k1[k + 1] + (h_jb - h_env[:, k]) * dz[:, k]
        ihat = ZMC.half * (k1[k + 1] + k1k)
        i2k = i2[k + 1] + ihat * dz[:, k]
        idag = ZMC.half * (i2[k + 1] + i2k)
        i3k = i3[k + 1] + idag * dz[:, k]
        iprm = ZMC.half * (i3[k + 1] + i3k)
        i4k = i4[k + 1] + iprm * dz[:, k]
        k1[k] = torch.where(m, k1k, k1[k])
        i2[k] = torch.where(m, i2k, i2[k])
        i3[k] = torch.where(m, i3k, i3[k])
        i4[k] = torch.where(m, i4k, i4[k])
    k1, i2, i3, i4 = (torch.stack(x, -1) for x in (k1, i2, i3, i4))

    # approximate lambda(z), levels msg+1 .. pver-1
    lambda_tmp = torch.zeros_like(z_mid)
    ks = slice(msg + 1, pver)
    km1 = slice(msg, pver - 1)
    in_cloud = ~((lev[:, ks] < jt_) | (lev[:, ks] >= jb_))
    expnum = h_jb.unsqueeze(-1) - (h_env_sat[:, km1] * (z_int[:, ks] - z_mid[:, ks])
                                   + h_env_sat[:, ks] * (z_mid[:, km1] - z_int[:, ks])) \
        / (z_mid[:, km1] - z_mid[:, ks])
    expnum = torch.where(in_cloud, expnum, torch.zeros_like(expnum))
    k1k, i2k, i3k, i4k = k1[:, ks], i2[:, ks], i3[:, ks], i4[:, ks]
    m = (((h_jb - h_env_min) > ZMC.mse_min_diff).unsqueeze(-1) & (expnum > 0)) & \
        (k1k > expnum * dz[:, ks])
    k1s = torch.where(m, k1k, torch.ones_like(k1k))
    tmp = expnum / k1s
    lt = (tmp
          + i2k / k1s * tmp * tmp
          + (2 * i2k * i2k - k1s * i3k) / (k1s * k1s) * tmp * tmp * tmp
          + (-5 * k1s * i2k * i3k + 5 * i2k * i2k * i2k + k1s * k1s * i4k)
          / (k1s * k1s * k1s) * tmp * tmp * tmp * tmp)
    lt = max_(lt, ZMC.lambda_limit_min)
    lt = min_(lt, ZMC.lambda_limit_max)
    lambda_tmp = torch.cat([lambda_tmp[:, :msg + 1], torch.where(m, lt, lambda_tmp[:, ks])], -1)

    # move the detrainment level down if entrainment there is too small
    lt_j0 = gather(lambda_tmp, j0)
    lt_j0p1 = gather(lambda_tmp, torch.clamp(j0 + 1, max=pver - 1))
    mv = (j0 < jb) & (lt_j0 < ZMC.lambda_threshold) & (lt_j0p1 > lt_j0)
    j0 = torch.where(mv, j0 + 1, j0)

    # entrainment must not increase above the level where detrainment starts
    cols = list(lambda_tmp.unbind(-1))
    for k in range(msg + 1, pver):
        m = (k >= jt) & (k <= j0)
        cols[k] = torch.where(m, max_(cols[k], cols[k - 1]), cols[k])
    lambda_tmp = torch.stack(cols, -1)

    lambda_max = gather(lambda_tmp, j0)
    lam = torch.zeros_like(z_mid)
    lam = torch.where(lev == jb_, lambda_max.unsqueeze(-1), lam)
    inr = lev >= msg
    lam = torch.where(inr & (lev >= j0.unsqueeze(-1)) & (lev <= jb_), lambda_max.unsqueeze(-1), lam)
    lam = torch.where(inr & (lev < j0.unsqueeze(-1)) & (lev >= jt_), lambda_tmp, lam)
    return j0, h_env_min, lam, lambda_max


def zm_downdraft_properties(rt: ZMRuntime, pver: int, msg: int, jb, jt, j0, jd, z_int, dz, s_mid,
                            q_mid, h_env, lambda_max, qsthat, hsthat, gamhat, mflx_up, mflx_dn,
                            entr_dn, s_dnd, q_dnd, h_dnd, q_dnd_sat, evp, totevp):
    """Downdraft mass flux, entrainment, thermodynamics and evaporation.
    Returns (jt, jd, mflx_dn, entr_dn, s_dnd, q_dnd, h_dnd, q_dnd_sat, evp, totevp)."""
    dev = z_int.device
    lev = _lev(pver, dev)
    lm_pos = lambda_max > 0
    jt = min_(jt, jb - 1)
    jd = max_(j0, jt + 1)
    jd = min_(jd, jb)
    h_dnd = torch.where(lev == jd.unsqueeze(-1), gather(h_env, jd - 1).unsqueeze(-1), h_dnd)
    has_dd = (jd < jb) & lm_pos
    lm_safe = torch.where(lm_pos, lambda_max, torch.ones_like(lambda_max))
    mflx_dn = torch.where((lev == jd.unsqueeze(-1)) & has_dd.unsqueeze(-1),
                          (-rt.alfa * lm_safe / lm_safe).unsqueeze(-1), mflx_dn)

    z_jd = gather(z_int, jd).unsqueeze(-1)
    m = (lev >= msg) & (lev > jd.unsqueeze(-1)) & (lev <= jb.unsqueeze(-1)) & lm_pos.unsqueeze(-1)
    dz_tmp = z_jd - z_int[:, :pver]
    dz_safe = torch.where(m, dz_tmp, torch.ones_like(dz_tmp))
    lmu = lm_safe.unsqueeze(-1)
    val = rdiv(-rt.alfa, 2 * lmu) * (mathfn.exp(2 * lmu * dz_safe) - 1) / dz_safe
    mflx_dn = torch.where(m, val, mflx_dn)

    mu_jb = gather(mflx_up, jb)
    md_jb = gather(mflx_dn, jb)
    md_jb = torch.where(has_dd, md_jb, torch.ones_like(md_jb))
    ratmjb = torch.where(has_dd, min_(torch.abs(mu_jb / md_jb), 1.0), torch.zeros_like(md_jb))
    m = (lev >= msg) & (lev >= jt.unsqueeze(-1)) & (lev <= jb.unsqueeze(-1)) & has_dd.unsqueeze(-1)
    mflx_dn = torch.where(m, mflx_dn * ratmjb.unsqueeze(-1), mflx_dn)

    # downdraft entrainment and MSE, top to bottom
    md = list(mflx_dn.unbind(-1))
    ed = list(entr_dn.unbind(-1))
    hd = list(h_dnd.unbind(-1))
    for k in range(max(msg, 1), pver):
        m = (k >= jt) & lm_pos
        ed_new = (md[k - 1] - md[k]) / dz[:, k - 1]
        ed[k - 1] = torch.where(m, ed_new, ed[k - 1])
        mdt = min_(md[k], -ZMC.small)
        hd_new = (md[k - 1] * hd[k - 1] - dz[:, k - 1] * ed[k - 1] * h_env[:, k - 1]) / mdt
        hd[k] = torch.where(m, hd_new, hd[k])
    entr_dn = torch.stack(ed, -1)
    h_dnd = torch.stack(hd, -1)

    m = (lev >= msg + 1) & (lev >= jd.unsqueeze(-1)) & (lev <= jb.unsqueeze(-1)) & has_dd.unsqueeze(-1)
    qds = qsthat + gamhat * (h_dnd - hsthat) / (PC.LatVap * (1 + gamhat))
    q_dnd_sat = torch.where(m, qds, q_dnd_sat)

    # downdraft quantities at the source level
    at_jd = lev == jd.unsqueeze(-1)
    q_jd = gather(q_dnd_sat, jd)
    q_dnd = torch.where(at_jd, q_jd.unsqueeze(-1), q_dnd)
    s_jd = (gather(h_dnd, jd) - PC.LatVap * q_jd) / PC.Cpair
    s_dnd = torch.where(at_jd, s_jd.unsqueeze(-1), s_dnd)

    # downdraft evaporation, top to bottom
    qd = list(q_dnd.unbind(-1))
    sd = list(s_dnd.unbind(-1))
    ev = list(evp.unbind(-1))
    qds_c = list(q_dnd_sat.unbind(-1))
    tot = torch.zeros_like(lambda_max)
    for k in range(msg + 1, pver - 1):   # k < jb <= pver-1
        m = (k >= jd) & (k < jb) & lm_pos
        qd[k + 1] = torch.where(m, qds_c[k + 1], qd[k + 1])
        e = -ed[k] * q_mid[:, k] + (md[k] * qd[k] - md[k + 1] * qd[k + 1]) / dz[:, k]
        e = max_(e, 0.0)
        ev[k] = torch.where(m, e, ev[k])
        mdt = min_(md[k + 1], -ZMC.small)
        s_new = ((PC.LatVap / PC.Cpair * ev[k] - ed[k] * s_mid[:, k]) * dz[:, k] + md[k] * sd[k]) / mdt
        sd[k + 1] = torch.where(m, s_new, sd[k + 1])
        tot = torch.where(m, tot + dz[:, k] * ed[k] * q_mid[:, k], tot)
    q_dnd = torch.stack(qd, -1)
    s_dnd = torch.stack(sd, -1)
    evp = torch.stack(ev, -1)

    totevp = totevp - tot
    totevp = totevp + (gather(mflx_dn, jd) * gather(q_dnd, jd) - gather(mflx_dn, jb) * gather(q_dnd, jb))
    return jt, jd, mflx_dn, entr_dn, s_dnd, q_dnd, h_dnd, q_dnd_sat, evp, totevp


def zm_cloud_properties(rt: ZMRuntime, pver: int, msg: int, limcnv: int, p_mid, z_mid, z_int,
                        t_mid, s_mid, s_int, q_mid, landfrac, tpert_g, jb, lel):
    """Updraft and downdraft properties for a unit cloud-base mass flux.

    Returns a dict with jt, jlcl, j0, jd and the profiles mflx_up, entr_up, detr_up,
    mflx_dn, entr_dn, mflx_net, s_upd, q_upd, ql, s_dnd, q_dnd, qst, cu, evp, pflx
    (pver+1), rprd.
    """
    ncol = p_mid.shape[0]
    dev = p_mid.device
    dt_ = p_mid.dtype
    lev = _lev(pver, dev)
    latvap, cpair, grav, epsilo, rdair = PC.LatVap, PC.Cpair, PC.gravit, PC.ep_2, PC.Rair
    zero2 = torch.zeros_like(p_mid)

    c0mask = rt.c0_ocn * (1 - landfrac) + rt.c0_lnd * landfrac

    # 1. saturation, gamma, layer thickness, environmental MSE
    _est, qst = qsat_hPa(t_mid, p_mid)
    gamma = qst * (1 + qst / epsilo) * epsilo * latvap / (rdair * t_mid * t_mid) * latvap / cpair
    dz = z_int[:, :pver] - z_int[:, 1:]
    h_env = cpair * t_mid + grav * z_mid + latvap * q_mid
    h_env_sat = cpair * t_mid + grav * z_mid + latvap * qst
    h_upd = h_env
    h_dnd = h_env
    s_upd = s_mid
    s_dnd = s_mid
    q_upd = q_mid
    q_dnd = q_mid
    q_dnd_sat = q_mid
    mflx_up = entr_up = detr_up = mflx_dn = entr_dn = mflx_net = zero2
    cu = evp = ql = rprd = zero2

    # 2. interface values (geometric mean where the jump is large), levels msg..pver-1
    qsthat = torch.where(lev >= msg, qst, zero2)
    hsthat = torch.where(lev >= msg, h_env_sat, zero2)
    gamhat = torch.where(lev >= msg, gamma, zero2)
    ks, km1 = slice(msg + 1, pver), slice(msg, pver - 1)
    a, b = qst[:, km1], qst[:, ks]
    big = torch.abs(a - b) > ZMC.interp_diff_min
    den = torch.where(big, a - b, torch.ones_like(a))
    qh = torch.where(big, mathfn.log(torch.where(big, a / b, torch.ones_like(a))) * a * b / den, b)
    a, b = gamma[:, km1], gamma[:, ks]
    big = torch.abs(a - b) > ZMC.interp_diff_min
    den = torch.where(big, a - b, torch.ones_like(a))
    gh = torch.where(big, mathfn.log(torch.where(big, a / b, torch.ones_like(a))) * a * b / den, b)
    qsthat = torch.cat([qsthat[:, :msg + 1], qh], -1)
    hsthat = torch.cat([hsthat[:, :msg + 1], cpair * s_int[:, ks] + latvap * qh], -1)
    gamhat = torch.cat([gamhat[:, :msg + 1], gh], -1)

    # 3. first estimate of cloud top; level of minimum saturated MSE (detrainment onset)
    jt = max_(lel, limcnv + 1)
    jt = min_(jt, pver - 1)
    jd = torch.full_like(jt, pver - 1)
    jlcl = lel
    start = max_(jt, msg)
    rng = (lev >= start.unsqueeze(-1)) & (lev < (jb + 1).unsqueeze(-1))
    bigv = torch.finfo(dt_).max
    hs = torch.where(rng, h_env_sat, torch.full_like(h_env_sat, bigv))
    j0_val, j0_loc = torch.min(hs, dim=-1)   # first index of the minimum (strict '<' scan)
    j0 = torch.full_like(jt, msg)
    j0 = torch.where(j0_val < 1.e6, j0_loc, j0)
    j0 = min_(j0, jb - 2)
    j0 = max_(j0, jt + 2)
    j0 = min_(j0, torch.full_like(j0, pver - 1))

    # 4. updraft MSE at and above cloud base, from the cloud-base value
    h_jb = gather(h_env, jb).unsqueeze(-1)
    s_jb = gather(s_mid, jb).unsqueeze(-1)
    m = (lev >= msg) & (lev >= jt.unsqueeze(-1)) & (lev <= jb.unsqueeze(-1))
    if rt.tpert_fix:
        h_new = h_jb + cpair * rt.tiedke_add
        s_new = s_jb + rt.tiedke_add
    else:
        tp = tpert_g.unsqueeze(-1)
        h_new = h_jb + cpair * (rt.tiedke_add + rt.tpert_fac * tp)
        s_new = s_jb + rt.tiedke_add + rt.tpert_fac * tp
    h_upd = torch.where(m, h_new.expand_as(h_upd), h_upd)
    s_upd = torch.where(m, s_new.expand_as(s_upd), s_upd)

    # 5. fractional entrainment
    j0, _h_env_min, lam, lambda_max = zm_calc_fractional_entrainment(
        pver, msg, jb, jt, j0, z_mid, z_int, dz, h_env, h_env_sat)
    lm_pos = lambda_max > 0
    lm_safe = torch.where(lm_pos, lambda_max, torch.ones_like(lambda_max))

    # 6b. updraft mass flux from cloud base upward
    mu = list(mflx_up.unbind(-1))
    eu = list(entr_up.unbind(-1))
    du = list(detr_up.unbind(-1))
    one = torch.ones(ncol, dtype=dt_, device=dev)
    at_jb = [(jb == k) & lm_pos for k in range(pver)]
    for k in range(pver):
        mu[k] = torch.where(at_jb[k], one, mu[k])
        eu[k] = torch.where(at_jb[k], rdiv(1.0, dz[:, k]), eu[k])
    z_jb = gather(z_int, jb)
    for k in range(pver - 2, msg - 1, -1):   # k < jb <= pver-1
        m = (k >= jt) & (k < jb) & lm_pos
        zuef = z_int[:, k] - z_jb
        zuef = torch.where(m, zuef, one)
        rmue = rdiv(1.0, lm_safe) * (mathfn.exp(lam[:, k + 1] * zuef) - 1) / zuef
        mu_k = rdiv(1.0, lm_safe) * (mathfn.exp(lam[:, k] * zuef) - 1) / zuef
        mu[k] = torch.where(m, mu_k, mu[k])
        eu[k] = torch.where(m, (rmue - mu[k + 1]) / dz[:, k], eu[k])
        du[k] = torch.where(m, (rmue - mu[k]) / dz[:, k], du[k])

    # 6c. updraft MSE from cloud base upward
    hu = list(h_upd.unbind(-1))
    diag_weak = torch.zeros(ncol, dtype=torch.bool, device=dev)
    for k in range(pver - 2, -1, -1):
        m = (k <= jb - 1) & (k >= lel) & lm_pos
        weak = m & (mu[k] < ZMC.mu_min)
        strong = m & ~(mu[k] < ZMC.mu_min)
        diag_weak = diag_weak | weak
        mu_safe = torch.where(strong, mu[k], one)
        h_new = mu[k + 1] / mu_safe * hu[k + 1] + dz[:, k] / mu_safe * (
            eu[k] * h_env[:, k] - du[k] * h_env_sat[:, k])
        hu[k] = torch.where(weak, h_env[:, k], torch.where(strong, h_new, hu[k]))
        mu[k] = torch.where(weak, torch.zeros_like(mu[k]), mu[k])
        eu[k] = torch.where(weak, torch.zeros_like(eu[k]), eu[k])
        du[k] = torch.where(weak, mu[k + 1] / dz[:, k], du[k])

    # 6d. freezing contribution: tmp_frz is zero without zm_microp, so tot_frz = 0
    tot_frz = 0.0

    # 6e. cloud top where the updraft MSE falls below the saturated environmental MSE
    doit = torch.ones(ncol, dtype=torch.bool, device=dev)
    diag_top_undershoot = diag_top_cross = diag_top_other = ~doit
    hu_jb = gather_cols(hu, jb)
    for k in range(pver - 3, -1, -1):
        m = doit & (k <= jb - 2) & (k >= lel - 1)
        hs_k = hsthat[:, k]
        cross = (hu[k] <= hs_k) & (hu[k + 1] > hsthat[:, k + 1]) & (mu[k] >= ZMC.mu_min)
        c1 = m & cross
        jt = torch.where(c1 & (hu[k] - hs_k < ZMC.hu_diff_min), torch.full_like(jt, k + 1),
                         torch.where(c1, torch.full_like(jt, k), jt))
        c2 = m & ~cross & (((hu[k] > hu_jb) & (tot_frz <= 0)) | (mu[k] < ZMC.mu_min))
        jt = torch.where(c2, torch.full_like(jt, k + 1), jt)
        diag_top_undershoot = diag_top_undershoot | (c1 & (hu[k] - hs_k < ZMC.hu_diff_min))
        diag_top_cross = diag_top_cross | (c1 & ~(hu[k] - hs_k < ZMC.hu_diff_min))
        diag_top_other = diag_top_other | c2
        doit = doit & ~(c1 | c2)

    # 6f. zero the updraft at and above the cloud top; detrain at the cloud top
    for k in range(msg, pver):
        z_ = lm_pos & (k >= lel) & (k <= jt)
        mu[k] = torch.where(z_, torch.zeros_like(mu[k]), mu[k])
        eu[k] = torch.where(z_, torch.zeros_like(eu[k]), eu[k])
        du[k] = torch.where(z_, torch.zeros_like(du[k]), du[k])
        hu[k] = torch.where(z_, h_env[:, k], hu[k])
        if k < pver - 1:
            top = lm_pos & (jt == k)
            du[k] = torch.where(top, mu[k + 1] / dz[:, k], du[k])
            eu[k] = torch.where(top, torch.zeros_like(eu[k]), eu[k])
            mu[k] = torch.where(top, torch.zeros_like(mu[k]), mu[k])

    # 6g. updraft s and q from cloud base up to the LCL
    su = list(s_upd.unbind(-1))
    qu = list(q_upd.unbind(-1))
    done = torch.zeros(ncol, dtype=torch.bool, device=dev)
    q_jb = gather(q_mid, jb)
    g_over_cp = grav / cpair
    for k in range(pver - 1, msg, -1):
        base = (jb == k) & lm_pos
        qu[k] = torch.where(base, q_jb, qu[k])
        su[k] = torch.where(base, (hu[k] - latvap * qu[k]) / cpair, su[k])
        if k == pver - 1:
            continue
        m = ~done & (k > jt) & (k < jb) & lm_pos
        mu_safe = torch.where(m, mu[k], one)
        s_new = mu[k + 1] / mu_safe * su[k + 1] + dz[:, k] / mu_safe * (eu[k] - du[k]) * s_mid[:, k]
        q_new = mu[k + 1] / mu_safe * qu[k + 1] + dz[:, k] / mu_safe * (
            eu[k] * q_mid[:, k] - du[k] * qst[:, k])
        su[k] = torch.where(m, s_new, su[k])
        qu[k] = torch.where(m, q_new, qu[k])
        tu = su[k] - g_over_cp * z_int[:, k]
        p_ifc = (p_mid[:, k] + p_mid[:, k - 1]) / 2
        _estu, qstu = qsat_hPa(torch.where(m, tu, t_mid[:, k]), p_ifc)
        sat = m & (qu[k] >= qstu)
        jlcl = torch.where(sat, torch.full_like(jlcl, k), jlcl)
        done = done | sat
    s_upd = torch.stack(su, -1)
    q_upd = torch.stack(qu, -1)
    h_upd = torch.stack(hu, -1)
    mflx_up = torch.stack(mu, -1)
    entr_up = torch.stack(eu, -1)
    detr_up = torch.stack(du, -1)

    # 6h. saturated updraft above the LCL
    m = (lev >= msg + 1) & (lev > jt.unsqueeze(-1)) & (lev <= jlcl.unsqueeze(-1)) & \
        lm_pos.unsqueeze(-1)
    s_upd = torch.where(m, s_int + (h_upd - hsthat) / (cpair * (1 + gamhat)), s_upd)
    q_upd = torch.where(m, qsthat + gamhat * (h_upd - hsthat) / (latvap * (1 + gamhat)), q_upd)

    # 6i. condensation from the dry static energy budget
    mu_kp1 = torch.cat([mflx_up[:, 1:], zero2[:, :1]], -1)
    su_kp1 = torch.cat([s_upd[:, 1:], zero2[:, :1]], -1)
    m = (lev >= msg + 1) & lm_pos.unsqueeze(-1) & (lev >= jt.unsqueeze(-1)) & \
        (lev < jb.unsqueeze(-1))
    cu_new = ((mflx_up * s_upd - mu_kp1 * su_kp1) / dz - (entr_up - detr_up) * s_mid) / (latvap / cpair)
    cu_new = torch.where(lev == jt.unsqueeze(-1), zero2, cu_new)
    cu_new = max_(0.0, cu_new)
    cu = torch.where(m, cu_new, cu)

    # 6j. liquid water, rain production, total precipitation (bottom to top)
    qlc = list(ql.unbind(-1))
    rp = list(rprd.unbind(-1))
    totpcp = torch.zeros(ncol, dtype=dt_, device=dev)
    for k in range(pver - 1, msg, -1):
        rp[k] = torch.zeros_like(rp[k])
        if k == pver - 1:
            continue
        m = (k >= jt) & (k < jb) & lm_pos & (mflx_up[:, k] >= 0)
        pos = mflx_up[:, k] > 0
        mu_safe = torch.where(m & pos, mflx_up[:, k], one)
        ql1 = (mflx_up[:, k + 1] * qlc[k + 1] - dz[:, k] * detr_up[:, k] * qlc[k + 1]
               + dz[:, k] * cu[:, k]) / mu_safe
        ql_new = torch.where(pos, ql1 / (1 + dz[:, k] * c0mask), torch.zeros_like(ql1))
        qlc[k] = torch.where(m, ql_new, qlc[k])
        totpcp = torch.where(m, totpcp + dz[:, k] * (cu[:, k] - detr_up[:, k] * qlc[k + 1]), totpcp)
        rp[k] = torch.where(m, c0mask * mflx_up[:, k] * qlc[k], rp[k])
    ql = torch.stack(qlc, -1)
    rprd = torch.stack(rp, -1)

    # 7. downdraft
    totevp = torch.zeros(ncol, dtype=dt_, device=dev)
    (jt, jd, mflx_dn, entr_dn, s_dnd, q_dnd, h_dnd, q_dnd_sat, evp, totevp) = \
        zm_downdraft_properties(rt, pver, msg, jb, jt, j0, jd, z_int, dz, s_mid, q_mid, h_env,
                                lambda_max, qsthat, hsthat, gamhat, mflx_up, mflx_dn, entr_dn,
                                s_dnd, q_dnd, h_dnd, q_dnd_sat, evp, totevp)

    # 8. scale the downdraft by the precipitation available (eq 4.106)
    totpcp = max_(0.0, totpcp)
    totevp = max_(0.0, totevp)
    both = ((totevp > 0) & (totpcp > 0)).unsqueeze(-1)
    tsum = torch.where(both.squeeze(-1), totevp + totpcp, one)
    scale = min_(1.0, totpcp / tsum).unsqueeze(-1)
    m = lev >= msg + 1
    mflx_dn = torch.where(m, torch.where(both, mflx_dn * scale, zero2), mflx_dn)
    entr_dn = torch.where(m, torch.where(both, entr_dn * scale, zero2), entr_dn)
    evp = torch.where(m, torch.where(both, evp * scale, zero2), evp)
    rprd = torch.where(m, rprd - evp, rprd)

    # 9. precipitation flux at interfaces (running sum from the top)
    pflx = torch.cat([torch.zeros_like(p_mid[:, :1]), torch.cumsum(rprd * dz, dim=-1)], -1)

    # 10. net mass flux
    mflx_net = torch.where(lev >= msg, mflx_up + mflx_dn, mflx_net)

    # branch diagnostics (not used by the scheme; for reference/scripts/zm_branch_coverage.py)
    diag = dict(weak_mass_flux=diag_weak, top_crossing=diag_top_cross,
                top_crossing_undershoot=diag_top_undershoot, top_buoyancy_or_flux=diag_top_other,
                top_not_found=doit, lcl_found=done)
    return dict(_diag=diag, jt=jt, jlcl=jlcl, j0=j0, jd=jd, mflx_up=mflx_up, entr_up=entr_up,
                detr_up=detr_up, mflx_dn=mflx_dn, entr_dn=entr_dn, mflx_net=mflx_net,
                s_upd=s_upd, q_upd=q_upd, ql=ql, s_dnd=s_dnd, q_dnd=q_dnd, qst=qst, cu=cu,
                evp=evp, pflx=pflx, rprd=rprd)
