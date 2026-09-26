"""ZM dilute CAPE: launch level, entraining parcel, CAPE.

Translation of, at E3SM 80fe0a71 (components/eamxx/src/physics/zm/impl/):
  zm_compute_dilute_cape_impl.hpp       compute_dilute_cape
  zm_find_mse_max_impl.hpp              find_mse_max (PERGRO off)
  zm_compute_dilute_parcel_impl.hpp     compute_dilute_parcel
  zm_compute_cape_from_parcel_impl.hpp  compute_cape_from_parcel

Tensors are (ncol, pver), level 0 at the top; per-column level indices are int64
(ncol,). The reference's serial level loops are Python loops over levels, with
every column advanced together under masks. Pressures are in mb.
"""
from __future__ import annotations

import torch

from .. import constants as PC
from .. import mathfn
from ..ekat import max_, rdiv
from .constants import ZMC, ZMRuntime
from .functions import entropy, invert_entropy
from .util import gather, seq_sum


def find_mse_max(rt: ZMRuntime, pver: int, num_msg: int, msemax_top_k, temperature, zmid,
                 sp_humidity):
    """Level of maximum moist static energy, searched upward from pver-1-mx_bot_lyr_adj.
    Returns (msemax_klev, mse_max_val)."""
    ncol = temperature.shape[0]
    dev = temperature.device
    mse_max_val = torch.zeros(ncol, dtype=temperature.dtype, device=dev)
    msemax_klev = torch.full((ncol,), pver - 1, dtype=torch.int64, device=dev)
    bot_layer = pver - 1 - rt.mx_bot_lyr_adj
    mse = PC.Cpair * temperature + PC.gravit * zmid + PC.LatVap * sp_humidity
    for k in range(bot_layer, num_msg - 1, -1):
        m = (k >= msemax_top_k) & (mse[:, k] > mse_max_val)
        mse_max_val = torch.where(m, mse[:, k], mse_max_val)
        msemax_klev = torch.where(m, torch.full_like(msemax_klev, k), msemax_klev)
    return msemax_klev, mse_max_val


def _invert(s, p, qt, tfg, active, env_t, env_q, nonconv):
    """invert_entropy on the active columns; the others get finite stand-in inputs."""
    s_ = torch.where(active, s, torch.zeros_like(s))
    qt_ = torch.where(active, qt, env_q)
    tfg_ = torch.where(active, tfg, env_t)
    t, qst, bad = invert_entropy(s_, p, qt_, tfg_, active=active)
    return t, qst, nonconv | (bad & active)


def compute_dilute_parcel(rt: ZMRuntime, pver: int, num_msg: int, klaunch, pmid, temperature,
                          sp_humidity, tpert, pblt, parcel_temp, parcel_vtemp, parcel_qsat,
                          lcl_pmid, lcl_temperature, lcl_klev):
    """Entraining parcel from the launch level klaunch upward (Raymond and Blyth 1986, 1992).
    Returns (parcel_temp, parcel_vtemp, parcel_qsat, lcl_pmid, lcl_temperature, lcl_klev,
    nonconverged)."""
    ncol = pmid.shape[0]
    z = torch.zeros(ncol, dtype=pmid.dtype, device=pmid.device)
    tpert_loc = tpert
    if rt.tpert_fix:
        tpert_loc = torch.where(klaunch < pblt, z, tpert)

    T = list(temperature.unbind(-1))
    q = list(sp_humidity.unbind(-1))
    p = list(pmid.unbind(-1))
    tmix, qtmix, qsmix, smix = ([z] * pver for _ in range(4))
    xsh2o, ds_xsh2o, ds_freeze = ([z] * pver for _ in range(3))
    pt, pv, pq = (list(x.unbind(-1)) for x in (parcel_temp, parcel_vtemp, parcel_qsat))
    mp = qtp = sp = sp0 = qtp0 = mp0 = z
    nonconv = torch.zeros(ncol, dtype=torch.bool, device=pmid.device)

    # Entrainment loop (from the launch level upward)
    for k in range(pver - 1, num_msg - 1, -1):
        at = klaunch == k
        above = k < klaunch
        mp0 = torch.where(at, torch.ones_like(z), mp0)
        qtp0 = torch.where(at, q[k], qtp0)
        sp0 = torch.where(at, entropy(T[k], p[k], torch.where(at, qtp0, q[k])), sp0)
        s_in = torch.where(at, sp0, smix[k])
        qt_in = torch.where(at, qtp0, qtmix[k])
        tfg = T[k]
        if k < pver - 1:
            dp = p[k] - p[k + 1]
            qtenv = ZMC.half * (q[k] + q[k + 1])
            tenv = ZMC.half * (T[k] + T[k + 1])
            penv = ZMC.half * (p[k] + p[k + 1])
            senv = entropy(tenv, penv, qtenv)
            dpdz = -(penv * PC.gravit) / (PC.Rair * tenv)
            dzdp = rdiv(1.0, dpdz)
            dmpdp = rt.dmpdz * dzdp
            sp = torch.where(above, sp - dmpdp * dp * senv, sp)
            qtp = torch.where(above, qtp - dmpdp * dp * qtenv, qtp)
            mp = torch.where(above, mp - dmpdp * dp, mp)
            mtot = torch.where(above, mp0 + mp, torch.ones_like(mp))   # 0 below launch: safe input
            s_above = (sp0 + sp) / mtot
            qt_above = (qtp0 + qtp) / mtot
            s_in = torch.where(above, s_above, s_in)
            qt_in = torch.where(above, qt_above, qt_in)
            tfg = torch.where(above, tmix[k + 1], tfg)
        act = at | above
        smix[k] = torch.where(act, s_in, smix[k])
        qtmix[k] = torch.where(act, qt_in, qtmix[k])
        t_new, qs_new, nonconv = _invert(smix[k], p[k], qtmix[k], tfg, act, T[k], q[k], nonconv)
        tmix[k] = torch.where(act, t_new, tmix[k])
        qsmix[k] = torch.where(act, qs_new, qsmix[k])

        if k < pver - 1:
            # LCL: first level where qsmix <= qtmix on ascent
            cond = above & (qsmix[k] <= qtmix[k]) & (qsmix[k + 1] > qtmix[k + 1])
            lcl_klev = torch.where(cond, torch.full_like(lcl_klev, k), lcl_klev)
            qxsk = qtmix[k] - qsmix[k]
            qxskp1 = qtmix[k + 1] - qsmix[k + 1]
            dqxsdp = (qxsk - qxskp1) / dp
            dqxsdp = torch.where(cond, dqxsdp, torch.ones_like(dqxsdp))
            lcl_p_new = p[k + 1] - qxskp1 / dqxsdp
            lcl_pmid = torch.where(cond, lcl_p_new, lcl_pmid)
            dsdp = (smix[k] - smix[k + 1]) / dp
            dqtdp = (qtmix[k] - qtmix[k + 1]) / dp
            lcl_p_safe = torch.where(cond, lcl_pmid, p[k])
            slcl = smix[k + 1] + dsdp * (lcl_p_safe - p[k + 1])
            qtlcl = qtmix[k + 1] + dqtdp * (lcl_p_safe - p[k + 1])
            t_lcl, _qslcl, nonconv = _invert(slcl, lcl_p_safe, qtlcl, tmix[k], cond, T[k], q[k],
                                             nonconv)
            lcl_temperature = torch.where(cond, t_lcl, lcl_temperature)

    # Precipitation/freezing loop: condensate above lwmax rains out, freezing below Tmelt
    for k in range(pver - 1, num_msg - 1, -1):
        at = klaunch == k
        pt[k] = torch.where(at, tmix[k], pt[k])
        pq[k] = torch.where(at, q[k], pq[k])
        pv_at = (pt[k] + rt.tpert_fac * tpert_loc) * (1 + PC.ZVIR * pq[k]) / (1 + pq[k])
        pv[k] = torch.where(at, pv_at, pv[k])
        if k == pver - 1:
            continue
        above = k < klaunch
        for _ in range(ZMC.nit_lheat):
            xsh2o[k] = torch.where(above, max_(0., qtmix[k] - qsmix[k] - ZMC.lwmax), xsh2o[k])
            tm_safe = torch.where(above, tmix[k], T[k])
            ds_x = ds_xsh2o[k + 1] - PC.CpLiq * mathfn.log(tm_safe / PC.Tmelt) * \
                max_(0., xsh2o[k] - xsh2o[k + 1])
            ds_xsh2o[k] = torch.where(above, ds_x, ds_xsh2o[k])
            cold = above & (tmix[k] <= PC.Tmelt)
            c1 = cold & (ds_freeze[k + 1] == 0)
            ds_freeze[k] = torch.where(
                c1, rdiv(PC.LatIce, tm_safe) * max_(0., qtmix[k] - qsmix[k] - xsh2o[k]), ds_freeze[k])
            c2 = cold & (ds_freeze[k + 1] != 0)
            ds_freeze[k] = torch.where(
                c2, ds_freeze[k + 1] + rdiv(PC.LatIce, tm_safe) * max_(0., qsmix[k + 1] - qsmix[k]),
                ds_freeze[k])
            new_s = smix[k] + ds_xsh2o[k] + ds_freeze[k]
            new_q = qtmix[k] - xsh2o[k]
            t_new, qs_new, nonconv = _invert(new_s, p[k], new_q, tmix[k], above, T[k], q[k],
                                             nonconv)
            tmix[k] = torch.where(above, t_new, tmix[k])
            qsmix[k] = torch.where(above, qs_new, qsmix[k])
        pt[k] = torch.where(above, tmix[k], pt[k])
        new_q = qtmix[k] - xsh2o[k]
        pq[k] = torch.where(above, torch.where(new_q > qsmix[k], qsmix[k], new_q), pq[k])
        pv_above = (pt[k] + rt.tpert_fac * tpert_loc) * (1 + PC.ZVIR * pq[k]) / (1 + new_q)
        pv[k] = torch.where(above, pv_above, pv[k])

    return (torch.stack(pt, -1), torch.stack(pv, -1), torch.stack(pq, -1), lcl_pmid,
            lcl_temperature, lcl_klev, nonconv)


def compute_cape_from_parcel(rt: ZMRuntime, pver: int, num_cin: int, num_msg: int, temperature,
                             tv, sp_humidity, pint, msemax_klev, lcl_pmid, lcl_klev, parcel_qsat,
                             parcel_temp, parcel_vtemp):
    """Buoyancy, equilibrium level and CAPE. Returns (parcel_qsat, parcel_temp, parcel_vtemp,
    eql_klev, cape)."""
    ncol = temperature.shape[0]
    dev = temperature.device
    lev = torch.arange(pver, device=dev)
    lcl_ok = (lcl_pmid >= ZMC.lcl_pressure_threshold).unsqueeze(-1)
    in_rng = (lev >= num_msg).unsqueeze(0)
    mask_b = in_rng & (lev <= msemax_klev.unsqueeze(-1)) & lcl_ok
    buoyancy = torch.where(mask_b, parcel_vtemp - tv + rt.tiedke_add, torch.zeros_like(tv))
    reset = in_rng & ~mask_b
    parcel_qsat = torch.where(reset, sp_humidity, parcel_qsat)
    parcel_temp = torch.where(reset, temperature, parcel_temp)
    parcel_vtemp = torch.where(reset, tv, parcel_vtemp)

    # equilibrium levels, counting negative-buoyancy crossings (at most num_cin)
    eql_tmp = torch.full((ncol, num_cin), pver - 1, dtype=torch.int64, device=dev)
    cnt = torch.zeros(ncol, dtype=torch.int64, device=dev)
    ok = lcl_pmid >= ZMC.lcl_pressure_threshold
    slot = torch.arange(num_cin, device=dev)
    for k in range(num_msg + 1, pver - 1):   # k < lcl_klev <= pver-2 (k+1 stays in range)
        m = ok & (k < lcl_klev) & (buoyancy[:, k + 1] > 0) & (buoyancy[:, k] <= 0)
        cnt = torch.where(m, torch.clamp(cnt + 1, max=num_cin), cnt)
        put = m.unsqueeze(-1) & (slot == (cnt - 1).unsqueeze(-1))
        eql_tmp = torch.where(put, torch.full_like(eql_tmp, k), eql_tmp)

    # candidate CAPE values: sum over levels below each equilibrium level (in level order)
    logp = mathfn.log(pint[:, 1:] / pint[:, :-1])
    term = PC.Rair * buoyancy * logp
    cape = torch.zeros(ncol, dtype=tv.dtype, device=dev)
    eql_klev = torch.full((ncol,), pver - 1, dtype=torch.int64, device=dev)
    for n in range(num_cin):
        m = in_rng & lcl_ok & (lev <= msemax_klev.unsqueeze(-1)) & (lev > eql_tmp[:, n:n + 1])
        cape_n = seq_sum(torch.where(m, term, torch.zeros_like(term))[:, num_msg:])
        better = cape_n > cape
        cape = torch.where(better, cape_n, cape)
        eql_klev = torch.where(better, eql_tmp[:, n], eql_klev)
    cape = max_(cape, 0.)
    return parcel_qsat, parcel_temp, parcel_vtemp, eql_klev, cape


def compute_dilute_cape(rt: ZMRuntime, pver: int, num_cin: int, num_msg: int, sp_humidity_in,
                        temperature_in, zmid, pmid, pint, pblt, tpert, calc_msemax_klev: bool,
                        prev_msemax_klev=None):
    """CAPE of a dilute (entraining) parcel (use_input_tq_mx = false, as zm_conv_main calls it).

    Returns a dict: parcel_qsat, msemax_klev, lcl_klev, eql_klev, cape, q_mx, t_mx,
    parcel_temp, lcl_temperature, nonconverged.
    """
    ncol = pmid.shape[0]
    dev = pmid.device
    temperature = temperature_in
    sp_humidity = sp_humidity_in
    msemax_klev = torch.full((ncol,), pver - 1, dtype=torch.int64, device=dev)
    tv = temperature * (1 + PC.ZVIR * sp_humidity) / (1 + sp_humidity)
    parcel_temp = temperature
    parcel_qsat = sp_humidity
    parcel_vtemp = tv

    pblt_ull = torch.zeros(ncol, dtype=torch.int64, device=dev)
    if rt.trig_ull:
        for k in range(pver - 2, num_msg - 1, -1):
            m = (pmid[:, k] <= ZMC.ull_upper_launch_pressure) & \
                (pmid[:, k + 1] > ZMC.ull_upper_launch_pressure)
            pblt_ull = torch.where(m, torch.full_like(pblt_ull, k), pblt_ull)

    if rt.trig_dcape and not calc_msemax_klev:
        msemax_klev = prev_msemax_klev
    else:
        top_k = pblt_ull if rt.trig_ull else pblt
        msemax_klev, _ = find_mse_max(rt, pver, num_msg, top_k, temperature, zmid, sp_humidity)

    q_mx = gather(sp_humidity, msemax_klev)
    t_mx = gather(temperature, msemax_klev)
    lcl_klev = msemax_klev
    lcl_pmid = gather(pmid, msemax_klev)
    lcl_temperature = gather(temperature, msemax_klev)

    (parcel_temp, parcel_vtemp, parcel_qsat, lcl_pmid, lcl_temperature, lcl_klev,
     nonconv) = compute_dilute_parcel(rt, pver, num_msg, msemax_klev, pmid, temperature,
                                      sp_humidity, tpert, pblt, parcel_temp, parcel_vtemp,
                                      parcel_qsat, lcl_pmid, lcl_temperature, lcl_klev)

    parcel_qsat, parcel_temp, parcel_vtemp, eql_klev, cape = compute_cape_from_parcel(
        rt, pver, num_cin, num_msg, temperature, tv, sp_humidity, pint, msemax_klev, lcl_pmid,
        lcl_klev, parcel_qsat, parcel_temp, parcel_vtemp)

    return dict(parcel_qsat=parcel_qsat, msemax_klev=msemax_klev, lcl_klev=lcl_klev,
                eql_klev=eql_klev, cape=cape, q_mx=q_mx, t_mx=t_mx, parcel_temp=parcel_temp,
                lcl_temperature=lcl_temperature, nonconverged=nonconv)
