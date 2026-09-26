"""ZM main driver.

Translation of components/eamxx/src/physics/zm/impl/zm_conv_main_impl.hpp
(zm_conv_main) at E3SM 80fe0a71.

EAMxx runs the scheme's kernels on all columns of an MPI rank; two values are
reduced over the active columns of that set: ktm (highest cloud top) and kbm
(highest cloud base). They only bound level loops that stay zero outside each
column's own cloud, so they cannot change nonzero results. Here they are reduced
over the columns of the call, as the reference does.
"""
from __future__ import annotations

import torch

from .. import constants as PC
from ..ekat import max_, min_, rdiv
from .cape import compute_dilute_cape
from .closure import zm_calc_output_tend, zm_closure
from .cloud import zm_cloud_properties
from .constants import ZMC, ZMRuntime
from .util import gather, seq_sum


def zm_conv_main(rt: ZMRuntime, limcnv: int, is_first_step: bool, time_step: float, t_mid,
                 q_mid_in, omega, p_mid_in, p_int_in, p_del_in, geos, z_mid_in, z_int_in,
                 pbl_hgt, tpert, landfrac, t_star, q_star) -> dict:
    """Zhang-McFarlane deep convection for one step. Pressures in Pa, heights above the
    surface. Returns the outputs of zm_conv_main as a dict (names as in the reference)."""
    ncol, pver = t_mid.shape
    dev = t_mid.device
    gravit, cpair = PC.gravit, PC.Cpair
    msg = limcnv - 1
    lev = torch.arange(pver, device=dev).unsqueeze(0)
    zero = torch.zeros_like(t_mid)
    zero1 = torch.zeros(ncol, dtype=t_mid.dtype, device=dev)

    # Kernel 1: unit conversion, PBL top, dry static energy
    z_srf = geos / gravit
    p_mid = p_mid_in * ZMC.pa_to_mb
    z_mid = z_mid_in + z_srf.unsqueeze(-1)
    p_int = p_int_in * ZMC.pa_to_mb
    z_int = z_int_in + z_srf.unsqueeze(-1)
    rng = (lev >= msg + 1) & (lev < pver - 1)
    near = torch.abs(z_mid - z_srf.unsqueeze(-1) - pbl_hgt.unsqueeze(-1)) < \
        (z_int[:, :pver] - z_int[:, 1:]) * ZMC.half
    cand = torch.where(rng & near, lev.expand(ncol, -1), torch.full((ncol, pver), pver - 1,
                                                                   dtype=torch.int64, device=dev))
    pbl_top = min_(cand.amin(-1), torch.full((ncol,), pver - 1, dtype=torch.int64, device=dev))
    q_mid = q_mid_in
    s_mid = t_mid + (gravit / cpair) * z_mid
    s_int = s_mid
    q_int = q_mid

    # Kernel 2: CAPE of the current state
    c = compute_dilute_cape(rt, pver, rt.num_cin, msg, q_mid, t_mid, z_mid, p_mid, p_int, pbl_top,
                            tpert, True)
    msemax_klev, lcl, lel, cape = c["msemax_klev"], c["lcl_klev"], c["eql_klev"], c["cape"]
    t_pcl, q_pcl_sat, t_pcl_lcl = c["parcel_temp"], c["parcel_qsat"], c["lcl_temperature"]
    nonconv = c["nonconverged"]

    # Kernel 3: CAPE of the previous state, at the same launch level (DCAPE)
    dcape = zero1
    use_dcape = rt.trig_dcape and not is_first_step
    if use_dcape:
        cm = compute_dilute_cape(rt, pver, rt.num_cin, msg, q_star, t_star, z_mid, p_mid, p_int,
                                 pbl_top, tpert, False, msemax_klev)
        dcape = (cape - cm["cape"]) / time_step
        nonconv = nonconv | cm["nonconverged"]

    # active columns
    if use_dcape:
        active = (cape > rt.cape_threshold) & (dcape > rt.dcape_threshold)
    else:
        active = cape > rt.cape_threshold
    act2 = active.unsqueeze(-1)

    # Kernel 4: layer thickness in mb, sub-cloud thickness, interface s and q
    p_del = torch.where(act2, p_del_in * ZMC.pa_to_mb, zero)
    max_idx = max_(torch.full_like(msemax_klev, msg), msemax_klev)
    dsubcld = torch.where(active, seq_sum(torch.where(lev >= max_idx.unsqueeze(-1), p_del, zero)),
                          zero1)
    s_int, q_int = _interfaces(s_mid, q_mid, msg, pver, active)

    # Kernel 5: cloud properties (computed for every column, kept for active ones)
    cp = zm_cloud_properties(rt, pver, msg, limcnv, p_mid, z_mid, z_int, t_mid, s_mid, s_int,
                             q_mid, landfrac, tpert, msemax_klev, lel)
    jt, jlcl, j0, jd = cp["jt"], cp["jlcl"], cp["j0"], cp["jd"]

    # Kernel 6: per-length to per-pressure units, levels msg+1 .. pver-1
    dz = z_int[:, :pver] - z_int[:, 1:]
    rng = lev >= msg + 1
    p_del_s = torch.where(act2, p_del, torch.ones_like(p_del))
    conv = lambda x: torch.where(rng, x * dz / p_del_s, x)   # noqa: E731
    detr_up, entr_up, entr_dn = conv(cp["detr_up"]), conv(cp["entr_up"]), conv(cp["entr_dn"])
    cu, rprd, evp = conv(cp["cu"]), conv(cp["rprd"]), conv(cp["evp"])
    mflx_up, mflx_dn, mflx_net = cp["mflx_up"], cp["mflx_dn"], cp["mflx_net"]
    s_upd, q_upd, ql, s_dnd, q_dnd, qst = (cp[k] for k in ("s_upd", "q_upd", "ql", "s_dnd",
                                                            "q_dnd", "qst"))
    pflx = cp["pflx"]

    # Kernel 7: closure
    dsub_s = torch.where(active, dsubcld, torch.ones_like(dsubcld))
    cbmf = zm_closure(rt, pver, msg, rt.cape_threshold, lcl, lel, jt, msemax_klev, dsub_s, z_mid,
                      z_int, p_mid, p_del_s, t_mid, s_mid, q_mid, qst, ql, s_int, q_int,
                      t_pcl_lcl, t_pcl, q_pcl_sat, s_upd, q_upd, mflx_net, detr_up, mflx_up,
                      mflx_dn, q_dnd, s_dnd, cape)

    # Kernel 8: limit the cloud-base mass flux; scale the fluxes
    mx_val = torch.zeros(ncol, dtype=t_mid.dtype, device=dev) - torch.finfo(t_mid.dtype).max
    ratio = mflx_up / p_del_s
    for k in range(msg + 1, pver):
        mx_val = max_(mx_val, ratio[:, k])
    pos = mx_val > 0
    lim = rdiv(1.0, time_step * torch.where(pos, mx_val, torch.ones_like(mx_val)))
    cbmf = torch.where(pos, min_(cbmf, lim), zero1)
    if rt.clos_dyn_adj:
        omega_pbl = gather(omega, pbl_top)
        cbmf = max_(cbmf - omega_pbl * ZMC.pa_to_mb, 0.0)
        cbmf = torch.where(pos, min_(cbmf, lim), cbmf)
    if rt.no_deep_pbl:
        cbmf = torch.where(gather(z_mid_in, jt) < pbl_hgt, zero1, cbmf)
    cb = cbmf.unsqueeze(-1)
    sc = lambda x: torch.where(rng & act2, x * cb, x)   # noqa: E731
    mflx_up, mflx_dn, mflx_net = sc(mflx_up), sc(mflx_dn), sc(mflx_net)
    detr_up, entr_up, entr_dn = sc(detr_up), sc(entr_up), sc(entr_dn)
    rprd, cu, evp = sc(rprd), sc(cu), sc(evp)
    lev_i = torch.arange(pver + 1, device=dev).unsqueeze(0)
    pflx = torch.where((lev_i >= msg + 2) & act2, pflx * (cb * ZMC.mb_to_pa / gravit), pflx)

    # ktm, kbm: highest cloud top and cloud base over the active columns of this call
    # (0-dim tensors; Kokkos Min reductions starting from pver-1)
    big = torch.full((ncol + 1,), pver - 1, dtype=torch.int64, device=dev)
    ktm = torch.cat([torch.where(active, jt, big[:ncol]), big[ncol:]]).amin()
    kbm = torch.cat([torch.where(active, msemax_klev, big[:ncol]), big[ncol:]]).amin()

    # Kernel 9: tendencies
    dsdt, dqdt, dl = zm_calc_output_tend(pver, msg, jt, msemax_klev, ktm, kbm, dsub_s, p_del_s,
                                         s_int, q_int, s_upd, q_upd, mflx_up, detr_up, mflx_dn,
                                         s_dnd, q_dnd, ql, evp, cu)

    # Kernel 10: outputs, precipitation, reserved liquid
    out_rng = rng & act2
    q_new = torch.where(out_rng, q_mid_in + time_step * dqdt, q_mid_in)
    qtnd = torch.where(out_rng, dqdt, zero)
    zdu = torch.where(out_rng, detr_up, zero)
    mcon = torch.where(out_rng, mflx_net, zero)
    heat = torch.where(out_rng, dsdt * cpair, zero)
    dlf = torch.where(out_rng, dl, zero)
    jctop = torch.where(active, jt, torch.full_like(jt, pver - 1))
    jcbot = torch.where(active, msemax_klev, torch.zeros_like(msemax_klev))
    local = p_del_in * (q_new - q_mid_in) + p_del_in * dlf * time_step
    prec_sum = seq_sum(torch.where(rng, -local, zero))    # local_sum -= local, from 0
    prec = torch.where(active, (1 / gravit) * max_(prec_sum, 0.0) / time_step / 1000, zero1)
    rliq = torch.where(active, seq_sum(dlf * p_del_in / gravit) / 1000, zero1)

    mcon = torch.cat([mcon, torch.zeros_like(mcon[:, :1])], -1)
    pflx = torch.where(act2, pflx, torch.zeros_like(pflx))
    keep = lambda x: torch.where(act2, x, zero)   # noqa: E731
    return dict(msemax_klev=msemax_klev, jctop=jctop, jcbot=jcbot, jt=jt, active=active,
                prec=prec, heat=heat, qtnd=qtnd, cape=cape, dcape=dcape, mcon=mcon, pflx=pflx,
                zdu=zdu, mflx_up=keep(mflx_up), entr_up=keep(entr_up), detr_up=keep(detr_up),
                mflx_dn=keep(mflx_dn), entr_dn=keep(entr_dn), p_del=p_del, dsubcld=dsubcld,
                ql=keep(ql), rliq=rliq, rprd=keep(rprd), dlf=dlf, ktm=ktm, kbm=kbm,
                nonconverged=nonconv)


def _interfaces(s_mid, q_mid, msg, pver, active):
    """Kernel 4: log-linear interpolation of s and q to interfaces where the jump is large."""
    ks, km1 = slice(msg + 1, pver), slice(msg, pver - 1)

    def one(x):
        a, b = x[:, km1], x[:, ks]           # k-1, k
        pos = (b > 0) | (a > 0)
        den = torch.where(pos, max_(a, b), torch.ones_like(a))
        difr = torch.where(pos, torch.abs((b - a) / den), torch.zeros_like(a))
        big = difr > ZMC.interp_diff_min
        ratio = torch.where(big, a / b, torch.ones_like(a))
        dd = torch.where(big, a - b, torch.ones_like(a))
        from .. import mathfn
        xi = torch.where(big, mathfn.log(ratio) * a * b / dd, ZMC.half * (b + a))
        new = torch.cat([x[:, :msg + 1], xi], -1)
        return torch.where(active.unsqueeze(-1), new, x)

    return one(s_mid), one(q_mid)
