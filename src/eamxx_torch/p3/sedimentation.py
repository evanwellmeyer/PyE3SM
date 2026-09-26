"""P3 sedimentation: upwind step, the Courant substep loop, and homogeneous freezing.

Translation of, at E3SM 80fe0a71:
  physics/p3/impl/p3_find_impl.hpp    (find_top, find_bottom)
  physics/p3/impl/p3_upwind_impl.hpp  (calc_first_order_upwind_step,
                                       generalized_sedimentation)
  physics/p3/impl/p3_cloud_sed_impl.hpp, p3_rain_sed_impl.hpp,
  physics/p3/impl/p3_ice_sed_impl.hpp (calc_bulk_rho_rime, ice_sedimentation,
                                       homogeneous_freezing)

Everything assumes EAMxx's own direction: kdir = -1, ktop = 0 (model top),
kbot = nlev-1.

Two things differ in form from the reference (see docs/plan.md, section 10):

1. Each column has its own hydrometeor top and bottom level, Courant number and
   therefore number of substeps (`while (dt_left > dt_left_tol)`). The port runs
   the substep loop a fixed number of times with finished columns masked off,
   and returns `hit_cap`, true for any column that still had time left.
2. find_top/find_bottom follow the `team.team_size() == 1` branch, the one a
   CPU (Kokkos Serial) build takes. It treats a NaN mixing ratio as "present"
   (`if (v(k) < small) continue`), where the multi-thread branch would not.
"""
from __future__ import annotations

import torch

from .. import constants as C
from .. import mathfn
from ..ekat import max_, min_, rdiv
from . import functions as F
from .runtime import P3Runtime
from .tables import apply_table, apply_table_ice, get_cloud_dsd2, get_rain_dsd2, lookup, lookup_ice

DT_LEFT_TOL = C.dt_left_tol  # 1e-4
# Substep cap. The loop takes at most one substep per level it walks down plus
# about one per unit of Courant number, so this is far above what the tuned
# configuration needs; `hit_cap` reports if it is ever reached.
MAX_SUBSTEPS = 400
# Stopping once every column has finished is an optimisation, not a change: a
# column with no time left is masked off and its state no longer moves. It costs
# a device sync per substep, so batched/compiled callers can turn it off.
EARLY_EXIT = True


def find_top_bottom(v: torch.Tensor, small: float):
    """find_top and find_bottom for kdir = -1. Returns (k_top, k_bot, log_present)."""
    nlev = v.shape[-1]
    idx = torch.arange(nlev, device=v.device)
    present = ~(v < small)
    k_top = torch.where(present, idx, torch.full_like(idx, nlev)).amin(dim=-1)
    log_present = k_top < nlev
    k_top = torch.where(log_present, k_top, torch.zeros_like(k_top))
    k_bot = torch.where(present, idx, torch.full_like(idx, -1)).amax(dim=-1)
    k_bot = torch.where(log_present, k_bot, torch.zeros_like(k_bot))
    return k_top, k_bot, log_present


def _range_mask(nlev: int, lo: torch.Tensor, hi: torch.Tensor, device) -> torch.Tensor:
    """Levels with lo <= k <= hi, per column."""
    idx = torch.arange(nlev, device=device)
    return (idx >= lo[:, None]) & (idx <= hi[:, None])


def calc_first_order_upwind_step(rho, inv_rho, inv_dz, k_bot, k_top, dt_sub, Vs, rs, active):
    """p3_upwind_impl.hpp: calc_first_order_upwind_step<kdir=-1>.

    k_top is the top of the step's range, k_bot its bottom (both (ncol,)).
    Returns (fluxes, rs).
    """
    nlev = rho.shape[-1]
    in_range = _range_mask(nlev, k_top, k_bot, rho.device) & active[:, None]
    idx = torch.arange(nlev, device=rho.device)
    at_top = idx == k_top[:, None]
    fluxes, out = [], []
    for V, r in zip(Vs, rs):
        flux = torch.where(in_range, V * r * rho, torch.zeros_like(r))
        # kdir == -1: shift_right(0, flux) is the level above, 0 at the top of the range
        flux_above = torch.cat([torch.zeros_like(flux[:, :1]), flux[:, :-1]], dim=-1)
        flux_above = torch.where(at_top, torch.zeros_like(flux_above), flux_above)
        fluxdiv = (flux_above - flux) * inv_dz
        out.append(torch.where(in_range, r + fluxdiv * dt_sub[:, None] * inv_rho, r))
        fluxes.append(flux)
    return fluxes, out


def generalized_sedimentation(rho, inv_rho, inv_dz, k_qxtop, k_qxbot, kbot: int, Co_max,
                              dt_left, prt_accum, Vs, rs, active):
    """p3_upwind_impl.hpp: generalized_sedimentation.

    Returns (fluxes, rs, k_qxbot, dt_left, prt_accum).
    """
    tmpint1 = torch.trunc(Co_max + 1).to(torch.int64)
    dt_sub = dt_left / tmpint1.to(dt_left.dtype).clamp(min=1)
    at_bot = k_qxbot == kbot
    k_temp = torch.where(at_bot, k_qxbot, k_qxbot + 1)
    fluxes, rs = calc_first_order_upwind_step(rho, inv_rho, inv_dz, k_temp, k_qxtop, dt_sub,
                                              Vs, rs, active)
    prt_accum = torch.where(at_bot & active, prt_accum + fluxes[0][:, kbot] * dt_sub, prt_accum)
    k_qxbot = torch.where(~at_bot & active, k_qxbot + 1, k_qxbot)
    dt_left = torch.where(active, dt_left - dt_sub, dt_left)
    return fluxes, rs, k_qxbot, dt_left, prt_accum


def calc_bulk_rho_rime(qi_tot, qi_rim, bi_rim, ro: P3Runtime, context=None):
    """p3_ice_sed_impl.hpp: calc_bulk_rho_rime. Returns (rho_rime, qi_rim, bi_rim)."""
    ctx = F._ctx(qi_tot, context)
    rho_rime = torch.zeros_like(qi_tot)
    bi_gt = (bi_rim >= C.BSMALL) & ctx
    bi_lt = (bi_rim < C.BSMALL) & ctx
    safe_bi = torch.where(bi_gt, bi_rim, torch.ones_like(bi_rim))
    rho_rime = F._set(rho_rime, bi_gt, qi_rim / safe_bi)
    rho_lt_min = rho_rime < ro.min_rime_rho
    rho_gt_max = rho_rime > ro.max_rime_rho
    rho_rime = F._set(rho_rime, bi_gt & rho_lt_min, ro.min_rime_rho)
    rho_rime = F._set(rho_rime, bi_gt & rho_gt_max, ro.max_rime_rho)
    adjust = bi_gt & (rho_gt_max | rho_lt_min)
    safe_rho = torch.where(rho_rime == 0, torch.ones_like(rho_rime), rho_rime)
    bi_rim = F._set(bi_rim, adjust, qi_rim / safe_rho)
    qi_rim = F._set(qi_rim, bi_lt, 0)
    bi_rim = F._set(bi_rim, bi_lt, 0)
    rho_rime = F._set(rho_rime, bi_lt, 0)
    qi_rim_gt = (qi_rim > qi_tot) & (rho_rime > 0) & ctx
    qi_rim = F._set(qi_rim, qi_rim_gt, qi_tot)
    safe_rho = torch.where(rho_rime == 0, torch.ones_like(rho_rime), rho_rime)
    bi_rim = F._set(bi_rim, qi_rim_gt, qi_rim / safe_rho)
    qi_rim_lt = (qi_rim < C.QSMALL) & ctx
    qi_rim = F._set(qi_rim, qi_rim_lt, 0)
    bi_rim = F._set(bi_rim, qi_rim_lt, 0)
    return rho_rime, qi_rim, bi_rim


def _co_max(V, dt_left, inv_dz, mask):
    """Co_max: max over the masked levels of V*dt_left*inv_dz, floored at 0."""
    co = torch.where(mask, V * dt_left[:, None] * inv_dz, torch.zeros_like(V))
    return max_(0.0, co.amax(dim=-1))


def cloud_sedimentation(qc_incld, rho, inv_rho, cld_frac_l, acn, inv_dz, dnu, dt, inv_dt,
                        do_predict_nc: bool, qc, nc, nc_incld, mu_c, lamc, qc_tend_in, nc_tend_in,
                        precip_liq_surf, max_substeps: int = MAX_SUBSTEPS):
    """p3_cloud_sed_impl.hpp: cloud_sedimentation."""
    nlev = qc.shape[-1]
    kbot = nlev - 1
    k_qxtop, k_qxbot, present = find_top_bottom(qc, C.QSMALL)
    dt_left = torch.where(present, torch.full_like(rho[:, 0], dt), torch.zeros_like(rho[:, 0]))
    prt_accum = torch.zeros_like(rho[:, 0])
    zero = torch.zeros_like(qc)
    for _ in range(max_substeps):
        active = present & (dt_left > DT_LEFT_TOL)
        if EARLY_EXIT and not bool(active.any()):
            break
        rng = _range_mask(nlev, k_qxtop, k_qxbot, rho.device) & active[:, None]
        qc_gt = rng & (qc_incld > C.QSMALL)
        nc_incld, mu_c, _nu, lamc, _cdist, _cdist1 = get_cloud_dsd2(
            qc_incld, nc_incld, mu_c, rho, zero, dnu, lamc, zero, zero, qc_gt)
        nc = torch.where(qc_gt, nc_incld * cld_frac_l, nc)
        safe_lamc = torch.where(qc_gt, lamc, torch.ones_like(lamc))
        # The source is 1/pow(lamc, bcn) with bcn a compile-time 2; clang rewrites
        # pow(x, 2.0) to x*x (LLVM SimplifyLibCalls), which differs from libm's pow
        # by 1 ulp for ~0.13% of values. Follow the build, not the call.
        dum = rdiv(1.0, safe_lamc * safe_lamc)
        V_qc = torch.where(qc_gt, acn * mathfn.tgamma(4 + C.bcn + mu_c) * dum
                           / mathfn.tgamma(mu_c + 4), zero)
        Vs, rs = [V_qc], [qc]
        if do_predict_nc:
            V_nc = torch.where(qc_gt, acn * mathfn.tgamma(1 + C.bcn + mu_c) * dum
                               / mathfn.tgamma(mu_c + 1), zero)
            Vs, rs = [V_qc, V_nc], [qc, nc]
        Co_max = _co_max(V_qc, dt_left, inv_dz, qc_gt)
        _fl, rs, k_qxbot, dt_left, prt_accum = generalized_sedimentation(
            rho, inv_rho, inv_dz, k_qxtop, k_qxbot, kbot, Co_max, dt_left, prt_accum, Vs, rs, active)
        if do_predict_nc:
            qc, nc = rs
        else:
            qc = rs[0]
        qc_incld = torch.where(active[:, None], qc / cld_frac_l, qc_incld)
        nc_incld = torch.where(active[:, None], nc / cld_frac_l, nc_incld)
    hit_cap = present & (dt_left > DT_LEFT_TOL)
    precip_liq_surf = torch.where(present, prt_accum * C.INV_RHO_H2O * inv_dt, precip_liq_surf)
    qc_tend = (qc - qc_tend_in) * inv_dt
    nc_tend = (nc - nc_tend_in) * inv_dt
    return (qc, nc, nc_incld, mu_c, lamc, qc_incld, qc_tend, nc_tend, precip_liq_surf, hit_cap)


def compute_rain_fall_velocity(vn_table_vals, vm_table_vals, qr_incld, rhofacr, nr_incld, mu_r,
                               lamr, V_qr, V_nr, ro: P3Runtime, context=None):
    """p3_rain_sed_impl.hpp: compute_rain_fall_velocity."""
    nr_incld, mu_r, lamr = get_rain_dsd2(qr_incld, nr_incld, mu_r, lamr, ro.constant_mu_rain, context)
    table = lookup(mu_r, lamr, context)
    ctx = F._ctx(qr_incld, context)
    V_qr = F._set(V_qr, ctx, apply_table(vm_table_vals, table) * rhofacr)
    V_nr = F._set(V_nr, ctx, apply_table(vn_table_vals, table) * rhofacr)
    return nr_incld, mu_r, lamr, V_qr, V_nr


def rain_sedimentation(rho, inv_rho, rhofacr, cld_frac_r, inv_dz, qr_incld, vn_table_vals,
                       vm_table_vals, dt, inv_dt, qr, nr, nr_incld, mu_r, lamr, precip_liq_flux,
                       qr_tend_in, nr_tend_in, precip_liq_surf, ro: P3Runtime,
                       max_substeps: int = MAX_SUBSTEPS):
    """p3_rain_sed_impl.hpp: rain_sedimentation."""
    nlev = qr.shape[-1]
    kbot = nlev - 1
    k_qxtop, k_qxbot, present = find_top_bottom(qr, C.QSMALL)
    dt_left = torch.where(present, torch.full_like(rho[:, 0], dt), torch.zeros_like(rho[:, 0]))
    prt_accum = torch.zeros_like(rho[:, 0])
    zero = torch.zeros_like(qr)
    idx_i = torch.arange(nlev + 1, device=rho.device)
    for _ in range(max_substeps):
        active = present & (dt_left > DT_LEFT_TOL)
        if EARLY_EXIT and not bool(active.any()):
            break
        rng = _range_mask(nlev, k_qxtop, k_qxbot, rho.device) & active[:, None]
        qr_gt = rng & (qr_incld > C.QSMALL)
        nr_incld, mu_r, lamr, V_qr, V_nr = compute_rain_fall_velocity(
            vn_table_vals, vm_table_vals, qr_incld, rhofacr, nr_incld, mu_r, lamr, zero, zero,
            ro, qr_gt)
        V_qr = torch.where(qr_gt, V_qr, zero)
        V_nr = torch.where(qr_gt, V_nr, zero)
        nr = torch.where(qr_gt, nr_incld * cld_frac_r, nr)
        Co_max = _co_max(V_qr, dt_left, inv_dz, qr_gt)
        fluxes, rs, k_qxbot, dt_left, prt_accum = generalized_sedimentation(
            rho, inv_rho, inv_dz, k_qxtop, k_qxbot, kbot, Co_max, dt_left, prt_accum,
            [V_qr, V_nr], [qr, nr], active)
        qr, nr = rs
        qr_incld = torch.where(active[:, None], qr / cld_frac_r, qr_incld)
        nr_incld = torch.where(active[:, None], nr / cld_frac_r, nr_incld)
        # precip_liq_flux(pk) += flux_qx(pk-1) over [k_qxtop+1, k_qxbot+1] (interface index)
        flux_qx = fluxes[0]
        flux_shift = torch.cat([flux_qx[:, :1], flux_qx], dim=-1)  # index pk -> flux(pk-1), clamped
        rng_i = ((idx_i >= (k_qxtop + 1)[:, None]) & (idx_i <= (k_qxbot + 1)[:, None])
                 & active[:, None])
        precip_liq_flux = torch.where(rng_i, precip_liq_flux + flux_shift, precip_liq_flux)
    hit_cap = present & (dt_left > DT_LEFT_TOL)
    precip_liq_surf = torch.where(present, precip_liq_surf + prt_accum * C.INV_RHO_H2O * inv_dt,
                                  precip_liq_surf)
    qr_tend = (qr - qr_tend_in) * inv_dt
    nr_tend = (nr - nr_tend_in) * inv_dt
    return (qr, nr, nr_incld, mu_r, lamr, qr_incld, precip_liq_flux, qr_tend, nr_tend,
            precip_liq_surf, hit_cap)


def ice_sedimentation(rho, inv_rho, rhofaci, cld_frac_i, inv_dz, dt, inv_dt, qi, qi_incld, ni,
                      ni_incld, qm, qm_incld, bm, bm_incld, qi_tend_in, ni_tend_in,
                      ice_table_vals, precip_ice_surf, ro: P3Runtime,
                      max_substeps: int = MAX_SUBSTEPS):
    """p3_ice_sed_impl.hpp: ice_sedimentation."""
    nlev = qi.shape[-1]
    kbot = nlev - 1
    k_qxtop, k_qxbot, present = find_top_bottom(qi, C.QSMALL)
    dt_left = torch.where(present, torch.full_like(rho[:, 0], dt), torch.zeros_like(rho[:, 0]))
    prt_accum = torch.zeros_like(rho[:, 0])
    zero = torch.zeros_like(qi)
    one = torch.ones_like(qi)
    for _ in range(max_substeps):
        active = present & (dt_left > DT_LEFT_TOL)
        if EARLY_EXIT and not bool(active.any()):
            break
        rng = _range_mask(nlev, k_qxtop, k_qxbot, rho.device) & active[:, None]
        qi_gt = rng & (qi_incld > C.QSMALL)
        ni_incld = torch.where(qi_gt, max_(ni_incld, C.NSMALL), ni_incld)
        rhop, qm_incld, bm_incld = calc_bulk_rho_rime(qi_incld, qm_incld, bm_incld, ro, qi_gt)
        qm = torch.where(qi_gt, qm_incld * cld_frac_i, qm)
        bm = torch.where(qi_gt, bm_incld * cld_frac_i, bm)
        tab = lookup_ice(torch.where(qi_gt, qi_incld, one), torch.where(qi_gt, ni_incld, one),
                         torch.where(qi_gt, qm_incld, zero), torch.where(qi_gt, rhop, one), qi_gt)
        table_val_ni_fallspd = apply_table_ice(0, ice_table_vals, tab)
        table_val_qi_fallspd = apply_table_ice(1, ice_table_vals, tab)
        table_val_ni_lammax = apply_table_ice(6, ice_table_vals, tab)
        table_val_ni_lammin = apply_table_ice(7, ice_table_vals, tab)
        ni_incld = torch.where(qi_gt, min_(ni_incld, table_val_ni_lammax * ni_incld), ni_incld)
        ni_incld = torch.where(qi_gt, max_(ni_incld, table_val_ni_lammin * ni_incld), ni_incld)
        ni = torch.where(qi_gt, ni_incld * cld_frac_i, ni)
        f = ro.ice_sedimentation_factor
        V_qit = torch.where(qi_gt, f * table_val_qi_fallspd * rhofaci, zero)
        V_nit = torch.where(qi_gt, f * table_val_ni_fallspd * rhofaci, zero)
        Co_max = _co_max(V_qit, dt_left, inv_dz, qi_gt)
        _fl, rs, k_qxbot, dt_left, prt_accum = generalized_sedimentation(
            rho, inv_rho, inv_dz, k_qxtop, k_qxbot, kbot, Co_max, dt_left, prt_accum,
            [V_qit, V_nit, V_qit, V_qit], [qi, ni, qm, bm], active)
        qi, ni, qm, bm = rs
        qi_incld = torch.where(active[:, None], qi / cld_frac_i, qi_incld)
        ni_incld = torch.where(active[:, None], ni / cld_frac_i, ni_incld)
        qm_incld = torch.where(active[:, None], qm / cld_frac_i, qm_incld)
        bm_incld = torch.where(active[:, None], bm / cld_frac_i, bm_incld)
    hit_cap = present & (dt_left > DT_LEFT_TOL)
    precip_ice_surf = torch.where(present, precip_ice_surf + prt_accum * C.INV_RHO_H2O * inv_dt,
                                  precip_ice_surf)
    qi_tend = (qi - qi_tend_in) * inv_dt
    ni_tend = (ni - ni_tend_in) * inv_dt
    return (qi, qi_incld, ni, ni_incld, qm, qm_incld, bm, bm_incld, qi_tend, ni_tend,
            precip_ice_surf, hit_cap)


def homogeneous_freezing(T_atm, inv_exner, qc, nc, qr, nr, qi, ni, qm, bm, th_atm):
    """p3_ice_sed_impl.hpp: homogeneous_freezing (the loop covers every level)."""
    t_lt = T_atm < C.T_homogfrz
    qc_ge = t_lt & (qc >= C.QSMALL)
    qr_ge = t_lt & (qr >= C.QSMALL)
    Qc_nuc, Qr_nuc = qc, qr
    Nc_nuc = max_(nc, C.NSMALL)
    Nr_nuc = max_(nr, C.NSMALL)
    qm = F._set(qm, qc_ge, qm + Qc_nuc)
    qi = F._set(qi, qc_ge, qi + Qc_nuc)
    bm = F._set(bm, qc_ge, bm + Qc_nuc * C.INV_RHO_RIMEMAX)
    ni = F._set(ni, qc_ge, ni + Nc_nuc)
    th_atm = F._set(th_atm, qc_ge, th_atm + inv_exner * Qc_nuc * C.LatIce * C.INV_CP)
    qm = F._set(qm, qr_ge, qm + Qr_nuc)
    qi = F._set(qi, qr_ge, qi + Qr_nuc)
    bm = F._set(bm, qr_ge, bm + Qr_nuc * C.INV_RHO_RIMEMAX)
    ni = F._set(ni, qr_ge, ni + Nr_nuc)
    th_atm = F._set(th_atm, qr_ge, th_atm + inv_exner * Qr_nuc * C.LatIce * C.INV_CP)
    qc = F._set(qc, qc_ge, 0)
    nc = F._set(nc, qc_ge, 0)
    qr = F._set(qr, qr_ge, 0)
    nr = F._set(nr, qr_ge, 0)
    return qc, nc, qr, nr, qi, ni, qm, bm, th_atm
