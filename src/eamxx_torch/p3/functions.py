"""P3 process rates, conservation limiters and prognostic updates.

Line-by-line translation of components/eamxx/src/physics/p3/impl/*_impl.hpp at
E3SM 80fe0a71, one function per source file, in source order:

  autoconversion, back_to_cell_average, calc_liq_relaxation_timescale,
  calc_rime_density, cldliq_imm_freezing, cloud_rain_acc, conservation
  (cloud/rain/ice water), droplet_self_coll, evaporate_rain,
  get_time_space_phys_variables, ice_classical_nucleation,
  ice_cldliq_wet_growth, ice_collection, ice_deposition_sublimation,
  ice_melting, ice_nucleation, ice_relaxation_timescale,
  ice_supersat_conservation, impose_max_total_ni, incloud_mixingratios,
  nc/ni/nr_conservation, prevent_liq_supersaturation, rain_imm_freezing,
  rain_self_collection, subgrid_variance_scaling, update_prognostics.

Every function is written so that values outside `context` (and outside each
inner mask) are the caller's inputs, exactly as the Kokkos `Pack::set(mask, v)`
calls leave them. `context=None` means "all lanes", matching `Mask(true)`.
`check_values` only prints and aborts in the reference, so it is not ported;
the port's budget checks cover its purpose.
"""
from __future__ import annotations

import dataclasses

import torch

from .. import constants as C
from .. import mathfn
from .. import saturation
from ..ekat import max_, min_, rdiv
from ..precision import double_device, to_precision
from .runtime import P3Runtime
from .tables import apply_table, lookup


def _ctx(x: torch.Tensor, context: torch.Tensor | None) -> torch.Tensor:
    return torch.ones_like(x, dtype=torch.bool) if context is None else context


def _set(dst: torch.Tensor, mask: torch.Tensor, val) -> torch.Tensor:
    """Pack::set(mask, val)."""
    if not isinstance(val, torch.Tensor):
        val = torch.full_like(dst, float(val))
    return torch.where(mask, val, dst)


def square(x):
    return x * x


def cube(x):
    return x * x * x


# ---------------------------------------------------------------- autoconversion
def cloud_water_autoconversion(rho, qc_incld, nc_incld, inv_qc_relvar,
                               qc2qr_autoconv_tend, nc2nr_autoconv_tend, ncautr,
                               ro: P3Runtime, context=None):
    """p3_autoconversion_impl.hpp: cloud_water_autoconversion."""
    ctx = _ctx(rho, context)
    qc_not_small = (qc_incld >= 1e-8) & ctx
    CONS3 = 1.0 / (C.CONS2 * ro.autoconversion_radius ** 3.0)
    sgs_var_coef = 1
    # masked-out entries get 1 (translation rule for masked branches)
    one = torch.ones_like(qc_incld)
    qc_s = torch.where(qc_not_small, qc_incld, one)
    nc_s = torch.where(qc_not_small, nc_incld, one)
    tend = (sgs_var_coef * ro.autoconversion_prefactor
            * mathfn.pow(qc_s, ro.autoconversion_qc_exponent)
            * mathfn.pow(nc_s * 1.e-6 * rho, -ro.autoconversion_nc_exponent))
    qc2qr_autoconv_tend = _set(qc2qr_autoconv_tend, qc_not_small, tend)
    ncautr = _set(ncautr, qc_not_small, qc2qr_autoconv_tend * CONS3)
    nc2nr_autoconv_tend = _set(nc2nr_autoconv_tend, qc_not_small,
                               qc2qr_autoconv_tend * nc_incld / qc_s)
    nc2nr_autoconv_tend = _set(nc2nr_autoconv_tend, (qc2qr_autoconv_tend == 0) & ctx, 0)
    qc2qr_autoconv_tend = _set(qc2qr_autoconv_tend, (nc2nr_autoconv_tend == 0) & ctx, 0)
    return qc2qr_autoconv_tend, nc2nr_autoconv_tend, ncautr


# ---------------------------------------------------------------- back to cell average
_BTCA_CLDL = ("nc_selfcollect_tend", "nc2nr_autoconv_tend", "qc2qr_autoconv_tend",
              "nc2ni_immers_freeze_tend", "ncheti_cnt", "qcheti_cnt", "nicnt", "qicnt",
              "ninuc_cnt", "qinuc_cnt")
_BTCA_CLDR = ("qr2qv_evap_tend", "nr_selfcollect_tend", "nr_evap_tend",
              "qr2qi_immers_freeze_tend", "nr2ni_immers_freeze_tend")
_BTCA_CLDI = ("qi2qr_melt_tend", "ni2nr_melt_tend", "ni_selfcollect_tend")
_BTCA_LR = ("qc2qr_accret_tend", "nc_accret_tend", "ncautr")
_BTCA_IL = ("nr_ice_shed_tend", "qc2qi_hetero_freeze_tend", "qc2qr_ice_shed_tend",
            "qc2qi_collect_tend", "nc_collect_tend", "ncshdc", "qc2qi_berg_tend")
_BTCA_IR = ("qr2qi_collect_tend", "nr_collect_tend")
_BTCA_GLACIATED_OR_I = ("qi2qv_sublim_tend", "qv2qi_vapdep_tend", "ni_sublim_tend")


def back_to_cell_average(cld_frac_l, cld_frac_r, cld_frac_i, tends: dict,
                         ro: P3Runtime, context=None):
    """p3_back_to_cell_average_impl.hpp: back_to_cell_average.

    `tends` maps the process-rate names to tensors; returns a new dict.
    """
    ctx = _ctx(cld_frac_l, context)
    ir_cldm = min_(cld_frac_i, cld_frac_r)
    il_cldm = min_(cld_frac_i, cld_frac_l)
    lr_cldm = min_(cld_frac_l, cld_frac_r)
    cld_frac_glaciated = max_(0.0001, cld_frac_i - il_cldm)
    out = dict(tends)
    for name, fac in [(_BTCA_LR, lr_cldm), (_BTCA_CLDR, cld_frac_r), (_BTCA_CLDL, cld_frac_l),
                      (_BTCA_CLDI, cld_frac_i), (_BTCA_IL, il_cldm), (_BTCA_IR, ir_cldm)]:
        for n in name:
            out[n] = _set(out[n], ctx, out[n] * fac)
    ice_fac = cld_frac_glaciated if ro.use_separate_ice_liq_frac else cld_frac_i
    for n in _BTCA_GLACIATED_OR_I:
        out[n] = _set(out[n], ctx, out[n] * ice_fac)
    return out


# ------------------------------------------------- liquid relaxation timescale
def calc_liq_relaxation_timescale(revap_table_vals, rho, f1r, f2r, dv, mu, sc, mu_r, lamr,
                                  cdistr, cdist, qr_incld, qc_incld, epsr, epsc, context=None):
    """p3_calc_liq_relaxation_timescale_impl.hpp. Returns (epsr, epsc)."""
    ctx = _ctx(rho, context)
    qr_not_small = (qr_incld >= C.QSMALL) & ctx
    epsr = torch.zeros_like(rho)  # epsr = 0 (unmasked in the reference)
    table = lookup(mu_r, lamr, qr_not_small)
    lamr_s = torch.where(qr_not_small, lamr, torch.ones_like(lamr))   # masked-out lanes: safe input
    epsr_v = (2 * C.Pi * cdistr * rho * dv
              * (f1r * mathfn.tgamma(mu_r + 2) / lamr_s
                 + f2r * torch.sqrt(rho / mu) * mathfn.cbrt(sc)
                 * apply_table(revap_table_vals, table)))
    epsr = _set(epsr, qr_not_small, epsr_v)
    qc_not_small = (qc_incld >= C.QSMALL) & ctx
    epsc = _set(epsc, qc_not_small, 2 * C.Pi * rho * dv * cdist)
    epsc = _set(epsc, ~qc_not_small & ctx, 0)
    return epsr, epsc


# ---------------------------------------------------------------- rime density
def calc_rime_density(T_atm, rhofaci, table_val_qi_fallspd, acn, lamc, mu_c, qc_incld,
                      qc2qi_collect_tend, vtrmi1, rho_qm_cloud, context=None):
    """p3_calc_rime_density_impl.hpp: calc_rime_density. Returns (vtrmi1, rho_qm_cloud)."""
    ctx = _ctx(T_atm, context)
    m = (qc2qi_collect_tend >= C.QSMALL) & (T_atm < C.T_zerodegc) & ctx
    vtrmi1 = _set(vtrmi1, m, table_val_qi_fallspd * rhofaci)
    m_qc = m & (qc_incld >= C.QSMALL)
    lamc = torch.where(m_qc, lamc, torch.ones_like(lamc))   # masked-out lanes: safe input
    Vt_qc = acn * mathfn.tgamma(4 + C.bcn + mu_c) / (mathfn.pow(lamc, C.bcn) * mathfn.tgamma(4 + mu_c))
    D_c = (4 + mu_c) / lamc
    V_impact = torch.abs(vtrmi1 - Vt_qc)
    inv_Tc = rdiv(1.0, min_(-0.001, T_atm - C.T_zerodegc))
    Ri = max_(1, min_(-0.5e+6 * D_c * V_impact * inv_Tc, 12))
    Ri_le_8 = Ri <= 8.0
    rho_qm_cloud = _set(rho_qm_cloud, m_qc & Ri_le_8, (0.051 + 0.114 * Ri - 0.0055 * square(Ri)) * 1000)
    rho_qm_cloud = _set(rho_qm_cloud, m_qc & ~Ri_le_8, 611. + 72.25 * (Ri - 8))
    rho_qm_cloud = _set(rho_qm_cloud, m & (qc_incld < C.QSMALL), 400)
    vtrmi1 = _set(vtrmi1, ~m & ctx, 0)
    rho_qm_cloud = _set(rho_qm_cloud, ~m & ctx, 400)
    return vtrmi1, rho_qm_cloud


# --------------------------------------------- cloud liquid immersion freezing
def cldliq_immersion_freezing(T_atm, lamc, mu_c, cdist1, qc_incld, inv_qc_relvar,
                              qc2qi_hetero_freeze_tend, nc2ni_immers_freeze_tend,
                              ro: P3Runtime, context=None):
    """p3_cldliq_imm_freezing_impl.hpp: cldliq_immersion_freezing."""
    if ro.immersion_freezing_in_float64 and T_atm.dtype != torch.float64:
        # option, not EAMxx: the same code in float64 (on the CPU if the device has none)
        dt, dev = T_atm.dtype, T_atm.device
        ddev = double_device(dev)
        x = to_precision((T_atm, lamc, mu_c, cdist1, qc_incld, inv_qc_relvar,
                          qc2qi_hetero_freeze_tend, nc2ni_immers_freeze_tend), torch.float64, ddev)
        ctx = None if context is None else context.to(ddev)
        out = cldliq_immersion_freezing(*x, dataclasses.replace(ro, immersion_freezing_in_float64=False), ctx)
        return to_precision(out, dt, dev)
    ctx = _ctx(T_atm, context)
    m = (qc_incld >= C.QSMALL) & (T_atm <= C.T_rainfrz) & ctx
    expAimmDt = mathfn.exp(ro.immersion_freezing_exponent * (C.T_zerodegc - T_atm))
    inv_lamc3 = cube(rdiv(1.0, torch.where(m, lamc, torch.ones_like(lamc))))   # masked-out lanes: safe input
    sgs_var_coef = 1
    qc2qi_hetero_freeze_tend = _set(
        qc2qi_hetero_freeze_tend, m,
        sgs_var_coef * C.CONS6 * cdist1 * mathfn.tgamma(7 + mu_c) * expAimmDt * square(inv_lamc3))
    nc2ni_immers_freeze_tend = _set(
        nc2ni_immers_freeze_tend, m,
        C.CONS5 * cdist1 * mathfn.tgamma(4.0 + mu_c) * expAimmDt * inv_lamc3)
    return qc2qi_hetero_freeze_tend, nc2ni_immers_freeze_tend


# ---------------------------------------------------------------- accretion
def cloud_rain_accretion(rho, inv_rho, qc_incld, nc_incld, qr_incld, inv_qc_relvar,
                         qc2qr_accret_tend, nc_accret_tend, ro: P3Runtime, context=None):
    """p3_cloud_rain_acc_impl.hpp: cloud_rain_accretion."""
    ctx = _ctx(rho, context)
    sgs_var_coef = 1
    m = (qr_incld >= C.QSMALL) & (qc_incld >= C.QSMALL) & ctx
    accret = (sgs_var_coef * ro.accretion_prefactor
              * mathfn.pow(qc_incld, ro.accretion_qc_exponent)
              * mathfn.pow(qr_incld, ro.accretion_qr_exponent))
    qc2qr_accret_tend = _set(qc2qr_accret_tend, m, accret)
    nc_accret_tend = _set(nc_accret_tend, m, qc2qr_accret_tend * nc_incld
                          / torch.where(m, qc_incld, torch.ones_like(qc_incld)))
    qc2qr_accret_tend = _set(qc2qr_accret_tend, (nc_accret_tend == 0) & ctx, 0)
    nc_accret_tend = _set(nc_accret_tend, (qc2qr_accret_tend == 0) & ctx, 0)
    return qc2qr_accret_tend, nc_accret_tend


# ---------------------------------------------------------------- conservation
def cloud_water_conservation(qc, dt, t: dict, use_hetfrz_classnuc: bool, ro: P3Runtime,
                             cld_frac_l=None, cld_frac_i=None, context=None):
    """p3_conservation_impl.hpp: cloud_water_conservation. Updates `t` in place-free style."""
    ctx = _ctx(qc, context)
    o = dict(t)
    if use_hetfrz_classnuc:
        sinks = (o["qc2qr_autoconv_tend"] + o["qc2qr_accret_tend"] + o["qc2qi_collect_tend"]
                 + o["qcheti_cnt"] + o["qc2qr_ice_shed_tend"] + o["qc2qi_berg_tend"]) * dt
    else:
        sinks = (o["qc2qr_autoconv_tend"] + o["qc2qr_accret_tend"] + o["qc2qi_collect_tend"]
                 + o["qc2qi_hetero_freeze_tend"] + o["qc2qr_ice_shed_tend"] + o["qc2qi_berg_tend"]) * dt
    sources = qc
    if ro.use_separate_ice_liq_frac:
        il_cldm = min_(cld_frac_i, cld_frac_l)
        cld_frac_glaciated = max_(cld_frac_i - il_cldm, 0.0001)
    else:
        il_cldm = torch.ones_like(qc)
        cld_frac_glaciated = torch.ones_like(qc)
    enforce = (sinks > sources) & (sinks >= C.QTENDSMALL) & ctx
    nothing_todo = ~enforce & ctx
    ratio = torch.zeros_like(qc)  # Pack ratio; default-constructed to 0
    safe_sinks = torch.where(enforce, sinks, torch.ones_like(sinks))
    ratio = _set(ratio, enforce, sources / safe_sinks)
    for n in ("qc2qr_autoconv_tend", "qc2qr_accret_tend", "qc2qi_collect_tend"):
        o[n] = _set(o[n], enforce, o[n] * ratio)
    if use_hetfrz_classnuc:
        o["qcheti_cnt"] = _set(o["qcheti_cnt"], enforce, o["qcheti_cnt"] * ratio)
        o["qicnt"] = _set(o["qicnt"], enforce, o["qicnt"] * ratio)
    else:
        o["qc2qi_hetero_freeze_tend"] = _set(o["qc2qi_hetero_freeze_tend"], enforce,
                                             o["qc2qi_hetero_freeze_tend"] * ratio)
    o["qc2qr_ice_shed_tend"] = _set(o["qc2qr_ice_shed_tend"], enforce, o["qc2qr_ice_shed_tend"] * ratio)
    o["qc2qi_berg_tend"] = _set(o["qc2qi_berg_tend"], enforce, o["qc2qi_berg_tend"] * ratio)
    ratio = _set(ratio, nothing_todo, 1)
    enforce2 = (sources > C.QTENDSMALL) & ctx
    if ro.use_separate_ice_liq_frac:
        fac = il_cldm / cld_frac_glaciated
        o["qv2qi_vapdep_tend"] = _set(o["qv2qi_vapdep_tend"], enforce2,
                                      o["qv2qi_vapdep_tend"] + o["qv2qi_vapdep_tend"] * (1 - ratio) * fac)
        o["qi2qv_sublim_tend"] = _set(o["qi2qv_sublim_tend"], enforce2,
                                      o["qi2qv_sublim_tend"] + o["qi2qv_sublim_tend"] * (1 - ratio) * fac)
    else:
        o["qv2qi_vapdep_tend"] = _set(o["qv2qi_vapdep_tend"], enforce2, o["qv2qi_vapdep_tend"] * (1 - ratio))
        o["qi2qv_sublim_tend"] = _set(o["qi2qv_sublim_tend"], enforce2, o["qi2qv_sublim_tend"] * (1 - ratio))
    return o


def rain_water_conservation(qr, qc2qr_autoconv_tend, qc2qr_accret_tend, qi2qr_melt_tend,
                            qc2qr_ice_shed_tend, dt, qr2qv_evap_tend, qr2qi_collect_tend,
                            qr2qi_immers_freeze_tend, context=None):
    """p3_conservation_impl.hpp: rain_water_conservation."""
    ctx = _ctx(qr, context)
    sinks = (qr2qv_evap_tend + qr2qi_collect_tend + qr2qi_immers_freeze_tend) * dt
    sources = qr + (qc2qr_autoconv_tend + qc2qr_accret_tend + qi2qr_melt_tend + qc2qr_ice_shed_tend) * dt
    enforce = (sinks > sources) & (sinks >= C.QTENDSMALL) & ctx
    safe_sinks = torch.where(enforce, sinks, torch.ones_like(sinks))
    ratio = sources / safe_sinks
    qr2qv_evap_tend = _set(qr2qv_evap_tend, enforce, qr2qv_evap_tend * ratio)
    qr2qi_collect_tend = _set(qr2qi_collect_tend, enforce, qr2qi_collect_tend * ratio)
    qr2qi_immers_freeze_tend = _set(qr2qi_immers_freeze_tend, enforce, qr2qi_immers_freeze_tend * ratio)
    return qr2qv_evap_tend, qr2qi_collect_tend, qr2qi_immers_freeze_tend


def ice_water_conservation(qi, qv2qi_vapdep_tend, qv2qi_nucleat_tend, qc2qi_berg_tend,
                           qr2qi_collect_tend, qc2qi_collect_tend, qr2qi_immers_freeze_tend,
                           qc2qi_hetero_freeze_tend, dt, qinuc_cnt, qcheti_cnt, qicnt,
                           qi2qv_sublim_tend, qi2qr_melt_tend, use_hetfrz_classnuc: bool,
                           context=None):
    """p3_conservation_impl.hpp: ice_water_conservation."""
    ctx = _ctx(qi, context)
    sinks = (qi2qv_sublim_tend + qi2qr_melt_tend) * dt
    if use_hetfrz_classnuc:
        sources = qi + (qv2qi_vapdep_tend + qv2qi_nucleat_tend + qr2qi_collect_tend + qc2qi_collect_tend
                        + qr2qi_immers_freeze_tend + qc2qi_berg_tend + qinuc_cnt + qcheti_cnt + qicnt) * dt
    else:
        sources = qi + (qv2qi_vapdep_tend + qv2qi_nucleat_tend + qr2qi_collect_tend + qc2qi_collect_tend
                        + qr2qi_immers_freeze_tend + qc2qi_hetero_freeze_tend + qc2qi_berg_tend) * dt
    enforce = (sinks > sources) & (sinks >= C.QTENDSMALL) & ctx
    safe_sinks = torch.where(enforce, sinks, torch.ones_like(sinks))
    ratio = sources / safe_sinks
    qi2qv_sublim_tend = _set(qi2qv_sublim_tend, enforce, qi2qv_sublim_tend * ratio)
    qi2qr_melt_tend = _set(qi2qr_melt_tend, enforce, qi2qr_melt_tend * ratio)
    return qi2qv_sublim_tend, qi2qr_melt_tend


# ---------------------------------------------------------------- droplet self collection
def droplet_self_collection(qc_incld, nc_selfcollect_tend, context=None):
    """p3_droplet_self_coll_impl.hpp: droplet_self_collection (sets 0 where qc >= qsmall)."""
    ctx = _ctx(qc_incld, context)
    return _set(nc_selfcollect_tend, (qc_incld >= C.QSMALL) & ctx, 0)


# ---------------------------------------------------------------- rain evaporation
def rain_evap_tscale_weight(dt_over_tau, weight, context=None):
    """p3_evaporate_rain_impl.hpp: rain_evap_tscale_weight."""
    ctx = _ctx(dt_over_tau, context)
    return _set(weight, ctx, -mathfn.expm1(-dt_over_tau) / dt_over_tau)


def rain_evap_equilib_tend(A_c, ab, tau_eff, tau_r, tend, context=None):
    """p3_evaporate_rain_impl.hpp: rain_evap_equilib_tend."""
    ctx = _ctx(A_c, context)
    return _set(tend, ctx, -A_c / ab * tau_eff / tau_r)


def rain_evap_instant_tend(ssat_r, ab, tau_r, tend, context=None):
    """p3_evaporate_rain_impl.hpp: rain_evap_instant_tend."""
    ctx = _ctx(ssat_r, context)
    return _set(tend, ctx, -ssat_r / (ab * tau_r))


def evaporate_rain(qr_incld, qc_incld, nr_incld, qi_incld, cld_frac_l, cld_frac_r, qv, qv_prev,
                   qv_sat_l, qv_sat_i, ab, abi, epsr, epsi_tot, t_atm, t_atm_prev, dqsdt, dt,
                   context=None):
    """p3_evaporate_rain_impl.hpp: evaporate_rain. Returns (qr2qv_evap_tend, nr_evap_tend)."""
    ctx = _ctx(qr_incld, context)
    zero = torch.zeros_like(qr_incld)
    qr2qv_evap_tend = zero
    nr_evap_tend = zero
    inv_dt = 1 / dt
    inv_cp = 1 / C.Cpair
    ssat_r = qv - qv_sat_l
    cld_frac = torch.zeros_like(qr_incld)
    set_cld_frac_zero = ((qc_incld + qi_incld) < 1.e-6) & ctx
    cld_frac = _set(cld_frac, set_cld_frac_zero, 0)
    cld_frac = _set(cld_frac, ~set_cld_frac_zero & ctx, cld_frac_l)
    qr_ge_qsmall = qr_incld >= C.QSMALL
    is_subsat = ssat_r < 0
    is_evap_area = cld_frac_r > cld_frac
    is_rain_evap = qr_ge_qsmall & is_subsat & is_evap_area & ctx

    safe_epsr = torch.where(is_rain_evap, epsr, torch.ones_like(epsr))
    tau_r = _set(zero, is_rain_evap, rdiv(1.0, safe_epsr))
    is_freezing = (t_atm < C.Tmelt) & ctx
    not_freezing = ~is_freezing & ctx
    eps_eff = torch.zeros_like(qr_incld)
    A_c = torch.zeros_like(qr_incld)
    latfac = (1 + (C.LatVap + C.LatIce) * inv_cp * dqsdt) / abi
    eps_eff = _set(eps_eff, is_freezing, epsr + epsi_tot * latfac)
    A_c = _set(A_c, is_freezing,
               (qv - qv_prev) * inv_dt - dqsdt * (t_atm - t_atm_prev) * inv_dt
               - (qv_sat_l - qv_sat_i) * latfac * epsi_tot)
    eps_eff = _set(eps_eff, not_freezing, epsr)
    A_c = _set(A_c, not_freezing, (qv - qv_prev) * inv_dt - dqsdt * (t_atm - t_atm_prev) * inv_dt)
    eps_eff = _set(eps_eff, (eps_eff < 1e-20) & ctx, 1e-20)
    tau_eff = rdiv(1.0, torch.where(ctx, eps_eff, torch.ones_like(eps_eff)))   # eps_eff = 0 outside ctx: safe input

    is_qr_tiny = (qr_incld < 1e-12) & (qv / qv_sat_l < 0.999)
    not_qr_tiny = ~is_qr_tiny & is_rain_evap
    is_qr_tiny = is_qr_tiny & is_rain_evap
    qr2qv_evap_tend = _set(qr2qv_evap_tend, is_qr_tiny, qr_incld * inv_dt)

    tscale_weight = rain_evap_tscale_weight(rdiv(dt, tau_eff), torch.zeros_like(qr_incld), is_rain_evap)
    safe_tau_r = torch.where(is_rain_evap, tau_r, torch.ones_like(tau_r))
    equilib_tend = rain_evap_equilib_tend(A_c, ab, tau_eff, safe_tau_r,
                                          torch.zeros_like(qr_incld), is_rain_evap)
    instant_tend = rain_evap_instant_tend(ssat_r, ab, safe_tau_r,
                                          torch.zeros_like(qr_incld), is_rain_evap)
    qr2qv_evap_tend = _set(qr2qv_evap_tend, not_qr_tiny,
                           instant_tend * tscale_weight + equilib_tend * (1 - tscale_weight))
    is_overevap = (qr2qv_evap_tend > -ssat_r * inv_dt / ab) & is_rain_evap
    qr2qv_evap_tend = _set(qr2qv_evap_tend, is_overevap, -ssat_r * inv_dt / ab)
    qr2qv_evap_tend = _set(qr2qv_evap_tend, is_rain_evap & (qr2qv_evap_tend < 0), 0)
    qr2qv_evap_tend = _set(qr2qv_evap_tend, is_rain_evap & (qr2qv_evap_tend > qr_incld * inv_dt),
                           qr_incld * inv_dt)
    qr2qv_evap_tend = _set(qr2qv_evap_tend, is_rain_evap,
                           qr2qv_evap_tend * (cld_frac_r - cld_frac) / cld_frac_r)
    safe_qr = torch.where(is_rain_evap, qr_incld, torch.ones_like(qr_incld))
    nr_evap_tend = _set(nr_evap_tend, is_rain_evap, qr2qv_evap_tend * (nr_incld / safe_qr))
    return qr2qv_evap_tend, nr_evap_tend


# ------------------------------------------------- time/space physical variables
def get_time_space_phys_variables(T_atm, pres, rho, qv_sat_l, qv_sat_i, context=None):
    """p3_get_time_space_phys_variables_impl.hpp. Returns mu, dv, sc, dqsdt, dqsidt, ab, abi, kap, eii."""
    ctx = _ctx(T_atm, context)
    zero = torch.zeros_like(T_atm)
    # The C++ sets these only on active levels (others stay 0 and are never used). The port
    # computes them on every level: the values on active levels are the same, and later
    # masked-out lanes divide by finite numbers instead of 0 (no 0/0, so no NaN gradients;
    # plan section 6). Checked bitwise on every P3 dump.
    mu = 1.496e-6 * mathfn.pow(T_atm, 1.5) / (T_atm + 120)
    dv = 8.794e-5 * mathfn.pow(T_atm, 1.81) / pres
    sc = mu / (rho * dv)
    tval1, tval2, dtval = 253.15, 273.15, 20
    dum = rdiv(1.0, C.RV * square(T_atm))
    dqsdt = C.LatVap * qv_sat_l * dum
    dqsidt = (C.LatVap + C.LatIce) * qv_sat_i * dum
    ab = 1 + dqsdt * C.LatVap * C.INV_CP
    abi = 1 + dqsidt * (C.LatVap + C.LatIce) * C.INV_CP
    kap = 1.414e+3 * mu
    t_lt_tval1 = T_atm < tval1
    t_lt_tval2 = T_atm < tval2
    eii = torch.zeros_like(T_atm)
    eii = _set(eii, t_lt_tval1 & ctx, 0.001)
    eii = _set(eii, ~t_lt_tval1 & t_lt_tval2 & ctx, 0.001 + (T_atm - tval1) * (0.3 - 0.001) / dtval)
    eii = _set(eii, ~t_lt_tval1 & ~t_lt_tval2 & ctx, 0.3)
    return mu, dv, sc, dqsdt, dqsidt, ab, abi, kap, eii


# ---------------------------------------------------- ice classical nucleation
def ice_classical_nucleation(frzimm, frzcnt, frzdep, rho, qc_incld, nc_incld, iflag: int,
                             ncheti_cnt, qcheti_cnt, nicnt, qicnt, ninuc_cnt, qinuc_cnt):
    """p3_ice_classical_nucleation_impl.hpp: ice_classical_nucleation (use_hetfrz_classnuc)."""
    qsmall = 1.0e-18
    piov3 = C.Pi / 3.0
    mi0 = 4.0 * piov3 * 900.0 * 1.0e-18
    mi0l_min = (4.0 / 3.0) * C.Pi * C.RHO_H2O * (4.0e-6) * (4.0e-6) * (4.0e-6)
    mi0l = qc_incld / max_(nc_incld, rdiv(1.0e6, rho))
    mi0l = max_(mi0l_min, mi0l)
    mask = qc_incld > qsmall
    zero = torch.zeros_like(rho)
    if iflag == 1:
        ncheti_cnt = torch.where(mask, frzimm * 1.0e6 / rho, zero)
        qcheti_cnt = torch.where(mask, ncheti_cnt * mi0l, zero)
    elif iflag == 2:
        nicnt = torch.where(mask, frzcnt * 1.0e6 / rho, zero)
        qicnt = torch.where(mask, nicnt * mi0l, zero)
        ninuc_cnt = torch.where(mask, frzdep * 1.0e6 / rho, zero)
        qinuc_cnt = torch.where(mask, ninuc_cnt * mi0, zero)
    else:
        raise ValueError("Unhandled Iflag in ice_classical_nucleation")
    return ncheti_cnt, qcheti_cnt, nicnt, qicnt, ninuc_cnt, qinuc_cnt


# ---------------------------------------------------------------- wet growth
def ice_cldliq_wet_growth(rho, temp, pres, rhofaci, table_val_qi2qr_melting,
                          table_val_qi2qr_vent_melt, dv, kap, mu, sc, qv, qc_incld, qi_incld,
                          ni_incld, qr_incld, log_wetgrowth, qr2qi_collect_tend,
                          qc2qi_collect_tend, qc_growth_rate, nr_ice_shed_tend,
                          qc2qr_ice_shed_tend, context=None):
    """p3_ice_cldliq_wet_growth_impl.hpp: ice_cldliq_wet_growth."""
    ctx = _ctx(rho, context)
    t_is_negative = temp < C.Tmelt
    qi_ge = qi_incld >= C.QSMALL
    qc_qr_ge = (qc_incld + qr_incld) >= 1.0e-6
    coll_ge = (qc2qi_collect_tend + qr2qi_collect_tend) >= 1.0e-10
    any_if = qi_ge & qc_qr_ge & t_is_negative & ctx
    any_if_col = any_if & coll_ge & ctx
    zerodeg = torch.full_like(rho, C.Tmelt)
    qsat0 = saturation.qv_sat_dry(zerodeg, pres, False, saturation.MURPHYKOOP)
    rate = (((table_val_qi2qr_melting + table_val_qi2qr_vent_melt * mathfn.cbrt(sc)
              * torch.sqrt(rhofaci * rho / mu))
             * 2 * C.Pi * (rho * C.LatVap * dv * (qsat0 - qv) - (temp - C.Tmelt) * kap)
             / (C.LatIce + C.CpLiq * (temp - C.Tmelt))) * ni_incld)
    qc_growth_rate = _set(qc_growth_rate, any_if, rate)
    qc_growth_rate = _set(qc_growth_rate, any_if, max_(qc_growth_rate, C.ZERO))
    dum = max_(C.ZERO, (qc2qi_collect_tend + qr2qi_collect_tend) - qc_growth_rate)
    dum_ge_small = (dum >= 1.0e-10) & ctx
    nr_ice_shed_tend = _set(nr_ice_shed_tend, any_if & dum_ge_small,
                            nr_ice_shed_tend + dum * 1.923e+6)
    coll_sum = qc2qi_collect_tend + qr2qi_collect_tend
    dum1 = rdiv(C.ONE, torch.where(coll_sum == 0, torch.ones_like(coll_sum), coll_sum))
    m = any_if_col & dum_ge_small
    qc2qr_ice_shed_tend = _set(qc2qr_ice_shed_tend, m, qc2qr_ice_shed_tend + dum * qc2qi_collect_tend * dum1)
    qc2qi_collect_tend_new = _set(qc2qi_collect_tend, m, max_(0, qc2qi_collect_tend - dum * qc2qi_collect_tend * dum1))
    qr2qi_collect_tend = _set(qr2qi_collect_tend, m, max_(0, qr2qi_collect_tend - dum * qr2qi_collect_tend * dum1))
    log_wetgrowth = torch.where(any_if, any_if & dum_ge_small, log_wetgrowth)
    return (log_wetgrowth, qr2qi_collect_tend, qc2qi_collect_tend_new, qc_growth_rate,
            nr_ice_shed_tend, qc2qr_ice_shed_tend)


# ---------------------------------------------------------------- ice collection
def ice_cldliq_collection(rho, temp, rhofaci, table_val_qc2qi_collect, qi_incld, qc_incld,
                          ni_incld, nc_incld, qc2qi_collect_tend, nc_collect_tend,
                          qc2qr_ice_shed_tend, ncshdc, ro: P3Runtime, context=None):
    """p3_ice_collection_impl.hpp: ice_cldliq_collection."""
    ctx = _ctx(rho, context)
    t_is_negative = temp <= C.Tmelt
    both_ge_small = (qi_incld >= C.QSMALL) & (qc_incld >= C.QSMALL) & ctx
    both_pos_t = both_ge_small & ~t_is_negative
    inv_dropmass = C.ONE / C.dropmass
    f = ro.cldliq_to_ice_collection_factor
    qc2qi_collect_tend = _set(qc2qi_collect_tend, both_ge_small & t_is_negative,
                              rhofaci * table_val_qc2qi_collect * qc_incld * f * rho * ni_incld)
    nc_collect_tend = _set(nc_collect_tend, both_ge_small,
                           rhofaci * table_val_qc2qi_collect * nc_incld * f * rho * ni_incld)
    qc2qr_ice_shed_tend = _set(qc2qr_ice_shed_tend, both_pos_t,
                               rhofaci * table_val_qc2qi_collect * qc_incld * f * rho * ni_incld)
    nc_collect_tend = _set(nc_collect_tend, both_pos_t,
                           rhofaci * table_val_qc2qi_collect * nc_incld * f * rho * ni_incld)
    ncshdc = _set(ncshdc, both_pos_t, qc2qr_ice_shed_tend * inv_dropmass)
    return qc2qi_collect_tend, nc_collect_tend, qc2qr_ice_shed_tend, ncshdc


def ice_rain_collection(rho, temp, rhofaci, logn0r, table_val_nr_collect, table_val_qr2qi_collect,
                        qi_incld, ni_incld, qr_incld, qr2qi_collect_tend, nr_collect_tend,
                        ro: P3Runtime, context=None):
    """p3_ice_collection_impl.hpp: ice_rain_collection."""
    ctx = _ctx(rho, context)
    t_is_negative = temp <= C.Tmelt
    both_ge_small = (qi_incld >= C.QSMALL) & (qr_incld >= C.QSMALL) & ctx
    both_neg_t = both_ge_small & t_is_negative
    f = ro.rain_to_ice_collection_factor
    # The source is pow(ten, ...); clang rewrites pow(10.0, x) to __exp10(x)
    qr2qi_collect_tend = _set(qr2qi_collect_tend, both_neg_t,
                              mathfn.exp10(table_val_qr2qi_collect + logn0r) * rho * rhofaci * f * ni_incld)
    nr_collect_tend = _set(nr_collect_tend, both_neg_t,
                           mathfn.exp10(table_val_nr_collect + logn0r) * rho * rhofaci * f * ni_incld)
    nr_collect_tend = _set(nr_collect_tend, both_ge_small & ~t_is_negative,
                           mathfn.exp10(table_val_nr_collect + logn0r) * rho * rhofaci * f * ni_incld)
    return qr2qi_collect_tend, nr_collect_tend


def ice_self_collection(rho, rhofaci, table_val_ni_self_collect, eii, qm_incld, qi_incld,
                        ni_incld, ni_selfcollect_tend, context=None):
    """p3_ice_collection_impl.hpp: ice_self_collection."""
    ctx = _ctx(rho, context)
    qm_pos = (qm_incld > C.ZERO) & ctx
    qi_ge = (qi_incld >= C.QSMALL) & ctx
    tmp1 = torch.zeros_like(rho)
    safe_qi = torch.where(qi_ge & qm_pos, qi_incld, torch.ones_like(qi_incld))
    tmp1 = _set(tmp1, qi_ge & qm_pos, qm_incld / safe_qi)
    tmp1_lt_six = tmp1 < 0.6
    tmp1_ge_six = tmp1 >= 0.6
    tmp1_lt_nine = tmp1 < 0.9
    tmp1_ge_nine = tmp1 >= 0.9
    Eii_fact = torch.zeros_like(rho)
    Eii_fact = _set(Eii_fact, tmp1_lt_six & qm_pos, 1)
    Eii_fact = _set(Eii_fact, tmp1_ge_six & tmp1_lt_nine & qm_pos, 1 - (tmp1 - 0.6) / 0.3)
    Eii_fact = _set(Eii_fact, tmp1_ge_nine & qm_pos, 0)
    Eii_fact = _set(Eii_fact, ~qm_pos & ctx, 1)
    ni_selfcollect_tend = _set(ni_selfcollect_tend, qi_ge,
                               table_val_ni_self_collect * rho * eii * Eii_fact * rhofaci
                               * ni_incld * ni_incld)
    return ni_selfcollect_tend


# --------------------------------------------- ice deposition and sublimation
def ice_deposition_sublimation(qi_incld, ni_incld, T_atm, qv_sat_l, qv_sat_i, epsi, abi, qv,
                               inv_dt, context=None):
    """p3_ice_deposition_sublimation_impl.hpp. Returns the four tendencies (all zeroed first)."""
    ctx = _ctx(qi_incld, context)
    zero = torch.zeros_like(qi_incld)
    qc2qi_berg_tend = zero
    qv2qi_vapdep_tend = zero
    qi2qv_sublim_tend = zero
    ni_sublim_tend = zero
    m = (qi_incld > C.QSMALL) & ctx
    qi_tend = min_(epsi / abi, inv_dt) * (qv - qv_sat_i)
    neg_qi_tend = qi_tend < 0
    qi2qv_sublim_tend = _set(qi2qv_sublim_tend, m & neg_qi_tend, -qi_tend)
    safe_qi = torch.where(m, qi_incld, torch.ones_like(qi_incld))
    ni_sublim_tend = _set(ni_sublim_tend, m & neg_qi_tend, qi2qv_sublim_tend * (ni_incld / safe_qi))
    T_lt_frz = T_atm < C.T_zerodegc
    qv2qi_vapdep_tend = _set(qv2qi_vapdep_tend, m & T_lt_frz & ~neg_qi_tend, qi_tend)
    qc2qi_berg_tend = _set(qc2qi_berg_tend, m & T_lt_frz, max_(epsi / abi * (qv_sat_l - qv_sat_i), 0))
    return qv2qi_vapdep_tend, qi2qv_sublim_tend, ni_sublim_tend, qc2qi_berg_tend


# ---------------------------------------------------------------- ice melting
def ice_melting(rho, T_atm, pres, rhofaci, table_val_qi2qr_melting, table_val_qi2qr_vent_melt,
                dv, sc, mu, kap, qv, qi_incld, ni_incld, qi2qr_melt_tend, ni2nr_melt_tend,
                context=None):
    """p3_ice_melting_impl.hpp: ice_melting."""
    ctx = _ctx(rho, context)
    has_melt_qi = (qi_incld >= C.QSMALL) & (T_atm > C.Tmelt) & ctx
    qsat0 = saturation.qv_sat_dry(torch.full_like(rho, C.Tmelt), pres, False, saturation.MURPHYKOOP)
    val = (((table_val_qi2qr_melting + table_val_qi2qr_vent_melt * mathfn.cbrt(sc)
             * torch.sqrt(rhofaci * rho / mu))
            * ((T_atm - C.Tmelt) * kap - rho * C.LatVap * dv * (qsat0 - qv))
            * 2 * C.Pi / C.LatIce) * ni_incld)
    qi2qr_melt_tend = _set(qi2qr_melt_tend, has_melt_qi, val)
    qi2qr_melt_tend = max_(qi2qr_melt_tend, 0)  # unmasked in the reference
    safe_qi = torch.where(has_melt_qi, qi_incld, torch.ones_like(qi_incld))
    ni2nr_melt_tend = _set(ni2nr_melt_tend, has_melt_qi, qi2qr_melt_tend * (ni_incld / safe_qi))
    return qi2qr_melt_tend, ni2nr_melt_tend


# ---------------------------------------------------------------- ice nucleation
def ice_nucleation(temp, inv_rho, ni, ni_activated, qv_supersat_i, inv_dt, do_predict_nc: bool,
                   do_prescribed_CCN: bool, qv2qi_nucleat_tend, ni_nucleat_tend,
                   ro: P3Runtime, context=None):
    """p3_ice_nucleation_impl.hpp: ice_nucleation.

    The reference picks its branch with `if (any_if_not_log.any())`; the two masks
    are separated by the runtime flags alone, so the branch is a scalar choice.
    """
    ctx = _ctx(temp, context)
    T_icenuc = C.Tmelt - 15.0
    mi0 = 4.0 * C.PIOV3 * 900.0 * 1.e-18
    t_lt = temp < T_icenuc
    sat_ge = qv_supersat_i >= 0.05
    do_log = (not do_predict_nc) or do_prescribed_CCN
    if do_log:
        m = t_lt & sat_ge & ctx
        dum = 0.005 * mathfn.exp(ro.deposition_nucleation_exponent * (C.Tmelt - temp)) * 1.0e3 * inv_rho
        dum = min_(dum, 1.0e5 * inv_rho)
        N_nuc = max_(C.ZERO, (dum - ni) * inv_dt)
        n_nuc_ge_nsmall = (N_nuc >= C.NSMALL) & ctx
        Q_nuc = max_(C.ZERO, (dum - ni) * mi0 * inv_dt)
        qv2qi_nucleat_tend = _set(qv2qi_nucleat_tend, m & n_nuc_ge_nsmall, Q_nuc)
        ni_nucleat_tend = _set(ni_nucleat_tend, m & n_nuc_ge_nsmall, N_nuc)
    else:
        m = t_lt & sat_ge & ctx
        ni_nucleat_tend = _set(ni_nucleat_tend, m, max_(C.ZERO, (ni_activated - ni) * inv_dt))
        qv2qi_nucleat_tend = _set(qv2qi_nucleat_tend, m, ni_nucleat_tend * mi0)
    return qv2qi_nucleat_tend, ni_nucleat_tend


# ------------------------------------------------- ice relaxation timescale
def ice_relaxation_timescale(rho, temp, rhofaci, table_val_qi2qr_melting,
                             table_val_qi2qr_vent_melt, dv, mu, sc, qi_incld, ni_incld,
                             epsi, epsi_tot, context=None):
    """p3_ice_relaxation_timescale_impl.hpp. Returns (epsi, epsi_tot)."""
    ctx = _ctx(rho, context)
    any_if = (qi_incld >= C.QSMALL) & (temp < C.Tmelt) & ctx
    val = ((table_val_qi2qr_melting + table_val_qi2qr_vent_melt * mathfn.cbrt(sc)
            * torch.sqrt(rhofaci * rho / mu)) * 2.0 * C.Pi * rho * dv) * ni_incld
    epsi = _set(epsi, any_if, val)
    epsi_tot = _set(epsi_tot, any_if, epsi_tot + epsi)
    epsi = _set(epsi, ~any_if & ctx, C.ZERO)
    return epsi, epsi_tot


# --------------------------------------------- ice supersaturation conservation
def ice_supersat_conservation(qv2qi_vapdep_tend, qv2qi_nucleat_tend, qinuc_cnt, cld_frac_i, qv,
                              qv_sat_i, t_atm, dt, qi2qv_sublim_tend, qr2qv_evap_tend,
                              use_hetfrz_classnuc: bool, context=None):
    """p3_ice_supersat_conservation_impl.hpp."""
    ctx = _ctx(qv, context)
    latsublim2 = (C.LatVap + C.LatIce) * (C.LatVap + C.LatIce)
    if use_hetfrz_classnuc:
        qv_sink = qv2qi_vapdep_tend + qv2qi_nucleat_tend + qinuc_cnt
    else:
        qv_sink = qv2qi_vapdep_tend + qv2qi_nucleat_tend
    mask = (qv_sink > C.QSMALL) & (cld_frac_i > 1e-20) & ctx
    qv_avail = ((qv + (qi2qv_sublim_tend + qr2qv_evap_tend) * dt - qv_sat_i)
                / (1 + latsublim2 * qv_sat_i / (C.CP * C.RH2O * square(t_atm))) / dt)
    qv_avail = max_(qv_avail, 0)
    sink_gt_avail = (qv_sink > qv_avail) & mask
    safe_sink = torch.where(sink_gt_avail, qv_sink, torch.ones_like(qv_sink))
    fract = qv_avail / safe_sink
    if use_hetfrz_classnuc:
        qinuc_cnt = _set(qinuc_cnt, sink_gt_avail, qinuc_cnt * fract)
    qv2qi_nucleat_tend = _set(qv2qi_nucleat_tend, sink_gt_avail, qv2qi_nucleat_tend * fract)
    qv2qi_vapdep_tend = _set(qv2qi_vapdep_tend, sink_gt_avail, qv2qi_vapdep_tend * fract)
    return qv2qi_vapdep_tend, qv2qi_nucleat_tend, qinuc_cnt


# ---------------------------------------------------------------- max total ni
def impose_max_total_ni(ni_local, max_total_ni: float, inv_rho_local, context=None):
    """p3_impose_max_total_ni_impl.hpp: impose_max_total_ni."""
    ctx = _ctx(ni_local, context)
    ni_not_small = (ni_local >= 1e-20) & ctx
    safe_ni = torch.where(ni_not_small, ni_local, torch.ones_like(ni_local))
    return _set(ni_local, ni_not_small, ni_local * min_(max_total_ni * inv_rho_local / safe_ni, 1))


# ---------------------------------------------------------------- in-cloud ratios
def calculate_incloud_mixingratios(qc, qr, qi, qm, nc, nr, ni, bm, inv_cld_frac_l,
                                   inv_cld_frac_i, inv_cld_frac_r, qc_incld, qr_incld, qi_incld,
                                   qm_incld, nc_incld, nr_incld, ni_incld, bm_incld, context=None):
    """p3_incloud_mixingratios_impl.hpp: calculate_incloud_mixingratios."""
    ctx = _ctx(qc, context)
    qc_ge = (qc >= C.QSMALL) & ctx
    qi_ge = (qi >= C.QSMALL) & ctx
    qi_qm_ge = (qi >= C.QSMALL) & (qm >= C.QSMALL) & ctx
    qr_ge = (qr >= C.QSMALL) & ctx
    qc_incld = _set(qc_incld, qc_ge, qc * inv_cld_frac_l)
    nc_incld = _set(nc_incld, qc_ge, max_(nc * inv_cld_frac_l, 0))
    qc_incld = _set(qc_incld, ~qc_ge & ctx, 0)
    nc_incld = _set(nc_incld, ~qc_ge & ctx, 0)
    qi_incld = _set(qi_incld, qi_ge, qi * inv_cld_frac_i)
    ni_incld = _set(ni_incld, qi_ge, max_(ni * inv_cld_frac_i, 0))
    qi_incld = _set(qi_incld, ~qi_ge & ctx, 0)
    ni_incld = _set(ni_incld, ~qi_ge & ctx, 0)
    qm_incld = _set(qm_incld, qi_qm_ge, qm * inv_cld_frac_i)
    bm_incld = _set(bm_incld, qi_qm_ge, max_(bm * inv_cld_frac_l, 0))  # note: _l, as in the source
    qm_incld = _set(qm_incld, ~qi_qm_ge & ctx, 0)
    bm_incld = _set(bm_incld, ~qi_qm_ge & ctx, 0)
    qr_incld = _set(qr_incld, qr_ge, qr * inv_cld_frac_r)
    nr_incld = _set(nr_incld, qr_ge, max_(nr * inv_cld_frac_r, 0))
    qr_incld = _set(qr_incld, ~qr_ge & ctx, 0)
    nr_incld = _set(nr_incld, ~qr_ge & ctx, 0)
    any_gt = ((qc_incld > C.incloud_limit) | (qi_incld > C.incloud_limit)
              | (qr_incld > C.precip_limit) | (bm_incld > C.incloud_limit)) & ctx
    qc_incld = _set(qc_incld, any_gt, min_(qc_incld, C.incloud_limit))
    qi_incld = _set(qi_incld, any_gt, min_(qi_incld, C.incloud_limit))
    bm_incld = _set(bm_incld, any_gt, min_(bm_incld, C.incloud_limit))
    qr_incld = _set(qr_incld, any_gt, min_(qr_incld, C.precip_limit))
    return qc_incld, qr_incld, qi_incld, qm_incld, nc_incld, nr_incld, ni_incld, bm_incld


# ---------------------------------------------------------------- number conservation
def nc_conservation(nc, nc_selfcollect_tend, dt, nc_collect_tend, nc2ni_immers_freeze_tend,
                    nc_accret_tend, nc2nr_autoconv_tend, ncheti_cnt, nicnt,
                    use_hetfrz_classnuc: bool, context=None):
    """p3_nc_conservation_impl.hpp."""
    ctx = _ctx(nc, context)
    if use_hetfrz_classnuc:
        sink_nc = (nc_collect_tend + ncheti_cnt + nc_accret_tend + nc2nr_autoconv_tend + nicnt) * dt
    else:
        sink_nc = (nc_collect_tend + nc2ni_immers_freeze_tend + nc_accret_tend + nc2nr_autoconv_tend) * dt
    source_nc = nc + nc_selfcollect_tend * dt
    mask = (sink_nc > source_nc) & ctx
    safe = torch.where(mask, sink_nc, torch.ones_like(sink_nc))
    ratio = source_nc / safe
    nc_collect_tend = _set(nc_collect_tend, mask, nc_collect_tend * ratio)
    if use_hetfrz_classnuc:
        ncheti_cnt = _set(ncheti_cnt, mask, ncheti_cnt * ratio)
        nicnt = _set(nicnt, mask, nicnt * ratio)
    else:
        nc2ni_immers_freeze_tend = _set(nc2ni_immers_freeze_tend, mask, nc2ni_immers_freeze_tend * ratio)
    nc_accret_tend = _set(nc_accret_tend, mask, nc_accret_tend * ratio)
    nc2nr_autoconv_tend = _set(nc2nr_autoconv_tend, mask, nc2nr_autoconv_tend * ratio)
    return nc_collect_tend, nc2ni_immers_freeze_tend, nc_accret_tend, nc2nr_autoconv_tend, ncheti_cnt, nicnt


def ni_conservation(ni, ni_nucleat_tend, nr2ni_immers_freeze_tend, nc2ni_immers_freeze_tend,
                    ncheti_cnt, nicnt, ninuc_cnt, dt, ni2nr_melt_tend, ni_sublim_tend,
                    ni_selfcollect_tend, use_hetfrz_classnuc: bool, context=None):
    """p3_ni_conservation_impl.hpp."""
    ctx = _ctx(ni, context)
    sink_ni = (ni2nr_melt_tend + ni_sublim_tend + ni_selfcollect_tend) * dt
    if use_hetfrz_classnuc:
        source_ni = ni + (ni_nucleat_tend + nr2ni_immers_freeze_tend + ncheti_cnt + nicnt + ninuc_cnt) * dt
    else:
        source_ni = ni + (ni_nucleat_tend + nr2ni_immers_freeze_tend + nc2ni_immers_freeze_tend) * dt
    mask = (sink_ni > source_ni) & ctx
    safe = torch.where(mask, sink_ni, torch.ones_like(sink_ni))
    ratio = source_ni / safe
    ni2nr_melt_tend = _set(ni2nr_melt_tend, mask, ni2nr_melt_tend * ratio)
    ni_sublim_tend = _set(ni_sublim_tend, mask, ni_sublim_tend * ratio)
    ni_selfcollect_tend = _set(ni_selfcollect_tend, mask, ni_selfcollect_tend * ratio)
    return ni2nr_melt_tend, ni_sublim_tend, ni_selfcollect_tend


def nr_conservation(nr, ni2nr_melt_tend, nr_ice_shed_tend, ncshdc, nc2nr_autoconv_tend, dt,
                    nmltratio, nr_collect_tend, nr2ni_immers_freeze_tend, nr_selfcollect_tend,
                    nr_evap_tend, context=None):
    """p3_nr_conservation_impl.hpp."""
    ctx = _ctx(nr, context)
    sink_nr = (nr_collect_tend + nr2ni_immers_freeze_tend + nr_selfcollect_tend + nr_evap_tend) * dt
    source_nr = nr + (ni2nr_melt_tend * nmltratio + nr_ice_shed_tend + ncshdc + nc2nr_autoconv_tend) * dt
    mask = (sink_nr > source_nr) & ctx
    safe = torch.where(mask, sink_nr, torch.ones_like(sink_nr))
    ratio = source_nr / safe
    nr_collect_tend = _set(nr_collect_tend, mask, nr_collect_tend * ratio)
    nr2ni_immers_freeze_tend = _set(nr2ni_immers_freeze_tend, mask, nr2ni_immers_freeze_tend * ratio)
    nr_selfcollect_tend = _set(nr_selfcollect_tend, mask, nr_selfcollect_tend * ratio)
    nr_evap_tend = _set(nr_evap_tend, mask, nr_evap_tend * ratio)
    return nr_collect_tend, nr2ni_immers_freeze_tend, nr_selfcollect_tend, nr_evap_tend


# ------------------------------------------- prevent liquid supersaturation
def prevent_liq_supersaturation(pres, t_atm, qv, dt, qv2qi_vapdep_tend, qinuc,
                                qi2qv_sublim_tend, qr2qv_evap_tend, context=None):
    """p3_prevent_liq_supersaturation_impl.hpp."""
    ctx = _ctx(qv, context)
    qv_sources = _set(torch.zeros_like(qv), ctx, qi2qv_sublim_tend + qr2qv_evap_tend)
    has_sources = (qv_sources >= C.QSMALL) & ctx
    qv_sinks = _set(torch.zeros_like(qv), has_sources, qv2qi_vapdep_tend + qinuc)
    qv_endstep = _set(torch.zeros_like(qv), has_sources, qv - qv_sinks * dt + qv_sources * dt)
    T_endstep = _set(torch.zeros_like(qv), has_sources,
                     t_atm + ((qv_sinks - qi2qv_sublim_tend) * (C.LatVap + C.LatIce) * C.INV_CP
                              - qr2qv_evap_tend * C.LatVap * C.INV_CP) * dt)
    # T_endstep is 0 outside has_sources; keep the saturation call finite there.
    T_safe = torch.where(has_sources, T_endstep, torch.full_like(T_endstep, C.Tmelt))
    qsl = saturation.qv_sat_dry(T_safe, pres, False, saturation.MURPHYKOOP)
    A = _set(torch.zeros_like(qv), has_sources,
             C.LatVap * qsl * dt * C.INV_CP / (C.RV * T_safe * T_safe)
             * ((C.LatVap + C.LatIce) * qi2qv_sublim_tend + C.LatVap * qr2qv_evap_tend))
    denom = qv_sources * dt + A
    safe_denom = torch.where(has_sources, denom, torch.ones_like(denom))
    frac = _set(torch.zeros_like(qv), has_sources, (qsl - qv + qv_sinks * dt + A) / safe_denom)
    frac = _set(frac, has_sources, max_(0, frac))
    frac = _set(frac, has_sources, min_(1, frac))
    qi2qv_sublim_tend = _set(qi2qv_sublim_tend, has_sources, frac * qi2qv_sublim_tend)
    qr2qv_evap_tend = _set(qr2qv_evap_tend, has_sources, frac * qr2qv_evap_tend)
    return qi2qv_sublim_tend, qr2qv_evap_tend


# ---------------------------------------------------------------- rain immersion freezing
def rain_immersion_freezing(T_atm, lamr, mu_r, cdistr, qr_incld, qr2qi_immers_freeze_tend,
                            nr2ni_immers_freeze_tend, ro: P3Runtime, context=None):
    """p3_rain_imm_freezing_impl.hpp: rain_immersion_freezing."""
    ctx = _ctx(T_atm, context)
    m = (qr_incld >= C.QSMALL) & (T_atm <= C.T_rainfrz) & ctx
    e = ro.immersion_freezing_exponent
    # masked-out entries get 1 in the logs (translation rule for masked branches)
    one = torch.ones_like(T_atm)
    cdistr = torch.where(m, cdistr, one)
    lamr = torch.where(m, lamr, one)
    qr2qi_immers_freeze_tend = _set(
        qr2qi_immers_freeze_tend, m,
        C.CONS6 * mathfn.exp(mathfn.log(cdistr) + mathfn.log(mathfn.tgamma(7. + mu_r))
                             - 6. * mathfn.log(lamr)) * mathfn.exp(e * (C.T_zerodegc - T_atm)))
    nr2ni_immers_freeze_tend = _set(
        nr2ni_immers_freeze_tend, m,
        C.CONS5 * mathfn.exp(mathfn.log(cdistr) + mathfn.log(mathfn.tgamma(4. + mu_r))
                             - 3. * mathfn.log(lamr)) * mathfn.exp(e * (C.T_zerodegc - T_atm)))
    return qr2qi_immers_freeze_tend, nr2ni_immers_freeze_tend


# ---------------------------------------------------------------- rain self collection
def rain_self_collection(rho, qr_incld, nr_incld, nr_selfcollect_tend, ro: P3Runtime, context=None):
    """p3_rain_self_collection_impl.hpp: rain_self_collection."""
    ctx = _ctx(rho, context)
    m = (qr_incld >= C.QSMALL) & ctx
    safe_nr = torch.where(m, nr_incld, torch.ones_like(nr_incld))
    dum2 = mathfn.cbrt(torch.where(m, qr_incld, torch.ones_like(qr_incld)) / (C.Pi * C.RHO_H2O * safe_nr))
    d1 = ro.rain_selfcollection_breakup_diameter
    lt = (dum2 < d1) & m
    gt = (dum2 >= d1) & m
    dum = torch.zeros_like(rho)
    dum = _set(dum, lt, 1)
    dum = _set(dum, gt, 2 - mathfn.exp(2300 * (dum2 - d1)))
    return _set(nr_selfcollect_tend, m, dum * ro.rain_selfcollection_prefactor * nr_incld * qr_incld * rho)


# ---------------------------------------------------------------- subgrid variance scaling
def subgrid_variance_scaling(relvar, expon: float):
    """p3_subgrid_variance_scaling_impl.hpp (not called in the default configuration)."""
    return mathfn.tgamma(relvar + expon) / (mathfn.tgamma(relvar) * mathfn.pow(relvar, expon))


# ---------------------------------------------------------------- prognostic updates
def update_prognostic_ice(t: dict, inv_exner, do_predict_nc: bool, log_wetgrowth, dt, nmltratio,
                          rho_qm_cloud, th_atm, qv, qi, ni, qm, bm, qc, nc, qr, nr,
                          use_hetfrz_classnuc: bool, context=None):
    """p3_update_prognostics_impl.hpp: update_prognostic_ice."""
    ctx = _ctx(qv, context)
    g = t.__getitem__
    if use_hetfrz_classnuc:
        qc = _set(qc, ctx, qc + (-g("qcheti_cnt") - g("qicnt") - g("qc2qi_collect_tend")
                                 - g("qc2qr_ice_shed_tend") - g("qc2qi_berg_tend")) * dt)
    else:
        qc = _set(qc, ctx, qc + (-g("qc2qi_hetero_freeze_tend") - g("qc2qi_collect_tend")
                                 - g("qc2qr_ice_shed_tend") - g("qc2qi_berg_tend")) * dt)
    if do_predict_nc:
        if use_hetfrz_classnuc:
            nc = _set(nc, ctx, nc + (-g("nc_collect_tend") - g("ncheti_cnt") - g("nicnt")) * dt)
        else:
            nc = _set(nc, ctx, nc + (-g("nc_collect_tend") - g("nc2ni_immers_freeze_tend")) * dt)
    qr = _set(qr, ctx, qr + (-g("qr2qi_collect_tend") + g("qi2qr_melt_tend")
                             - g("qr2qi_immers_freeze_tend") + g("qc2qr_ice_shed_tend")) * dt)
    nr = _set(nr, ctx, nr + (-g("nr_collect_tend") - g("nr2ni_immers_freeze_tend")
                             + nmltratio * g("ni2nr_melt_tend") + g("nr_ice_shed_tend")
                             + g("ncshdc")) * dt)
    qi_not_small = (qi >= C.QSMALL) & ctx
    safe_qi = torch.where(qi_not_small, qi, torch.ones_like(qi))
    sub_melt = g("qi2qv_sublim_tend") + g("qi2qr_melt_tend")
    bm = _set(bm, qi_not_small, bm - (sub_melt / safe_qi) * dt * bm)
    qm = _set(qm, qi_not_small, qm - (sub_melt * qm / safe_qi) * dt)
    qi = _set(qi, qi_not_small, qi - sub_melt * dt)
    if use_hetfrz_classnuc:
        dum = (g("qr2qi_collect_tend") + g("qc2qi_collect_tend") + g("qr2qi_immers_freeze_tend")
               + g("qcheti_cnt") + g("qicnt")) * dt
        qi = _set(qi, ctx, qi + (g("qv2qi_vapdep_tend") + g("qv2qi_nucleat_tend")
                                 + g("qc2qi_berg_tend") + g("qinuc_cnt")) * dt + dum)
        qm = _set(qm, ctx, qm + dum)
        bm = _set(bm, ctx, bm + (g("qr2qi_collect_tend") * C.INV_RHO_RIMEMAX
                                 + g("qc2qi_collect_tend") / rho_qm_cloud
                                 + (g("qr2qi_immers_freeze_tend") + g("qcheti_cnt") + g("qicnt"))
                                 * C.INV_RHO_RIMEMAX) * dt)
        ni = _set(ni, ctx, ni + (g("ni_nucleat_tend") - g("ni2nr_melt_tend") - g("ni_sublim_tend")
                                 - g("ni_selfcollect_tend") + g("nr2ni_immers_freeze_tend")
                                 + g("ncheti_cnt") + g("nicnt") + g("ninuc_cnt")) * dt)
    else:
        dum = (g("qr2qi_collect_tend") + g("qc2qi_collect_tend") + g("qr2qi_immers_freeze_tend")
               + g("qc2qi_hetero_freeze_tend")) * dt
        qi = _set(qi, ctx, qi + (g("qv2qi_vapdep_tend") + g("qv2qi_nucleat_tend")
                                 + g("qc2qi_berg_tend")) * dt + dum)
        qm = _set(qm, ctx, qm + dum)
        bm = _set(bm, ctx, bm + (g("qr2qi_collect_tend") * C.INV_RHO_RIMEMAX
                                 + g("qc2qi_collect_tend") / rho_qm_cloud
                                 + (g("qr2qi_immers_freeze_tend") + g("qc2qi_hetero_freeze_tend"))
                                 * C.INV_RHO_RIMEMAX) * dt)
        ni = _set(ni, ctx, ni + (g("ni_nucleat_tend") - g("ni2nr_melt_tend") - g("ni_sublim_tend")
                                 - g("ni_selfcollect_tend") + g("nr2ni_immers_freeze_tend")
                                 + g("nc2ni_immers_freeze_tend")) * dt)
    qm_lt = (qm < 0) & ctx
    qm = _set(qm, qm_lt, 0)
    bm = _set(bm, qm_lt, 0)
    qm = _set(qm, log_wetgrowth & ctx, qi)
    bm = _set(bm, log_wetgrowth & ctx, qm * C.INV_RHO_RIMEMAX)
    if use_hetfrz_classnuc:
        qv = _set(qv, ctx, qv + (-g("qv2qi_vapdep_tend") + g("qi2qv_sublim_tend")
                                 - g("qv2qi_nucleat_tend") - g("qinuc_cnt")) * dt)
        th_atm = _set(th_atm, ctx, th_atm + inv_exner * (
            (g("qv2qi_vapdep_tend") - g("qi2qv_sublim_tend") + g("qv2qi_nucleat_tend")
             + g("qinuc_cnt")) * (C.LatVap + C.LatIce) * C.INV_CP
            + (g("qr2qi_collect_tend") + g("qc2qi_collect_tend") + g("qcheti_cnt") + g("qicnt")
               + g("qr2qi_immers_freeze_tend") - g("qi2qr_melt_tend") + g("qc2qi_berg_tend"))
            * C.LatIce * C.INV_CP) * dt)
    else:
        qv = _set(qv, ctx, qv + (-g("qv2qi_vapdep_tend") + g("qi2qv_sublim_tend")
                                 - g("qv2qi_nucleat_tend")) * dt)
        th_atm = _set(th_atm, ctx, th_atm + inv_exner * (
            (g("qv2qi_vapdep_tend") - g("qi2qv_sublim_tend") + g("qv2qi_nucleat_tend"))
            * (C.LatVap + C.LatIce) * C.INV_CP
            + (g("qr2qi_collect_tend") + g("qc2qi_collect_tend") + g("qc2qi_hetero_freeze_tend")
               + g("qr2qi_immers_freeze_tend") - g("qi2qr_melt_tend") + g("qc2qi_berg_tend"))
            * C.LatIce * C.INV_CP) * dt)
    return th_atm, qv, qi, ni, qm, bm, qc, nc, qr, nr


def update_prognostic_liquid(qc2qr_accret_tend, nc_accret_tend, qc2qr_autoconv_tend,
                             nc2nr_autoconv_tend, ncautr, nc_selfcollect_tend, qr2qv_evap_tend,
                             nr_evap_tend, nr_selfcollect_tend, do_predict_nc: bool,
                             do_prescribed_CCN: bool, inv_rho, inv_exner, dt, th_atm, qv, qc, nc,
                             qr, nr, context=None):
    """p3_update_prognostics_impl.hpp: update_prognostic_liquid."""
    ctx = _ctx(qv, context)
    qc = _set(qc, ctx, qc + (-qc2qr_accret_tend - qc2qr_autoconv_tend) * dt)
    qr = _set(qr, ctx, qr + (qc2qr_accret_tend + qc2qr_autoconv_tend - qr2qv_evap_tend) * dt)
    if do_predict_nc or do_prescribed_CCN:
        nc = _set(nc, ctx, nc + (-nc_accret_tend - nc2nr_autoconv_tend + nc_selfcollect_tend) * dt)
    else:
        nc = _set(nc, ctx, C.NCCNST * inv_rho)
    if C.IPARAM in (1, 2):
        nr = _set(nr, ctx, nr + (0.5 * nc2nr_autoconv_tend - nr_selfcollect_tend - nr_evap_tend) * dt)
    else:
        nr = _set(nr, ctx, nr + (ncautr - nr_selfcollect_tend - nr_evap_tend) * dt)
    qv = _set(qv, ctx, qv + qr2qv_evap_tend * dt)
    th_atm = _set(th_atm, ctx, th_atm + inv_exner * (-qr2qv_evap_tend * C.LatVap * C.INV_CP) * dt)
    return th_atm, qv, qc, nc, qr, nr
