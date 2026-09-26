"""P3 main: p3_main_init, p3_main_part1/2/3 and p3_main.

Translation of components/eamxx/src/physics/p3/impl/p3_main_impl.hpp,
p3_main_impl_part1.hpp, p3_main_impl_part2.hpp and p3_main_impl_part3.hpp at
E3SM 80fe0a71 (SCREAM_P3_SMALL_KERNELS off).

The reference returns early from a column when no nucleation is possible and no
hydrometeors are present, and again after part2 if no hydrometeors are left.
Those `return`s become column masks here: a stage runs for every column and its
result is then selected with `torch.where`, so a skipped column keeps exactly
the state the reference leaves it with.

check_values is not ported: it only prints and (with force_abort, which is off)
aborts.
"""
from __future__ import annotations

import torch

from .. import constants as C
from ..gradsafe import nonsingular
from .. import mathfn
from .. import saturation
from ..ekat import max_, min_, rdiv
from . import functions as F
from . import sedimentation as S
from .runtime import P3Runtime
from .tables import (P3Tables, apply_table_ice, apply_table_coll, get_cdistr_logn0r,
                     get_cloud_dsd2, get_rain_dsd2, lookup_ice, lookup_rain)

# Process-rate names carried through part2 (all start at 0 at every level)
TEND_NAMES = (
    "qc2qr_accret_tend", "qr2qv_evap_tend", "qc2qr_autoconv_tend", "nc_accret_tend",
    "nc_selfcollect_tend", "nc2nr_autoconv_tend", "nr_selfcollect_tend", "nr_evap_tend",
    "ncautr", "qi2qv_sublim_tend", "nr_ice_shed_tend", "qc2qi_hetero_freeze_tend",
    "qr2qi_collect_tend", "qc2qr_ice_shed_tend", "qi2qr_melt_tend", "qc2qi_collect_tend",
    "qr2qi_immers_freeze_tend", "qv2qi_nucleat_tend", "ni2nr_melt_tend", "nc_collect_tend",
    "ncshdc", "nc2ni_immers_freeze_tend", "nr_collect_tend", "ni_selfcollect_tend",
    "ni_nucleat_tend", "qv2qi_vapdep_tend", "qc2qi_berg_tend", "nr2ni_immers_freeze_tend",
    "ni_sublim_tend", "ncheti_cnt", "qcheti_cnt", "nicnt", "qicnt", "ninuc_cnt", "qinuc_cnt",
)


def p3_main_init(cld_frac_i, cld_frac_l, cld_frac_r, inv_exner, th_atm, dz, qv):
    """p3_main_impl.hpp: p3_main_init. Returns a dict of the initialised fields."""
    zero = torch.zeros_like(th_atm)
    exner = rdiv(1.0, inv_exner)
    return dict(
        precip_liq_surf=torch.zeros_like(th_atm[:, 0]),
        precip_ice_surf=torch.zeros_like(th_atm[:, 0]),
        diag_equiv_reflectivity=torch.full_like(th_atm, -99.0),
        ze_ice=torch.full_like(th_atm, 1.e-22),
        ze_rain=torch.full_like(th_atm, 1.e-22),
        diag_eff_radius_qc=torch.full_like(th_atm, 10.e-6),
        diag_eff_radius_qi=torch.full_like(th_atm, 25.e-6),
        diag_eff_radius_qr=torch.full_like(th_atm, 500.e-6),
        inv_cld_frac_i=rdiv(1.0, cld_frac_i),
        inv_cld_frac_l=rdiv(1.0, cld_frac_l),
        inv_cld_frac_r=rdiv(1.0, cld_frac_r),
        exner=exner,
        T_atm=th_atm * exner,
        qv=max_(qv, 0),
        inv_dz=rdiv(1.0, dz),
        # zero_init
        mu_r=zero, lamr=zero, logn0r=zero, nu=zero, cdist=zero, cdist1=zero, cdistr=zero,
        qc_incld=zero, qr_incld=zero, qi_incld=zero, qm_incld=zero,
        nc_incld=zero, nr_incld=zero, ni_incld=zero, bm_incld=zero,
        inv_rho=zero, prec=zero, rho=zero, rhofacr=zero, rhofaci=zero, acn=zero,
        qv_sat_l=zero, qv_sat_i=zero, sup=zero, qv_supersat_i=zero, tmparr1=zero,
        qtend_ignore=zero, ntend_ignore=zero, mu_c=zero, lamc=zero,
        rho_qi=zero, qv2qi_depos_tend=zero, precip_total_tend=zero, nevapr=zero,
        precip_liq_flux=th_atm.new_zeros((th_atm.shape[0], th_atm.shape[-1] + 1)),
        precip_ice_flux=th_atm.new_zeros((th_atm.shape[0], th_atm.shape[-1] + 1)),
        diag_vm_qi=zero, diag_diam_qi=zero, pratot=zero, prctot=zero, qr_evap_tend=zero,
    )


def p3_main_part1(predictNc: bool, do_prescribed_CCN: bool, dt, pres, dpres, dz, nc_nuceat_tend,
                  nccn_prescribed, inv_exner, exner, inv_cld_frac_l, inv_cld_frac_i,
                  inv_cld_frac_r, qv, th_atm, qc, nc, qr, nr, qi, ni, qm, bm, ro: P3Runtime):
    """p3_main_impl.hpp part1. Returns a dict of updated state plus the two column flags."""
    g = C.gravit
    rho = dpres / dz / g
    inv_rho = rdiv(1.0, rho)
    T_atm = th_atm * exner
    qv_sat_l = saturation.qv_sat_dry(T_atm, pres, False, saturation.MURPHYKOOP)
    qv_sat_i = saturation.qv_sat_dry(T_atm, pres, True, saturation.MURPHYKOOP)
    qv_supersat_i = qv / qv_sat_i - 1
    rhofacr = mathfn.pow(C.RHO_1000MB * inv_rho, .54)
    rhofaci = mathfn.pow(C.RHO_600MB * inv_rho, .54)
    dum = 1.496e-6 * mathfn.pow(T_atm, 1.5) / (T_atm + 120)
    acn = rdiv(g * C.RHO_H2O, 18 * dum)
    nucleationPossible = ((T_atm < C.T_zerodegc) & (qv_supersat_i >= -0.05)).any(dim=-1)

    hydrometeorsPresent = torch.zeros_like(nucleationPossible)
    # cloud liquid
    drymass = qc < C.QSMALL
    qv = F._set(qv, drymass, qv + qc)
    th_atm = F._set(th_atm, drymass, th_atm - inv_exner * qc * C.LatVap * C.INV_CP)
    qc = F._set(qc, drymass, 0)
    nc = F._set(nc, drymass, 0)
    not_drymass = ~drymass
    hydrometeorsPresent = hydrometeorsPresent | not_drymass.any(dim=-1)
    if do_prescribed_CCN:
        nccn_scaled = nccn_prescribed / inv_cld_frac_l
        # d(x^0.55)/dx is infinite at x = 0 (no CCN); gradient 0 there, values unchanged (gradsafe)
        nccn_scaled = nonsingular(nccn_scaled == 0, lambda x: mathfn.pow(x, ro.spa_ccn_to_nc_exponent), nccn_scaled)
        nc = F._set(nc, not_drymass, max_(nc, ro.spa_ccn_to_nc_factor * nccn_scaled))
    elif predictNc:
        nc = F._set(nc, not_drymass, max_(nc + nc_nuceat_tend * dt, 0.0))
    else:
        nc = F._set(nc, not_drymass, C.NCCNST * inv_rho)
    # rain
    drymass = qr < C.QSMALL
    qv = F._set(qv, drymass, qv + qr)
    th_atm = F._set(th_atm, drymass, th_atm - inv_exner * qr * C.LatVap * C.INV_CP)
    qr = F._set(qr, drymass, 0)
    nr = F._set(nr, drymass, 0)
    hydrometeorsPresent = hydrometeorsPresent | (~drymass).any(dim=-1)
    # ice
    drymass = (qi < C.QSMALL) | ((qi < 1.e-8) & (qv_supersat_i < -0.1))
    qv = F._set(qv, drymass, qv + qi)
    th_atm = F._set(th_atm, drymass, th_atm - inv_exner * qi * (C.LatVap + C.LatIce) * C.INV_CP)
    qi = F._set(qi, drymass, 0)
    ni = F._set(ni, drymass, 0)
    qm = F._set(qm, drymass, 0)
    bm = F._set(bm, drymass, 0)
    hydrometeorsPresent = hydrometeorsPresent | (~drymass).any(dim=-1)
    # small ice above freezing goes to rain
    drymass = (qi >= C.QSMALL) & (qi < 1.e-8) & (T_atm >= C.T_zerodegc)
    qr = F._set(qr, drymass, qr + qi)
    th_atm = F._set(th_atm, drymass, th_atm - inv_exner * qi * C.LatIce * C.INV_CP)
    qi = F._set(qi, drymass, 0)
    ni = F._set(ni, drymass, 0)
    qm = F._set(qm, drymass, 0)
    bm = F._set(bm, drymass, 0)

    T_atm = th_atm * exner
    zero = torch.zeros_like(qc)
    (qc_incld, qr_incld, qi_incld, qm_incld, nc_incld, nr_incld, ni_incld,
     bm_incld) = F.calculate_incloud_mixingratios(
        qc, qr, qi, qm, nc, nr, ni, bm, inv_cld_frac_l, inv_cld_frac_i, inv_cld_frac_r,
        zero, zero, zero, zero, zero, zero, zero, zero)
    return dict(T_atm=T_atm, rho=rho, inv_rho=inv_rho, qv_sat_l=qv_sat_l, qv_sat_i=qv_sat_i,
                qv_supersat_i=qv_supersat_i, rhofacr=rhofacr, rhofaci=rhofaci, acn=acn,
                qv=qv, th_atm=th_atm, qc=qc, nc=nc, qr=qr, nr=nr, qi=qi, ni=ni, qm=qm, bm=bm,
                qc_incld=qc_incld, qr_incld=qr_incld, qi_incld=qi_incld, qm_incld=qm_incld,
                nc_incld=nc_incld, nr_incld=nr_incld, ni_incld=ni_incld, bm_incld=bm_incld,
                nucleationPossible=nucleationPossible, hydrometeorsPresent=hydrometeorsPresent)


def p3_main_part2(ro: P3Runtime, predictNc: bool, do_prescribed_CCN: bool, dt, inv_dt, tables:
                  P3Tables, st: dict, dgn: dict):
    """p3_main_impl.hpp part2. `st` is the working state, `dgn` the diagnostic inputs.

    Returns (state, hydrometeorsPresent, diagnostics).
    """
    s = dict(st)
    qsmall, nsmall = C.QSMALL, C.NSMALL
    zero = torch.zeros_like(s["qc"])
    skip_all = ((s["qc"] < qsmall) & (s["qr"] < qsmall) & (s["qi"] < qsmall)
                & (s["T_atm"] < C.T_zerodegc) & (s["qv_supersat_i"] < -0.05))
    not_skip_all = ~skip_all
    t = {n: zero for n in TEND_NAMES}
    t["rho_qm_cloud"] = torch.full_like(zero, 400.0)
    qc_growth_rate = zero
    epsi = zero
    epsi_tot = zero
    epsc = zero
    epsr = zero
    vtrmi1 = zero
    wetgrowth = torch.zeros_like(zero, dtype=torch.bool)
    tv = {n: zero for n in ("table_val_qi_fallspd", "table_val_ni_self_collect",
                            "table_val_qc2qi_collect", "table_val_qi2qr_melting",
                            "table_val_nr_collect", "table_val_qr2qi_collect",
                            "table_val_ni_lammax", "table_val_ni_lammin",
                            "table_val_qi2qr_vent_melt")}

    skip_micro = skip_all | ~((s["qc_incld"] >= qsmall) | (s["qr_incld"] >= qsmall)
                              | (s["qi_incld"] >= qsmall))
    nsm = ~skip_micro  # not_skip_micro

    mu, dv, sc, dqsdt, dqsidt, ab, abi, kap, eii = F.get_time_space_phys_variables(
        s["T_atm"], dgn["pres"], s["rho"], s["qv_sat_l"], s["qv_sat_i"], nsm)
    (s["nc_incld"], s["mu_c"], s["nu"], s["lamc"], s["cdist"],
     s["cdist1"]) = get_cloud_dsd2(s["qc_incld"], s["nc_incld"], s["mu_c"], s["rho"], s["nu"],
                                   tables.dnu_table_vals, s["lamc"], s["cdist"], s["cdist1"], nsm)
    s["nc"] = F._set(s["nc"], nsm, s["nc_incld"] * dgn["cld_frac_l"])
    s["nr_incld"], s["mu_r"], s["lamr"] = get_rain_dsd2(
        s["qr_incld"], s["nr_incld"], s["mu_r"], s["lamr"], ro.constant_mu_rain, nsm)
    s["cdistr"], s["logn0r"] = get_cdistr_logn0r(
        s["qr_incld"], s["nr_incld"], s["mu_r"], s["lamr"], s["cdistr"], s["logn0r"], nsm)
    s["nr"] = F._set(s["nr"], nsm, s["nr_incld"] * dgn["cld_frac_r"])
    s["ni_incld"] = F.impose_max_total_ni(s["ni_incld"], ro.max_total_ni, s["inv_rho"], nsm)

    qi_gt_small = (s["qi_incld"] >= qsmall) & nsm
    s["ni_incld"] = F._set(s["ni_incld"], qi_gt_small, max_(s["ni_incld"], nsmall))
    s["nr_incld"] = F._set(s["nr_incld"], qi_gt_small, max_(s["nr_incld"], nsmall))
    rhop, s["qm_incld"], s["bm_incld"] = S.calc_bulk_rho_rime(
        s["qi_incld"], s["qm_incld"], s["bm_incld"], ro, qi_gt_small)
    s["qm"] = F._set(s["qm"], qi_gt_small, s["qm_incld"] * dgn["cld_frac_i"])
    s["bm"] = F._set(s["bm"], qi_gt_small, s["bm_incld"] * dgn["cld_frac_i"])
    one = torch.ones_like(zero)
    ti = lookup_ice(torch.where(qi_gt_small, s["qi_incld"], one),
                    torch.where(qi_gt_small, s["ni_incld"], one),
                    torch.where(qi_gt_small, s["qm_incld"], zero),
                    torch.where(qi_gt_small, rhop, one), qi_gt_small)
    tr = lookup_rain(torch.where(qi_gt_small, s["qr_incld"], zero),
                     torch.where(qi_gt_small, s["nr_incld"], one), qi_gt_small)
    for name, idx in (("table_val_qi_fallspd", 1), ("table_val_ni_self_collect", 2),
                      ("table_val_qc2qi_collect", 3), ("table_val_qi2qr_melting", 4),
                      ("table_val_ni_lammax", 6), ("table_val_ni_lammin", 7),
                      ("table_val_qi2qr_vent_melt", 9)):
        tv[name] = F._set(tv[name], qi_gt_small, apply_table_ice(idx, tables.ice_table_vals, ti))
    qr_gt_small = (s["qr_incld"] >= qsmall) & qi_gt_small
    for name, idx in (("table_val_nr_collect", 0), ("table_val_qr2qi_collect", 1)):
        tv[name] = F._set(tv[name], qr_gt_small,
                          apply_table_coll(idx, tables.collect_table_vals, ti, tr))
    s["ni_incld"] = F._set(s["ni_incld"], qi_gt_small,
                           min_(s["ni_incld"], tv["table_val_ni_lammax"] * s["ni_incld"]))
    s["ni_incld"] = F._set(s["ni_incld"], qi_gt_small,
                           max_(s["ni_incld"], tv["table_val_ni_lammin"] * s["ni_incld"]))

    if ro.do_ice_production:
        (t["qc2qi_collect_tend"], t["nc_collect_tend"], t["qc2qr_ice_shed_tend"],
         t["ncshdc"]) = F.ice_cldliq_collection(
            s["rho"], s["T_atm"], s["rhofaci"], tv["table_val_qc2qi_collect"], s["qi_incld"],
            s["qc_incld"], s["ni_incld"], s["nc_incld"], t["qc2qi_collect_tend"],
            t["nc_collect_tend"], t["qc2qr_ice_shed_tend"], t["ncshdc"], ro, nsm)
        t["qr2qi_collect_tend"], t["nr_collect_tend"] = F.ice_rain_collection(
            s["rho"], s["T_atm"], s["rhofaci"], s["logn0r"], tv["table_val_nr_collect"],
            tv["table_val_qr2qi_collect"], s["qi_incld"], s["ni_incld"], s["qr_incld"],
            t["qr2qi_collect_tend"], t["nr_collect_tend"], ro, nsm)
        t["ni_selfcollect_tend"] = F.ice_self_collection(
            s["rho"], s["rhofaci"], tv["table_val_ni_self_collect"], eii, s["qm_incld"],
            s["qi_incld"], s["ni_incld"], t["ni_selfcollect_tend"], nsm)
    t["qi2qr_melt_tend"], t["ni2nr_melt_tend"] = F.ice_melting(
        s["rho"], s["T_atm"], dgn["pres"], s["rhofaci"], tv["table_val_qi2qr_melting"],
        tv["table_val_qi2qr_vent_melt"], dv, sc, mu, kap, s["qv"], s["qi_incld"], s["ni_incld"],
        t["qi2qr_melt_tend"], t["ni2nr_melt_tend"], nsm)
    if ro.do_ice_production:
        (wetgrowth, t["qr2qi_collect_tend"], t["qc2qi_collect_tend"], qc_growth_rate,
         t["nr_ice_shed_tend"], t["qc2qr_ice_shed_tend"]) = F.ice_cldliq_wet_growth(
            s["rho"], s["T_atm"], dgn["pres"], s["rhofaci"], tv["table_val_qi2qr_melting"],
            tv["table_val_qi2qr_vent_melt"], dv, kap, mu, sc, s["qv"], s["qc_incld"],
            s["qi_incld"], s["ni_incld"], s["qr_incld"], wetgrowth, t["qr2qi_collect_tend"],
            t["qc2qi_collect_tend"], qc_growth_rate, t["nr_ice_shed_tend"],
            t["qc2qr_ice_shed_tend"], nsm)
    epsi, epsi_tot = F.ice_relaxation_timescale(
        s["rho"], s["T_atm"], s["rhofaci"], tv["table_val_qi2qr_melting"],
        tv["table_val_qi2qr_vent_melt"], dv, mu, sc, s["qi_incld"], s["ni_incld"], epsi,
        epsi_tot, nsm)
    vtrmi1, t["rho_qm_cloud"] = F.calc_rime_density(
        s["T_atm"], s["rhofaci"], tv["table_val_qi_fallspd"], s["acn"], s["lamc"], s["mu_c"],
        s["qc_incld"], t["qc2qi_collect_tend"], vtrmi1, t["rho_qm_cloud"], nsm)
    if ro.do_ice_production:
        if ro.use_hetfrz_classnuc:
            for iflag in (1, 2):
                (t["ncheti_cnt"], t["qcheti_cnt"], t["nicnt"], t["qicnt"], t["ninuc_cnt"],
                 t["qinuc_cnt"]) = F.ice_classical_nucleation(
                    dgn["hetfrz_immersion_nucleation_tend"], dgn["hetfrz_contact_nucleation_tend"],
                    dgn["hetfrz_deposition_nucleation_tend"], s["rho"], s["qc_incld"],
                    s["nc_incld"], iflag, t["ncheti_cnt"], t["qcheti_cnt"], t["nicnt"],
                    t["qicnt"], t["ninuc_cnt"], t["qinuc_cnt"])
        else:
            t["qc2qi_hetero_freeze_tend"], t["nc2ni_immers_freeze_tend"] = F.cldliq_immersion_freezing(
                s["T_atm"], s["lamc"], s["mu_c"], s["cdist1"], s["qc_incld"],
                dgn["inv_qc_relvar"], t["qc2qi_hetero_freeze_tend"],
                t["nc2ni_immers_freeze_tend"], ro, nsm)
        t["qr2qi_immers_freeze_tend"], t["nr2ni_immers_freeze_tend"] = F.rain_immersion_freezing(
            s["T_atm"], s["lamr"], s["mu_r"], s["cdistr"], s["qr_incld"],
            t["qr2qi_immers_freeze_tend"], t["nr2ni_immers_freeze_tend"], ro, nsm)
    epsr, epsc = F.calc_liq_relaxation_timescale(
        tables.revap_table_vals, s["rho"], C.f1r, C.f2r, dv, mu, sc, s["mu_r"], s["lamr"],
        s["cdistr"], s["cdist"], s["qr_incld"], s["qc_incld"], epsr, epsc, nsm)
    t["qr2qv_evap_tend"], t["nr_evap_tend"] = F.evaporate_rain(
        s["qr_incld"], s["qc_incld"], s["nr_incld"], s["qi_incld"], dgn["cld_frac_l"],
        dgn["cld_frac_r"], s["qv"], dgn["qv_prev"], s["qv_sat_l"], s["qv_sat_i"], ab, abi, epsr,
        epsi_tot, s["T_atm"], dgn["t_prev"], dqsdt, dt, nsm)
    if ro.do_ice_production:
        (t["qv2qi_vapdep_tend"], t["qi2qv_sublim_tend"], t["ni_sublim_tend"],
         t["qc2qi_berg_tend"]) = F.ice_deposition_sublimation(
            s["qi_incld"], s["ni_incld"], s["T_atm"], s["qv_sat_l"], s["qv_sat_i"], epsi, abi,
            s["qv"], inv_dt, nsm)

    if ro.do_ice_production:
        t["qv2qi_nucleat_tend"], t["ni_nucleat_tend"] = F.ice_nucleation(
            s["T_atm"], s["inv_rho"], s["ni"], dgn["ni_activated"], s["qv_supersat_i"], inv_dt,
            predictNc, do_prescribed_CCN, t["qv2qi_nucleat_tend"], t["ni_nucleat_tend"], ro,
            not_skip_all)
    (t["qc2qr_autoconv_tend"], t["nc2nr_autoconv_tend"],
     t["ncautr"]) = F.cloud_water_autoconversion(
        s["rho"], s["qc_incld"], s["nc_incld"], dgn["inv_qc_relvar"], t["qc2qr_autoconv_tend"],
        t["nc2nr_autoconv_tend"], t["ncautr"], ro, not_skip_all)
    t["nc_selfcollect_tend"] = F.droplet_self_collection(s["qc_incld"], t["nc_selfcollect_tend"],
                                                         not_skip_all)
    t["qc2qr_accret_tend"], t["nc_accret_tend"] = F.cloud_rain_accretion(
        s["rho"], s["inv_rho"], s["qc_incld"], s["nc_incld"], s["qr_incld"],
        dgn["inv_qc_relvar"], t["qc2qr_accret_tend"], t["nc_accret_tend"], ro, not_skip_all)
    t["nr_selfcollect_tend"] = F.rain_self_collection(
        s["rho"], s["qr_incld"], s["nr_incld"], t["nr_selfcollect_tend"], ro, not_skip_all)
    t = F.back_to_cell_average(dgn["cld_frac_l"], dgn["cld_frac_r"], dgn["cld_frac_i"], t, ro,
                              not_skip_all)
    t = F.cloud_water_conservation(s["qc"], dt, t, ro.use_hetfrz_classnuc, ro,
                                   dgn["cld_frac_l"], dgn["cld_frac_i"], not_skip_all)
    (t["qr2qv_evap_tend"], t["qr2qi_collect_tend"],
     t["qr2qi_immers_freeze_tend"]) = F.rain_water_conservation(
        s["qr"], t["qc2qr_autoconv_tend"], t["qc2qr_accret_tend"], t["qi2qr_melt_tend"],
        t["qc2qr_ice_shed_tend"], dt, t["qr2qv_evap_tend"], t["qr2qi_collect_tend"],
        t["qr2qi_immers_freeze_tend"], not_skip_all)
    t["qi2qv_sublim_tend"], t["qi2qr_melt_tend"] = F.ice_water_conservation(
        s["qi"], t["qv2qi_vapdep_tend"], t["qv2qi_nucleat_tend"], t["qc2qi_berg_tend"],
        t["qr2qi_collect_tend"], t["qc2qi_collect_tend"], t["qr2qi_immers_freeze_tend"],
        t["qc2qi_hetero_freeze_tend"], dt, t["qinuc_cnt"], t["qcheti_cnt"], t["qicnt"],
        t["qi2qv_sublim_tend"], t["qi2qr_melt_tend"], ro.use_hetfrz_classnuc, not_skip_all)
    (t["nc_collect_tend"], t["nc2ni_immers_freeze_tend"], t["nc_accret_tend"],
     t["nc2nr_autoconv_tend"], t["ncheti_cnt"], t["nicnt"]) = F.nc_conservation(
        s["nc"], t["nc_selfcollect_tend"], dt, t["nc_collect_tend"],
        t["nc2ni_immers_freeze_tend"], t["nc_accret_tend"], t["nc2nr_autoconv_tend"],
        t["ncheti_cnt"], t["nicnt"], ro.use_hetfrz_classnuc, not_skip_all)
    (t["nr_collect_tend"], t["nr2ni_immers_freeze_tend"], t["nr_selfcollect_tend"],
     t["nr_evap_tend"]) = F.nr_conservation(
        s["nr"], t["ni2nr_melt_tend"], t["nr_ice_shed_tend"], t["ncshdc"],
        t["nc2nr_autoconv_tend"], dt, C.nmltratio, t["nr_collect_tend"],
        t["nr2ni_immers_freeze_tend"], t["nr_selfcollect_tend"], t["nr_evap_tend"], not_skip_all)
    (t["ni2nr_melt_tend"], t["ni_sublim_tend"],
     t["ni_selfcollect_tend"]) = F.ni_conservation(
        s["ni"], t["ni_nucleat_tend"], t["nr2ni_immers_freeze_tend"],
        t["nc2ni_immers_freeze_tend"], t["ncheti_cnt"], t["nicnt"], t["ninuc_cnt"], dt,
        t["ni2nr_melt_tend"], t["ni_sublim_tend"], t["ni_selfcollect_tend"],
        ro.use_hetfrz_classnuc, not_skip_all)
    (t["qv2qi_vapdep_tend"], t["qv2qi_nucleat_tend"],
     t["qinuc_cnt"]) = F.ice_supersat_conservation(
        t["qv2qi_vapdep_tend"], t["qv2qi_nucleat_tend"], t["qinuc_cnt"], dgn["cld_frac_i"],
        s["qv"], s["qv_sat_i"], s["th_atm"] / dgn["inv_exner"], dt, t["qi2qv_sublim_tend"],
        t["qr2qv_evap_tend"], ro.use_hetfrz_classnuc, not_skip_all)
    t["qi2qv_sublim_tend"], t["qr2qv_evap_tend"] = F.prevent_liq_supersaturation(
        dgn["pres"], s["T_atm"], s["qv"], dt, t["qv2qi_vapdep_tend"], t["qv2qi_nucleat_tend"],
        t["qi2qv_sublim_tend"], t["qr2qv_evap_tend"], not_skip_all)
    (s["th_atm"], s["qv"], s["qi"], s["ni"], s["qm"], s["bm"], s["qc"], s["nc"], s["qr"],
     s["nr"]) = F.update_prognostic_ice(
        t, dgn["inv_exner"], predictNc, wetgrowth, dt, C.nmltratio, t["rho_qm_cloud"],
        s["th_atm"], s["qv"], s["qi"], s["ni"], s["qm"], s["bm"], s["qc"], s["nc"], s["qr"],
        s["nr"], ro.use_hetfrz_classnuc, not_skip_all)
    (s["th_atm"], s["qv"], s["qc"], s["nc"], s["qr"], s["nr"]) = F.update_prognostic_liquid(
        t["qc2qr_accret_tend"], t["nc_accret_tend"], t["qc2qr_autoconv_tend"],
        t["nc2nr_autoconv_tend"], t["ncautr"], t["nc_selfcollect_tend"], t["qr2qv_evap_tend"],
        t["nr_evap_tend"], t["nr_selfcollect_tend"], predictNc, do_prescribed_CCN, s["inv_rho"],
        dgn["inv_exner"], dt, s["th_atm"], s["qv"], s["qc"], s["nc"], s["qr"], s["nr"],
        not_skip_all)

    d = {}
    if ro.use_hetfrz_classnuc:
        depos = (t["qv2qi_vapdep_tend"] - t["qi2qv_sublim_tend"] + t["qv2qi_nucleat_tend"]
                 + t["qinuc_cnt"])
    else:
        depos = t["qv2qi_vapdep_tend"] - t["qi2qv_sublim_tend"] + t["qv2qi_nucleat_tend"]
    d["qv2qi_depos_tend"] = F._set(s["qv2qi_depos_tend"], not_skip_all, depos)
    d["precip_total_tend"] = F._set(s["precip_total_tend"], not_skip_all,
                                    t["qc2qr_accret_tend"] + t["qc2qr_autoconv_tend"]
                                    + t["qc2qr_ice_shed_tend"] + t["qc2qi_collect_tend"])
    d["nevapr"] = F._set(s["nevapr"], not_skip_all, t["qi2qv_sublim_tend"] + t["qr2qv_evap_tend"])
    d["qr_evap_tend"] = F._set(s["qr_evap_tend"], not_skip_all, t["qr2qv_evap_tend"])
    d["vap_ice_exchange"] = F._set(s["vap_ice_exchange"], not_skip_all,
                                   t["qv2qi_vapdep_tend"] - t["qi2qv_sublim_tend"]
                                   + t["qv2qi_nucleat_tend"])
    d["vap_liq_exchange"] = F._set(s["vap_liq_exchange"], not_skip_all, -t["qr2qv_evap_tend"])
    d["liq_ice_exchange"] = F._set(s["liq_ice_exchange"], not_skip_all,
                                   t["qc2qi_hetero_freeze_tend"] + t["qr2qi_immers_freeze_tend"]
                                   - t["qi2qr_melt_tend"] + t["qc2qi_berg_tend"]
                                   + t["qc2qi_collect_tend"] + t["qr2qi_collect_tend"])
    if ro.extra_p3_diags:
        for name in ("qr2qv_evap", "qi2qv_sublim", "qc2qr_accret", "qc2qr_autoconv",
                     "qv2qi_vapdep", "qc2qi_berg", "qc2qr_ice_shed", "qc2qi_collect",
                     "qr2qi_collect", "qc2qi_hetero_freeze", "qr2qi_immers_freeze", "qi2qr_melt"):
            d[name] = F._set(s[name], not_skip_all, t[name + "_tend"])
    s.update(d)

    qc_small = (s["qc"] < qsmall) & not_skip_all
    qr_small = (s["qr"] < qsmall) & not_skip_all
    qi_small = (s["qi"] < qsmall) & not_skip_all
    hydrometeorsPresent = (((s["qc"] >= qsmall) & not_skip_all).any(dim=-1)
                           | ((s["qr"] >= qsmall) & not_skip_all).any(dim=-1)
                           | ((s["qi"] >= qsmall) & not_skip_all).any(dim=-1))
    s["qv"] = F._set(s["qv"], qc_small, s["qv"] + s["qc"])
    s["th_atm"] = F._set(s["th_atm"], qc_small,
                         s["th_atm"] - dgn["inv_exner"] * s["qc"] * C.LatVap * C.INV_CP)
    s["qc"] = F._set(s["qc"], qc_small, 0)
    s["nc"] = F._set(s["nc"], qc_small, 0)
    s["qv"] = F._set(s["qv"], qr_small, s["qv"] + s["qr"])
    s["th_atm"] = F._set(s["th_atm"], qr_small,
                         s["th_atm"] - dgn["inv_exner"] * s["qr"] * C.LatVap * C.INV_CP)
    s["qr"] = F._set(s["qr"], qr_small, 0)
    s["nr"] = F._set(s["nr"], qr_small, 0)
    s["qv"] = F._set(s["qv"], qi_small, s["qv"] + s["qi"])
    s["th_atm"] = F._set(s["th_atm"], qi_small,
                         s["th_atm"] - dgn["inv_exner"] * s["qi"] * (C.LatVap + C.LatIce) * C.INV_CP)
    s["qi"] = F._set(s["qi"], qi_small, 0)
    s["ni"] = F._set(s["ni"], qi_small, 0)
    s["qm"] = F._set(s["qm"], qi_small, 0)
    s["bm"] = F._set(s["bm"], qi_small, 0)
    s["pratot"] = F._set(s["pratot"], not_skip_all, t["qc2qr_accret_tend"])
    s["prctot"] = F._set(s["prctot"], not_skip_all, t["qc2qr_autoconv_tend"])
    s["ni_incld"] = F._set(s["ni_incld"], not_skip_all, s["ni"] / dgn["cld_frac_i"])
    s["ni_incld"] = F.impose_max_total_ni(s["ni_incld"], ro.max_total_ni, s["inv_rho"], not_skip_all)
    s["ni"] = F._set(s["ni"], not_skip_all, s["ni_incld"] * dgn["cld_frac_i"])
    (s["qc_incld"], s["qr_incld"], s["qi_incld"], s["qm_incld"], s["nc_incld"], s["nr_incld"],
     s["ni_incld"], s["bm_incld"]) = F.calculate_incloud_mixingratios(
        s["qc"], s["qr"], s["qi"], s["qm"], s["nc"], s["nr"], s["ni"], s["bm"],
        s["inv_cld_frac_l"], s["inv_cld_frac_i"], s["inv_cld_frac_r"], s["qc_incld"],
        s["qr_incld"], s["qi_incld"], s["qm_incld"], s["nc_incld"], s["nr_incld"],
        s["ni_incld"], s["bm_incld"], not_skip_all)
    return s, hydrometeorsPresent


def p3_main_part3(ro: P3Runtime, tables: P3Tables, s: dict, dgn: dict):
    """p3_main_impl.hpp part3."""
    s = dict(s)
    qsmall, nsmall = C.QSMALL, C.NSMALL
    zero = torch.zeros_like(s["qc"])
    one = torch.ones_like(zero)
    # cloud liquid
    qc_gt_small = s["qc"] >= qsmall
    qc_small = ~qc_gt_small
    qc_incld = s["qc"] / dgn["cld_frac_l"]
    nc_incld = s["nc"] / dgn["cld_frac_l"]
    nc_incld, s["mu_c"], s["nu"], s["lamc"], _i1, _i2 = get_cloud_dsd2(
        qc_incld, nc_incld, s["mu_c"], s["rho"], s["nu"], tables.dnu_table_vals, s["lamc"],
        zero, zero, qc_gt_small)
    s["nc"] = F._set(s["nc"], qc_gt_small, nc_incld * dgn["cld_frac_l"])
    s["diag_eff_radius_qc"] = F._set(s["diag_eff_radius_qc"], qc_gt_small,
                                     0.5 * (s["mu_c"] + 3) / torch.where(qc_gt_small, s["lamc"], torch.ones_like(s["lamc"])))
    s["qv"] = F._set(s["qv"], qc_small, s["qv"] + s["qc"])
    s["th_atm"] = F._set(s["th_atm"], qc_small,
                         s["th_atm"] - dgn["inv_exner"] * s["qc"] * C.LatVap * C.INV_CP)
    s["vap_liq_exchange"] = F._set(s["vap_liq_exchange"], qc_small,
                                   s["vap_liq_exchange"] - s["qc"])
    s["qc"] = F._set(s["qc"], qc_small, 0)
    s["nc"] = F._set(s["nc"], qc_small, 0)
    # rain
    qr_gt_small = s["qr"] >= qsmall
    qr_small = ~qr_gt_small
    qr_incld = s["qr"] / dgn["cld_frac_r"]
    nr_incld = s["nr"] / dgn["cld_frac_r"]
    nr_incld, s["mu_r"], s["lamr"] = get_rain_dsd2(qr_incld, nr_incld, s["mu_r"], s["lamr"],
                                                   ro.constant_mu_rain, qr_gt_small)
    s["nr"] = F._set(s["nr"], qr_gt_small, nr_incld * dgn["cld_frac_r"])
    mu_r, lamr = s["mu_r"], s["lamr"]
    ze_rain_v = (s["nr"] * (mu_r + 6) * (mu_r + 5) * (mu_r + 4) * (mu_r + 3) * (mu_r + 2)
                 * (mu_r + 1) / mathfn.pow(torch.where(qr_gt_small, lamr, one), 6.0))
    s["ze_rain"] = F._set(s["ze_rain"], qr_gt_small, ze_rain_v)
    s["ze_rain"] = F._set(s["ze_rain"], qr_gt_small, max_(s["ze_rain"], 1.e-22))
    s["diag_eff_radius_qr"] = F._set(s["diag_eff_radius_qr"], qr_gt_small, rdiv(1.5, torch.where(qr_gt_small, lamr, one)))
    s["qv"] = F._set(s["qv"], qr_small, s["qv"] + s["qr"])
    s["th_atm"] = F._set(s["th_atm"], qr_small,
                         s["th_atm"] - dgn["inv_exner"] * s["qr"] * C.LatVap * C.INV_CP)
    s["vap_liq_exchange"] = F._set(s["vap_liq_exchange"], qr_small, s["vap_liq_exchange"] - s["qr"])
    s["qr"] = F._set(s["qr"], qr_small, 0)
    s["nr"] = F._set(s["nr"], qr_small, 0)
    # ice
    qi_gt_small = s["qi"] >= qsmall
    qi_small = ~qi_gt_small
    s["ni"] = max_(s["ni"], nsmall)
    qi_incld = s["qi"] / dgn["cld_frac_i"]
    ni_incld = s["ni"] / dgn["cld_frac_i"]
    qm_incld = s["qm"] / dgn["cld_frac_i"]
    bm_incld = s["bm"] / dgn["cld_frac_i"]
    rhop, qm_incld, bm_incld = S.calc_bulk_rho_rime(qi_incld, qm_incld, bm_incld, ro, qi_gt_small)
    s["qm"] = F._set(s["qm"], qi_gt_small, qm_incld * dgn["cld_frac_i"])
    s["bm"] = F._set(s["bm"], qi_gt_small, bm_incld * dgn["cld_frac_i"])
    ni_incld = F.impose_max_total_ni(ni_incld, ro.max_total_ni, s["inv_rho"])
    ti = lookup_ice(torch.where(qi_gt_small, qi_incld, one), torch.where(qi_gt_small, ni_incld, one),
                    torch.where(qi_gt_small, qm_incld, zero), torch.where(qi_gt_small, rhop, one),
                    qi_gt_small)
    tvals = {}
    for name, idx in (("qi_fallspd", 1), ("ice_eff_radius", 5), ("ni_lammax", 6),
                      ("ni_lammin", 7), ("ice_reflectivity", 8), ("ice_mean_diam", 10),
                      ("ice_bulk_dens", 11)):
        tvals[name] = F._set(zero, qi_gt_small, apply_table_ice(idx, tables.ice_table_vals, ti))
    ni_incld = F._set(ni_incld, qi_gt_small, min_(ni_incld, tvals["ni_lammax"] * ni_incld))
    ni_incld = F._set(ni_incld, qi_gt_small, max_(ni_incld, tvals["ni_lammin"] * ni_incld))
    s["ni"] = ni_incld * dgn["cld_frac_i"]
    qm_small = (s["qm"] < qsmall) & qi_gt_small
    s["qm"] = F._set(s["qm"], qm_small, 0)
    s["bm"] = F._set(s["bm"], qm_small, 0)
    s["diag_vm_qi"] = F._set(s["diag_vm_qi"], qi_gt_small, tvals["qi_fallspd"] * s["rhofaci"])
    s["diag_eff_radius_qi"] = F._set(s["diag_eff_radius_qi"], qi_gt_small, tvals["ice_eff_radius"])
    s["diag_diam_qi"] = F._set(s["diag_diam_qi"], qi_gt_small, tvals["ice_mean_diam"])
    s["rho_qi"] = F._set(s["rho_qi"], qi_gt_small, tvals["ice_bulk_dens"])
    s["ze_ice"] = F._set(s["ze_ice"], qi_gt_small,
                         s["ze_ice"] + 0.1892 * tvals["ice_reflectivity"] * ni_incld * s["rho"])
    s["ze_ice"] = F._set(s["ze_ice"], qi_gt_small, max_(s["ze_ice"], 1.e-22))
    s["ze_ice"] = F._set(s["ze_ice"], qi_gt_small, s["ze_ice"] * dgn["cld_frac_i"])
    s["qv"] = F._set(s["qv"], qi_small, s["qv"] + s["qi"])
    s["th_atm"] = F._set(s["th_atm"], qi_small,
                         s["th_atm"] - dgn["inv_exner"] * s["qi"] * (C.LatVap + C.LatIce) * C.INV_CP)
    s["qi"] = F._set(s["qi"], qi_small, 0)
    s["ni"] = F._set(s["ni"], qi_small, 0)
    s["qm"] = F._set(s["qm"], qi_small, 0)
    s["bm"] = F._set(s["bm"], qi_small, 0)
    s["diag_diam_qi"] = F._set(s["diag_diam_qi"], qi_small, 0)
    s["diag_equiv_reflectivity"] = 10 * mathfn.log10((s["ze_rain"] + s["ze_ice"]) * 1.e18)
    s["nr"] = F._set(s["nr"], s["qr"] < qsmall, 0)
    return s


def _select(old: dict, new: dict, col_mask: torch.Tensor) -> dict:
    """Keep `old` in columns where col_mask is false: the reference's early `return`."""
    out = dict(old)
    for k, v in new.items():
        if not isinstance(v, torch.Tensor):
            out[k] = v
            continue
        o = old.get(k)
        if not isinstance(o, torch.Tensor) or o.shape != v.shape:
            out[k] = v
            continue
        m = col_mask if v.dim() == 1 else col_mask[:, None]
        out[k] = torch.where(m, v, o)
    return out


def p3_main(ro: P3Runtime, tables: P3Tables, state: dict, dgn: dict, dt: float, predictNc: bool,
            prescribedCCN: bool, max_substeps: int = S.MAX_SUBSTEPS):
    """p3_main_impl.hpp: p3_main_internal.

    state: qv, th_atm, qc, nc, qr, nr, qi, ni, qm, bm  (ncol, nlev), dry mixing ratios.
    dgn: pres, dpres, dz, nc_nuceat_tend, nccn, ni_activated, inv_qc_relvar, cld_frac_i,
         cld_frac_l, cld_frac_r, inv_exner, qv_prev, t_prev (+ hetfrz_* if used).
    Returns (state, diagnostics).
    """
    inv_dt = 1 / dt
    s = p3_main_init(dgn["cld_frac_i"], dgn["cld_frac_l"], dgn["cld_frac_r"], dgn["inv_exner"],
                     state["th_atm"], dgn["dz"], state["qv"])
    for k in ("th_atm", "qc", "nc", "qr", "nr", "qi", "ni", "qm", "bm"):
        s[k] = state[k]
    for k in ("liq_ice_exchange", "vap_liq_exchange", "vap_ice_exchange"):
        s[k] = state.get(k, torch.zeros_like(state["qc"]))
    if ro.extra_p3_diags:
        for k in ("qr2qv_evap", "qi2qv_sublim", "qc2qr_accret", "qc2qr_autoconv", "qv2qi_vapdep",
                  "qc2qi_berg", "qc2qr_ice_shed", "qc2qi_collect", "qr2qi_collect",
                  "qc2qi_hetero_freeze", "qr2qi_immers_freeze", "qi2qr_melt", "qr_sed", "qc_sed",
                  "qi_sed"):
            s[k] = state.get(k, torch.zeros_like(state["qc"]))

    p1 = p3_main_part1(predictNc, prescribedCCN, dt, dgn["pres"], dgn["dpres"], dgn["dz"],
                       dgn["nc_nuceat_tend"], dgn["nccn"], dgn["inv_exner"], s["exner"],
                       s["inv_cld_frac_l"], s["inv_cld_frac_i"], s["inv_cld_frac_r"], s["qv"],
                       s["th_atm"], s["qc"], s["nc"], s["qr"], s["nr"], s["qi"], s["ni"],
                       s["qm"], s["bm"], ro)
    nucleationPossible = p1.pop("nucleationPossible")
    hydrometeorsPresent = p1.pop("hydrometeorsPresent")
    s.update(p1)
    col_active = nucleationPossible | hydrometeorsPresent

    s2, hyd2 = p3_main_part2(ro, predictNc, prescribedCCN, dt, inv_dt, tables, s, dgn)
    s = _select(s, s2, col_active)
    hyd2 = hyd2 & col_active
    sed_active = col_active & hyd2

    s3 = dict(s)
    (s3["qc"], s3["nc"], s3["nc_incld"], s3["mu_c"], s3["lamc"], s3["qc_incld"], qc_sed,
     nc_tend, s3["precip_liq_surf"], cap_c) = S.cloud_sedimentation(
        s["qc_incld"], s["rho"], s["inv_rho"], dgn["cld_frac_l"], s["acn"], s["inv_dz"],
        tables.dnu_table_vals, dt, inv_dt, predictNc, s["qc"], s["nc"], s["nc_incld"],
        s["mu_c"], s["lamc"], s.get("qc_sed", s["qtend_ignore"]), s["ntend_ignore"],
        s["precip_liq_surf"], max_substeps)
    (s3["qr"], s3["nr"], s3["nr_incld"], s3["mu_r"], s3["lamr"], s3["qr_incld"],
     s3["precip_liq_flux"], qr_sed, nr_tend, s3["precip_liq_surf"], cap_r) = S.rain_sedimentation(
        s["rho"], s["inv_rho"], s["rhofacr"], dgn["cld_frac_r"], s["inv_dz"], s["qr_incld"],
        tables.vn_table_vals, tables.vm_table_vals, dt, inv_dt,
        s3["qr"], s3["nr"], s["nr_incld"], s["mu_r"], s["lamr"], s3["precip_liq_flux"],
        s.get("qr_sed", s["qtend_ignore"]), s["ntend_ignore"], s3["precip_liq_surf"], ro,
        max_substeps)
    (s3["qi"], s3["qi_incld"], s3["ni"], s3["ni_incld"], s3["qm"], s3["qm_incld"], s3["bm"],
     s3["bm_incld"], qi_sed, ni_tend, s3["precip_ice_surf"], cap_i) = S.ice_sedimentation(
        s["rho"], s["inv_rho"], s["rhofaci"], dgn["cld_frac_i"], s["inv_dz"], dt, inv_dt,
        s3["qi"], s["qi_incld"], s3["ni"], s["ni_incld"], s3["qm"], s["qm_incld"], s3["bm"],
        s["bm_incld"], s.get("qi_sed", s["qtend_ignore"]), s["ntend_ignore"],
        tables.ice_table_vals, s["precip_ice_surf"], ro, max_substeps)
    if ro.extra_p3_diags:
        s3["qc_sed"], s3["qr_sed"], s3["qi_sed"] = qc_sed, qr_sed, qi_sed
    if ro.do_ice_production:
        (s3["qc"], s3["nc"], s3["qr"], s3["nr"], s3["qi"], s3["ni"], s3["qm"], s3["bm"],
         s3["th_atm"]) = S.homogeneous_freezing(
            s3["T_atm"], dgn["inv_exner"], s3["qc"], s3["nc"], s3["qr"], s3["nr"], s3["qi"],
            s3["ni"], s3["qm"], s3["bm"], s3["th_atm"])
    s3 = p3_main_part3(ro, tables, s3, dgn)
    hit_cap = cap_c | cap_r | cap_i
    s = _select(s, s3, sed_active)
    s["hit_substep_cap"] = hit_cap & sed_active
    return s
