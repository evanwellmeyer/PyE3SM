"""P3 atmosphere process: pre-processing, p3_main, post-processing, property checks.

Translation of components/eamxx/src/physics/p3/eamxx_p3_process_interface.{hpp,cpp}
and eamxx_p3_run.cpp at E3SM 80fe0a71 (p3_preamble, run_impl, p3_postamble and
the checks registered in initialize_impl).

P3 works in dry mixing ratios: the pre-process converts the host's wet mixing
ratios with the pressure-thickness ratio, and the post-process converts back.
Its pressure inputs are the dry ones (p_dry_mid, pseudo_density_dry), while dz
comes from the wet state.
"""
from __future__ import annotations

import torch

from .. import constants as C
from .. import physics_functions as PF
from ..ekat import max_, rdiv
from ..property_checks import S_MAX, within_interval
from .main import p3_main
from .runtime import P3Runtime
from .tables import P3Tables

MINCLD = 0.0001  # p3_preamble::mincld

WET_TO_DRY = ("qc", "nc", "qr", "nr", "qi", "ni", "qm", "bm", "qv")


def preprocess(T_mid, p_mid, p_dry_mid, pseudo_density, pseudo_density_dry, qv, qc, nc, qr, nr,
               qi, ni, qm, bm, qv_prev, cld_frac_t_in, cld_frac_l_in, cld_frac_i_in,
               ro: P3Runtime):
    """p3_preamble::operator(). Returns (state, diagnostic inputs)."""
    dz = PF.calculate_dz(pseudo_density, p_mid, T_mid, qv)
    state = dict(qc=qc, nc=nc, qr=qr, nr=nr, qi=qi, ni=ni, qm=qm, bm=bm, qv=qv)
    for k in WET_TO_DRY:
        state[k] = PF.calculate_drymmr_from_wetmmr_dp_based(state[k], pseudo_density,
                                                            pseudo_density_dry)
    qv_prev = PF.calculate_drymmr_from_wetmmr_dp_based(qv_prev, pseudo_density, pseudo_density_dry)
    exner = PF.exner_function(p_mid)
    inv_exner = rdiv(1.0, exner)
    state["th_atm"] = PF.calculate_theta_from_T(T_mid, p_mid)

    one = torch.ones_like(T_mid)
    if ro.use_separate_ice_liq_frac:
        cld_frac_l = one if ro.set_cld_frac_l_to_one else max_(cld_frac_l_in, MINCLD)
        cld_frac_i = one if ro.set_cld_frac_i_to_one else max_(cld_frac_i_in, MINCLD)
    else:
        cld_frac_l = one if ro.set_cld_frac_l_to_one else max_(cld_frac_t_in, MINCLD)
        cld_frac_i = one if ro.set_cld_frac_i_to_one else max_(cld_frac_t_in, MINCLD)
    if ro.set_cld_frac_r_to_one:
        cld_frac_r = one
    else:
        cld_frac_r = max_(cld_frac_t_in, MINCLD)
        # levels 1..nlev-1: use the level above if its total cloud fraction is larger
        above = torch.cat([cld_frac_t_in[:, :1], cld_frac_t_in[:, :-1]], dim=-1)
        idx = torch.arange(T_mid.shape[-1], device=T_mid.device)
        active = idx > 0
        cld_frac_r = torch.where(active & (above > cld_frac_r), above, cld_frac_r)
    dgn = dict(pres=p_dry_mid, dpres=pseudo_density_dry, dz=dz, inv_exner=inv_exner,
               cld_frac_l=cld_frac_l, cld_frac_i=cld_frac_i, cld_frac_r=cld_frac_r,
               qv_prev=qv_prev)
    return state, dgn, exner


def postprocess(T_mid_in, th_atm, p_mid, pseudo_density, pseudo_density_dry, state: dict,
                diag_eff_radius_qc, diag_eff_radius_qi, diag_eff_radius_qr, precip_liq_surf_flux,
                precip_ice_surf_flux, precip_liq_surf_mass, precip_ice_surf_mass, dt):
    """p3_postamble::operator()."""
    T_atm = ((PF.calculate_T_from_theta(th_atm, p_mid) - T_mid_in)
             * pseudo_density_dry / pseudo_density)
    T_atm = T_atm + T_mid_in
    out = {}
    for k in WET_TO_DRY:
        out[k] = PF.calculate_wetmmr_from_drymmr_dp_based(state[k], pseudo_density,
                                                          pseudo_density_dry)
    out["T_mid"] = T_atm
    out["T_prev_micro_step"] = T_atm
    out["qv_prev_micro_step"] = out["qv"]
    out["eff_radius_qc"] = diag_eff_radius_qc * 1e6
    out["eff_radius_qi"] = diag_eff_radius_qi * 1e6
    out["eff_radius_qr"] = diag_eff_radius_qr * 1e6
    out["precip_liq_surf_mass"] = precip_liq_surf_mass + precip_liq_surf_flux * C.RHO_H2O * dt
    out["precip_ice_surf_mass"] = precip_ice_surf_mass + precip_ice_surf_flux * C.RHO_H2O * dt
    return out


def invariant_checks(fields: dict) -> tuple[dict, dict]:
    """The two invariant checks of initialize_impl (run before and after run_impl)."""
    out = dict(fields)
    status = {}
    out["T_mid"], status["T_mid"] = within_interval(out["T_mid"], 100.0, 500.0, False)
    out["qv"], status["qv"] = within_interval(out["qv"], 1e-13, 0.2, True)
    return out, status


def postconditions(fields: dict) -> tuple[dict, dict]:
    """The post-condition checks of initialize_impl, in registration order."""
    out = dict(fields)
    status = {}
    specs = [("qc", 0.0, 0.1, False), ("qi", 0.0, 0.1, False), ("qr", 0.0, 0.1, False),
             ("qm", 0.0, 0.1, False), ("nc", 0.0, 1.e11, False), ("nr", 0.0, 1.e10, False),
             ("ni", 0.0, 1.e10, False), ("bm", 0.0, 1.0, False),
             ("precip_liq_surf_mass", 0.0, S_MAX, False),
             ("precip_ice_surf_mass", 0.0, S_MAX, False),
             ("eff_radius_qc", 0.0, 1.0e2, False), ("eff_radius_qi", 0.0, 5.0e3, False),
             ("eff_radius_qr", 0.0, 5.0e3, False)]
    for name, lb, ub, rep in specs:
        if name in out:
            out[name], status[name] = within_interval(out[name], lb, ub, rep)
    return out, status


def p3_run(state: dict, dt: float, tables: P3Tables, ro: P3Runtime = P3Runtime(),
           predictNc: bool = True, prescribedCCN: bool = True, max_substeps: int | None = None):
    """P3Microphysics::run_impl on EAMxx-named fields.

    state: T_mid, p_mid, p_dry_mid, pseudo_density, pseudo_density_dry, qv, qc, nc, qr, nr,
      qi, ni, qm, bm, nc_nuceat_tend, nccn, ni_activated, inv_qc_relvar, cldfrac_tot,
      qv_prev_micro_step, T_prev_micro_step, precip_liq_surf_mass, precip_ice_surf_mass
      (plus cldfrac_liq / cldfrac_ice if use_separate_ice_liq_frac).
    Returns the updated/computed fields under their EAMxx names (before the checks).
    """
    s = state
    if tables.vn_table_vals.dtype != s["T_mid"].dtype:
        raise ValueError(f"P3 tables are {tables.vn_table_vals.dtype}, the state is {s['T_mid'].dtype}: "
                         "load them with load_p3_tables(dtype=...) (single precision reads *.dat4)")
    if tables.vn_table_vals.device != s["T_mid"].device:
        from ..precision import cast
        tables = cast(tables, s["T_mid"].dtype, s["T_mid"].device)
    kwargs = {} if max_substeps is None else {"max_substeps": max_substeps}
    pstate, dgn, _exner = preprocess(
        s["T_mid"], s["p_mid"], s["p_dry_mid"], s["pseudo_density"], s["pseudo_density_dry"],
        s["qv"], s["qc"], s["nc"], s["qr"], s["nr"], s["qi"], s["ni"], s["qm"], s["bm"],
        s["qv_prev_micro_step"], s["cldfrac_tot"], s.get("cldfrac_liq"), s.get("cldfrac_ice"), ro)
    dgn.update(nc_nuceat_tend=s["nc_nuceat_tend"], nccn=s.get("nccn"),
               ni_activated=s["ni_activated"], inv_qc_relvar=s["inv_qc_relvar"],
               t_prev=s["T_prev_micro_step"])
    if ro.use_hetfrz_classnuc:
        for k in ("hetfrz_immersion_nucleation_tend", "hetfrz_contact_nucleation_tend",
                  "hetfrz_deposition_nucleation_tend"):
            dgn[k] = s[k]
    # run_impl zeroes the three exchange diagnostics before p3_main
    for k in ("liq_ice_exchange", "vap_liq_exchange", "vap_ice_exchange"):
        pstate[k] = torch.zeros_like(s["qc"])
    m = p3_main(ro, tables, pstate, dgn, dt, predictNc, prescribedCCN, **kwargs)

    out = postprocess(s["T_mid"], m["th_atm"], s["p_mid"], s["pseudo_density"],
                      s["pseudo_density_dry"], m, m["diag_eff_radius_qc"],
                      m["diag_eff_radius_qi"], m["diag_eff_radius_qr"], m["precip_liq_surf"],
                      m["precip_ice_surf"], s["precip_liq_surf_mass"], s["precip_ice_surf_mass"],
                      dt)
    out.update(
        micro_liq_ice_exchange=m["liq_ice_exchange"], micro_vap_liq_exchange=m["vap_liq_exchange"],
        micro_vap_ice_exchange=m["vap_ice_exchange"], rainfrac=dgn["cld_frac_r"],
        precip_total_tend=m["precip_total_tend"], nevapr=m["nevapr"],
        diag_equiv_reflectivity=m["diag_equiv_reflectivity"],
    )
    if ro.extra_p3_diags:
        for k in ("qr2qv_evap", "qi2qv_sublim", "qc2qr_accret", "qc2qr_autoconv", "qv2qi_vapdep",
                  "qc2qi_berg", "qc2qr_ice_shed", "qc2qi_collect", "qr2qi_collect",
                  "qc2qi_hetero_freeze", "qr2qi_immers_freeze", "qi2qr_melt", "qr_sed", "qc_sed",
                  "qi_sed"):
            out[k] = m[k]
    out["_internal"] = dict(main=m, dgn=dgn, pre=pstate)
    return out
