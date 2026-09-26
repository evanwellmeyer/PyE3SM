"""SHOC atmosphere process: pre-processing, TMS coupling, shoc_main, post-processing, checks.

Translation of components/eamxx/src/physics/shoc/eamxx_shoc_process_interface.{hpp,cpp}
at E3SM 80fe0a71 (SHOCPreprocess, SHOCMacrophysics::run_impl,
apply_turbulent_mountain_stress, SHOCPostprocess, and the post-condition checks
added in initialize_impl).

Not ported (off in the standard configuration; see docs/plan.md):
  * do_3d_turbulence_shoc (horizontal eddy diffusivities for the dycore),
  * check_flux_state_consistency (namelist default false),
  * column conservation checks (enable_column_conservation_checks false).
"""
from __future__ import annotations

import torch

from .. import constants as C
from .. import physics_functions as PF
from ..ekat import max_, min_, rdiv
from ..property_checks import S_MAX, within_interval
from . import functions as F
from .constants import SC
from .main import SHOCRuntime, shoc_init, shoc_main

__all__ = ["SHOCRuntime", "shoc_init", "npbl_from_hybrid", "preprocess", "apply_turbulent_mountain_stress",
           "postprocess", "postconditions", "shoc_run"]


def npbl_from_hybrid(hyam, hybm) -> int:
    """initialize_impl: pref_mid = ps0*hyam + psref*hybm, then shoc_init(nlev, 0, pref_mid)."""
    ps0 = psref = C.P0
    pref_mid = [ps0 * float(a) + psref * float(b) for a, b in zip(hyam, hybm)]
    return shoc_init(len(pref_mid), 0, pref_mid)


def preprocess(T_mid, p_mid, p_int, pseudo_density, omega, phis, surf_sens_flux, surf_evap,
               surf_mom_flux, qv, qc, tke, um_pert_diff, vm_pert_diff, z_surf: float = 0.0):
    """SHOCPreprocess::operator(). Returns a dict of the preprocessed variables.

    `tke` here is the tracer field; the returned "tke" is its clipped value
    (the preprocess writes it back into the tracer group).
    """
    zvir, cpair, ggr = C.ZVIR, C.Cpair, C.gravit
    inv_ggr = 1 / ggr
    mintke = SC.mintke
    nlev = T_mid.shape[-1]

    exner = PF.exner_function(p_mid)
    # Reference: inv_exner.set(exner != 0, 1/exner); where exner == 0 the buffer keeps
    # stale data (and debug builds abort). The port divides unconditionally.
    inv_exner = rdiv(1.0, exner)
    tke = max_(mintke, tke)
    tke_copy = tke
    qc_copy = qc
    qw = qv + qc
    theta_zt = PF.calculate_theta_from_T(T_mid, p_mid)
    thlm = PF.calculate_thetal_from_theta(theta_zt, T_mid, qc)
    thv = theta_zt * (1 + zvir * qv - qc)
    dz = PF.calculate_dz(pseudo_density, p_mid, T_mid, qv)
    rrho = inv_ggr * (pseudo_density / dz)
    wm_zt = -1 * omega / (rrho * ggr)
    um_pert = um_pert_diff
    vm_pert = vm_pert_diff

    z_int = PF.calculate_z_int(dz, z_surf)
    z_mid = PF.calculate_z_mid(z_int)
    z_sfc = z_int[:, nlev:nlev + 1]
    zt_grid = z_mid - z_sfc
    zi_grid = torch.cat([z_int[:, :nlev] - z_sfc, torch.zeros_like(z_sfc)], dim=-1)
    shoc_s = PF.calculate_dse(T_mid, z_mid, phis[:, None])

    rrho_i = F.linear_interp(zt_grid, zi_grid, rrho, nlev, nlev + 1, 0.0)
    rrho_i_sfc = rrho_i[:, nlev]
    exner_int = PF.exner_function(p_int[:, nlev])
    inv_exner_int_surf = rdiv(1.0, exner_int)

    wpthlp_sfc = (surf_sens_flux / (cpair * rrho_i_sfc)) * inv_exner_int_surf
    wprtp_sfc = surf_evap / rrho_i_sfc
    taux, tauy = surf_mom_flux[:, 0], surf_mom_flux[:, 1]
    upwp_sfc = taux / rrho_i_sfc
    vpwp_sfc = tauy / rrho_i_sfc

    stress_is_small = (torch.abs(taux) < 1.e-12) & (torch.abs(tauy) < 1.e-12)
    tau_mag = torch.sqrt(torch.where(stress_is_small, torch.ones_like(taux), taux * taux + tauy * tauy))
    # Masked-out branch uses 1 so it stays finite (no NaN in gradients).
    pert_scale_fac = rdiv(C.tau_pert_mag, torch.where(stress_is_small, torch.ones_like(tau_mag), tau_mag))
    upwp_sfc_pert = torch.where(stress_is_small, rdiv(C.tau_pert_mag, rrho_i_sfc), upwp_sfc * pert_scale_fac)
    vpwp_sfc_pert = torch.where(stress_is_small, torch.zeros_like(vpwp_sfc), vpwp_sfc * pert_scale_fac)

    return dict(inv_exner=inv_exner, tke=tke, tke_copy=tke_copy, qc_copy=qc_copy, qw=qw, thlm=thlm,
                thv=thv, dz=dz, rrho=rrho, wm_zt=wm_zt, um_pert=um_pert, vm_pert=vm_pert,
                z_int=z_int, z_mid=z_mid, zt_grid=zt_grid, zi_grid=zi_grid, shoc_s=shoc_s,
                rrho_i=rrho_i, wpthlp_sfc=wpthlp_sfc, wprtp_sfc=wprtp_sfc, upwp_sfc=upwp_sfc,
                vpwp_sfc=vpwp_sfc, upwp_sfc_pert=upwp_sfc_pert, vpwp_sfc_pert=vpwp_sfc_pert)


def apply_turbulent_mountain_stress(upwp_sfc, vpwp_sfc, surf_drag_coeff_tms, horiz_winds, rrho_i):
    """SHOCMacrophysics::apply_turbulent_mountain_stress."""
    rho_sfc = rrho_i[:, -1]
    upwp_sfc = upwp_sfc - surf_drag_coeff_tms * horiz_winds[:, 0, -1] / rho_sfc
    vpwp_sfc = vpwp_sfc - surf_drag_coeff_tms * horiz_winds[:, 1, -1] / rho_sfc
    return upwp_sfc, vpwp_sfc


def postprocess(qw, qc_copy, tke_copy, cldfrac_liq, shoc_ql2, dse, z_mid, phis, surf_mom_flux,
                um_pert, vm_pert):
    """SHOCPostprocess::operator() (compute_mass_and_energy_fluxes off)."""
    inv_qc_relvar_max, inv_qc_relvar_min = 10.0, 0.001
    tke = tke_copy
    qc = qc_copy
    qv = qw - qc
    cldfrac_liq = min_(cldfrac_liq, 1.0)
    condition = (qc != 0) & (shoc_ql2 != 0)
    # Masked-out branch uses 1 for the denominator so it stays finite.
    qc2_safe = torch.where(condition, shoc_ql2, torch.ones_like(shoc_ql2))
    relvar = min_(inv_qc_relvar_max, max_(inv_qc_relvar_min, (qc * qc) / qc2_safe))
    inv_qc_relvar = torch.where(condition, relvar, torch.ones_like(relvar))
    T_mid = PF.calculate_temperature_from_dse(dse, z_mid, phis[:, None])
    tau_est = F._sqrt0(surf_mom_flux[:, 0] * surf_mom_flux[:, 0] + surf_mom_flux[:, 1] * surf_mom_flux[:, 1])
    return dict(tke=tke, qc=qc, qv=qv, cldfrac_liq=cldfrac_liq, inv_qc_relvar=inv_qc_relvar,
                T_mid=T_mid, um_pert_diff=um_pert, vm_pert_diff=vm_pert, tau_est=tau_est)


def postconditions(out: dict) -> tuple[dict, dict]:
    """Post-condition checks registered in initialize_impl, in registration order.

    Returns (fields after any repair, {check name: status}).
    """
    out = dict(out)
    status = {}
    specs = [  # (field, lb, ub, can_repair)
        ("T_mid", 100.0, 500.0, False),
        ("qc", 0.0, 0.1, False),
        ("horiz_winds", -400.0, 400.0, False),
        ("pbl_height", 0.0, S_MAX, False),
        ("cldfrac_liq", 0.0, 1.0, False),
        ("tke", 0.0, S_MAX, False),
        ("tau_est", 0.0, S_MAX, False),
        ("qv", 0.0, 0.2, True),
    ]
    for name, lb, ub, rep in specs:
        out[name], status[name] = within_interval(out[name], lb, ub, rep)
    return out, status


def shoc_run(state: dict, dt: float, cell_length, npbl: int, rt: SHOCRuntime = SHOCRuntime(),
             apply_tms: bool = True, tracer_names: tuple[str, ...] = ()) -> dict:
    """SHOCMacrophysics::run_impl on EAMxx-named fields.

    state: EAMxx fields (ncol, ...) as tensors: T_mid, p_mid, p_int, pseudo_density,
      omega, phis, surf_sens_flux, surf_evap, surf_mom_flux (ncol,2), horiz_winds
      (ncol,2,nlev), qv, qc, tke, cldfrac_liq, sgs_buoy_flux, eddy_diff_mom,
      um_pert_diff, vm_pert_diff, surf_drag_coeff_tms (if apply_tms), plus the other
      members of turbulence_advected_tracers named in tracer_names.
    cell_length: (ncol,) dx = dy [m].
    Returns the updated/computed fields under their EAMxx names (before property checks).
    """
    if dt > 300:
        raise ValueError("SHOC is intended to run with a timestep no longer than 5 minutes.")
    s = state
    pre = preprocess(s["T_mid"], s["p_mid"], s["p_int"], s["pseudo_density"], s["omega"], s["phis"],
                     s["surf_sens_flux"], s["surf_evap"], s["surf_mom_flux"], s["qv"], s["qc"], s["tke"],
                     s["um_pert_diff"], s["vm_pert_diff"])
    cldfrac_liq_prev = s["cldfrac_liq"]
    upwp_sfc, vpwp_sfc = pre["upwp_sfc"], pre["vpwp_sfc"]
    if apply_tms:
        upwp_sfc, vpwp_sfc = apply_turbulent_mountain_stress(upwp_sfc, vpwp_sfc, s["surf_drag_coeff_tms"],
                                                             s["horiz_winds"], pre["rrho_i"])
    # turbulence_advected_tracers: qv, qc and tke are overwritten in postprocess,
    # so only the other members' diffused values survive. The tracer solves are
    # independent of each other, so the member order does not matter.
    trac_fields = [s["qv"], s["qc"], pre["tke"]] + [s[n] for n in tracer_names]
    qtracers = torch.stack(trac_fields, dim=1)
    wtracer_sfc = torch.zeros_like(qtracers[:, :, 0])
    hdtime = dt
    nadv = max(int(round(hdtime / dt)), 1)

    m = shoc_main(rt, npbl, nadv, dt,
                  cell_length, cell_length, pre["zt_grid"], pre["zi_grid"], s["p_mid"], s["p_int"],
                  s["pseudo_density"], pre["thv"], pre["wm_zt"], pre["wpthlp_sfc"], pre["wprtp_sfc"],
                  upwp_sfc, vpwp_sfc, wtracer_sfc, pre["inv_exner"], s["phis"],
                  pre["upwp_sfc_pert"], pre["vpwp_sfc_pert"],
                  pre["shoc_s"], pre["tke_copy"], pre["thlm"], pre["qw"],
                  s["horiz_winds"][:, 0], s["horiz_winds"][:, 1], s["sgs_buoy_flux"], qtracers,
                  s["eddy_diff_mom"], s["cldfrac_liq"], pre["qc_copy"], pre["um_pert"], pre["vm_pert"])

    post = postprocess(m["qw"], m["shoc_ql"], m["tke"], m["shoc_cldfrac"], m["shoc_ql2"], m["host_dse"],
                       pre["z_mid"], s["phis"], s["surf_mom_flux"], m["um_pert"], m["vm_pert"])
    out = dict(post)
    out.update(
        horiz_winds=torch.stack([m["u_wind"], m["v_wind"]], dim=1),
        sgs_buoy_flux=m["wthv_sec"], eddy_diff_mom=m["tk"], eddy_diff_heat=m["tkh"],
        cldfrac_liq_prev=cldfrac_liq_prev, pbl_height=m["pblh"], ustar=m["ustar"], obklen=m["obklen"],
        w_variance=m["w_sec"], thl_sec=m["thl_sec"],
    )
    for i, n in enumerate(tracer_names):
        out[n] = m["qtracers"][:, 3 + i]
    if rt.extra_diags:
        out.update(shoc_mix=m["shoc_mix"], brunt=m["brunt"], w3=m["w3"], isotropy=m["isotropy"],
                   wthl_sec=m["wthl_sec"], wqw_sec=m["wqw_sec"], uw_sec=m["uw_sec"], vw_sec=m["vw_sec"],
                   qw_sec=m["qw_sec"], shoc_cond=m["shoc_cond"], shoc_evap=m["shoc_evap"])
    out["_internal"] = dict(pre=pre, main=m, upwp_sfc=upwp_sfc, vpwp_sfc=vpwp_sfc)
    return out
