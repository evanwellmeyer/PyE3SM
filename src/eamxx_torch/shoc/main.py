"""SHOC driver: shoc_init and shoc_main.

Translation of components/eamxx/src/physics/shoc/impl/shoc_main_impl.hpp at
E3SM 80fe0a71, the default build (SCREAM_SHOC_SMALL_KERNELS off): shoc_init
and shoc_main_internal. The 3D-turbulence path (do_3d_turb) is not part of the
standard configuration and is not ported.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from . import functions as F
from .constants import SC


@dataclass(frozen=True)
class SHOCRuntime:
    """SHOCRuntime options (defaults = EAMxx namelist defaults at 80fe0a71)."""
    lambda_low: float = 0.001
    lambda_high: float = 0.08
    lambda_slope: float = 2.65
    lambda_thresh: float = 0.02
    thl2tune: float = 1.0
    qw2tune: float = 1.0
    qwthl2tune: float = 1.0
    w2tune: float = 1.0
    length_fac: float = 0.5
    c_diag_3rd_mom: float = 7.0
    Ckh: float = 0.1
    Ckm: float = 0.1
    shoc_1p5tke: bool = False
    extra_diags: bool = False
    # Not an EAMxx option (docs/differentiability.md). The energy fixer's shoctop is the
    # highest level with tke != mintke, an exact equality that rounding in the implicit
    # TKE solve can flip. With a tolerance, |tke - mintke| <= tol * mintke counts as
    # 'at the floor'. 0 (default) is EAMxx's exact test.
    energy_fixer_tke_tol: float = 0.0


def shoc_init(nbot_shoc: int, ntop_shoc: int, pref_mid) -> int:
    """Maximum number of levels in the PBL from the surface (npbl).

    max over k in [ntop, nbot) with pref_mid(k) >= pblmaxp of (nbot - k); 1 if none.
    pref_mid is a 1D sequence of floats (a grid constant, not a state tensor).
    """
    npbl = 1
    for k in range(ntop_shoc, nbot_shoc):
        if pref_mid[k] >= SC.pblmaxp:
            npbl = max(npbl, nbot_shoc - k)
    return npbl


def shoc_main(rt: SHOCRuntime, npbl: int, nadv: int, dtime: float,
              # Input
              dx, dy, zt_grid, zi_grid, pres, presi, pdel, thv, w_field,
              wthl_sfc, wqw_sfc, uw_sfc, vw_sfc, wtracer_sfc, inv_exner, phis,
              uw_sfc_pert, vw_sfc_pert,
              # Input/Output
              host_dse, tke, thetal, qw, u_wind, v_wind, wthv_sec, qtracers, tk,
              shoc_cldfrac, shoc_ql, um_pert, vm_pert):
    """shoc_main_internal for all columns at once.

    Columns: (ncol,) scalars, (ncol, nlev) midpoints, (ncol, nlevi) interfaces,
    qtracers (ncol, ntrac, nlev), wtracer_sfc (ncol, ntrac).
    Returns a dict of all in/out and output variables.
    """
    nlev = zt_grid.shape[-1]

    # Energy integrals before SHOC
    se_b, ke_b, wv_b, wl_b = F.shoc_energy_integrals(host_dse, pdel, qw, shoc_ql, u_wind, v_wind)

    for _ in range(nadv):
        tke = F.check_tke(tke)
        dz_zt, dz_zi, rho_zt = F.shoc_grid(zt_grid, zi_grid, pdel)
        shoc_qv = F.compute_shoc_vapor(qw, shoc_ql)
        shoc_tabs = F.compute_shoc_temperature(thetal, shoc_ql, inv_exner)
        ustar, kbfs, obklen = F.shoc_diag_obklen(uw_sfc, vw_sfc, wthl_sfc, wqw_sfc,
                                                 thetal[:, nlev - 1], shoc_ql[:, nlev - 1],
                                                 shoc_qv[:, nlev - 1])
        pblh = F.pblintd(npbl, zt_grid, zi_grid, thetal, shoc_ql, shoc_qv, u_wind, v_wind,
                         ustar, obklen, kbfs, shoc_cldfrac)
        brunt, shoc_mix = F.shoc_length(rt.length_fac, dx, dy, zt_grid, zi_grid, dz_zt, tke, thv)
        tke, tk, tkh, isotropy, shear_strain3d = F.shoc_tke(
            dtime, rt.lambda_low, rt.lambda_high, rt.lambda_slope, rt.lambda_thresh, rt.Ckh, rt.Ckm,
            rt.shoc_1p5tke, wthv_sec, shoc_mix, dz_zi, dz_zt, pres, shoc_tabs, u_wind, v_wind, brunt,
            zt_grid, zi_grid, pblh, tke, tk)
        thetal, qw, qtracers, tke, u_wind, v_wind, um_pert, vm_pert = F.update_prognostics_implicit(
            dtime, dz_zt, dz_zi, rho_zt, zt_grid, zi_grid, tk, tkh, uw_sfc, vw_sfc, wthl_sfc, wqw_sfc,
            wtracer_sfc, thetal, qw, qtracers, tke, u_wind, v_wind, uw_sfc_pert, vw_sfc_pert,
            um_pert, vm_pert)
        sm = F.diag_second_shoc_moments(rt.thl2tune, rt.qw2tune, rt.qwthl2tune, rt.w2tune, rt.shoc_1p5tke,
                                        thetal, qw, u_wind, v_wind, tke, isotropy, tkh, tk, dz_zi,
                                        zt_grid, zi_grid, shoc_mix, wthl_sfc, wqw_sfc, uw_sfc, vw_sfc)
        w3 = F.diag_third_shoc_moments(rt.c_diag_3rd_mom, rt.shoc_1p5tke, sm["w_sec"], sm["thl_sec"],
                                       sm["wthl_sec"], isotropy, brunt, thetal, tke, dz_zt, dz_zi,
                                       zt_grid, zi_grid)
        pdf = F.shoc_assumed_pdf(thetal, qw, w_field, sm["thl_sec"], sm["qw_sec"], dtime, rt.extra_diags,
                                 sm["wthl_sec"], sm["w_sec"], sm["wqw_sec"], sm["qwthl_sec"], w3, pres,
                                 zt_grid, zi_grid, shoc_ql)
        shoc_cldfrac, shoc_ql = pdf["shoc_cldfrac"], pdf["shoc_ql"]
        wthv_sec = pdf["wthv_sec"]
        tke = F.check_tke(tke)

    # Update host dry static energy and apply the energy fixer
    host_dse = F.update_host_dse(thetal, shoc_ql, inv_exner, zt_grid, phis)
    se_a, ke_a, wv_a, wl_a = F.shoc_energy_integrals(host_dse, pdel, qw, shoc_ql, u_wind, v_wind)
    host_dse = F.shoc_energy_fixer(dtime, nadv, zt_grid, zi_grid, se_b, ke_b, wv_b, wl_b,
                                   se_a, ke_a, wv_a, wl_a, wthl_sfc, wqw_sfc, rho_zt, tke, presi, host_dse,
                                   tke_tol=rt.energy_fixer_tke_tol)

    # Diagnose PBL quantities from the final state
    shoc_qv = F.compute_shoc_vapor(qw, shoc_ql)
    ustar, kbfs, obklen = F.shoc_diag_obklen(uw_sfc, vw_sfc, wthl_sfc, wqw_sfc,
                                             thetal[:, nlev - 1], shoc_ql[:, nlev - 1],
                                             shoc_qv[:, nlev - 1])
    pblh = F.pblintd(npbl, zt_grid, zi_grid, thetal, shoc_ql, shoc_qv, u_wind, v_wind,
                     ustar, obklen, kbfs, shoc_cldfrac)

    return dict(
        host_dse=host_dse, tke=tke, thetal=thetal, qw=qw, u_wind=u_wind, v_wind=v_wind,
        wthv_sec=wthv_sec, qtracers=qtracers, tk=tk, shoc_cldfrac=shoc_cldfrac, shoc_ql=shoc_ql,
        um_pert=um_pert, vm_pert=vm_pert,
        pblh=pblh, ustar=ustar, obklen=obklen, shoc_ql2=pdf["shoc_ql2"], tkh=tkh,
        shoc_cond=pdf["shoc_cond"], shoc_evap=pdf["shoc_evap"], shoc_mix=shoc_mix,
        w_sec=sm["w_sec"], thl_sec=sm["thl_sec"], qw_sec=sm["qw_sec"], qwthl_sec=sm["qwthl_sec"],
        wthl_sec=sm["wthl_sec"], wqw_sec=sm["wqw_sec"], wtke_sec=sm["wtke_sec"],
        uw_sec=sm["uw_sec"], vw_sec=sm["vw_sec"], w3=w3, wqls_sec=pdf["wqls"], brunt=brunt,
        isotropy=isotropy, shear_strain3d=shear_strain3d,
        # budget terms (not EAMxx outputs; used by the budget checks)
        energy_integrals_before=(se_b, ke_b, wv_b, wl_b),
    )
