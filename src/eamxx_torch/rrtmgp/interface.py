"""EAMxx radiation process: RRTMGPRadiation::run_impl and the rrtmgp_interface driver.

Translation of, E3SM 80fe0a71:
  - components/eamxx/src/physics/rrtmgp/eamxx_rrtmgp_process_interface.cpp
      (initialize_impl settings, run_impl)
  - components/eamxx/src/physics/rrtmgp/eamxx_rrtmgp_interface.hpp
      (compute_band_by_band_surface_albedos, compute_broadband_surface_fluxes,
       rrtmgp_main, rrtmgp_sw, rrtmgp_lw, compute_cloud_area, compute_aerocom_cloudtop,
       mixing_ratio_to_cloud_mass, limit_to_bounds_k, get_wavelength_index)
  - components/eamxx/src/physics/rrtmgp/rrtmgp_utils.hpp
      (compute_heating_rate, radiation_do)

Form changes (results unchanged, see docs/plan.md):
  - Columns are not split into chunks: every column is independent.
  - The SW driver does not compact daytime columns. It computes every column
    with mu0 replaced by 1 on night columns and then sets their fluxes to zero,
    which is what the reference returns for them.
  - The vertical axis runs from the model top (top_at_1 = true). The reference
    derives this from the pressures of column 0; the host contract fixes it.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .. import constants as C
from .. import physics_functions as PF
from ..ekat import max_, min_
from ..orbit import SHR_ORB_UNDEF_INT, shr_orb_cosz, shr_orb_decl, shr_orb_params
from ..precision import to_precision
from ..property_checks import within_interval
from ..timestamp import TimeStamp
from ..trcmix import trcmix
from .cloud_optics import get_cloud_optics, limit_to_bounds
from .data import GasOpticsData, RRTMGPData
from .gas_optics import gas_optics_lw, gas_optics_sw
from .mcica import get_subsampled_clouds
from .rte import (delta_scale_2str, inc_1scalar_by_1scalar_bybnd, inc_2stream_by_2stream_bybnd,
                  increment_1scalar_by_1scalar, increment_2stream_by_2stream, rte_lw, rte_sw,
                  seq_sum, sum_broadband, sum_byband)

NSWBANDS = 14
NLWBANDS = 16
VISIBLE_WAVENUMBER_THRESHOLD = 14286.0   # 0.7 micron
DBL_MAX = 1.7976931348623157e308


@dataclass(frozen=True)
class RRTMGPRuntime:
    """The rrtmgp parameters EAMxx reads (defaults as in the C++ constructor/initialize)."""
    rad_frequency: int = 1
    orbital_year: int = -9999
    orbital_eccentricity: float = -9999.
    orbital_obliquity: float = -9999.
    orbital_mvelp: float = -9999.
    fixed_total_solar_irradiance: float = -9999.
    fixed_solar_zenith_angle: float = -9999.
    co2vmr: float = 388.717e-6
    n2ovmr: float = 323.141e-9
    ch4vmr: float = 1807.851e-9
    f11vmr: float = 768.7644e-12
    f12vmr: float = 531.2820e-12
    n2vmr: float = 0.7906
    covmr: float = 1.0e-7
    do_subcol_sampling: bool = True
    do_aerosol_rad: bool = True
    extra_clnclrsky_diag: bool = False
    extra_clnsky_diag: bool = False
    active_gases: tuple = ("h2o", "co2", "o3", "n2o", "co", "ch4", "o2", "n2")
    # Not an EAMxx option (docs/differentiability.md). EAMxx seeds McICA from the digits of
    # the lowest layer pressures, so any change of pressure redraws every subcolumn (a jump
    # in the fluxes). False: seeds from the column index and the step count instead
    # (state['mcica_seed'], int64 (ncol,), if given, replaces the column index). The draws
    # then do not depend on the state. True (default) is EAMxx.
    mcica_seed_from_pressure: bool = True

    @property
    def gas_names(self) -> list[str]:
        out = []
        for g in self.active_gases:
            if g not in out:
                out.append(g)
        return out


def radiation_do(irad: int, nstep: int) -> bool:
    if irad == 0:
        return False
    return nstep == 0 or nstep % irad == 0


# ---------------------------------------------------------------- small interface routines
def _band_class(k: GasOpticsData):
    """Per band: 'vis', 'nir' or 'both' (straddles 0.7 micron)."""
    wl = k.band_lims_wvn
    out = []
    for b in range(wl.shape[1]):
        v1 = float(wl[0, b]) > VISIBLE_WAVENUMBER_THRESHOLD
        v2 = float(wl[1, b]) > VISIBLE_WAVENUMBER_THRESHOLD
        out.append("vis" if v1 and v2 else ("nir" if not v1 and not v2 else "both"))
    return out


def compute_band_by_band_surface_albedos(k_sw, dir_vis, dir_nir, dif_vis, dif_nir):
    cols_dir, cols_dif = [], []
    for c in _band_class(k_sw):
        if c == "vis":
            cols_dir.append(dir_vis)
            cols_dif.append(dif_vis)
        elif c == "nir":
            cols_dir.append(dir_nir)
            cols_dif.append(dif_nir)
        else:
            cols_dir.append(0.5 * (dir_vis + dir_nir))
            cols_dif.append(0.5 * (dif_vis + dif_nir))
    return torch.stack(cols_dir, dim=-1), torch.stack(cols_dif, dim=-1)


def compute_broadband_surface_fluxes(k_sw, bnd_dir, bnd_dif):
    """bnd_dir, bnd_dif: (ncol, nbnd) at the surface. Returns dir_vis, dir_nir, dif_vis, dif_nir."""
    z = torch.zeros_like(bnd_dir[:, 0])
    dir_vis, dir_nir, dif_vis, dif_nir = z, z, z, z
    for b, c in enumerate(_band_class(k_sw)):
        if c == "vis":
            dir_vis = dir_vis + bnd_dir[:, b]
            dif_vis = dif_vis + bnd_dif[:, b]
        elif c == "nir":
            dir_nir = dir_nir + bnd_dir[:, b]
            dif_nir = dif_nir + bnd_dif[:, b]
        else:
            dir_vis = dir_vis + 0.5 * bnd_dir[:, b]
            dif_vis = dif_vis + 0.5 * bnd_dif[:, b]
            dir_nir = dir_nir + 0.5 * bnd_dir[:, b]
            dif_nir = dif_nir + 0.5 * bnd_dif[:, b]
    return dir_vis, dir_nir, dif_vis, dif_nir


def mixing_ratio_to_cloud_mass(q, cf, dp):
    incloud = min_(q / max_(0.0001, cf), 0.005)
    return torch.where(cf > 0, incloud * dp / C.gravit, torch.zeros_like(q))


def compute_heating_rate(flux_up, flux_dn, pdel):
    return ((flux_up[:, 1:] - flux_up[:, :-1] - flux_dn[:, 1:] + flux_dn[:, :-1])
            * C.gravit / (C.Cpair * pdel))


def compute_cloud_area(pmin: float, pmax: float, pmid, cld_tau_gpt):
    ngpt = cld_tau_gpt.shape[-1]
    inrange = (pmid >= pmin) & (pmid < pmax)
    mask = ((cld_tau_gpt > 0) & inrange[..., None]).any(dim=1)
    ngpt_inv = 1.0 / ngpt
    return seq_sum(mask.to(pmid.dtype) * ngpt_inv, -1)


def compute_aerocom_cloudtop(tmid, pmid, p_del, z_del, qc, qi, rel, rei, cldfrac_tot, nc):
    """Cloud-top diagnostics (maximum-random overlap, column form). Returns a dict."""
    q_threshold = 0.0
    cldfrac_tot_threshold = 0.001
    ncol, nlay = tmid.shape
    z = torch.zeros(ncol, dtype=tmid.dtype, device=tmid.device)
    T_top, p_top, ice_top, liq_top, cdnc_top, reqc_top, reqi_top = (z,) * 7
    clr = torch.ones_like(z)
    for k in range(1, nlay):
        cf, cfm = cldfrac_tot[:, k], cldfrac_tot[:, k - 1]
        q = qc[:, k] + qi[:, k]
        cond = (q > q_threshold) & (cf > cldfrac_tot_threshold)
        one = torch.ones_like(cf)
        tmp = (clr * (1.0 - max_(cfm, cf))
               / torch.where(cond, 1.0 - min_(cfm, 1.0 - cldfrac_tot_threshold), one))
        wts = clr - tmp
        phi = qc[:, k] / torch.where(cond, q, one)
        cdnc = nc[:, k] * p_del[:, k] / z_del[:, k] / C.gravit / torch.where(cond, cf, one)
        T_top = torch.where(cond, T_top + tmid[:, k] * wts, T_top)
        p_top = torch.where(cond, p_top + pmid[:, k] * wts, p_top)
        ice_top = torch.where(cond, ice_top + (1.0 - phi) * wts, ice_top)
        liq_top = torch.where(cond, liq_top + phi * wts, liq_top)
        cdnc_top = torch.where(cond, cdnc_top + cdnc * phi * wts, cdnc_top)
        reqc_top = torch.where(cond, reqc_top + rel[:, k] * phi * wts, reqc_top)
        reqi_top = torch.where(cond, reqi_top + rei[:, k] * (1.0 - phi) * wts, reqi_top)
        clr = torch.where(cond, tmp, clr)
    return dict(T_mid_at_cldtop=T_top, p_mid_at_cldtop=p_top, cldfrac_ice_at_cldtop=ice_top,
                cldfrac_liq_at_cldtop=liq_top, cldfrac_tot_at_cldtop=1.0 - clr,
                cdnc_at_cldtop=cdnc_top, eff_radius_qc_at_cldtop=reqc_top,
                eff_radius_qi_at_cldtop=reqi_top)


def get_wavelength_index(k: GasOpticsData, wavelength: float) -> int:
    wl = k.band_lims_wvn
    w = wavelength * 1e2
    idx = -2**31
    for b in range(wl.shape[1]):
        lo, hi = 1. / float(wl[0, b]), 1. / float(wl[1, b])
        if lo < hi:
            if lo <= w <= hi:
                idx = max(idx, b)
        elif lo >= w >= hi:
            idx = max(idx, b)
    return idx


# ---------------------------------------------------------------- SW and LW drivers
def rrtmgp_sw(k: GasOpticsData, p_lay, t_lay, p_lev, t_lev, vmr, sfc_alb_dir, sfc_alb_dif, mu0,
              aerosol, clouds, tsi_scaling, extra_clnclrsky_diag=False, extra_clnsky_diag=False):
    """aerosol: (tau, ssa, g) by band; clouds: (tau, ssa, g) by g-point."""
    day = mu0 > 0
    mu0_d = torch.where(day, mu0, torch.ones_like(mu0))
    t_lay_limited = limit_to_bounds(t_lay, k.temp_ref_min, k.temp_ref_max)
    (tau, ssa, g), toa_flux = gas_optics_sw(k, p_lay, p_lev, t_lay_limited, vmr)
    toa_flux = tsi_scaling * toa_flux
    gb = k.gpt2band.to(p_lay.device)
    alb_dir, alb_dif = sfc_alb_dir[:, gb], sfc_alb_dif[:, gb]
    d2 = day[:, None]
    d3 = day[:, None, None]
    zero = torch.zeros((), dtype=p_lay.dtype, device=p_lay.device)
    out = {}

    def solve(tau, ssa, g, prefix, bands=False):
        fu, fd, fdir = rte_sw(tau, ssa, g, mu0_d, toa_flux, alb_dir, alb_dif)
        out[prefix + "flux_up"] = torch.where(d2, sum_broadband(fu), zero)
        out[prefix + "flux_dn"] = torch.where(d2, sum_broadband(fd), zero)
        out[prefix + "flux_dn_dir"] = torch.where(d2, sum_broadband(fdir), zero)
        if bands:
            out["bnd_flux_up"] = torch.where(d3, sum_byband(fu, k.band2gpt), zero)
            out["bnd_flux_dn"] = torch.where(d3, sum_byband(fd, k.band2gpt), zero)
            out["bnd_flux_dn_dir"] = torch.where(d3, sum_byband(fdir, k.band2gpt), zero)

    gas_only = (tau, ssa, g)
    if extra_clnclrsky_diag:
        solve(tau, ssa, g, "clnclrsky_")
    atau, assa, ag = delta_scale_2str(*aerosol)
    tau, ssa, g = inc_2stream_by_2stream_bybnd(tau, ssa, g, atau, assa, ag, k.gpt2band)
    solve(tau, ssa, g, "clrsky_")
    ctau, cssa, cg = delta_scale_2str(*clouds)
    tau, ssa, g = increment_2stream_by_2stream(tau, ssa, g, ctau, cssa, cg)
    solve(tau, ssa, g, "", bands=True)
    if extra_clnsky_diag:
        solve(*increment_2stream_by_2stream(*gas_only, ctau, cssa, cg), "clnsky_")
    nlev = p_lev.shape[1]
    zeros2 = torch.zeros((p_lay.shape[0], nlev), dtype=p_lay.dtype, device=p_lay.device)
    for pre in ("clnclrsky_", "clnsky_"):
        for f in ("flux_up", "flux_dn", "flux_dn_dir"):
            out.setdefault(pre + f, zeros2)
    return out


def rrtmgp_lw(k: GasOpticsData, p_lay, t_lay, p_lev, t_lev, vmr, aer_tau, cld_tau,
              extra_clnclrsky_diag=False, extra_clnsky_diag=False):
    """aer_tau by band; cld_tau by g-point."""
    nlay = p_lay.shape[1]
    t_sfc = t_lev[:, nlay]
    sfc_emis = torch.full((p_lay.shape[0], k.ngpt), 0.98, dtype=p_lay.dtype, device=p_lay.device)
    t_lay_limited = limit_to_bounds(t_lay, k.temp_ref_min, k.temp_ref_max)
    t_lev_limited = limit_to_bounds(t_lev, k.temp_ref_min, k.temp_ref_max)
    tau, sources = gas_optics_lw(k, p_lay, p_lev, t_lay_limited, t_sfc, vmr, t_lev_limited)
    out = {}

    def solve(tau, prefix, bands=False):
        fu, fd = rte_lw(tau, sources, sfc_emis)
        out[prefix + "flux_up"] = sum_broadband(fu)
        out[prefix + "flux_dn"] = sum_broadband(fd)
        if bands:
            out["bnd_flux_up"] = sum_byband(fu, k.band2gpt)
            out["bnd_flux_dn"] = sum_byband(fd, k.band2gpt)

    gas_only = tau
    if extra_clnclrsky_diag:
        solve(tau, "clnclrsky_")
    tau = inc_1scalar_by_1scalar_bybnd(tau, aer_tau, k.gpt2band)
    solve(tau, "clrsky_")
    tau = increment_1scalar_by_1scalar(tau, cld_tau)
    solve(tau, "", bands=True)
    if extra_clnsky_diag:
        solve(increment_1scalar_by_1scalar(gas_only, cld_tau), "clnsky_")
    zeros2 = torch.zeros((p_lay.shape[0], nlay + 1), dtype=p_lay.dtype, device=p_lay.device)
    for pre in ("clnclrsky_", "clnsky_"):
        for f in ("flux_up", "flux_dn"):
            out.setdefault(pre + f, zeros2)
    return out


def rrtmgp_main(data: RRTMGPData, p_lay, t_lay, p_lev, t_lev, vmr, sfc_alb_dir, sfc_alb_dif,
                mu0, lwp, iwp, rel, rei, cldfrac, aer_sw, aer_tau_lw, tsi_scaling,
                extra_clnclrsky_diag=False, extra_clnsky_diag=False, seeds=None):
    """seeds: None (EAMxx: from pressure digits) or (sw_seeds, lw_seeds), int64 (ncol,)."""
    ksw, klw = data.k_dist_sw, data.k_dist_lw
    nlay = p_lay.shape[1]
    clouds_sw = get_cloud_optics(data.cloud_optics_sw, lwp, iwp, rel, rei, True)
    clouds_lw = get_cloud_optics(data.cloud_optics_lw, lwp, iwp, rel, rei, False)
    sw_seeds, lw_seeds = seeds if seeds is not None else (None, None)
    sw_gpt = get_subsampled_clouds(*clouds_sw, cldfrac, p_lay, ksw.gpt2band, nlay - 1, sw_seeds)
    lw_gpt, _, _ = get_subsampled_clouds(clouds_lw, None, None, cldfrac, p_lay, klw.gpt2band,
                                        nlay - 2, lw_seeds)
    sw = rrtmgp_sw(ksw, p_lay, t_lay, p_lev, t_lev, vmr, sfc_alb_dir, sfc_alb_dif, mu0, aer_sw,
                   sw_gpt, tsi_scaling, extra_clnclrsky_diag, extra_clnsky_diag)
    lw = rrtmgp_lw(klw, p_lay, t_lay, p_lev, t_lev, vmr, aer_tau_lw, lw_gpt,
                   extra_clnclrsky_diag, extra_clnsky_diag)
    return dict(sw=sw, lw=lw, cld_tau_sw_bnd=clouds_sw[0], cld_tau_lw_bnd=clouds_lw,
                cld_tau_sw_gpt=sw_gpt[0], cld_tau_lw_gpt=lw_gpt)


# ---------------------------------------------------------------- the process
def gas_vmrs(state, ro: RRTMGPRuntime, lat):
    """Volume mixing ratios of the active gases (run_impl, before the chunk loop)."""
    vmr = {}
    for name in ro.gas_names:
        if name == "o3":
            vmr[name] = state["o3_volume_mix_ratio"]
        elif name == "n2":
            vmr[name] = torch.full_like(state["p_mid"], ro.n2vmr)   # set in initialize_impl
        elif name == "co":
            vmr[name] = torch.full_like(state["p_mid"], ro.covmr)
        elif name == "h2o":
            qv = state["qv"]
            vmr[name] = qv / (1.0 - qv) * C.MWdry / C.GAS_MOL_WEIGHT["h2o"]
        else:
            q = trcmix(name, lat, state["p_mid"], ro.co2vmr, ro.n2ovmr, ro.ch4vmr, ro.f11vmr,
                       ro.f12vmr)
            vmr[name] = C.MWdry / C.GAS_MOL_WEIGHT[name] * q
    return vmr


def orbital_state(ro: RRTMGPRuntime, t_start: TimeStamp):
    """Orbital parameters -> (calday, delta, eccf), as run_impl computes them."""
    orbital_year = ro.orbital_year
    eccen, obliq, mvelp = ro.orbital_eccentricity, ro.orbital_obliquity, ro.orbital_mvelp
    if eccen >= 0 and obliq >= 0 and mvelp >= 0:
        orbital_year = SHR_ORB_UNDEF_INT
    elif orbital_year < 0:
        orbital_year = t_start.date[0]
    eccen, obliq, mvelp, obliqr, lambm0, mvelpp = shr_orb_params(orbital_year, eccen, obliq, mvelp)
    calday = t_start.frac_of_year_in_days() + 1
    delta, eccf = shr_orb_decl(calday, eccen, mvelpp, lambm0, obliqr)
    if ro.fixed_total_solar_irradiance >= 0:
        eccf = ro.fixed_total_solar_irradiance / 1360.9
    return calday, delta, eccf


def mcica_state_free_seeds(column_seed, ncol: int, nstep: int, device) -> tuple:
    """McICA seeds that do not depend on the state (mcica_seed_from_pressure = False).

    seed = nstep * 2**40 + column * 2**20 (+ 2**19 for LW). The generator adds up to
    nlay*ngpt (16384) to a seed, so columns, steps and SW/LW never share a stream while
    column < 2**20 and nstep < 2**22.
    """
    col = (torch.arange(ncol, dtype=torch.int64, device=device) if column_seed is None
           else torch.as_tensor(column_seed, dtype=torch.int64, device=device))
    sw = col * 2**20 + int(nstep) * 2**40
    return sw, sw + 2**19


def rrtmgp_run(state: dict, dt: float, data: RRTMGPData, ro: RRTMGPRuntime, t_start: TimeStamp,
               nstep: int, lat: torch.Tensor, lon: torch.Tensor) -> dict:
    """RRTMGPRadiation::run_impl on EAMxx-named fields.

    state: p_mid, p_int, pseudo_density, T_mid, qv, qc, nc, qi, cldfrac_tot, eff_radius_qc,
      eff_radius_qi, surf_lw_flux_up, sfc_alb_{dir,dif}_{vis,nir}, o3_volume_mix_ratio,
      aero_{tau,ssa,g}_sw (ncol, nswbands, nlay), aero_tau_lw (ncol, nlwbands, nlay), and
      rad_heating_pdel (needed on steps without a radiation update).
    t_start: start-of-step time stamp; nstep: its step count. lat, lon: degrees, (ncol,).
    Returns the fields run_impl writes. On steps without a radiation update only T_mid changes.
    """
    s = state
    T_mid, pdel, p_mid = s["T_mid"], s["pseudo_density"], s["p_mid"]
    from ..precision import cast
    data = cast(data, T_mid.dtype, T_mid.device)   # coefficients in the state's precision/device
    lat, lon = to_precision(lat, T_mid.dtype, T_mid.device), to_precision(lon, T_mid.dtype, T_mid.device)
    if not radiation_do(ro.rad_frequency, nstep):
        return {"T_mid": T_mid + s["rad_heating_pdel"] / pdel * dt}

    calday, delta, eccf = orbital_state(ro, t_start)
    vmr = gas_vmrs(s, ro, lat)
    ncol, nlay = T_mid.shape
    if ro.fixed_solar_zenith_angle > 0:
        mu0 = torch.full((ncol,), ro.fixed_solar_zenith_angle, dtype=T_mid.dtype,
                         device=T_mid.device)
    else:
        mu0 = shr_orb_cosz(calday, lat * C.Pi / 180.0, lon * C.Pi / 180.0, delta,
                           ro.rad_frequency * dt)

    dz = PF.calculate_dz(pdel, p_mid, T_mid, s["qv"])
    bc_top = T_mid[:, :1]
    bc_bot = torch.sqrt(torch.sqrt(s["surf_lw_flux_up"] / C.stebol))[:, None]
    interior = (T_mid[:, 1:] * dz[:, :-1] + T_mid[:, :-1] * dz[:, 1:]) / (dz[:, :-1] + dz[:, 1:])
    t_int = torch.cat([bc_top, interior, bc_bot], dim=1)

    if ro.do_aerosol_rad:
        aer_sw = tuple(s[f"aero_{n}_sw"].permute(0, 2, 1) for n in ("tau", "ssa", "g"))
        aer_lw = s["aero_tau_lw"].permute(0, 2, 1)
    else:
        z = torch.zeros((ncol, nlay, NSWBANDS), dtype=T_mid.dtype, device=T_mid.device)
        aer_sw = (z, z, z)
        aer_lw = torch.zeros((ncol, nlay, NLWBANDS), dtype=T_mid.dtype, device=T_mid.device)

    if ro.do_subcol_sampling:
        cldfrac = s["cldfrac_tot"]
    else:
        cldfrac = torch.where(s["cldfrac_tot"] > 0, 1., 0.).to(T_mid.dtype)
    lwp = mixing_ratio_to_cloud_mass(s["qc"], cldfrac, pdel) * 1e3
    iwp = mixing_ratio_to_cloud_mass(s["qi"], cldfrac, pdel) * 1e3

    ksw = data.k_dist_sw
    alb_dir, alb_dif = compute_band_by_band_surface_albedos(
        ksw, s["sfc_alb_dir_vis"], s["sfc_alb_dir_nir"], s["sfc_alb_dif_vis"], s["sfc_alb_dif_nir"])

    seeds = None
    if not ro.mcica_seed_from_pressure:
        seeds = mcica_state_free_seeds(s.get("mcica_seed"), T_mid.shape[0], nstep, T_mid.device)
    r = rrtmgp_main(data, p_mid, T_mid, s["p_int"], t_int, vmr, alb_dir, alb_dif, mu0, lwp, iwp,
                    s["eff_radius_qc"], s["eff_radius_qi"], cldfrac, aer_sw, aer_lw, eccf,
                    ro.extra_clnclrsky_diag, ro.extra_clnsky_diag, seeds)
    sw, lw = r["sw"], r["lw"]

    sw_heating = compute_heating_rate(sw["flux_up"], sw["flux_dn"], pdel)
    lw_heating = compute_heating_rate(lw["flux_up"], lw["flux_dn"], pdel)
    rad_heating = sw_heating + lw_heating

    bnd_dir = sw["bnd_flux_dn_dir"][:, nlay]
    bnd_dif = sw["bnd_flux_dn"][:, nlay] - bnd_dir
    dir_vis, dir_nir, dif_vis, dif_nir = compute_broadband_surface_fluxes(ksw, bnd_dir, bnd_dif)

    tau_lw_gpt = r["cld_tau_lw_gpt"]
    out = {
        "SW_flux_up": sw["flux_up"], "SW_flux_dn": sw["flux_dn"],
        "SW_flux_dn_dir": sw["flux_dn_dir"],
        "LW_flux_up": lw["flux_up"], "LW_flux_dn": lw["flux_dn"],
    }
    for sky in ("clnclrsky", "clrsky", "clnsky"):
        out[f"SW_{sky}_flux_up"] = sw[f"{sky}_flux_up"]
        out[f"SW_{sky}_flux_dn"] = sw[f"{sky}_flux_dn"]
        out[f"SW_{sky}_flux_dn_dir"] = sw[f"{sky}_flux_dn_dir"]
        out[f"LW_{sky}_flux_up"] = lw[f"{sky}_flux_up"]
        out[f"LW_{sky}_flux_dn"] = lw[f"{sky}_flux_dn"]
    out.update(
        T_mid=T_mid + rad_heating * dt,
        rad_heating_pdel=pdel * rad_heating,
        sfc_flux_dir_vis=dir_vis, sfc_flux_dir_nir=dir_nir,
        sfc_flux_dif_vis=dif_vis, sfc_flux_dif_nir=dif_nir,
        sfc_flux_sw_net=sw["flux_dn"][:, nlay] - sw["flux_up"][:, nlay],
        sfc_flux_lw_dn=lw["flux_dn"][:, nlay],
        cldlow=compute_cloud_area(700e2, DBL_MAX, p_mid, tau_lw_gpt),
        cldmed=compute_cloud_area(400e2, 700e2, p_mid, tau_lw_gpt),
        cldhgh=compute_cloud_area(0, 400e2, p_mid, tau_lw_gpt),
        cldtot=compute_cloud_area(0, DBL_MAX, p_mid, tau_lw_gpt),
        dtau067=r["cld_tau_sw_bnd"][..., get_wavelength_index(ksw, 0.67e-6)],
        dtau105=r["cld_tau_lw_bnd"][..., get_wavelength_index(data.k_dist_lw, 10.5e-6)],
        sunlit_mask=(sw["clrsky_flux_dn"][:, 0] > 0).to(T_mid.dtype),
        cldfrac_rad=cldfrac,
        cosine_solar_zenith_angle=mu0,
    )
    out.update(compute_aerocom_cloudtop(T_mid, p_mid, pdel, dz, s["qc"], s["qi"],
                                        s["eff_radius_qc"], s["eff_radius_qi"], cldfrac, s["nc"]))
    for name in ro.gas_names:
        if name != "o3":
            out[name + "_volume_mix_ratio"] = vmr[name]
    out["_internal"] = dict(mu0=mu0, eccf=eccf, calday=calday, delta=delta, t_int=t_int, dz=dz,
                            lwp=lwp, iwp=iwp, rrtmgp=r)
    return out


def invariant_checks(fields: dict) -> tuple[dict, dict]:
    """The invariant check of initialize_impl: T_mid within [100, 500], no repair."""
    out = dict(fields)
    status = {}
    out["T_mid"], status["T_mid"] = within_interval(out["T_mid"], 100.0, 500.0, False)
    return out, status
