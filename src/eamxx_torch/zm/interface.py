"""ZM atmosphere process: run_impl and the property checks.

Translation of, at E3SM 80fe0a71:
  components/eamxx/src/physics/zm/eamxx_zm_process_interface.cpp
      initialize_impl (checks), run_impl
  components/eamxx/src/physics/zm/impl/zm_input_state_impl.hpp   calculate_tpert
  components/eamxx/src/physics/zm/impl/zm_output_tend_impl.hpp   init_all, init_tmp

Order inside run_impl: heights from the state; the PBL temperature perturbation from
SHOC's theta-l variance; zm_conv_main; MCSP added to the tendencies; rain
evaporation on the state updated by those tendencies; momentum transport; then the
state update, detrainment of cloud water and number, and the diagnostics.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .. import constants as PC
from .. import physics_functions as PF
from ..ekat import max_, min_
from ..property_checks import S_MAX, within_interval
from .constants import ZMC, ZMRuntime, build_estbl, set_limcnv
from .evap_mcsp import zm_conv_evap, zm_conv_mcsp_tend, zm_transport_momentum
from .main import zm_conv_main
from .util import gather

# 4/3 pi r^3 rho of the assumed detrained droplets and crystals (constants folded by the
# compiler with the host's libm pow; math.pow is the same libm)
_LIQ_MASS = 4.0 / 3.0 * PC.Pi * math.pow(ZMC.cld_liq_radius, 3.0) * ZMC.cld_liq_density
_ICE_MASS = 4.0 / 3.0 * PC.Pi * math.pow(ZMC.cld_ice_radius, 3.0) * ZMC.cld_ice_density


@dataclass
class ZMSetup:
    """What initialize_impl and load_runtime_options set up once."""
    runtime: ZMRuntime
    limcnv: int
    estbl: torch.Tensor


def zm_setup(runtime: ZMRuntime, hyai, hybi, dtype=torch.float64, device="cpu") -> ZMSetup:
    return ZMSetup(runtime, set_limcnv(runtime.upper_limit_pref, np.asarray(hyai), np.asarray(hybi)),
                   torch.as_tensor(build_estbl(), dtype=dtype, device=device))


def calculate_tpert(is_first_step: bool, pblh, z_int, p_mid, qc, thl_sec):
    """PBL-top temperature perturbation: std deviation of theta-l plus condensate heating."""
    ncol, nlev = p_mid.shape
    zero = torch.zeros(ncol, dtype=p_mid.dtype, device=p_mid.device)
    if is_first_step:
        return zero
    ind = torch.full((ncol,), -1, dtype=torch.int64, device=p_mid.device)
    for k in range(nlev):
        m = (z_int[:, k] > pblh) & (z_int[:, k + 1] <= pblh)
        ind = torch.where(m, torch.full_like(ind, k), ind)
    found = ind >= 0
    kk = torch.clamp(ind, min=0)
    exner_pbl = PF.exner_function(gather(p_mid, kk))
    qc_pbl = gather(qc, kk)
    thl_sec_pbl = gather(thl_sec, kk)
    thl_std_pbl = torch.sqrt(torch.where(found, thl_sec_pbl, zero))
    tpert = (thl_std_pbl + (PC.LatVap / PC.Cpair) * qc_pbl) / exner_pbl
    tpert = min_(ZMC.tpert_limiter, tpert)
    return torch.where(found, tpert, zero)


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
    for name, lb, ub in (("qc", 0.0, 0.1), ("qi", 0.0, 0.1), ("nc", 0.0, 1.e12), ("ni", 0.0, 1.e12),
                         ("precip_liq_surf_mass", 0.0, S_MAX), ("precip_ice_surf_mass", 0.0, S_MAX)):
        if name in out:
            out[name], status[name] = within_interval(out[name], lb, ub, False)
    return out, status


def zm_run(state: dict, dt: float, setup: ZMSetup, is_first_step: bool) -> dict:
    """ZMDeepConvection::run_impl on EAMxx-named fields.

    state: T_mid, qv, qc, qi, nc, ni, horiz_winds, p_mid, p_int, pseudo_density, phis,
      omega, cldfrac_tot, pbl_height, landfrac, thl_sec, precip_liq_surf_mass,
      precip_ice_surf_mass, zm_t_prev, zm_q_prev.
    Returns the fields ZM writes (before the post-condition checks), plus "_internal".
    """
    rt = setup.runtime
    s = state
    if setup.estbl.dtype != s["T_mid"].dtype or setup.estbl.device != s["T_mid"].device:
        from ..precision import cast
        setup = cast(setup, s["T_mid"].dtype, s["T_mid"].device)
    T_mid, qv, qc = s["T_mid"], s["qv"], s["qc"]
    uwind, vwind = s["horiz_winds"][:, 0], s["horiz_winds"][:, 1]
    p_mid, p_int, p_del = s["p_mid"], s["p_int"], s["pseudo_density"]
    t_prev, q_prev = s["zm_t_prev"], s["zm_q_prev"]
    if is_first_step:
        t_prev, q_prev = T_mid, qv
    ncol, nlev = T_mid.shape
    zero = torch.zeros_like(T_mid)

    # heights above the surface
    z_del = PF.calculate_dz(p_del, p_mid, T_mid, qv)
    z_int = PF.calculate_z_int(z_del, 0.0)
    z_mid = PF.calculate_z_mid(z_int)

    tpert = calculate_tpert(is_first_step, s["pbl_height"], z_int, p_mid, qc, s["thl_sec"])

    m = zm_conv_main(rt, setup.limcnv, is_first_step, dt, T_mid, qv, s["omega"], p_mid, p_int, p_del,
                     s["phis"], z_mid, z_int, s["pbl_height"], tpert, s["landfrac"], t_prev, q_prev)
    active = m["active"]
    act2 = active.unsqueeze(-1)
    tend_out_s, tend_out_qv = m["heat"], m["qtnd"]
    tend_out_u, tend_out_v = zero, zero

    # MCSP (all columns)
    mc = dict(mcsp_ds_out=zero, mcsp_dq_out=zero, mcsp_du_out=zero, mcsp_dv_out=zero,
              mcsp_freq=zero[:, 0], mcsp_shear=zero[:, 0], zm_depth=zero[:, 0])
    if rt.mcsp_enabled:
        mc = zm_conv_mcsp_tend(rt, dt, m["jctop"], p_mid, p_int, p_del, uwind, vwind,
                               tend_out_s, tend_out_qv)
        tend_out_s = tend_out_s + mc["ptend_s"]
        tend_out_qv = tend_out_qv + mc["ptend_q"]
        tend_out_u = tend_out_u + mc["ptend_u"]
        tend_out_v = tend_out_v + mc["ptend_v"]

    # state updated by the tendencies so far, for evaporation and momentum transport
    tmp_T = T_mid + tend_out_s / PC.CP * dt
    tmp_qv = qv + tend_out_qv * dt
    tmp_winds = torch.stack([uwind + tend_out_u * dt, vwind + tend_out_v * dt], 1)

    # rain evaporation and snow (active columns)
    ev = zm_conv_evap(rt, setup.estbl, dt, p_mid, p_del, tmp_T, tmp_qv, m["rprd"],
                      s["cldfrac_tot"], m["prec"])
    tend_tmp_s = torch.where(act2, ev["tend_s"], zero)
    tend_tmp_qv = torch.where(act2, ev["tend_q"], zero)
    prec = torch.where(active, ev["prec"], m["prec"])
    snow = torch.where(active, ev["snow"], torch.zeros_like(m["prec"]))
    tend_out_s = torch.where(act2, tend_out_s + tend_tmp_s, tend_out_s)
    tend_out_qv = torch.where(act2, tend_out_qv + tend_tmp_qv, tend_out_qv)
    evap_ds_out, evap_dq_out = tend_tmp_s, tend_tmp_qv

    # convective momentum transport (active columns)
    dp_mb = torch.where(act2, m["p_del"], torch.ones_like(m["p_del"]))
    wind_tend, seten = zm_transport_momentum(dt, tmp_winds, m["mflx_up"], m["mflx_dn"],
                                             m["detr_up"], m["entr_up"], m["entr_dn"], dp_mb,
                                             m["jt"], m["msemax_klev"], m["ktm"], m["kbm"])
    tend_out_s = torch.where(act2, tend_out_s + seten, tend_out_s)
    tend_out_u = torch.where(act2, tend_out_u + wind_tend[:, 0], tend_out_u)
    tend_out_v = torch.where(act2, tend_out_v + wind_tend[:, 1], tend_out_v)

    # surface precipitation
    prec_liq = prec - snow
    precip_liq_surf_mass = s["precip_liq_surf_mass"] + max_(0.0, prec_liq) * PC.RHO_H2O * dt
    precip_ice_surf_mass = s["precip_ice_surf_mass"] + snow * PC.RHO_H2O * dt

    # detrained condensate split into liquid and ice by temperature; state update
    ice_frac = torch.where(T_mid > ZMC.dlf_tk1, zero, torch.where(
        T_mid < ZMC.dlf_tk2, torch.ones_like(T_mid),
        (ZMC.dlf_tk1 - T_mid) / (ZMC.dlf_tk1 - ZMC.dlf_tk2)))
    dlf_pos = max_(m["dlf"], 0.0)
    zm_detr_qc = (1 - ice_frac) * dlf_pos
    zm_detr_qi = ice_frac * dlf_pos
    zm_detr_nc = zm_detr_qc / _LIQ_MASS
    zm_detr_ni = zm_detr_qi / _ICE_MASS
    tend_out_t = tend_out_s / PC.CP
    T_new = T_mid + tend_out_t * dt
    qv_new = qv + tend_out_qv * dt
    out = {}
    if rt.apply_detr_tend:
        out["qc"] = qc + zm_detr_qc * dt
        out["qi"] = s["qi"] + zm_detr_qi * dt
        out["nc"] = s["nc"] + zm_detr_nc * dt
        out["ni"] = s["ni"] + zm_detr_ni * dt
        T_new = T_new + zm_detr_qi * PC.LatIce / PC.CP * dt
    winds = torch.stack([uwind + tend_out_u * dt, vwind + tend_out_v * dt], 1)

    out.update(
        T_mid=T_new, qv=qv_new, horiz_winds=winds, zm_t_prev=T_new, zm_q_prev=qv_new,
        precip_liq_surf_mass=precip_liq_surf_mass, precip_ice_surf_mass=precip_ice_surf_mass,
        deep_conv_t_tend=tend_out_t, zm_detr_qc=zm_detr_qc, zm_detr_qi=zm_detr_qi,
        zm_detr_nc=zm_detr_nc, zm_detr_ni=zm_detr_ni,
        zm_prec=prec, zm_snow=snow, zm_cape=m["cape"], zm_dcape=m["dcape"],
        zm_activity=active.to(T_mid.dtype),
        zm_mflx_up=m["mflx_up"], zm_entr_up=m["entr_up"], zm_detr_up=m["detr_up"],
        zm_mflx_dn=m["mflx_dn"], zm_entr_dn=m["entr_dn"], zm_mass_flux_int=m["mcon"],
        mcsp_ds_out=mc["mcsp_ds_out"], mcsp_dq_out=mc["mcsp_dq_out"],
        mcsp_du_out=mc["mcsp_du_out"], mcsp_dv_out=mc["mcsp_dv_out"],
        mcsp_freq=mc["mcsp_freq"], mcsp_shear=mc["mcsp_shear"], zm_depth=mc["zm_depth"],
        evap_ds_out=evap_ds_out, evap_dq_out=evap_dq_out,
    )
    out["_internal"] = dict(main=m, tpert=tpert, z_mid=z_mid, z_int=z_int,
                            nonconverged=m["nonconverged"])
    return out
