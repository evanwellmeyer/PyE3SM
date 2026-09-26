"""ZM constants and runtime options.

Translation of, at E3SM 80fe0a71:
  components/eamxx/src/physics/zm/zm_functions.hpp   struct ZMC, struct ZmRuntimeOpt
      (load_runtime_options, set_limcnv; the saturation vapor pressure table estbl)
  components/eamxx/src/physics/zm/impl/zm_inline_functions.hpp   svp_trans and the
      Goff-Gratch functions it calls (only for building estbl; the per-point versions
      are in functions.py)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .. import constants as PC


class ZMC:
    nwind = 2
    tboil = 373.16
    pref = 1000.0
    cpwv = 1.810e3
    LOOPMAX = 100
    nit_lheat = 2
    omeps = 1 - PC.ep_2
    omsm = 0.99999
    tol_coeff = 0.001
    tol_eps = 3.e-8
    half = 0.5
    small = 1.e-36
    cdifr_min = 1.e-6
    maxc_factor = 1.e-12
    flux_factor = 1.e-12
    mbsth = 1.e-15
    small_conv = 1.e-20
    interp_diff_min = 1.E-6
    zvir = 0.608
    lcl_coeff_a = 2840.0
    lcl_coeff_b = 3.5
    lcl_coeff_c = 4.805
    pa_to_mb = 0.01
    mb_to_pa = 100.0
    cld_liq_radius = 8.e-6
    cld_ice_radius = 14.e-6
    cld_liq_density = 997.0
    cld_ice_density = 500.0
    lcl_pressure_threshold = 600.0
    lwmax = 1.e-3
    ull_upper_launch_pressure = 600.0
    cape_threshold = 0.
    dcape_threshold = 0.
    beta = 0.0
    mu_min = 0.02
    hu_diff_min = -2000.0
    lambda_limit_min = 0.0
    lambda_limit_max = 0.0002
    lambda_threshold = 1.e-6
    pergro_rhd_threshold = -1.e-4
    pergro_perturbation = 8.64e-11
    momcu = 0.4
    momcd = 0.4
    mse_min_diff = 100.0
    tpert_limiter = 2.0
    MCSP_storm_speed_pref = 600e2
    MCSP_conv_depth_min = 700e2
    MCSP_shear_min = 3.0
    MCSP_shear_max = 200.0
    MCSP_t_coeff_default = 0.3
    MCSP_q_coeff_default = 0.0
    MCSP_u_coeff_default = 0.0
    MCSP_v_coeff_default = 0.0
    alfa = 0.14
    ke = 2.5E-6
    dmpdz = -0.7e-3
    tiedke_add = 0.8
    c0 = 0.0020
    tmin = 127.16
    tmax = 375.16
    h2otrip = 273.16
    ttrice = 20.0
    dlf_tk1 = 268.15
    dlf_tk2 = 238.15


# --------------------------------------------------------------------- estbl (host, once)
def _svp_water(t: float) -> float:
    """goffgratch_svp_water on a Python float, with libm (as the table is built).

    clang turns pow(10, x) into __exp10(x) in the reference build, so this calls
    __exp10; log10(magic7) is folded at compile time with the same libm.
    """
    from ..mathfn import _libm_fn
    exp10 = _libm_fn("__exp10")
    magic1, magic2, magic3, magic4 = -7.90298, 5.02808, 1.3816e-7, 11.344
    magic5, magic6, magic7 = 8.1328e-3, -3.49149, 1013.246
    arg = (magic1 * (ZMC.tboil / t - 1)
           + magic2 * math.log10(ZMC.tboil / t)
           - magic3 * (float(exp10(magic4 * (1 - t / ZMC.tboil))) - 1)
           + magic5 * (float(exp10(magic6 * (ZMC.tboil / t - 1))) - 1)
           + math.log10(magic7))
    return float(exp10(arg)) * 100


def _svp_ice(t: float) -> float:
    from ..mathfn import _libm_fn
    exp10 = _libm_fn("__exp10")
    magic1, magic2, magic3, magic4 = -9.09718, 3.56654, 0.876793, 6.1071
    arg = (magic1 * (ZMC.h2otrip / t - 1)
           - magic2 * math.log10(ZMC.h2otrip / t)
           + magic3 * (1 - t / ZMC.h2otrip)
           + math.log10(magic4))
    return float(exp10(arg)) * 100


def svp_trans(t: float) -> float:
    """svp_trans: water above Tmelt-ttrice, ice below Tmelt, blended in between."""
    es = _svp_water(t) if t >= (PC.Tmelt - ZMC.ttrice) else 0.0
    if t < PC.Tmelt:
        esice = _svp_ice(t)
        weight = 1.0 if (PC.Tmelt - t) > ZMC.ttrice else (PC.Tmelt - t) / ZMC.ttrice
        es = weight * esice + (1 - weight) * es
    return es


def build_estbl() -> np.ndarray:
    """ZmRuntimeOpt::load_runtime_options: estbl(i) = svp_trans(tmin + i), i < plenest."""
    plenest = int(ZMC.tmax - ZMC.tmin) + 3
    return np.array([svp_trans(ZMC.tmin + i) for i in range(plenest)], dtype=np.float64)


# --------------------------------------------------------------------- runtime options
@dataclass(frozen=True)
class ZMRuntime:
    """ZmRuntimeOpt as load_runtime_options fills it (defaults = EAMxx defaults,
    which are also the CIME defaults for the F2010xx-ZM compsets)."""
    apply_detr_tend: bool = True
    upper_limit_pref: float = 40e2
    tau: float = 3600.0
    alfa: float = ZMC.alfa
    ke: float = ZMC.ke
    dmpdz: float = ZMC.dmpdz
    tpert_fix: bool = True
    tpert_fac: float = 2.0
    tiedke_add: float = ZMC.tiedke_add
    c0_lnd: float = ZMC.c0
    c0_ocn: float = ZMC.c0
    num_cin: int = 1
    mx_bot_lyr_adj: int = 1
    trig_dcape: bool = True
    trig_ull: bool = True
    clos_dyn_adj: bool = False
    no_deep_pbl: bool = False
    cape_threshold: float = ZMC.cape_threshold
    dcape_threshold: float = ZMC.dcape_threshold
    zm_microp: bool = False
    old_snow: bool = True
    mcsp_enabled: bool = True
    mcsp_t_coeff: float = ZMC.MCSP_t_coeff_default
    mcsp_q_coeff: float = ZMC.MCSP_q_coeff_default
    mcsp_u_coeff: float = ZMC.MCSP_u_coeff_default
    mcsp_v_coeff: float = ZMC.MCSP_v_coeff_default


def set_limcnv(upper_limit_pref: float, hyai, hybi) -> int:
    """ZmRuntimeOpt::set_limcnv: the interface level index where the reference pressure
    profile (ps0 = P0) crosses upper_limit_pref."""
    if not (0 < upper_limit_pref < PC.P0):
        raise ValueError(f"upper_limit_pref must be in (0, P0), got {upper_limit_pref}")
    hyai = [float(x) for x in hyai]
    hybi = [float(x) for x in hybi]
    nlev = len(hyai) - 1
    ps0 = PC.P0
    if ps0 * hyai[0] + ps0 * hybi[0] >= upper_limit_pref:
        return 0
    for k in range(nlev):
        pk0 = ps0 * hyai[k] + ps0 * hybi[k]
        pk1 = ps0 * hyai[k + 1] + ps0 * hybi[k + 1]
        if pk0 < upper_limit_pref and pk1 >= upper_limit_pref:
            return k
    return nlev + 1
