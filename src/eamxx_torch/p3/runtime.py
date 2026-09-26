"""P3 runtime options: the P3Runtime struct of p3_functions.hpp at E3SM 80fe0a71.

Defaults are the struct's defaults. EAMxx reads them from the namelist
(`load_runtime_options_from_file`); the DP-EAMxx configuration sets
do_predict_nc and do_prescribed_ccn (which live in P3Infrastructure) and
leaves these at their defaults.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class P3Runtime:
    max_total_ni: float = 740.0e3
    autoconversion_prefactor: float = 1350.0
    autoconversion_qc_exponent: float = 2.47
    autoconversion_nc_exponent: float = 1.79
    autoconversion_radius: float = 25.0e-6
    accretion_prefactor: float = 67.0
    accretion_qc_exponent: float = 1.15
    accretion_qr_exponent: float = 1.15
    rain_selfcollection_prefactor: float = 5.78
    rain_selfcollection_breakup_diameter: float = 0.00028
    constant_mu_rain: float = 1.0
    spa_ccn_to_nc_factor: float = 1.0
    spa_ccn_to_nc_exponent: float = 1.0
    cldliq_to_ice_collection_factor: float = 0.5
    rain_to_ice_collection_factor: float = 1.0
    min_rime_rho: float = 50.0
    max_rime_rho: float = 900.0
    immersion_freezing_exponent: float = 0.65
    deposition_nucleation_exponent: float = 0.304
    ice_sedimentation_factor: float = 1.0
    do_ice_production: bool = True
    set_cld_frac_l_to_one: bool = False
    set_cld_frac_i_to_one: bool = False
    set_cld_frac_r_to_one: bool = False
    use_hetfrz_classnuc: bool = False
    use_separate_ice_liq_frac: bool = False
    extra_p3_diags: bool = False
    # Not EAMxx (off by default; docs/float32_gpu.md). In a float32 run, evaluate
    # cldliq_immersion_freezing in float64 and round its two tendencies to float32. In float,
    # CONS6*cdist1*tgamma(7+mu_c)*expAimmDt overflows below about 195 K (as in EAMxx's float
    # build), and cloud_water_conservation turns the inf into NaN. No effect in float64.
    immersion_freezing_in_float64: bool = False
