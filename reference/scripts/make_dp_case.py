#!/usr/bin/env python3
"""Create a run directory for a DP-EAMxx IOP case on the standalone reference build.

The settings reproduce what CIME's EAMxx buildnml would generate for compset
FIOP-SCREAMv1-DP on grid ne30pg2_ne30pg2 (components/eamxx/cime_config/
namelist_defaults_eamxx.xml at the pinned commit), plus the per-case
atmchange lines of E3SM-Project/scmlib DPxx_SCREAM_SCRIPTS/run_dpxx_scream_<CASE>.csh.

Known differences from a CIME DP-EAMxx run (see docs/plan.md, "Reference"):
  * No coupler. Surface fields that the coupler would supply come from the
    IOP file (iop_srf_prop) or from the IC defaults below.
  * SPA aerosol data comes from the ne4 file (nearest column) instead of the
    4.4 GB ne30pg2 file.
  * sc_export is omitted (it only sends fields to the coupler).

Usage:
  python reference/scripts/make_dp_case.py bomex [--ne 5] [--hours 6] [--out DIR]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

REF_ROOT = Path(__file__).resolve().parents[1]

# Per-case settings, copied from scmlib DPxx_SCREAM_SCRIPTS (commit in docs/plan.md).
CASES = {
    "bomex": dict(
        lat=15.0, lon=300.0, iop_file="BOMEX_iopfile_4scam.nc",
        srf_prop=True, nudge_tq=False, nudge_uv=False, coriolis=False, subsidence=True,
        start_date="1969-06-25", start_sec=0, hours=6, ne=5, length=50000.0,
    ),
    "rico": dict(
        lat=17.97, lon=298.54, iop_file="RICO_iopfile_4scam.nc",
        srf_prop=False, nudge_tq=False, nudge_uv=False, coriolis=False, subsidence=True,
        start_date="2004-12-16", start_sec=0, hours=24, ne=5, length=50000.0,
    ),
    "dycomsrf01": dict(
        lat=31.5, lon=238.5, iop_file="DYCOMSrf01_iopfile_4scam.nc",
        srf_prop=True, nudge_tq=False, nudge_uv=False, coriolis=False, subsidence=True,
        start_date="1999-07-10", start_sec=0, hours=6, ne=5, length=50000.0,
    ),
    "arm97": dict(
        lat=36.605, lon=262.515, iop_file="ARM97_iopfile_4scam.nc",
        srf_prop=True, nudge_tq=False, nudge_uv=True, coriolis=False, subsidence=False,
        start_date="1997-06-19", start_sec=84585, hours=26 * 24, ne=20, length=200000.0,
    ),
    "twp06": dict(
        lat=-12.425, lon=130.891, iop_file="TWP06_iopfile_4scam.nc",
        srf_prop=True, nudge_tq=False, nudge_uv=True, coriolis=False, subsidence=False,
        start_date="2006-01-17", start_sec=10800, hours=26 * 24, ne=20, length=200000.0,
    ),
    "cass": dict(  # EAMxx's own DP test case (tests/.../homme_shoc_cld_spa_p3_rrtmgp_pg2_dp)
        lat=36.605, lon=262.515, iop_file="CASS_iopfile_4scam.nc",
        srf_prop=True, nudge_tq=True, nudge_uv=True, coriolis=True, subsidence=True,
        start_date="2000-07-24", start_sec=61100, hours=9, ne=5, length=50000.0,  # IOP data end 9.03 h after start
    ),
    "cass_scmlib": dict(  # scmlib run_dpxx_scream_CASS.csh (06:00 LST start, 14 h, wind nudging only)
        lat=36.605, lon=262.515, iop_file="CASS_iopfile_4scam.nc",
        srf_prop=True, nudge_tq=False, nudge_uv=True, coriolis=False, subsidence=True,
        start_date="2000-07-24", start_sec=43200, hours=14, ne=5, length=50000.0,
    ),
    "arm97_active": dict(  # ARM97 settings; the 8-day window from 23 June 1997 of Bogenschutz et al. (2023)
        lat=36.605, lon=262.515, iop_file="ARM97_iopfile_4scam.nc",
        srf_prop=True, nudge_tq=False, nudge_uv=True, coriolis=False, subsidence=False,
        start_date="1997-06-23", start_sec=0, hours=8 * 24, ne=20, length=200000.0,
    ),
    "gateiii": dict(  # scmlib run_dpxx_scream_GATEIII.csh; no IOP surface fluxes (ocean, Tg given)
        lat=9.0, lon=336.0, iop_file="GATEIII_iopfile_4scam.nc",
        srf_prop=False, nudge_tq=False, nudge_uv=True, coriolis=False, subsidence=True,
        start_date="1974-08-30", start_sec=0, hours=20 * 24, ne=20, length=200000.0,
    ),
    "gabls": dict(
        lat=73.0, lon=180.0, iop_file="GABLS_iopfile_4scam.nc",
        srf_prop=True, nudge_tq=False, nudge_uv=False, coriolis=False, subsidence=False,
        start_date="1999-07-01", start_sec=0, hours=9, ne=5, length=50000.0,
    ),
}

INPUT_YAML = """%YAML 1.1
---
driver_options:
  atmosphere_dag_verbosity_level: 0
  atm_log_level: info
  atm_flush_level: warn
  output_to_screen: false
  mass_column_conservation_error_tolerance: 1e-10
  energy_column_conservation_error_tolerance: 1e-14
  column_conservation_checks_fail_handling_type: warning
  check_all_computed_fields_for_nans: true
  enable_iop: true

iop_options:
  doubly_periodic_mode: true
  iop_file: {iop_path}
  target_latitude: {lat}
  target_longitude: {lon}
  iop_dosubsidence: {subsidence}
  iop_srf_prop: {srf_prop}
  iop_nudge_uv: {nudge_uv}
  iop_nudge_tq: {nudge_tq}
  iop_nudge_tq_low: 1100.0
  iop_nudge_tq_high: 0.0
  iop_nudge_tscale: 10800.0
  iop_coriolis: {coriolis}

time_stepping:
  time_step: {dt}
  run_t0: {run_t0}
  number_of_steps: {nsteps}

initial_conditions:
  filename: {ic_file}
  topography_filename: {topo_file}
  surf_evap: 0.0
  precip_liq_surf_mass: 0.0
  precip_ice_surf_mass: 0.0
  cldfrac_liq: 0.0
  sgs_buoy_flux: 0.0
  eddy_diff_mom: 0.0
  eddy_diff_heat: 0.0
  eddy_diff_mom_horiz: 0.0
  eddy_diff_heat_horiz: 0.0
  T_prev_micro_step: 0.0
  qv_prev_micro_step: 0.0
  qr: 0.0
  nr: 0.0
  qm: 0.0
  bm: 0.0
  ni_activated: 0.0
  nc_nuceat_tend: 0.0
  tke: 0.0
  pbl_height: 2000.0
  sfc_alb_dir_vis: 0.0
  sfc_alb_dir_nir: 0.0
  sfc_alb_dif_vis: 0.0
  sfc_alb_dif_nir: 0.0
  surf_sens_flux: 0.0
  surf_lw_flux_up: 0.0
  perturbed_fields: [T_mid]
  generate_perturbation_random_seed: false
  perturbation_random_seed: 0
  perturbation_limit: 0.001
  perturbation_minimum_pressure: 900.0

eamxx:
  atm_procs_list: [sc_import,homme,physics]
  schedule_type: sequential
  type: group
  sc_import:
    number_of_subcycles: 1
  homme:
    moisture: moist
    compute_tendencies: [T_mid,qv]
  physics:
    atm_procs_list: [{physics_list}]
{zm_block}    schedule_type: sequential
    type: group
    iop_forcing:
      compute_tendencies: [T_mid,qv]
    mac_aero_mic:
      atm_procs_list: [tms,shoc,cld_fraction,spa,p3]
      schedule_type: sequential
      type: group
      number_of_subcycles: {mac_subcycles}
      tms: {{}}
      shoc:
        enable_column_conservation_checks: false
        check_flux_state_consistency: false
        lambda_low: 0.001
        lambda_high: 0.08
        lambda_slope: 2.65
        lambda_thresh: 0.02
        thl2tune: 1.0
        qw2tune: 1.0
        qwthl2tune: 1.0
        w2tune: 1.0
        length_fac: 0.5
        c_diag_3rd_mom: 7.0
        coeff_kh: 0.1
        coeff_km: 0.1
        coeff_kh_horiz: 0.1
        coeff_km_horiz: 0.1
        extra_shoc_diags: true
        shoc_1p5tke: false
        compute_tendencies: [T_mid,qv]
      cld_fraction: {{}}
      spa:
        spa_remap_file: none
        spa_data_file: {spa_file}
        time_interpolation_method: yearly_periodic
      p3:
        do_prescribed_ccn: true
        use_hetfrz_classnuc: false
        do_predict_nc: true
        enable_column_conservation_checks: false
        max_total_ni: 740.0e3
        tables: [{p3_tables}]
        autoconversion_prefactor: 1350.0
        autoconversion_qc_exponent: 2.47
        autoconversion_nc_exponent: 1.79
        autoconversion_radius: 25.0e-6
        accretion_prefactor: 67.0
        accretion_qc_exponent: 1.15
        accretion_qr_exponent: 1.15
        rain_selfcollection_prefactor: 5.78
        rain_selfcollection_breakup_diameter: 0.00028
        constant_mu_rain: 1.0
        spa_ccn_to_nc_factor: 2000.0
        spa_ccn_to_nc_exponent: 0.55
        cldliq_to_ice_collection_factor: 0.5
        rain_to_ice_collection_factor: 1.0
        min_rime_rho: 50.0
        max_rime_rho: 900.0
        immersion_freezing_exponent: 0.65
        deposition_nucleation_exponent: 0.304
        ice_sedimentation_factor: 1.0
        do_ice_production: true
        set_cld_frac_l_to_one: false
        set_cld_frac_r_to_one: true
        set_cld_frac_i_to_one: false
        use_separate_ice_liq_frac: false
        extra_p3_diags: false
        compute_tendencies: [T_mid,qv]
    rrtmgp:
      rrtmgp_coefficients_file_sw: {data}/scream/init/rrtmgp-data-sw-g112-210809.nc
      rrtmgp_coefficients_file_lw: {data}/scream/init/rrtmgp-data-lw-g128-210809.nc
      rrtmgp_cloud_optics_file_sw: {data}/scream/init/rrtmgp-cloud-optics-coeffs-sw.nc
      rrtmgp_cloud_optics_file_lw: {data}/scream/init/rrtmgp-cloud-optics-coeffs-lw.nc
      column_chunk_size: 1280
      active_gases: [h2o, co2, o3, n2o, co, ch4, o2, n2]
      ch4vmr: 1807.851e-9
      co2vmr: 388.717e-6
      n2ovmr: 323.141e-9
      f11vmr: 768.7644e-12
      f12vmr: 531.2820e-12
      n2vmr: 0.7906
      covmr: 1.0e-7
      orbital_year: 2010
      orbital_eccentricity: -9999.0
      orbital_obliquity: -9999.0
      orbital_mvelp: -9999.0
      fixed_total_solar_irradiance: -9999.0
      rad_frequency: {rad_frequency}
      do_aerosol_rad: true
      enable_column_conservation_checks: false
      extra_clnclrsky_diag: false
      extra_clnsky_diag: false
      do_subcol_sampling: true
      pool_size_multiplier: 1.0
      compute_tendencies: [T_mid]

grids_manager:
  type: homme
  physics_grid_type: pg2
  physics_grid_rebalance: none
  dynamics_namelist_file_name: namelist.nl
  vertical_coordinate_filename: {vcoord_file}

scorpio:
  output_yaml_files: [{output_yamls}]
...
"""


# ZM settings: CIME defaults (namelist_defaults_eamxx.xml, <zm>), used by the
# EAMxx low-resolution development compsets F2010xx-ZM (physics: zm,mac_aero_mic,rrtmgp).
ZM_BLOCK = """    zm:
      apply_detr_tend: true
      trig_dcape: true
      trig_ull: true
      clos_dyn_adj: false
      upper_limit_pref: 40e2
      tau: 3600.0
      alfa: 0.14
      ke: 2.5E-6
      dmpdz: -0.7e-3
      tpert_fix: true
      tpert_fac: 2.0
      tiedke_add: 0.8
      c0_lnd: 0.002
      c0_ocn: 0.002
      num_cin: 1
      cape_threshold: 0.0
      dcape_threshold: 0.0
      mcsp_enabled: true
      mcsp_t_coeff: 0.3
      mcsp_q_coeff: 0.0
      mcsp_u_coeff: 0.0
      mcsp_v_coeff: 0.0
"""

# HOMME ctl_nl: CIME defaults for DP-EAMxx on ne30pg2 (pg2 selectors apply).
NAMELIST = """&ctl_nl
cubed_sphere_map = 2
disable_diagnostics = .false.
dt_remap_factor = 2
dt_tracer_factor = 6
hv_ref_profiles = 6
hypervis_order = 2
hypervis_scaling = 3.0
laplace_scaling = 0.0
hypervis_subcycle = 1
horiz_turb_subcycle = 1
hypervis_subcycle_tom = 1
hypervis_subcycle_q = 6
horiz_turb_subcycle_q = 1
nu = 0.216784
nu_top = 1.0e4
tom_sponge_start = 0.0
pgrad_correction = 1
se_ftype = 2
se_geometry = "plane"
se_limiter_option = 9
se_ne = 1024
se_ne_x = {ne}
se_ne_y = {ne}
se_lx = {length}
se_ly = {length}
se_nsplit = -1
se_partmethod = 4
se_topology = "plane"
se_tstep = {se_tstep}
statefreq = 9999
theta_advect_form = 1
theta_hydrostatic_mode = .false.
tstep_type = 9
vert_remap_q_alg = 10
transport_alg = 12
vtheta_thresh = 100.0
internal_diagnostics_level = 0
semi_lagrange_trajectory_nsubstep = 0
semi_lagrange_trajectory_nvelocity = -1
semi_lagrange_halo = -1
semi_lagrange_diagnostics = 0
mesh_file = "none"
do_3d_turbulence = .false.
/
"""

HORIZ_AVG_YAML = """%YAML 1.1
---
averaging_type: {avg}
max_snapshots_per_file: 100000
filename_prefix: {prefix}
fields:
  physics_pg2:
    field_names:
{fields}
output_control:
  frequency: {freq}
  frequency_units: {units}
...
"""

PROFILE_FIELDS = [
    "z_mid", "p_mid", "ps", "omega", "T_mid", "qv", "qc", "qr", "qi", "nc", "nr", "ni",
    "qm", "bm", "horiz_winds", "cldfrac_liq", "cldfrac_ice", "cldfrac_tot", "tke",
    "eddy_diff_mom", "eddy_diff_heat", "sgs_buoy_flux", "inv_qc_relvar", "pbl_height",
    "PotentialTemperature", "LiqPotentialTemperature", "RelativeHumidity",
    "LiqWaterPath", "IceWaterPath", "RainWaterPath", "VapWaterPath",
    "precip_liq_surf_mass_flux", "precip_ice_surf_mass_flux", "precip_total_surf_mass_flux",
    "rad_heating_pdel", "sfc_flux_lw_dn", "sfc_flux_sw_net",
    "LW_flux_up_at_model_top", "SW_flux_up_at_model_top", "SW_flux_dn_at_model_top",
    "surf_sens_flux", "surf_evap", "surf_mom_flux", "surf_radiative_T",
    "shoc_T_mid_tend", "p3_T_mid_tend", "rrtmgp_T_mid_tend", "homme_T_mid_tend",
    "iop_forcing_T_mid_tend", "shoc_qv_tend", "p3_qv_tend", "homme_qv_tend",
    "iop_forcing_qv_tend",
    "shoc_mix", "wthl_sec", "wqw_sec", "qw_sec", "thl_sec", "w3", "w_variance",
]

ZM_FIELDS = ["zm_prec", "zm_snow", "zm_cape", "zm_dcape", "zm_activity", "mcsp_freq", "mcsp_shear",
             "zm_depth", "deep_conv_t_tend"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case", choices=sorted(CASES))
    ap.add_argument("--ne", type=int, help="elements per side (default: case value)")
    ap.add_argument("--length", type=float, help="domain side [m] (default: case value)")
    ap.add_argument("--hours", type=float, help="run length [h] (default: case value)")
    ap.add_argument("--dt", type=int, help="physics step [s] (default 100; 1800 with --lr)")
    ap.add_argument("--lr", action="store_true",
                    help="EAMxx-LR development config (F2010xx-ZM on ne30pg2): ZM on, dt=1800 s, "
                         "mac_aero_mic x12 subcycles, rad every 2 steps, 5x5 elements of 333 km (HOMME SL transport needs ne>=5)")
    ap.add_argument("--out", type=Path, help="run directory (default: reference/runs/dp_<case>)")
    args = ap.parse_args()

    c = dict(CASES[args.case])
    if args.lr:
        c.update(ne=5, length=5 * 333000.0)
    args.dt = args.dt or (1800 if args.lr else 100)
    ne = args.ne or c["ne"]
    length = args.length or c["length"]
    hours = args.hours or c["hours"]
    nsteps = int(round(hours * 3600 / args.dt))
    out = args.out or REF_ROOT / "runs" / f"dp_{args.case}_{'lr_' if args.lr else ''}ne{ne}"
    out.mkdir(parents=True, exist_ok=True)

    data = Path(os.environ.get("SCREAM_INPUT_ROOT", REF_ROOT / "inputdata")) / "atm"
    tables = ", ".join(
        str(data / "scream/tables" / t)
        for t in ("p3_lookup_table_1.dat-v4.1.1", "mu_r_table_vals_v2.dat8",
                  "revap_table_vals_v2.dat8", "vn_table_vals_v2.dat8", "vm_table_vals_v2.dat8")
    )
    yb = lambda b: "true" if b else "false"  # noqa: E731

    (out / "input.yaml").write_text(INPUT_YAML.format(
        iop_path=data / "cam/scam/iop" / c["iop_file"],
        lat=c["lat"], lon=c["lon"], subsidence=yb(c["subsidence"]), srf_prop=yb(c["srf_prop"]),
        nudge_uv=yb(c["nudge_uv"]), nudge_tq=yb(c["nudge_tq"]), coriolis=yb(c["coriolis"]),
        dt=args.dt, run_t0=f"{c['start_date']}-{c['start_sec']:05d}", nsteps=nsteps,
        ic_file=data / "scream/init/screami_ne30np4L128_20260813.nc",
        topo_file=data / "cam/topo/USGS-gtopo30_ne30np4pg2_x6t-SGH.c20210614.nc",
        spa_file=data / "scream/init/spa_file_unified_and_complete_ne4_20220428.nc",
        p3_tables=tables, data=data,
        vcoord_file=data / "scream/init/vertical_coordinates_L128_20220927.nc",
        output_yamls="./profiles_horiz_avg.yaml",
        physics_list="iop_forcing,zm,mac_aero_mic,rrtmgp" if args.lr else "iop_forcing,mac_aero_mic,rrtmgp",
        zm_block=ZM_BLOCK if args.lr else "",
        mac_subcycles=12 if args.lr else 1,
        rad_frequency=2 if args.lr else 3,
    ))
    (out / "namelist.nl").write_text(NAMELIST.format(
        ne=ne, length=length, se_tstep=300 if args.lr else "8.3333333333333"))
    (out / "profiles_horiz_avg.yaml").write_text(HORIZ_AVG_YAML.format(
        avg="average", prefix=f"dp_{args.case}.horiz_avg",
        fields="\n".join(f"    - {f}_horiz_avg" for f in PROFILE_FIELDS + (ZM_FIELDS if args.lr else [])),
        freq=args.dt // 60 if args.lr else 15, units="nmins",
    ))
    print(out)


if __name__ == "__main__":
    main()
