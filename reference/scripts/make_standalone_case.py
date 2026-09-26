#!/usr/bin/env python3
"""Build a standalone (no dycore, point grid) EAMxx case from dumped columns plus overrides.

  python reference/scripts/make_standalone_case.py PRESET OUTDIR

Writes OUTDIR/ic.nc and OUTDIR/input.yaml for the physics_standalone executable
(reference/eamxx_ext/physics_standalone). Each column of the case is a column of
a DP-EAMxx dump ("pre" state of a process call) with some fields overridden, so
the reference runs the physics on realistic profiles in regimes the DP cases
never reach. Presets:

  shoc_tms  tms + shoc on 8 ARM97 columns (day and night) with nonzero surface
            stress (incl. |tau| < 1e-12 and one zero component), land fraction
            0 / 0.5 / 1 (TMS drag on), a column with a surface layer colder than
            182 K (SHOC's stable-PBL fallback), and strong surface fluxes.

  p3        p3 alone on 8 columns (ARM97 ice, BOMEX warm) with a supercooled
            liquid column (homogeneous freezing), a tiny-nc column, a very dry
            ice column (sublimation limiters) and a very wet column
            (conservation limiters). extra_p3_diags is on.

  rrtmgp, rrtmgp_alt
            rrtmgp alone on 8 columns (ARM97 ice cloud, CASS day, DYCOMS
            stratocumulus) with temperatures outside the gas-optics tables,
            effective radii outside the cloud-optics tables, an optically very
            thin top layer, and latitudes for polar day, polar night, a very short
            night and |lat| > 45. rrtmgp: fixed orbit and solar constant, extra
            clean/clear-sky calls. rrtmgp_alt: orbit from the date, fixed zenith
            angle, no subcolumn sampling, no aerosol.

  shoc_sp   shoc alone on the shoc_tms columns (for the single-precision build, which
            has no TMS; SHOC's required surf_drag_coeff_tms is 0 in the IC).

  mac_sp    shoc, cld_fraction, p3 chained for 36 steps on 8 realistic columns (BOMEX
            and DYCOMS-RF01 warm cloud, ARM97 ice and rain): the processes EAMxx builds
            in single precision, fed by each other as in a run. nccn, ni_activated and
            nc_nuceat_tend stay at their IC values (no SPA).

--single writes a case for the single-precision build: P3 reads its *.dat4 tables.

The dump fields are read at the column's own values; nothing is interpolated.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import netCDF4
import numpy as np

HERE = Path(__file__).resolve().parent
REF = HERE.parent
sys.path.insert(0, str(HERE))
from read_dump import read_dump  # noqa: E402

VCOORD = REF / "inputdata/atm/scream/init/vertical_coordinates_L128_20220927.nc"

# (run dir, dump stem under dump/shoc, column) + overrides applied to that column
DAY = ("dumps_strict_arm97_7p5d", "s3456_c0_r3_pre")
NIGHT = ("dumps_strict_arm97_7p5d", "s4752_c0_r3_pre")
PRESETS = {
    "shoc_tms": dict(
        procs=["tms", "shoc"], dt=100, nsteps=3, sgh30=300.0,
        columns=[
            (DAY, 0, dict(surf_mom_flux=(0.2, -0.1), landfrac=1.0)),
            (DAY, 1, dict(surf_mom_flux=(1e-13, -1e-13), landfrac=1.0)),
            (DAY, 2, dict(surf_mom_flux=(-0.05, 0.3), landfrac=0.0)),
            (DAY, 3, dict(surf_mom_flux=(0.0, 0.12), landfrac=0.5)),
            (NIGHT, 0, dict(surf_mom_flux=(0.3, 0.3), landfrac=1.0)),
            (NIGHT, 1, dict(surf_mom_flux=(0.0, 0.0), landfrac=1.0)),
            (NIGHT, 2, dict(surf_mom_flux=(-0.02, -0.01), landfrac=0.0, cold_surface=175.0)),
            (NIGHT, 3, dict(surf_mom_flux=(1.5, -0.7), landfrac=1.0, surf_sens_flux=400.0,
                            surf_evap=3e-4)),
        ],
    ),
}

P3_DAY = ("dumps_strict_arm97_7p5d", "s4752_c0_r1_pre")
P3_WARM = ("dumps_strict_bomex", "s216_c0_r0_pre")
PRESETS["p3"] = dict(
    procs=["p3"], dt=100, nsteps=3, sgh30=300.0, proc_dump="p3",
    columns=[
        (P3_DAY, 0, dict(cold_liquid=220.0)),   # supercooled liquid -> homogeneous freezing
        (P3_DAY, 1, dict(tiny_nc=1.0e-20, no_ccn=True)),  # nc below nsmall with cloud present
        (P3_DAY, 2, dict(subliming_ice=0.98)),  # lots of ice just below ice saturation
        (P3_DAY, 3, dict(big_qc=0.02)),         # very wet: conservation limiters
        (P3_WARM, 0, {}),
        (P3_WARM, 1, dict(wet_rain=0.9999)),      # rain evaporating into near-saturated air
        (P3_DAY, 2, dict(saturate=0.999)),      # near liquid saturation: supersaturation limiter
        (P3_DAY, 3, {}),
    ],
)

MID_FIELDS = ["T_mid", "p_mid", "pseudo_density", "omega", "qv", "qc", "tke", "sgs_buoy_flux",
              "eddy_diff_mom", "cldfrac_liq", "um_pert_diff", "vm_pert_diff"]
COL_FIELDS = ["surf_sens_flux", "surf_evap"]

# p3 preset: fields come from a p3 dump (p3's own inputs)
P3_MID_FIELDS = ["T_mid", "p_mid", "p_dry_mid", "pseudo_density", "pseudo_density_dry", "qv",
                 "qc", "nc", "qr", "nr", "qi", "ni", "qm", "bm", "nccn", "nc_nuceat_tend",
                 "ni_activated", "inv_qc_relvar", "cldfrac_tot", "qv_prev_micro_step",
                 "T_prev_micro_step"]
P3_COL_FIELDS = ["precip_liq_surf_mass", "precip_ice_surf_mass"]

PRESETS["shoc_sp"] = dict(PRESETS["shoc_tms"], procs=["shoc"], no_tms=True)

MAC_COLUMNS = [
    (("dumps_strict_bomex", "s108_c0_r0_pre"), 0), (("dumps_strict_bomex", "s108_c0_r0_pre"), 1),
    (("dumps_strict_dycomsrf01", "s108_c0_r0_pre"), 0), (("dumps_strict_dycomsrf01", "s108_c0_r0_pre"), 1),
    (("dumps_strict_arm97_7p5d", "s4752_c0_r1_pre"), 0), (("dumps_strict_arm97_7p5d", "s4752_c0_r1_pre"), 1),
    (("dumps_strict_arm97_7p5d", "s4752_c0_r1_pre"), 2), (("dumps_strict_arm97_7p5d", "s4752_c0_r1_pre"), 3),
]
PRESETS["mac_sp"] = dict(procs=["shoc", "cld_fraction", "p3"], dt=100, nsteps=36, sgh30=300.0,
                         proc_dump="mac", columns=MAC_COLUMNS)

def build(preset: str, out: Path, single: bool = False) -> None:
    spec = PRESETS[preset]
    if spec.get("proc_dump") == "mac":
        build_mac(preset, spec, out, single)
        return
    if spec.get("proc_dump") == "p3":
        build_p3(preset, spec, out)
        if single:
            _single(out)
        return
    if spec.get("proc_dump") == "rrtmgp":
        build_rrtmgp(preset, spec, out)
        return
    cols = []
    for (run, stem), icol, ov in spec["columns"]:
        _, pre = read_dump(REF / "runs" / run / "dump" / "shoc" / stem)
        c = {k: np.array(pre[k][icol], dtype=np.float64) for k in MID_FIELDS + COL_FIELDS + ["p_int", "phis"]}
        c["U"] = pre["horiz_winds"][icol, 0].copy()
        c["V"] = pre["horiz_winds"][icol, 1].copy()
        c["surf_mom_flux"] = pre["surf_mom_flux"][icol].copy()
        c["landfrac"] = np.array(0.0)
        for k, v in ov.items():
            if k == "cold_surface":
                # Surface layer colder than SHOC's tabs_crit (182 K): cap T at v in the lowest 6 levels.
                c["T_mid"][-6:] = np.minimum(c["T_mid"][-6:], v)
            else:
                c[k] = np.array(v, dtype=np.float64)
        cols.append(c)
    ncol = len(cols)
    nlev = cols[0]["T_mid"].shape[0]
    # phis is a single YAML constant on a point grid (the dumps differ by 1 ulp
    # between columns from HOMME's remap); use the first column's value.
    phis = float(cols[0]["phis"])

    out.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(VCOORD) as v, netCDF4.Dataset(out / "ic.nc", "w", format="NETCDF3_64BIT_DATA") as d:  # CDF5, like EAMxx inputdata
        d.createDimension("time", None)
        d.createDimension("ncol", ncol)
        d.createDimension("lev", nlev)
        d.createDimension("ilev", nlev + 1)
        tv = d.createVariable("time", "f8", ("time",))
        tv[0] = 0.0
        for name in ("hyam", "hybm", "hyai", "hybi", "lev", "ilev"):
            var = d.createVariable(name, "f8", v[name].dimensions)
            var[:] = v[name][:]
        d.createVariable("P0", "f8", ())[...] = 100000.0
        lat = d.createVariable("lat", "f8", ("ncol",))
        lon = d.createVariable("lon", "f8", ("ncol",))
        lat[:] = 36.605   # one latitude for every column, so the logged cell_length[0] applies to all
        lon[:] = 262.515
        for k in MID_FIELDS + ["U", "V"]:
            d.createVariable(k, "f8", ("time", "ncol", "lev"))[0] = np.stack([c[k] for c in cols])
        d.createVariable("p_int", "f8", ("time", "ncol", "ilev"))[0] = np.stack([c["p_int"] for c in cols])
        for k in COL_FIELDS + ["landfrac"]:
            d.createVariable(k, "f8", ("time", "ncol"))[0] = np.array([c[k] for c in cols])
        if spec.get("no_tms"):   # SHOC requires TMS's drag coefficient; without TMS it is 0
            d.createVariable("surf_drag_coeff_tms", "f8", ("time", "ncol"))[0] = np.zeros(len(cols))
        # Vector fields are read per component: <name>_U, <name>_V
        for i, comp in enumerate("UV"):
            d.createVariable(f"surf_mom_flux_{comp}", "f8", ("time", "ncol"))[0] = np.array(
                [c["surf_mom_flux"][i] for c in cols])

    shoc = """  shoc:
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
"""
    procs = spec["procs"]
    yaml = f"""%YAML 1.1
---
# Generated by reference/scripts/make_standalone_case.py {preset}
driver_options:
  atmosphere_dag_verbosity_level: 0
time_stepping:
  time_step: {spec['dt']}
  run_t0: 1997-06-26-00000
  number_of_steps: {spec['nsteps']}
eamxx:
  atm_procs_list: [{', '.join(procs)}]
  schedule_type: sequential
  type: group
{shoc if 'shoc' in procs else ''}grids_manager:
  type: mesh_free
  geo_data_source: IC_FILE
  grids_names: [physics_pg2]  # TMS requires this grid name
  physics_pg2:
    type: point_grid
    aliases: [physics]
    number_of_global_columns: {ncol}
    number_of_vertical_levels: {nlev}
initial_conditions:
  filename: {out.resolve() / 'ic.nc'}
  phis: {phis!r}
  sgh30: {spec['sgh30']!r}
scorpio:
  output_yaml_files: []
...
"""
    (out / "input.yaml").write_text(yaml)
    print(f"wrote {out/'ic.nc'} ({ncol} columns) and {out/'input.yaml'}")


def build_p3(preset: str, spec: dict, out: Path) -> None:
    """IC and input.yaml for a p3-only standalone case."""
    cols = []
    for (run, stem), icol, ov in spec["columns"]:
        _, pre = read_dump(REF / "runs" / run / "dump" / "p3" / stem)
        c = {k: np.array(pre[k][icol], dtype=np.float64) for k in P3_MID_FIELDS + P3_COL_FIELDS}
        for k, v in ov.items():
            if k == "cold_liquid":     # cool the cloudy layer below the homogeneous-freezing point
                m = c["qc"] > 1e-8
                c["T_mid"] = np.where(m, np.minimum(c["T_mid"], v), c["T_mid"])
                c["T_prev_micro_step"] = c["T_mid"].copy()
            elif k == "tiny_nc":       # cloud present but almost no droplets
                c["nc"] = np.where(c["qc"] > 1e-8, v, c["nc"])
            elif k == "subliming_ice":  # much ice at ~0 C, air just under ice saturation
                import torch as _t
                from eamxx_torch import saturation as _sat
                m = c["qi"] > 1e-8
                c["T_mid"] = np.where(m, 272.0, c["T_mid"])
                c["T_prev_micro_step"] = c["T_mid"].copy()
                qsat_i = _sat.qv_sat_dry(_t.as_tensor(c["T_mid"]), _t.as_tensor(c["p_dry_mid"]),
                                         True, _sat.MURPHYKOOP).numpy()
                c["qi"] = np.where(m, 5.0e-3, c["qi"])
                c["ni"] = np.where(m, 1.0e7, c["ni"])
                c["qv"] = np.where(m, v * qsat_i, c["qv"])
                c["qv_prev_micro_step"] = c["qv"].copy()
            elif k == "big_qc":        # far more cloud water than the sinks can consume
                c["qc"] = np.where(c["qc"] > 1e-8, v, c["qc"])
            elif k == "wet_rain":      # heavy rain in air just below liquid saturation
                import torch as _t
                from eamxx_torch import saturation as _sat
                qsat = _sat.qv_sat_dry(_t.as_tensor(c["T_mid"]), _t.as_tensor(c["p_dry_mid"]),
                                       False, _sat.MURPHYKOOP).numpy()
                m = (c["T_mid"] > 273.15) & (c["p_dry_mid"] > 7.0e4)
                c["qr"] = np.where(m, 5.0e-3, c["qr"])
                c["nr"] = np.where(m, np.maximum(c["nr"], 1.0e4), c["nr"])
                c["qv"] = np.where(m, v * qsat, c["qv"])
                c["qv_prev_micro_step"] = c["qv"].copy()
            elif k == "no_ccn":        # switch off the prescribed-CCN floor on nc
                c["nccn"] = np.zeros_like(c["nccn"])
            elif k == "saturate":      # air at v * liquid saturation, so sublimation oversaturates
                import torch as _t
                from eamxx_torch import saturation as _sat
                qsat = _sat.qv_sat_dry(_t.as_tensor(c["T_mid"]), _t.as_tensor(c["p_dry_mid"]),
                                       False, _sat.MURPHYKOOP).numpy()
                c["qv"] = np.where(c["qi"] > 1e-8, v * qsat, c["qv"])
                c["qv_prev_micro_step"] = c["qv"].copy()
            else:
                raise ValueError(k)
        cols.append(c)
    ncol = len(cols)
    nlev = cols[0]["T_mid"].shape[0]
    out.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(VCOORD) as v, netCDF4.Dataset(out / "ic.nc", "w",
                                                       format="NETCDF3_64BIT_DATA") as d:
        d.createDimension("time", None)
        d.createDimension("ncol", ncol)
        d.createDimension("lev", nlev)
        d.createDimension("ilev", nlev + 1)
        d.createVariable("time", "f8", ("time",))[0] = 0.0
        for name in ("hyam", "hybm", "hyai", "hybi", "lev", "ilev"):
            d.createVariable(name, "f8", v[name].dimensions)[:] = v[name][:]
        d.createVariable("P0", "f8", ())[...] = 100000.0
        d.createVariable("lat", "f8", ("ncol",))[:] = 36.605
        d.createVariable("lon", "f8", ("ncol",))[:] = 262.515
        for k in P3_MID_FIELDS:
            d.createVariable(k, "f8", ("time", "ncol", "lev"))[0] = np.stack([c[k] for c in cols])
        for k in P3_COL_FIELDS:
            d.createVariable(k, "f8", ("time", "ncol"))[0] = np.array([c[k] for c in cols])
    p3_opts = (REF / "runs/dumps_strict_bomex/input.yaml").read_text()
    blk = re.search(r"^( +)p3:\n(.*?)(?=^\1\S)", p3_opts, re.M | re.S).group(2)
    body = "\n".join("  " + ln[6:] for ln in blk.rstrip().split("\n") if ln.strip())
    body = body.replace("extra_p3_diags: false", "extra_p3_diags: true")
    body = "\n".join(ln for ln in body.split("\n") if "compute_tendencies" not in ln)
    yaml = f"""%YAML 1.1
---
# Generated by reference/scripts/make_standalone_case.py {preset}
driver_options:
  atmosphere_dag_verbosity_level: 0
time_stepping:
  time_step: {spec['dt']}
  run_t0: 1997-06-26-00000
  number_of_steps: {spec['nsteps']}
eamxx:
  atm_procs_list: [p3]
  schedule_type: sequential
  type: group
  p3:
{body}
grids_manager:
  type: mesh_free
  geo_data_source: IC_FILE
  grids_names: [physics_pg2]
  physics_pg2:
    type: point_grid
    aliases: [physics]
    number_of_global_columns: {ncol}
    number_of_vertical_levels: {nlev}
initial_conditions:
  filename: {out.resolve() / 'ic.nc'}
  phis: 0.0
scorpio:
  output_yaml_files: []
...
"""
    (out / "input.yaml").write_text(yaml)
    print(f"wrote {out/'ic.nc'} ({ncol} columns) and {out/'input.yaml'}")


# rrtmgp presets: fields from rrtmgp dumps (its own inputs). One latitude/longitude per
# column (point grids read them from the IC file).
RAD_ICE = ("dumps_strict_arm97_7p5d", "s6394_c0_r0_pre")   # night, deep cloud with ice
RAD_DAY = ("dumps_strict_cass_rad", "s109_c0_r1_pre")      # day, shallow liquid
RAD_SC = ("dumps_strict_dycomsrf01_rad", "s109_c0_r2_pre")  # stratocumulus
RAD_MID_FIELDS = ["T_mid", "p_mid", "pseudo_density", "qv", "qc", "nc", "qi", "cldfrac_tot",
                  "eff_radius_qc", "eff_radius_qi", "o3_volume_mix_ratio"]
RAD_COL_FIELDS = ["surf_lw_flux_up", "sfc_alb_dir_vis", "sfc_alb_dir_nir", "sfc_alb_dif_vis",
                  "sfc_alb_dif_nir"]
RAD_BAND_FIELDS = {"aero_tau_sw": "swband", "aero_ssa_sw": "swband", "aero_g_sw": "swband",
                   "aero_tau_lw": "lwband"}
STEBOL = 5.670374419e-8


def _short_night_lat(decl_deg: float, gap: float = 0.005) -> float:
    """Latitude where the night is `gap` radians of hour angle long (pi - h = gap)."""
    import math
    cos_h = -math.cos(gap)
    return math.degrees(math.atan(-cos_h / math.tan(math.radians(decl_deg))))


# Column list shared by both rrtmgp presets: (source, column, lat, lon, overrides)
RAD_COLUMNS = [
    (RAD_ICE, 0, 36.605, 262.515, {}),
    (RAD_ICE, 1, 36.605, 262.515, dict(cold_top=150.0, rei=250.0, rel=30.0)),
    (RAD_DAY, 0, 36.605, 262.515, dict(hot_surface=360.0, rel=1.0)),
    (RAD_SC, 0, 80.0, 0.0, {}),             # polar day
    (RAD_SC, 1, -80.0, 0.0, {}),            # polar night
    # a night shorter than the 300 s averaging window, centred on local midnight at 0Z
    (RAD_SC, 2, "short_night", 359.375, {}),
    # optically very thin top layer; the averaging window crosses local midnight at 0Z
    (RAD_ICE, 2, 36.605, 359.5, dict(thin_top=1.0e-5)),
    (RAD_DAY, 1, 50.0, 100.0, {}),          # |lat| > 45: other stratospheric trace-gas slope
]
RAD_OPTS_COMMON = dict(rad_frequency=3, column_chunk_size=8, pool_size_multiplier=1.0,
                       active_gases="[h2o, co2, o3, n2o, co, ch4, o2, n2]",
                       ch4vmr="1807.851e-9", co2vmr="388.717e-6", n2ovmr="323.141e-9",
                       f11vmr="768.7644e-12", f12vmr="531.2820e-12", n2vmr="0.7906",
                       covmr="1.0e-7", enable_column_conservation_checks="false")
PRESETS["rrtmgp"] = dict(  # fixed orbit and solar constant, extra clean/clear-sky calls
    procs=["rrtmgp"], dt=100, nsteps=4, proc_dump="rrtmgp", columns=RAD_COLUMNS,
    opts=dict(orbital_year=-9999, orbital_eccentricity="0.0167", orbital_obliquity="23.44",
              orbital_mvelp="102.7", fixed_total_solar_irradiance="1361.0",
              do_aerosol_rad="true", do_subcol_sampling="true", extra_clnclrsky_diag="true",
              extra_clnsky_diag="true"))
PRESETS["rrtmgp_alt"] = dict(  # orbit from the date, fixed zenith angle, no McICA, no aerosol
    procs=["rrtmgp"], dt=100, nsteps=4, proc_dump="rrtmgp", columns=RAD_COLUMNS,
    opts=dict(orbital_year=-9999, orbital_eccentricity="-9999.0", orbital_obliquity="-9999.0",
              orbital_mvelp="-9999.0", fixed_total_solar_irradiance="-9999.0",
              fixed_solar_zenith_angle="0.6", do_aerosol_rad="false", do_subcol_sampling="false",
              extra_clnclrsky_diag="false", extra_clnsky_diag="false"))


def build_rrtmgp(preset: str, spec: dict, out: Path) -> None:
    """IC and input.yaml for an rrtmgp-only standalone case."""
    sys.path.insert(0, str(REF.parent / "src"))
    from eamxx_torch.orbit import shr_orb_decl, shr_orb_params
    from eamxx_torch.timestamp import TimeStamp
    run_t0 = "1997-06-26-00000"
    e, _, _, obr, l0, mp = shr_orb_params(2000000000, 0.0167, 23.44, 102.7)
    decl, _ = shr_orb_decl(TimeStamp.from_string(run_t0).frac_of_year_in_days() + 1, e, mp, l0, obr)
    import math
    cols, lats, lons = [], [], []
    for (run, stem), icol, lat, lon, ov in spec["columns"]:
        _, pre = read_dump(REF / "runs" / run / "dump" / "rrtmgp" / stem)
        c = {k: np.array(pre[k][icol], dtype=np.float64)
             for k in RAD_MID_FIELDS + RAD_COL_FIELDS + list(RAD_BAND_FIELDS) + ["p_int"]}
        for k, v in ov.items():
            if k == "cold_top":      # below the gas-optics tables' 160 K
                c["T_mid"][:12] = v
            elif k == "hot_surface":  # above the tables' 355 K, surface and lowest layers
                c["T_mid"][-4:] = v
                c["surf_lw_flux_up"] = np.array(STEBOL * (v + 5.0) ** 4)
            elif k == "rei":          # ice larger than the cloud-optics table
                c["qi"] = np.where(c["cldfrac_tot"] > 0.5, np.maximum(c["qi"], 1e-5), c["qi"])
                c["eff_radius_qi"] = np.where(c["qi"] > 0, v, c["eff_radius_qi"])
            elif k == "rel":          # liquid radius outside the table
                c["eff_radius_qc"] = np.where(c["qc"] > 0, v, c["eff_radius_qc"])
            elif k == "thin_top":     # a top layer this many Pa thick: tiny gas optical depth
                c["p_int"][0] = c["p_int"][1] - v
            else:
                raise ValueError(k)
        cols.append(c)
        lats.append(_short_night_lat(math.degrees(decl)) if lat == "short_night" else lat)
        lons.append(lon)
    ncol = len(cols)
    nlev = cols[0]["T_mid"].shape[0]
    out.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(VCOORD) as v, netCDF4.Dataset(out / "ic.nc", "w",
                                                       format="NETCDF3_64BIT_DATA") as d:
        d.createDimension("time", None)
        d.createDimension("ncol", ncol)
        d.createDimension("lev", nlev)
        d.createDimension("ilev", nlev + 1)
        d.createDimension("swband", 14)
        d.createDimension("lwband", 16)
        d.createVariable("time", "f8", ("time",))[0] = 0.0
        for name in ("hyam", "hybm", "hyai", "hybi", "lev", "ilev"):
            d.createVariable(name, "f8", v[name].dimensions)[:] = v[name][:]
        d.createVariable("P0", "f8", ())[...] = 100000.0
        d.createVariable("lat", "f8", ("ncol",))[:] = np.array(lats)
        d.createVariable("lon", "f8", ("ncol",))[:] = np.array(lons)
        for k in RAD_MID_FIELDS:
            d.createVariable(k, "f8", ("time", "ncol", "lev"))[0] = np.stack([c[k] for c in cols])
        d.createVariable("p_int", "f8", ("time", "ncol", "ilev"))[0] = np.stack(
            [c["p_int"] for c in cols])
        for k in RAD_COL_FIELDS:
            d.createVariable(k, "f8", ("time", "ncol"))[0] = np.array([c[k] for c in cols])
        for k, bdim in RAD_BAND_FIELDS.items():
            d.createVariable(k, "f8", ("time", "ncol", bdim, "lev"))[0] = np.stack(
                [c[k] for c in cols])
    data = REF / "inputdata/atm/scream/init"
    opts = dict(rrtmgp_coefficients_file_sw=data / "rrtmgp-data-sw-g112-210809.nc",
                rrtmgp_coefficients_file_lw=data / "rrtmgp-data-lw-g128-210809.nc",
                rrtmgp_cloud_optics_file_sw=data / "rrtmgp-cloud-optics-coeffs-sw.nc",
                rrtmgp_cloud_optics_file_lw=data / "rrtmgp-cloud-optics-coeffs-lw.nc",
                **RAD_OPTS_COMMON, **spec["opts"])
    body = "\n".join(f"    {k}: {v}" for k, v in opts.items())
    yaml = f"""%YAML 1.1
---
# Generated by reference/scripts/make_standalone_case.py {preset}
driver_options:
  atmosphere_dag_verbosity_level: 0
time_stepping:
  time_step: {spec['dt']}
  run_t0: {run_t0}
  number_of_steps: {spec['nsteps']}
eamxx:
  atm_procs_list: [rrtmgp]
  schedule_type: sequential
  type: group
  rrtmgp:
{body}
grids_manager:
  type: mesh_free
  geo_data_source: IC_FILE
  grids_names: [physics_pg2]
  physics_pg2:
    type: point_grid
    aliases: [physics]
    number_of_global_columns: {ncol}
    number_of_vertical_levels: {nlev}
initial_conditions:
  filename: {out.resolve() / 'ic.nc'}
scorpio:
  output_yaml_files: []
...
"""
    (out / "input.yaml").write_text(yaml)
    print(f"wrote {out/'ic.nc'} ({ncol} columns, lat {np.round(lats, 4).tolist()}) "
          f"and {out/'input.yaml'}")


def _single(out: Path) -> None:
    """Point P3 at the single-precision tables (*.dat4), as EAMxx's float build reads."""
    y = out / "input.yaml"
    y.write_text(y.read_text().replace(".dat8", ".dat4"))


def build_mac(preset: str, spec: dict, out: Path, single: bool) -> None:
    """shoc, cld_fraction, p3 on columns built from a shoc dump and the p3 dump of the same call."""
    cols = []
    for (run, stem), icol in spec["columns"]:
        _, sh = read_dump(REF / "runs" / run / "dump" / "shoc" / stem)
        _, p3 = read_dump(REF / "runs" / run / "dump" / "p3" / stem)
        c = {k: np.array(sh[k][icol], dtype=np.float64) for k in MID_FIELDS + COL_FIELDS + ["p_int", "phis"]}
        for k in P3_MID_FIELDS + P3_COL_FIELDS:
            if k not in c:
                c[k] = np.array(p3[k][icol], dtype=np.float64)
        c["U"], c["V"] = sh["horiz_winds"][icol, 0].copy(), sh["horiz_winds"][icol, 1].copy()
        c["surf_mom_flux"] = sh["surf_mom_flux"][icol].copy()
        c["surf_drag_coeff_tms"] = np.array(0.0)
        cols.append(c)
    ncol, nlev = len(cols), cols[0]["T_mid"].shape[0]
    out.mkdir(parents=True, exist_ok=True)
    mids = sorted(set(MID_FIELDS + P3_MID_FIELDS)) + ["U", "V"]
    with netCDF4.Dataset(VCOORD) as v, netCDF4.Dataset(out / "ic.nc", "w", format="NETCDF3_64BIT_DATA") as d:
        d.createDimension("time", None)
        d.createDimension("ncol", ncol)
        d.createDimension("lev", nlev)
        d.createDimension("ilev", nlev + 1)
        d.createVariable("time", "f8", ("time",))[0] = 0.0
        for name in ("hyam", "hybm", "hyai", "hybi", "lev", "ilev"):
            d.createVariable(name, "f8", v[name].dimensions)[:] = v[name][:]
        d.createVariable("P0", "f8", ())[...] = 100000.0
        d.createVariable("lat", "f8", ("ncol",))[:] = 36.605
        d.createVariable("lon", "f8", ("ncol",))[:] = 262.515
        for k in mids:
            d.createVariable(k, "f8", ("time", "ncol", "lev"))[0] = np.stack([c[k] for c in cols])
        d.createVariable("p_int", "f8", ("time", "ncol", "ilev"))[0] = np.stack([c["p_int"] for c in cols])
        for k in COL_FIELDS + P3_COL_FIELDS + ["surf_drag_coeff_tms"]:
            d.createVariable(k, "f8", ("time", "ncol"))[0] = np.array([c[k] for c in cols])
        d.createVariable("landfrac", "f8", ("time", "ncol"))[0] = np.zeros(ncol)
        for i, comp in enumerate("UV"):
            d.createVariable(f"surf_mom_flux_{comp}", "f8", ("time", "ncol"))[0] = np.array(
                [c["surf_mom_flux"][i] for c in cols])
    dp_yaml = (REF / "runs/dumps_strict_bomex/input.yaml").read_text()
    blk = lambda name: re.search(rf"^( +){name}:.*\n(?:\1 +.*\n)*", dp_yaml, re.M).group(0)  # noqa: E731
    phys = "".join(blk(n) for n in ("shoc", "cld_fraction", "p3"))
    ind = len(re.match(r"^( *)", phys).group(1))
    phys = "".join(line[ind - 2:] + "\n" for line in phys.splitlines() if line.strip()
                   and "compute_tendencies" not in line)
    yaml = f"""%YAML 1.1
---
# Generated by reference/scripts/make_standalone_case.py {preset}{' --single' if single else ''}
driver_options:
  atmosphere_dag_verbosity_level: 0
time_stepping:
  time_step: {spec['dt']}
  run_t0: 1997-06-26-00000
  number_of_steps: {spec['nsteps']}
eamxx:
  atm_procs_list: [{', '.join(spec['procs'])}]
  schedule_type: sequential
  type: group
{phys}grids_manager:
  type: mesh_free
  geo_data_source: IC_FILE
  grids_names: [physics_pg2]
  physics_pg2:
    type: point_grid
    aliases: [physics]
    number_of_global_columns: {ncol}
    number_of_vertical_levels: {nlev}
initial_conditions:
  filename: {out.resolve() / 'ic.nc'}
  phis: {float(cols[0]['phis'])!r}
  sgh30: {spec['sgh30']!r}
scorpio:
  output_yaml_files: []
...
"""
    (out / "input.yaml").write_text(yaml)
    if single:
        _single(out)
    print(f"wrote {out/'ic.nc'} ({ncol} columns) and {out/'input.yaml'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("preset", choices=sorted(PRESETS))
    ap.add_argument("out", type=Path)
    ap.add_argument("--single", action="store_true", help="for the single-precision build (*.dat4 tables)")
    a = ap.parse_args()
    build(a.preset, a.out, a.single)


if __name__ == "__main__":
    main()
