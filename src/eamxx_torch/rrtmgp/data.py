"""RRTMGP coefficient data: k-distributions and cloud-optics lookup tables.

Translation of the Kokkos code paths EAMxx uses, E3SM 80fe0a71 (rte-rrtmgp
submodule 2a887acc):
  - examples/mo_load_coefficients.h        load_and_init
  - rrtmgp/mo_gas_optics_rrtmgp.h          GasOpticsRRTMGPK::load / init_abs_coeffs and
                                           reduce_minor_arrays, create_idx_minor(_scaling),
                                           create_key_species_reduce, create_flavor,
                                           create_gpoint_flavor
  - rte/mo_optical_props.h                 OpticalPropsK::init (band/g-point maps)
  - examples/all-sky/mo_load_cloud_coefficients.h  load_cld_lutcoeff
  - extensions/cloud_optics/mo_cloud_optics.h      CloudOpticsK::load (LUT)

The files are read the way conv::SimpleNetCDF::read reads them: dimensions in
Fortran order (the reverse of the netCDF order), integer arrays shifted to
0-based, integer flags read as `value == 1`, strings with whitespace removed.
The data files are E3SM input data and are not part of this package.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch


def _nc(path):
    import netCDF4
    d = netCDF4.Dataset(str(path))
    d.set_auto_maskandscale(False)
    return d


def _read(d, name):
    """SimpleNetCDF::read: Fortran dimension order; ints made 0-based."""
    a = np.asarray(d.variables[name][...])
    a = np.ascontiguousarray(a.transpose()) if a.ndim > 1 else a
    if a.dtype.kind in "iu":
        a = a.astype(np.int64) - 1
    return a


def _read_bool(d, name):
    a = np.asarray(d.variables[name][...])
    return np.ascontiguousarray(a.transpose()) == 1 if a.ndim > 1 else a == 1


def _read_scalar(d, name):
    return float(np.asarray(d.variables[name][...]))


def _read_strings(d, name):
    """char2d_to_string1d: every non-whitespace character, in order."""
    a = np.asarray(d.variables[name][...])
    return ["".join(c.decode() for c in row if not c.decode().isspace()) for row in a]


def _loc(s: str, arr: list[str]) -> int:
    """string_loc_in_array (0-based, -1 if absent, case-insensitive)."""
    ls = s.lower()
    for i, item in enumerate(arr):
        if ls == item.lower():
            return i
    return -1


def _in(s: str, arr: list[str]) -> bool:
    return _loc(s, arr) != -1


def _t(a, dtype=torch.float64):
    return torch.as_tensor(np.ascontiguousarray(a), dtype=dtype)


@dataclass
class MinorSet:
    """Minor absorbers for one part of the atmosphere (lower or upper), reduced to the
    gases the host provides."""
    kminor: torch.Tensor                 # (ncontrib, neta, ntemp)
    limits_gpt: list[tuple[int, int]]    # per minor: (first g-point, last g-point)
    scales_with_density: list[bool]
    scale_by_complement: list[bool]
    idx_minor: list[int]                 # index into gas_names
    idx_minor_scaling: list[int]         # index into gas_names, -1 if none
    kminor_start: list[int]
    max_gpt_diff: int


@dataclass
class GasOpticsData:
    """GasOpticsRRTMGPK after load(): one k-distribution (SW or LW)."""
    gas_names: list[str]                 # this->gas_names (reduced, file order)
    band_lims_wvn: torch.Tensor          # (2, nbnd)
    band2gpt: list[tuple[int, int]]      # per band: first and last g-point
    gpt2band: torch.Tensor               # (ngpt,) int64
    press_ref: torch.Tensor
    press_ref_log: torch.Tensor
    temp_ref: torch.Tensor
    press_ref_min: float
    press_ref_max: float
    temp_ref_min: float
    temp_ref_max: float
    press_ref_log_delta: float
    temp_ref_delta: float
    press_ref_trop_log: float
    vmr_ref: torch.Tensor                # (2, ngas+1, ntemp)
    flavor: torch.Tensor                 # (2, nflav) int64
    gpoint_flavor: torch.Tensor          # (2, ngpt) int64
    kmajor: torch.Tensor                 # (ngpt, neta, npres+1, ntemp)
    lower: MinorSet
    upper: MinorSet
    krayl: torch.Tensor | None           # (ngpt, neta, ntemp, 2)
    planck_frac: torch.Tensor | None     # (ngpt, neta, npres+1, ntemp)
    totplnk: torch.Tensor | None         # (nPlanckTemp, nbnd)
    totplnk_delta: float
    solar_src: torch.Tensor | None       # (ngpt,)
    is_key: list[bool] = field(default_factory=list)

    @property
    def ngpt(self) -> int:
        return int(self.gpt2band.shape[0])

    @property
    def nband(self) -> int:
        return len(self.band2gpt)

    @property
    def ngas(self) -> int:
        return len(self.gas_names)

    @property
    def nflav(self) -> int:
        return int(self.flavor.shape[1])

    @property
    def neta(self) -> int:
        return int(self.kmajor.shape[1])

    @property
    def npres(self) -> int:
        return int(self.kmajor.shape[2]) - 1

    @property
    def ntemp(self) -> int:
        return int(self.kmajor.shape[3])


def _band_maps(band_lims_gpt: np.ndarray):
    """OpticalPropsK::init with explicit g-point limits: band2gpt and gpt2band."""
    nbnd = band_lims_gpt.shape[1]
    band2gpt = [(int(band_lims_gpt[0, b]), int(band_lims_gpt[1, b])) for b in range(nbnd)]
    ngpt = int(band_lims_gpt.max()) + 1
    gpt2band = np.zeros(ngpt, dtype=np.int64)
    for b, (g0, g1) in enumerate(band2gpt):
        gpt2band[g0:g1 + 1] = b
    return band2gpt, torch.as_tensor(gpt2band)


def _reduce_minor(available, gas_names, gas_minor, identifier_minor, kminor_atm, minor_gases_atm,
                  minor_limits_gpt_atm, minor_scales_with_density_atm, scaling_gas_atm,
                  scale_by_complement_atm, kminor_start_atm):
    """reduce_minor_arrays: keep only minor contributions whose gas the host provides."""
    nm = len(minor_gases_atm)
    present = []
    tot_g = 0
    for i in range(nm):
        idx_mnr = _loc(minor_gases_atm[i], identifier_minor)
        p = _in(gas_minor[idx_mnr], available)
        present.append(p)
        if p:
            tot_g += int(minor_limits_gpt_atm[1, i] - minor_limits_gpt_atm[0, i] + 1)
    red_nm = sum(present)
    if red_nm == nm:
        return (kminor_atm.copy(), list(minor_gases_atm), minor_limits_gpt_atm.copy(),
                minor_scales_with_density_atm.copy(), list(scaling_gas_atm),
                scale_by_complement_atm.copy(), kminor_start_atm.copy())
    gases_red, scaling_red, dens_red, compl_red = [], [], [], []
    for i in range(nm):
        if present[i]:
            gases_red.append(minor_gases_atm[i])
            scaling_red.append(scaling_gas_atm[i])
            dens_red.append(bool(minor_scales_with_density_atm[i]))
            compl_red.append(bool(scale_by_complement_atm[i]))
    limits_red = np.zeros((2, red_nm), dtype=np.int64)
    start_red = np.zeros(red_nm, dtype=np.int64)
    kminor_red = np.zeros((tot_g, kminor_atm.shape[1], kminor_atm.shape[2]))
    slot = -1
    n_elim = 0
    for i in range(nm):
        ng = int(minor_limits_gpt_atm[1, i] - minor_limits_gpt_atm[0, i] + 1)
        if present[i]:
            slot += 1
            limits_red[0, slot] = minor_limits_gpt_atm[0, i]
            limits_red[1, slot] = minor_limits_gpt_atm[1, i]
            start_red[slot] = kminor_start_atm[i] - n_elim
            s0, s1 = int(start_red[slot]), int(kminor_start_atm[i])
            kminor_red[s0:s0 + ng] = kminor_atm[s1:s1 + ng]
        else:
            n_elim += ng
    return (kminor_red, gases_red, limits_red, np.array(dens_red, dtype=bool), scaling_red,
            np.array(compl_red, dtype=bool), start_red)


def _minor_set(red, red_gas_names, gas_minor, identifier_minor) -> MinorSet:
    kminor, gases, limits, dens, scaling, compl, start = red
    # create_idx_minor / create_idx_minor_scaling
    idx_minor = [_loc(gas_minor[_loc(g, identifier_minor)], red_gas_names) for g in gases]
    idx_scaling = [_loc(s, red_gas_names) for s in scaling]
    max_diff = -2**31  # std::numeric_limits<int>::lowest()
    for i in range(limits.shape[1]):
        max_diff = max(max_diff, int(limits[1, i] - limits[0, i]))
    return MinorSet(kminor=_t(kminor), limits_gpt=[(int(limits[0, i]), int(limits[1, i]))
                                                  for i in range(limits.shape[1])],
                    scales_with_density=[bool(x) for x in dens],
                    scale_by_complement=[bool(x) for x in compl],
                    idx_minor=idx_minor, idx_minor_scaling=idx_scaling,
                    kminor_start=[int(x) for x in start], max_gpt_diff=max_diff)


def _create_flavor(key_species: np.ndarray) -> np.ndarray:
    """create_flavor: unique key-species pairs, in first-seen order."""
    lst = []
    for ibnd in range(key_species.shape[2]):
        for iatm in range(key_species.shape[1]):
            lst.append([int(key_species[0, iatm, ibnd]), int(key_species[1, iatm, ibnd])])
    for pair in lst:
        if pair[0] == -1 and pair[1] == -1:
            pair[0] = pair[1] = 1
    flav = []
    for i, pair in enumerate(lst):
        if pair not in lst[:i]:
            flav.append(pair)
    return np.array(flav, dtype=np.int64).T.copy()  # (2, nflav)


def _create_gpoint_flavor(key_species, gpt2band, flavor) -> np.ndarray:
    ngpt = gpt2band.shape[0]
    out = np.zeros((2, ngpt), dtype=np.int64)
    for igpt in range(ngpt):
        for iatm in range(2):
            k1 = int(key_species[0, iatm, gpt2band[igpt]])
            k2 = int(key_species[1, iatm, gpt2band[igpt]])
            if k1 == -1 and k2 == -1:
                k1 = k2 = 1
            r = -1
            for iflav in range(flavor.shape[1]):
                if flavor[0, iflav] == k1 and flavor[1, iflav] == k2:
                    r = iflav
            out[iatm, igpt] = r
    return out


def load_gas_optics(path: str | Path, available_gases: list[str]) -> GasOpticsData:
    """load_and_init + GasOpticsRRTMGPK::load for one coefficients file."""
    with _nc(path) as d:
        gas_names = _read_strings(d, "gas_names")
        gas_minor = _read_strings(d, "gas_minor")
        identifier_minor = _read_strings(d, "identifier_minor")
        minor_gases_lower = _read_strings(d, "minor_gases_lower")
        minor_gases_upper = _read_strings(d, "minor_gases_upper")
        scaling_gas_lower = _read_strings(d, "scaling_gas_lower")
        scaling_gas_upper = _read_strings(d, "scaling_gas_upper")
        key_species = _read(d, "key_species")                 # (2, 2, nbnd)
        band_lims = _read(d, "bnd_limits_wavenumber")         # (2, nbnd)
        band2gpt = _read(d, "bnd_limits_gpt")                 # (2, nbnd)
        press_ref = _read(d, "press_ref")
        temp_ref = _read(d, "temp_ref")
        press_ref_trop = _read_scalar(d, "press_ref_trop")
        kminor_lower = _read(d, "kminor_lower")
        kminor_upper = _read(d, "kminor_upper")
        minor_limits_gpt_lower = _read(d, "minor_limits_gpt_lower")
        minor_limits_gpt_upper = _read(d, "minor_limits_gpt_upper")
        dens_lower = _read_bool(d, "minor_scales_with_density_lower")
        dens_upper = _read_bool(d, "minor_scales_with_density_upper")
        compl_lower = _read_bool(d, "scale_by_complement_lower")
        compl_upper = _read_bool(d, "scale_by_complement_upper")
        kminor_start_lower = _read(d, "kminor_start_lower")
        kminor_start_upper = _read(d, "kminor_start_upper")
        vmr_ref = _read(d, "vmr_ref")                         # (2, ngas+1, ntemp)
        kmajor = _read(d, "kmajor")                           # (ngpt, neta, npres+1, ntemp)
        rayl_lower = _read(d, "rayl_lower") if "rayl_lower" in d.variables else None
        rayl_upper = _read(d, "rayl_upper") if "rayl_upper" in d.variables else None
        internal = "totplnk" in d.variables
        totplnk = _read(d, "totplnk") if internal else None
        planck_frac = _read(d, "plank_fraction") if internal else None
        solar_src = None
        if not internal:
            if "solar_source" in d.variables:
                solar_src = _read(d, "solar_source")
            else:
                solar_src = _read(d, "solar_source_quiet")

    # init_abs_coeffs
    band2gpt_l, gpt2band = _band_maps(band2gpt)
    red_names = [g for g in gas_names if _in(g, available_gases)]
    ngas = len(red_names)
    vmr_ref_red = np.zeros((vmr_ref.shape[0], ngas + 1, vmr_ref.shape[2]))
    vmr_ref_red[:, 0, :] = vmr_ref[:, 0, :]
    for i, g in enumerate(red_names):
        vmr_ref_red[:, i + 1, :] = vmr_ref[:, _loc(g, gas_names) + 1, :]

    lower = _minor_set(_reduce_minor(available_gases, gas_names, gas_minor, identifier_minor,
                                     kminor_lower, minor_gases_lower, minor_limits_gpt_lower,
                                     dens_lower, scaling_gas_lower, compl_lower,
                                     kminor_start_lower), red_names, gas_minor, identifier_minor)
    upper = _minor_set(_reduce_minor(available_gases, gas_names, gas_minor, identifier_minor,
                                     kminor_upper, minor_gases_upper, minor_limits_gpt_upper,
                                     dens_upper, scaling_gas_upper, compl_upper,
                                     kminor_start_upper), red_names, gas_minor, identifier_minor)
    krayl = None
    if rayl_lower is not None:
        krayl = _t(np.stack([rayl_lower, rayl_upper], axis=-1))
    # press_ref_log(i) = log(press_ref(i)); scalars with libm log, as the reference
    press_ref_log = np.log(press_ref)
    press_ref_trop_log = math.log(press_ref_trop)

    # create_key_species_reduce, create_flavor, create_gpoint_flavor
    key_red = np.full_like(key_species, -1)
    for ip in range(key_species.shape[0]):
        for ia in range(key_species.shape[1]):
            for it in range(key_species.shape[2]):
                if key_species[ip, ia, it] != -1:
                    key_red[ip, ia, it] = _loc(gas_names[key_species[ip, ia, it]], red_names)
    flavor = _create_flavor(key_red)
    gpoint_flavor = _create_gpoint_flavor(key_red, gpt2band.numpy(), flavor)

    temp_ref_min = float(temp_ref[0])
    temp_ref_max = float(temp_ref[-1])
    press_ref_min = float(press_ref[-1])
    press_ref_max = float(press_ref[0])
    press_ref_log_delta = (math.log(press_ref_min) - math.log(press_ref_max)) / (press_ref.shape[0] - 1)
    temp_ref_delta = (temp_ref_max - temp_ref_min) / (temp_ref.shape[0] - 1)
    is_key = [False] * ngas
    for i in range(flavor.shape[0]):
        for j in range(flavor.shape[1]):
            if flavor[i, j] != -1:
                is_key[flavor[i, j]] = True
    totplnk_delta = 0.0
    if internal:
        totplnk_delta = (temp_ref_max - temp_ref_min) / (totplnk.shape[0] - 1)

    return GasOpticsData(
        gas_names=red_names, band_lims_wvn=_t(band_lims), band2gpt=band2gpt_l, gpt2band=gpt2band,
        press_ref=_t(press_ref), press_ref_log=_t(press_ref_log), temp_ref=_t(temp_ref),
        press_ref_min=press_ref_min, press_ref_max=press_ref_max, temp_ref_min=temp_ref_min,
        temp_ref_max=temp_ref_max, press_ref_log_delta=press_ref_log_delta,
        temp_ref_delta=temp_ref_delta, press_ref_trop_log=press_ref_trop_log,
        vmr_ref=_t(vmr_ref_red), flavor=torch.as_tensor(flavor),
        gpoint_flavor=torch.as_tensor(gpoint_flavor), kmajor=_t(kmajor), lower=lower, upper=upper,
        krayl=krayl, planck_frac=None if planck_frac is None else _t(planck_frac),
        totplnk=None if totplnk is None else _t(totplnk), totplnk_delta=totplnk_delta,
        solar_src=None if solar_src is None else _t(solar_src), is_key=is_key)


@dataclass
class CloudOpticsData:
    """CloudOpticsK after load() (lookup-table form, the one EAMxx loads)."""
    band_lims_wvn: torch.Tensor   # (2, nbnd)
    radliq_lwr: float
    radliq_upr: float
    radice_lwr: float
    radice_upr: float
    liq_nsteps: int
    ice_nsteps: int
    liq_step_size: float
    ice_step_size: float
    lut_extliq: torch.Tensor      # (nsize_liq, nbnd)
    lut_ssaliq: torch.Tensor
    lut_asyliq: torch.Tensor
    lut_extice: torch.Tensor      # (nsize_ice, nbnd, nrghice)
    lut_ssaice: torch.Tensor
    lut_asyice: torch.Tensor
    icergh: int = 1               # set_ice_roughness(1) in load


def load_cloud_optics(path: str | Path) -> CloudOpticsData:
    """load_cld_lutcoeff + CloudOpticsK::load (LUT)."""
    with _nc(path) as d:
        band_lims = _read(d, "bnd_limits_wavenumber")
        s = {k: _read_scalar(d, k) for k in ("radliq_lwr", "radliq_upr", "radliq_fac",
                                             "radice_lwr", "radice_upr", "radice_fac")}
        luts = {k: _read(d, k).astype(np.float64) for k in (
            "lut_extliq", "lut_ssaliq", "lut_asyliq", "lut_extice", "lut_ssaice", "lut_asyice")}
    nsize_liq = luts["lut_extliq"].shape[0]
    nsize_ice = luts["lut_extice"].shape[0]
    return CloudOpticsData(
        band_lims_wvn=_t(band_lims), radliq_lwr=s["radliq_lwr"], radliq_upr=s["radliq_upr"],
        radice_lwr=s["radice_lwr"], radice_upr=s["radice_upr"], liq_nsteps=nsize_liq,
        ice_nsteps=nsize_ice,
        liq_step_size=(s["radliq_upr"] - s["radliq_lwr"]) / (nsize_liq - 1.),
        ice_step_size=(s["radice_upr"] - s["radice_lwr"]) / (nsize_ice - 1.),
        **{k: _t(v) for k, v in luts.items()})


@dataclass
class RRTMGPData:
    """What rrtmgp_initialize loads: two k-distributions and two cloud-optics tables."""
    k_dist_sw: GasOpticsData
    k_dist_lw: GasOpticsData
    cloud_optics_sw: CloudOpticsData
    cloud_optics_lw: CloudOpticsData


def load_rrtmgp(coefficients_file_sw, coefficients_file_lw, cloud_optics_file_sw,
                cloud_optics_file_lw, gas_names: list[str]) -> RRTMGPData:
    """rrtmgp_interface::rrtmgp_initialize. gas_names: the active gases, in order."""
    return RRTMGPData(load_gas_optics(coefficients_file_sw, gas_names),
                      load_gas_optics(coefficients_file_lw, gas_names),
                      load_cloud_optics(cloud_optics_file_sw),
                      load_cloud_optics(cloud_optics_file_lw))
