"""Intensive observation period (IOP) data and forcing.

Translation of, E3SM 80fe0a71:
  - components/eamxx/src/share/data_managers/IOPDataManager.{hpp,cpp}
      (constructor defaults, initialize_iop_file, TimeInfo::get_iop_file_time_idx,
       read_iop_file_data, set_fields_from_iop_data, correct_temperature_and_water_vapor)
  - components/eamxx/src/physics/iop_forcing/eamxx_iop_forcing_process_interface.{hpp,cpp}
      (advance_iop_subsidence, advance_iop_forcing, iop_apply_coriolis, run_impl,
       the postcondition checks)
  - externals/ekat/src/algorithm/ekat_lin_interp_impl.hpp (LinInterp setup, lin_interp)
  - components/eamxx/src/share/field/utils/horiz_contraction.cpp (the nudging means)
  - components/eamxx/src/control/atmosphere_surface_coupling_importer.cpp
      (overwrite_iop_imports)

The data manager works on the host in float64 (numpy), as the reference does:
its profiles are small and read once per IOP time interval. It keeps its fields
between reads, because a read only rewrites the levels inside the file's
pressure range (except for T, q, u, v, u_ls, v_ls, whose ends are filled).

The IOP data are piecewise constant in time: the interval containing the
start-of-step time stamp is used, with no time interpolation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .. import constants as C
from .. import mathfn
from ..ekat import max_
from ..property_checks import within_interval
from ..timestamp import TimeStamp


@dataclass(frozen=True)
class IOPParams:
    """The iop_options parameters, with IOPDataManager's defaults."""
    iop_file: str
    target_latitude: float
    target_longitude: float
    doubly_periodic_mode: bool = True
    iop_srf_prop: bool = False
    iop_dosubsidence: bool = False
    iop_coriolis: bool = False
    iop_nudge_tq: bool = False
    iop_nudge_uv: bool = False
    iop_nudge_tq_low: float = 1050.
    iop_nudge_tq_high: float = 0.
    iop_nudge_tscale: float = 10800.
    zero_non_iop_tracers: bool = False


# setup_iop_field calls, in order: (names to try, rank, surface variable name)
_SCALARS = [(["Ps"],), (["Tg"],), (["lhflx", "lh"],), (["shflx", "sh"],)]
_LEVELS = [(["T"], "Tsair"), (["q"], "qsrf"), (["cld"], None), (["clwp"], None),
           (["divq"], "divqsrf"), (["vertdivq"], "vertdivqsrf"), (["NUMLIQ"], None),
           (["CLDLIQ"], None), (["CLDICE"], None), (["NUMICE"], None), (["divu"], "divusrf"),
           (["divv"], "divvsrf"), (["divT"], "divtsrf"), (["vertdivT"], "vertdivTsrf"),
           (["divT3d"], "divT3dsrf"), (["u"], "usrf"), (["u_ls"], "usrf"), (["v"], "vsrf"),
           (["v_ls"], "vsrf"), (["Q1"], None), (["Q2"], None), (["omega"], "Ptend")]
_FILLED_ENDS = ("T", "q", "u", "u_ls", "v", "v_ls")
INT_MIN = -2147483648
INT_MAX = 2147483647


def lin_interp_1d(x1: np.ndarray, x2: np.ndarray, y1: np.ndarray) -> np.ndarray:
    """ekat::LinInterp on one column (setup + lin_interp), float64 numpy.

    setup: k1 = std::upper_bound(x1, x2) - 1, not below 0; lin_interp uses the
    neighbour above (below at the last source level) and computes, per element,
    ((y1[k1ph] - y1[k1]) * (x2 - x1[k1])) / (x1[k1ph] - x1[k1]) + y1[k1].
    """
    km1 = x1.shape[0]
    k1 = np.searchsorted(x1, x2, side="right")
    k1 = np.where(k1 > 0, k1 - 1, k1)
    k1ph = np.where(k1 == km1 - 1, k1 - 1, k1 + 1)
    y = y1[k1ph] - y1[k1]
    y = y * (x2 - x1[k1])
    y = y / (x1[k1ph] - x1[k1])
    return y + y1[k1]


class IOPDataManager:
    """IOPDataManager for one IOP file (doubly periodic or single column)."""

    def __init__(self, params: IOPParams, run_t0: TimeStamp, model_nlevs: int,
                 hyam: np.ndarray, hybm: np.ndarray):
        import netCDF4
        p = params
        if not p.doubly_periodic_mode:
            raise ValueError("doubly_periodic_mode is the only supported use of IOP files")
        if not (-90 <= p.target_latitude <= 90) or not (0 <= p.target_longitude <= 360):
            raise ValueError("IOP target latitude/longitude out of range")
        self.params = p
        self.hyam = np.asarray(hyam, dtype=np.float64)
        self.hybm = np.asarray(hybm, dtype=np.float64)
        self.model_nlevs = model_nlevs
        self._nc = netCDF4.Dataset(str(p.iop_file))
        self._nc.set_auto_maskandscale(False)
        v = self._nc.variables
        self.fields: dict[str, np.ndarray] = {}
        self.field_type: dict[str, str] = {}
        self.file_varnames: dict[str, str] = {}
        self.surface_varnames: dict[str, str] = {}

        def setup(names, rank, srf=None):
            for name in names:
                if name in v:
                    if name != names[0]:
                        self.file_varnames[names[0]] = name
                    if srf is not None and srf in v:
                        self.surface_varnames[names[0]] = srf
                    self.field_type[names[0]] = "FromFile"
                    self.fields[names[0]] = np.zeros(()) if rank == 0 else np.zeros(model_nlevs)
                    return

        for (names,) in _SCALARS:
            setup(names, 0)
        for names, srf in _LEVELS:
            setup(names, 1, srf)
        for req in ("Ps", "T", "q", "divT", "divq"):
            if req not in self.fields:
                raise ValueError(f'IOP file required to contain variable "{req}"')
        both_ls = "u_ls" in self.fields and "v_ls" in self.fields
        neither_ls = not ("u_ls" in self.fields or "v_ls" in self.fields)
        if not (both_ls or neither_ls):
            raise ValueError("Either u_ls and v_ls both defined in IOP file, or neither")
        self.use_large_scale_wind = both_ls
        if p.iop_coriolis and not both_ls:
            raise ValueError("Large scale winds required for coriolis forcing")
        for name, base in (("divT3d", "vertdivT"), ("divq3d", "vertdivq")):
            if base in self.fields:
                self.fields[name] = np.zeros(model_nlevs)
                self.field_type[name] = "Computed"
        both_3d = "divT3d" in self.fields and "divq3d" in self.fields
        neither_3d = not ("divT3d" in self.fields or "divq3d" in self.fields)
        if not (both_3d or neither_3d):
            raise ValueError("Either T and q both have 3d forcing, or neither")
        self.use_3d_forcing = both_3d

        for bdate_name in ("bdate", "basedate", "nbdate"):
            if bdate_name in v:
                break
        else:
            raise ValueError("No valid name for bdate")
        bdate = int(np.asarray(v[bdate_name][...]))
        yr = bdate // 10000
        mo = bdate // 100 - yr * 100
        day = bdate - (yr * 10000 + mo * 100)
        self.iop_file_begin_time = TimeStamp.make(yr, mo, day, 0, 0, 0)
        if "time" in self._nc.dimensions:
            self._time_dim = "time"
        elif "tsec" in self._nc.dimensions:
            self._time_dim = "tsec"
        else:
            raise ValueError("No valid dimension for tsec")
        # tsec is read into an int: conversion truncates toward zero
        self.iop_file_times_in_sec = [int(t) for t in np.asarray(v["tsec"][:], dtype=np.float64)]
        self.time_idx_of_current_data = -1

        lat = float(np.asarray(v["lat"][:]).ravel()[0])
        lon = float(np.asarray(v["lon"][:]).ravel()[0])
        rel_lat_err = abs(lat - p.target_latitude) / max(abs(p.target_latitude), 0.1)
        rel_lon_err = abs(math.fmod(lon + 360.0, 360.0) - p.target_longitude) / \
            max(p.target_longitude, 0.1)
        feps = float(np.finfo(np.float32).eps)
        if not (rel_lat_err < feps and rel_lon_err < feps):
            raise ValueError("IOP file lat/lon do not match the target latitude/longitude")
        self.file_levs = int(self._nc.dimensions["lev"].size)
        self.iop_file_pressure = np.zeros(self.file_levs + 1)
        self.iop_file_pressure[:self.file_levs] = self._read_lev()
        self.model_pressure = np.zeros(model_nlevs)

    # ------------------------------------------------------------------ file access
    def _read_lev(self) -> np.ndarray:
        return np.asarray(self._nc["lev"][:], dtype=np.float64) / 100

    def _read_at(self, name: str, t: int):
        var = self._nc[name]
        a = np.asarray(var[t], dtype=np.float64) if var.dimensions[0] == self._time_dim \
            else np.asarray(var[...], dtype=np.float64)
        return a.reshape(-1) if a.ndim else a

    def has_iop_field(self, name: str) -> bool:
        return name in self.fields

    def get_iop_file_time_idx(self, ts: TimeStamp) -> int:
        """The last t with begin + tsec(t) <= ts < begin + tsec(t+1).

        Compared in whole seconds from the file's base date (time stamps here carry
        no fractional seconds), which orders exactly as TimeStamp comparisons do.
        """
        times = np.asarray(self.iop_file_times_in_sec, dtype=np.int64)
        off = ts.seconds_from(self.iop_file_begin_time)
        ok = np.nonzero((times[:-1] <= off) & (off < times[1:]))[0]
        if ok.size == 0:
            raise ValueError(f"Current model time ({ts.to_string()}) is not within the IOP "
                             "time period")
        return int(ok[-1])

    # ------------------------------------------------------------------ reading
    def read_iop_file_data(self, current_ts: TimeStamp) -> None:
        t = self.get_iop_file_time_idx(current_ts)
        if t < self.time_idx_of_current_data:
            raise ValueError("Attempting to read previous iop file data time index")
        if t == self.time_idx_of_current_data:
            return
        file_levs = self.file_levs
        has_level_data = any(a.ndim == 1 and self.field_type[n] == "FromFile"
                             for n, a in self.fields.items())
        if has_level_data:
            ps = float(self._read_at("Ps", t)[0])
            self.fields["Ps"] = np.array(ps)
            pres = self.iop_file_pressure
            pres[:file_levs] = self._read_lev()
            ps_mb = ps / 100
            lmin = INT_MAX
            for ilev in range(file_levs + 1):
                if ilev == file_levs:
                    pres[ilev] = ps_mb
                if pres[ilev] > ps_mb:
                    pres[ilev] = ps_mb
                if pres[ilev] == ps_mb and ilev < lmin:
                    lmin = ilev + 1
            adj = lmin
            if not adj > 1:
                raise ValueError("Pressures in the iop file are incorrectly set")
            mp = 1000 * self.hyam + ps * self.hybm / 100
            self.model_pressure = mp
            nm = self.model_nlevs
            start, end = INT_MIN, INT_MAX
            for ilev in range(adj):
                if pres[ilev] <= mp[0] and ilev > start:
                    start = ilev
                if pres[ilev] >= mp[nm - 1] and ilev + 1 < end:
                    end = ilev + 1
            if start == INT_MIN:
                start = 0
            if end == INT_MAX:
                end = adj
            mstart, mend = INT_MAX, INT_MIN
            for ilev in range(nm):
                if mp[ilev] >= pres[start] and ilev < mstart:
                    mstart = ilev
                if mp[ilev] <= pres[end - 1] and ilev + 1 > mend:
                    mend = ilev + 1
            if mstart == INT_MAX:
                mstart = nm - 1
            if mend == INT_MIN:
                mend = 1
            self._interp_ranges = (adj, start, end, mstart, mend)

        for fname in sorted(self.fields):
            if self.field_type[fname] == "Computed":
                continue
            file_varname = self.file_varnames.get(fname, fname)
            if self.fields[fname].ndim == 0:
                self.fields[fname] = np.array(float(self._read_at(file_varname, t)[0]))
                continue
            adj, start, end, mstart, mend = self._interp_ranges
            data = self._read_at(file_varname, t)
            fv = np.zeros(adj)
            fv[:adj - 1] = data[:adj - 1]
            if fname in self.surface_varnames:
                fv[adj - 1] = float(self._read_at(self.surface_varnames[fname], t)[0])
            else:
                dx = fv[adj - 2] - fv[adj - 3]
                if dx == 0:
                    fv[adj - 1] = fv[adj - 2]
                else:
                    pres = self.iop_file_pressure
                    dy = pres[adj - 2] - pres[adj - 3]
                    scale = dy / dx
                    fv[adj - 1] = (pres[adj - 1] - pres[adj - 2]) / scale + fv[adj - 2]
            out = self.fields[fname]
            out[mstart:mend] = lin_interp_1d(self.iop_file_pressure[start:end],
                                             self.model_pressure[mstart:mend], fv[start:end])
            if fname in _FILLED_ENDS:
                out[:mstart + 1] = fv[0]
                out[mend - 1:] = fv[adj - 1]
        if self.field_type.get("divT3d") == "Computed":
            self.fields["divT3d"] = self.fields["divT"] + self.fields["vertdivT"]
        if self.field_type.get("divq3d") == "Computed":
            self.fields["divq3d"] = self.fields["divq"] + self.fields["vertdivq"]
        self.time_idx_of_current_data = t

    # ------------------------------------------------------------------ initial conditions
    def correct_temperature_and_water_vapor(self, T_mid_col0: np.ndarray, qv_col0: np.ndarray):
        """Replace IOP T (and q) above the first non-zero T by the model's column 0."""
        t_iop = self.fields["T"]
        nz = np.nonzero(t_iop > 0)[0]
        first_valid_idx = int(nz[0]) if nz.size else INT_MAX
        if 0 < first_valid_idx:
            k = min(first_valid_idx, self.model_nlevs)
            t_iop[:k] = T_mid_col0[:k]
            self.fields["q"][:k] = qv_col0[:k]

    def set_fields_from_iop_data(self, state: dict) -> dict:
        """set_fields_from_iop_data: ps, T_mid, horiz_winds, qv (and nc, qc, qi, ni) from IOP.

        state: EAMxx-named tensors (ncol, ...). Returns a new dict.
        """
        out = dict(state)
        if self.params.zero_non_iop_tracers:
            for k in ("qv", "qc", "nc", "qr", "nr", "qi", "ni", "qm", "bm", "tke"):
                if k in out:
                    out[k] = torch.zeros_like(out[k])
        ref = out["T_mid"]
        if "T_mid" in out and "qv" in out:
            self.correct_temperature_and_water_vapor(out["T_mid"][0].detach().cpu().numpy(),
                                                     out["qv"][0].detach().cpu().numpy())

        def col(name):
            return torch.as_tensor(self.fields[name], dtype=ref.dtype, device=ref.device)

        ncol = ref.shape[0]
        if "ps" in out and self.has_iop_field("Ps"):
            out["ps"] = torch.full((ncol,), float(self.fields["Ps"]), dtype=ref.dtype,
                                   device=ref.device)
        if self.has_iop_field("T"):
            out["T_mid"] = col("T").expand(ncol, -1).clone()
        if "horiz_winds" in out and (self.has_iop_field("u") or self.has_iop_field("v")):
            hw = out["horiz_winds"]
            u = col("u").expand(ncol, -1) if self.has_iop_field("u") else hw[:, 0]
            v = col("v").expand(ncol, -1) if self.has_iop_field("v") else hw[:, 1]
            out["horiz_winds"] = torch.stack([u, v], dim=1).clone()
        for fld, iop in (("qv", "q"), ("nc", "NUMLIQ"), ("qc", "CLDLIQ"), ("qi", "CLDICE"),
                         ("ni", "NUMICE")):
            if fld in out and self.has_iop_field(iop):
                out[fld] = col(iop).expand(ncol, -1).clone()
        return out

    def advance(self, run_t0: TimeStamp, dt: float, nsteps: int) -> None:
        """Replay the reads a run makes: at run_t0 (initialization), then at the start of
        each of the first nsteps steps. Used to rebuild the state for a given step."""
        self.read_iop_file_data(run_t0)
        ts = run_t0
        for _ in range(nsteps):
            self.read_iop_file_data(ts)
            ts = ts + dt


# ---------------------------------------------------------------------- the process
def _lin_interp_cols(x1: torch.Tensor, x2: torch.Tensor, y1: torch.Tensor) -> torch.Tensor:
    """ekat::LinInterp per column: x1, x2, y1 (ncol, nlev), x1 ascending."""
    km1 = x1.shape[-1]
    k1 = torch.searchsorted(x1.contiguous(), x2.contiguous(), right=True)
    k1 = torch.where(k1 > 0, k1 - 1, k1)
    k1ph = torch.where(k1 == km1 - 1, k1 - 1, k1 + 1)
    x1_k1, x1_k1ph = torch.gather(x1, -1, k1), torch.gather(x1, -1, k1ph)
    y1_k1, y1_k1ph = torch.gather(y1, -1, k1), torch.gather(y1, -1, k1ph)
    y = y1_k1ph - y1_k1
    y = y * (x2 - x1_k1)
    y = y / (x1_k1ph - x1_k1)
    return y + y1_k1


def advance_iop_subsidence(dt, ref_p_mid, omega, u, v, T, Q: dict):
    p_dep = ref_p_mid - dt * omega
    u_new = _lin_interp_cols(ref_p_mid, p_dep, u)
    v_new = _lin_interp_cols(ref_p_mid, p_dep, v)
    T_new = _lin_interp_cols(ref_p_mid, p_dep, T)
    T_new = T_new * (1.0 + (dt * C.Rair / C.Cpair) * omega / ref_p_mid)
    Q_new = {k: _lin_interp_cols(ref_p_mid, p_dep, q) for k, q in Q.items()}
    return u_new, v_new, T_new, Q_new


def horiz_mean(x: torch.Tensor, weight: float) -> torch.Tensor:
    """horiz_contraction over dim 0 with a constant weight: acc = 0; acc += w*x_i (in order)."""
    return torch.cumsum(weight * x, dim=0)[-1]


def iop_forcing_run(state: dict, dt: float, iop: IOPDataManager, hyam, hybm,
                    tracer_names: tuple, num_global_cols: int | None = None) -> dict:
    """IOPForcing::run_impl, after the data manager has read the data for this step.

    state: ps (ncol,), horiz_winds (ncol, 2, nlev), T_mid, and the tracers (ncol, nlev),
      which include qv. num_global_cols: the columns the nudging means run over (the
      domain of a DP run); None means each column is its own domain (a single-column run).
    Returns the updated T_mid, horiz_winds and tracers.
    """
    p = iop.params
    T = state["T_mid"]
    ref = T
    ncol = T.shape[0]
    hyam_t = torch.as_tensor(hyam, dtype=ref.dtype, device=ref.device)
    hybm_t = torch.as_tensor(hybm, dtype=ref.dtype, device=ref.device)

    def f(name):
        return torch.as_tensor(iop.fields[name], dtype=ref.dtype, device=ref.device)

    ps0 = C.P0
    ref_p_mid = hyam_t * ps0 + hybm_t * state["ps"][:, None]
    u, v = state["horiz_winds"][:, 0], state["horiz_winds"][:, 1]
    Q = {k: state[k] for k in tracer_names}
    if p.iop_dosubsidence:
        u, v, T, Q = advance_iop_subsidence(dt, ref_p_mid, f("omega"), u, v, T, Q)
    divT = f("divT3d") if iop.use_3d_forcing else f("divT")
    divq = f("divq3d") if iop.use_3d_forcing else f("divq")
    T = 1.0 * T + dt * divT
    Q["qv"] = 1.0 * Q["qv"] + dt * divq
    if p.iop_coriolis:
        u_ls, v_ls = f("u_ls"), f("v_ls")
        fcor = 2 * C.omega * math.sin(p.target_latitude * C.Pi / 180)
        u_cor = v - v_ls
        v_cor = u - u_ls
        u = 1.0 * u + (dt * fcor) * u_cor
        v = 1.0 * v + (-dt * fcor) * v_cor

    if p.iop_nudge_tq or p.iop_nudge_uv:
        rtau = p.iop_nudge_tscale if dt < p.iop_nudge_tscale else dt   # std::max(dt, tscale)
        alpha = -dt / rtau

        def mean(x):
            if num_global_cols is None:   # each column is its own domain: weight 1/1
                return 0.0 + 1.0 * x
            return horiz_mean(x, 1.0 / num_global_cols).expand_as(x)

        if p.iop_nudge_tq:
            qv_mean = mean(Q["qv"])
            t_mean = mean(T)
            ps_iop = float(iop.fields["Ps"])
            pressure_from_iop = torch.as_tensor(iop.hyam * ps0 + iop.hybm * ps_iop,
                                                dtype=ref.dtype, device=ref.device)
            nudge_level = ((pressure_from_iop <= p.iop_nudge_tq_low * 100)
                           & (pressure_from_iop >= p.iop_nudge_tq_high * 100))
            Q["qv"] = torch.where(nudge_level, 1.0 * Q["qv"] + alpha * (qv_mean - f("q")), Q["qv"])
            T = torch.where(nudge_level, 1.0 * T + alpha * (t_mean - f("T")), T)
        if p.iop_nudge_uv:
            u_iop = f("u_ls") if iop.use_large_scale_wind else f("u")
            v_iop = f("v_ls") if iop.use_large_scale_wind else f("v")
            u_mean, v_mean = mean(u), mean(v)
            u = 1.0 * u + alpha * (u_mean - u_iop)
            v = 1.0 * v + alpha * (v_mean - v_iop)

    out = {"T_mid": T, "horiz_winds": torch.stack([u, v], dim=1)}
    out.update(Q)
    return out


def iop_forcing_postconditions(fields: dict) -> tuple[dict, dict]:
    """The postcondition checks of IOPForcing::initialize_impl, in registration order."""
    out = dict(fields)
    status = {}
    out["T_mid"], status["T_mid"] = within_interval(out["T_mid"], 100.0, 500.0, False)
    out["horiz_winds"], status["horiz_winds"] = within_interval(out["horiz_winds"], -400.0,
                                                                400.0, False)
    out["qv"], status["qv"] = within_interval(out["qv"], 0, 0.2, True)
    return out, status


# ---------------------------------------------------------------------- surface imports
IOP_SURFACE_IMPORTS = ("surf_evap", "surf_sens_flux", "surf_radiative_T", "surf_lw_flux_up")


def overwrite_iop_imports(state: dict, iop: IOPDataManager, import_names=IOP_SURFACE_IMPORTS):
    """SurfaceCouplingImporter::overwrite_iop_imports (run-time imports)."""
    out = dict(state)
    ref = next(v for v in state.values() if isinstance(v, torch.Tensor) and v.is_floating_point())
    ncol = state[import_names[0]].shape[0] if import_names[0] in state else ref.shape[0]

    def full(x):
        return torch.full((ncol,), x, dtype=ref.dtype, device=ref.device)   # value in double, stored as Real

    for fname in import_names:
        if fname == "surf_evap" and iop.has_iop_field("lhflx"):
            out[fname] = full(float(iop.fields["lhflx"]) / C.LatVap)
        elif fname == "surf_sens_flux" and iop.has_iop_field("shflx"):
            out[fname] = full(float(iop.fields["shflx"]))
        elif fname == "surf_radiative_T" and iop.has_iop_field("Tg"):
            out[fname] = full(float(iop.fields["Tg"]))
        elif fname == "surf_lw_flux_up" and iop.has_iop_field("Tg"):
            tg = torch.as_tensor(float(iop.fields["Tg"]), dtype=torch.float64)
            out[fname] = full(float(C.stebol * mathfn.pow(tg, torch.tensor(4.0,
                                                                           dtype=torch.float64))))
    return out


def load_hybrid(vcoord_file: str | Path) -> dict:
    """hyam, hybm, hyai, hybi from a vertical-coordinate file (float64 numpy)."""
    import netCDF4
    with netCDF4.Dataset(str(vcoord_file)) as d:
        return {k: np.asarray(d[k][:], dtype=np.float64) for k in ("hyam", "hybm", "hyai", "hybi")}
