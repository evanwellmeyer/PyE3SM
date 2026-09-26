"""Simple Prescribed Aerosols (EAMxx process ``spa``).

Translation of, at E3SM 80fe0a71:
  components/eamxx/src/physics/spa/eamxx_spa_process_interface.cpp
      SPA::create_requests, initialize_impl, run_impl
  components/eamxx/src/share/remap/iop_remapper.cpp
      IOPRemapper::setup_closest_col_info (nearest data column for IOP runs)
plus the DataInterpolation pieces in data_interpolation.py.

Per step: interpolate the monthly climatology to the end-of-step time,
rebuild the data pressure from its surface pressure, interpolate linearly in
pressure to the model p_mid (constant extrapolation beyond the data range),
then apply the post-condition range checks, which repair out-of-range values.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np
import torch

from . import data_interpolation as di
from .property_checks import within_interval
from .timestamp import TimeStamp

# File variable -> EAMxx field (SPA::initialize_impl aliases).
FILE_TO_FIELD = {
    "CCN3": "nccn",
    "AER_G_SW": "aero_g_sw",
    "AER_SSA_SW": "aero_ssa_sw",
    "AER_TAU_SW": "aero_tau_sw",
    "AER_TAU_LW": "aero_tau_lw",
}
EPS = sys.float_info.epsilon  # std::numeric_limits<double>::epsilon()


@dataclass
class SPAColumnData:
    """SPA data on the model columns, for all retained time slices.

    fields[name]: (nslice, ncol, [nband,] nlev_src) float64
    ps: (nslice, ncol); hyam, hybm: (nlev_src,)
    """

    db: di.TimeDatabase
    fields: dict[str, torch.Tensor]
    ps: torch.Tensor
    hyam: torch.Tensor
    hybm: torch.Tensor


def nearest_column_iop(lat: np.ndarray, lon: np.ndarray, target_lat: float, target_lon: float) -> int:
    """IOPRemapper::setup_closest_col_info: L1 distance in degrees, first minimum wins."""
    if not (-90 <= target_lat <= 90 and 0 <= target_lon <= 360):
        raise ValueError("IOP target lat/lon out of range")
    dist = np.abs(lat - target_lat) + np.abs(lon - target_lon)
    best, loc = np.finfo(np.float64).max, -1
    for icol, d in enumerate(dist):  # Kokkos::MinLoc: strict '<', first index on ties
        if d < best:
            best, loc = d, icol
    return loc


def load_spa_columns(path: str, target_lat: float, target_lon: float, ncol: int,
                     dtype=torch.float64, device="cpu") -> SPAColumnData:
    """Read the SPA file and copy the IOP nearest column to `ncol` model columns."""
    import netCDF4

    with netCDF4.Dataset(path) as d:
        icol = nearest_column_iop(d["lat"][:].data, d["lon"][:].data, target_lat, target_lon)
        db = di.periodic_time_database(d["time"][:].data, d["time"].units)
        sl = db.indices
        fields = {}
        for fname, name in FILE_TO_FIELD.items():
            v = d[fname]
            if v.dimensions[:2] != ("time", "ncol"):
                raise ValueError(f"unexpected dims for {fname}: {v.dimensions}")
            col = np.asarray(v[sl, icol].data, dtype=np.float64)  # float32 -> double, exact
            fields[name] = torch.as_tensor(col, dtype=dtype, device=device)[:, None].expand(
                (len(sl), ncol) + col.shape[1:]).contiguous()
        ps = np.asarray(d["PS"][sl, icol].data, dtype=np.float64)
        ps = torch.as_tensor(ps, dtype=dtype, device=device)[:, None].expand(len(sl), ncol).contiguous()
        hyam = torch.as_tensor(np.asarray(d["hyam"][:].data, np.float64), dtype=dtype, device=device)
        hybm = torch.as_tensor(np.asarray(d["hybm"][:].data, np.float64), dtype=dtype, device=device)
    return SPAColumnData(db, fields, ps, hyam, hybm)


class SPA:
    """SPA process. Construct at the run's start time; call once per step."""

    def __init__(self, data: SPAColumnData, t0: TimeStamp):
        self.data = data
        self.time = di.TimeInterpState(data.db, t0)

    def run(self, ts_end: TimeStamp, p_mid: torch.Tensor):
        """SPA::run_impl(dt) -> DataInterpolation::run(end_of_step_ts()), then post-conditions.

        Returns (fields, status): fields keyed by EAMxx names; status maps each
        field to its post-condition result (0 pass, 1 repaired, 2 fail).
        """
        i0, i1, alpha = self.time.weights(ts_end)
        d = self.data
        src = {n: di.time_interp_linear(f[i0], f[i1], alpha) for n, f in d.fields.items()}
        ps = di.time_interp_linear(d.ps[i0], d.ps[i1], alpha)
        p_data = di.reconstruct_p_from_ps(ps, d.hyam, d.hybm)
        out = di.vertical_remap(p_data, p_mid, src)
        self.last_raw = out  # output of run_impl, before the post-condition checks
        return self.postconditions(out)

    @staticmethod
    def postconditions(out: dict[str, torch.Tensor]):
        """SPA::initialize_impl property checks, applied in the order they were added."""
        status = {}
        out = dict(out)
        out["nccn"], status["nccn"] = within_interval(out["nccn"], 0, 1e11, True, -1e11 * EPS)
        for n in ("aero_g_sw", "aero_ssa_sw", "aero_tau_sw", "aero_tau_lw"):
            out[n], status[n] = within_interval(out[n], 0.0, 1.0, True)
        return out, status
