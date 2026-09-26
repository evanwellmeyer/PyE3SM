"""Time and vertical interpolation of prescribed data (used by SPA).

Translation of, at E3SM 80fe0a71:
  components/eamxx/src/share/algorithm/eamxx_data_interpolation.cpp
      TimeDatabase (build_time_database_slices, setup_periodic_time_database,
      find_interval, get_next_idx), init_time_interpolation,
      shift_data_interval, run (time interpolation), reconstruct_p_from_ps
  components/eamxx/src/share/remap/vertical_remapper.cpp
      remap_fwd_impl, apply_vertical_interpolation, extrapolate (P0 mode)
  externals/ekat/src/algorithm/ekat_lin_interp_impl.hpp
      LinInterp::setup_impl, lin_interp_impl
  components/eamxx/src/share/field/field_update_impl.hpp,
  components/eamxx/src/share/util/eamxx_combine_ops.hpp
      Field::update(x, alpha, beta): y *= beta; y += alpha*x; y += 0

Host-side logic (which time slices, the interpolation weight) is plain
Python on time stamps. Per-step arithmetic is torch.

Horizontal remapping is not here: the caller supplies the data already on the
model columns (for IOP cases, the nearest data column; see spa.load_spa_columns).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .timestamp import YEARLY_PERIODIC, TimeInterval, TimeStamp, parse_cf_time_units


# ----------------------------------------------------------------- time database
@dataclass
class TimeDatabase:
    times: list[TimeStamp]  # retained slice times, in file order
    indices: list[int]      # time index in the file of each retained slice
    timeline: str

    def size(self) -> int:
        return len(self.times)

    def get_next_idx(self, prev: int) -> int:
        nxt = prev + 1
        if nxt >= self.size():
            if self.timeline != YEARLY_PERIODIC:
                raise RuntimeError("Requesting slice past the database end")
            nxt = nxt % self.size()
        return nxt

    def find_interval(self, t: TimeStamp) -> int:
        def contains(b, e):
            return TimeInterval.make(self.times[b], self.times[e], self.timeline).contains(t)

        for beg in range(self.size() - 1):
            if contains(beg, beg + 1):
                return beg
        if not (self.timeline == YEARLY_PERIODIC and contains(self.size() - 1, 0)):
            raise RuntimeError(f"No interval contains {t.to_string()}")
        return self.size() - 1


def periodic_time_database(raw_times, time_units: str,
                           year_start_ts: TimeStamp | None = None) -> TimeDatabase:
    """build_time_database_slices + setup_periodic_time_database (one file).

    raw_times: values of the file's time variable; time_units: its 'units'.
    """
    t_ref, mult = parse_cf_time_units(time_units)
    times = [t_ref + float(t) * mult for t in raw_times]
    if any(times[i + 1] < times[i] for i in range(len(times) - 1)):
        raise RuntimeError("time slices not sorted")
    slices = list(zip(times, range(len(times))))
    if len(slices) < 2:
        raise RuntimeError("need 2+ time slices")

    start_ts = year_start_ts if (year_start_ts is not None and year_start_ts.is_valid()) else slices[0][0]
    start_p1y = TimeStamp.make(start_ts.date[0] + 1, start_ts.date[1], start_ts.date[2], *start_ts.time)
    slices = [s for s in slices if not (start_p1y <= s[0])]
    last = slices[-1][0]
    first_valid_past = TimeStamp.make(last.date[0] - 1, last.date[1], last.date[2], *last.time)
    slices = [s for s in slices if not (s[0] <= first_valid_past)]
    if len(slices) < 2:
        raise RuntimeError("Not enough time slices within 12 months")
    return TimeDatabase([s[0] for s in slices], [s[1] for s in slices], YEARLY_PERIODIC)


class TimeInterpState:
    """The data-interval bookkeeping of DataInterpolation (host side).

    Holds which two database slices bracket the current time. `weights(ts)`
    reproduces DataInterpolation::run(ts) up to the field update: it may shift
    the interval once, then returns (slice_beg, slice_end, alpha).
    """

    def __init__(self, db: TimeDatabase, t0: TimeStamp):
        # init_time_interpolation(t0)
        self.db = db
        t0_interval = db.find_interval(t0)
        t_beg = db.times[t0_interval]
        self.interval = TimeInterval.make(TimeStamp(), t_beg, db.timeline, do_compute_length=False)
        self.idx = [None, t0_interval]
        self._shift()

    def _shift(self) -> None:
        # shift_data_interval
        self.idx[0] = self.idx[1]
        self.idx[1] = self.db.get_next_idx(self.idx[0])
        self.interval.advance(self.db.times[self.idx[1]])

    def weights(self, ts: TimeStamp) -> tuple[int, int, float]:
        if not self.interval.contains(ts):
            self._shift()
        beg_to_ts = TimeInterval.make(self.interval.beg, ts, self.interval.timeline)
        alpha = beg_to_ts.length / self.interval.length
        if not (0 <= alpha <= 1):
            raise RuntimeError(f"time {ts.to_string()} outside the data interval")
        return self.idx[0], self.idx[1], alpha


# ------------------------------------------------------------- per-step math
def time_interp_linear(beg: torch.Tensor, end: torch.Tensor, alpha: float) -> torch.Tensor:
    """out.deep_copy(beg); out.update(end, alpha, 1-alpha) (CombineMode::Update, gamma=0)."""
    beta = 1 - alpha
    out = beg * beta
    out = out + alpha * end
    out = out + 0.0
    return out


P0 = 100000.0  # physics::Constants::P0 [Pa]


def reconstruct_p_from_ps(ps: torch.Tensor, hyam: torch.Tensor, hybm: torch.Tensor) -> torch.Tensor:
    """p_data(icol,k) = ps(icol)*hybm(k) + P0*hyam(k).   ps: (ncol,) -> (ncol, nlev_src)."""
    return ps[..., None] * hybm + P0 * hyam


def lin_interp_setup(x1: torch.Tensor, x2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """LinInterp::setup_impl: k1 = upper_bound(x1, x2) - 1 (clamped at 0); k1ph = k1 +/- 1.

    x1: (ncol, km1) source coordinate (ascending); x2: (ncol, km2) target.
    """
    km1 = x1.shape[-1]
    ub = torch.searchsorted(x1.contiguous(), x2.contiguous(), right=True)
    k1 = torch.where(ub > 0, ub - 1, ub)
    k1ph = torch.where(k1 == km1 - 1, k1 - 1, k1 + 1)
    return k1, k1ph


def lin_interp(x1, x2, y1, k1, k1ph) -> torch.Tensor:
    """LinInterp::lin_interp_impl.

    y2 = y1_k1ph - y1_k1; y2 *= x2 - x1_k1; y2 /= x1_k1ph - x1_k1; y2 += y1_k1
    y1 may have extra dims between column and level: (ncol, ..., km1).
    """
    x1_k1 = torch.gather(x1, -1, k1)
    x1_k1ph = torch.gather(x1, -1, k1ph)
    extra = y1.shape[1:-1]
    idx_shape = (k1.shape[0],) + (1,) * len(extra) + (k1.shape[-1],)
    exp_shape = (k1.shape[0],) + tuple(extra) + (k1.shape[-1],)
    y1_k1 = torch.gather(y1, -1, k1.reshape(idx_shape).expand(exp_shape))
    y1_k1ph = torch.gather(y1, -1, k1ph.reshape(idx_shape).expand(exp_shape))
    if extra:
        bshape = (k1.shape[0],) + (1,) * len(extra) + (k1.shape[-1],)
        x2b, x1_k1b, x1_k1phb = x2.reshape(bshape), x1_k1.reshape(bshape), x1_k1ph.reshape(bshape)
    else:
        x2b, x1_k1b, x1_k1phb = x2, x1_k1, x1_k1ph
    y2 = y1_k1ph - y1_k1
    y2 = y2 * (x2b - x1_k1b)
    y2 = y2 / (x1_k1phb - x1_k1b)
    y2 = y2 + y1_k1
    return y2


def extrapolate_p0(x1, x2, y1, y2) -> torch.Tensor:
    """VerticalRemapper::extrapolate with P0 at both ends.

    Lower half of target levels (ilev >= nlevs_tgt/2): x2 > x1[-1] -> y1[-1].
    Upper half: x2 < x1[0] -> y1[0].
    """
    nlev_tgt = x2.shape[-1]
    mid = nlev_tgt // 2
    ilev = torch.arange(nlev_tgt, device=x2.device)
    x_min = x1[..., :1]
    x_max = x1[..., -1:]
    below = (ilev >= mid) & (x2 > x_max)
    above = (ilev < mid) & (x2 < x_min)
    extra = y1.ndim - 2
    if extra:
        shp = (x2.shape[0],) + (1,) * extra + (nlev_tgt,)
        below, above = below.reshape(shp), above.reshape(shp)
    y2 = torch.where(below, y1[..., -1:], y2)
    y2 = torch.where(above, y1[..., :1], y2)
    return y2


def vertical_remap(x1, x2, fields: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """VerticalRemapper::remap_fwd_impl for LEV fields with P0 extrapolation."""
    k1, k1ph = lin_interp_setup(x1, x2)
    out = {}
    for name, y1 in fields.items():
        y2 = lin_interp(x1, x2, y1, k1, k1ph)
        out[name] = extrapolate_p0(x1, x2, y1, y2)
    return out
