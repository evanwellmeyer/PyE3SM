"""The EAMxx physics suite: process order, checks, sub-cycling, radiation frequency.

Translation of how EAMxx runs its physics group, E3SM 80fe0a71:
  - share/atm_process/atmosphere_process.cpp       AtmosphereProcess::run: pre-condition
        checks, sub-cycles of run_impl with their own time stamps, post-condition checks
        (an invariant check is registered as both)
  - share/atm_process/atmosphere_process_group.cpp  sequential groups
  - cime_config/namelist_defaults_eamxx.xml         the standard process list
        physics = [iop_forcing (DP/SCM only), mac_aero_mic, rrtmgp],
        mac_aero_mic = [tms, shoc, cld_fraction, spa, p3];
        and the low-resolution list with ZM (F2010xx-ZM): physics = [zm, mac_aero_mic,
        rrtmgp] with mac_aero_mic sub-cycled 12 times and radiation every 2 steps at 1800 s

Each process adapter reads EAMxx-named fields from the state dict and returns the
fields it writes, after its own property checks. The adapters call the same
scheme entry points the per-scheme replays validate.

Public entry point: Suite.step (or physics_step), which runs one physics step and
returns the new state, tendencies, surface quantities and diagnostics. The host
provides pressures (p_mid, p_int, pseudo_density and the dry versions), surface
fluxes and grid geometry; index 0 is the model top.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .cld_fraction import cld_fraction
from .p3 import P3Runtime, load_p3_tables, p3_run
from .p3 import invariant_checks as p3_invariant_checks
from .p3 import postconditions as p3_postconditions
from .property_checks import within_interval
from .rrtmgp import RRTMGPRuntime, load_rrtmgp, rrtmgp_run
from .rrtmgp import invariant_checks as rrtmgp_invariant_checks
from .shoc import SHOCRuntime, npbl_from_hybrid, shoc_run
from .shoc.interface import postconditions as shoc_postconditions
from .precision import cast, double_device, to_precision
from .spa import SPA, load_spa_columns
from .timestamp import TimeStamp
from .tms import tms
from .zm import ZMRuntime, zm_run, zm_setup
from .zm import invariant_checks as zm_invariant_checks
from .zm import postconditions as zm_postconditions

SHOC_TRACERS = ("bm", "nc", "ni", "nr", "qi", "qm", "qr")   # turbulence-advected, besides qv/qc/tke
ADVECTED_TRACERS = ("qv", "bm", "nc", "ni", "nr", "qc", "qi", "qm", "qr", "tke")
MAC_AERO_MIC = ("tms", "shoc", "cld_fraction", "spa", "p3")
DOUBLE_ONLY = frozenset({"iop_forcing", "zm", "tms", "rrtmgp"})   # not built in EAMxx single precision


@dataclass
class Grid:
    """Host geometry. lat/lon in degrees, cell_length in m, all (ncol,); hybrid
    coefficients (nlev) and (nlev+1) as numpy float64."""
    lat: torch.Tensor
    lon: torch.Tensor
    cell_length: torch.Tensor
    hyam: np.ndarray
    hybm: np.ndarray
    hyai: np.ndarray
    hybi: np.ndarray


@dataclass
class SuiteConfig:
    """Options of the standard suite (EAMxx defaults unless set)."""
    shoc: SHOCRuntime = field(default_factory=SHOCRuntime)
    p3: P3Runtime = field(default_factory=P3Runtime)
    p3_predict_nc: bool = True
    p3_prescribed_ccn: bool = True
    rrtmgp: RRTMGPRuntime = field(default_factory=RRTMGPRuntime)
    mac_aero_mic_subcycles: int = 1
    p3_tables_dir: str | None = None
    rrtmgp_files: tuple | None = None       # sw coeffs, lw coeffs, sw cloud optics, lw cloud optics
    spa_data_file: str | None = None
    spa_target: tuple | None = None         # (lat, lon) for the nearest SPA column
    zm: ZMRuntime | None = None             # ZM on (EAMxx-LR) when set; runs before mac_aero_mic
    dtype: torch.dtype = torch.float64      # working precision (float32: P3 reads *.dat4 tables)
    device: str = "cpu"                     # e.g. "mps" or "cuda"
    # EAMxx builds RRTMGP, ZM, TMS and IOP forcing only in double: its single-precision
    # build has float P3, SHOC, cld_fraction and SPA with those processes absent. With
    # dtype float32 these run in float64 (on the CPU when the device has no float64) and
    # their outputs return to dtype. False runs them in float32 too, which has no
    # reference; RRTMGP's minor-gas scaling then overflows (docs/float32_gpu.md).
    double_only_in_float64: bool = True


@dataclass
class Clock:
    """A process's own time stamps (EAMxx keeps one pair per process)."""
    start: TimeStamp
    end: TimeStamp


class Suite:
    """The physics group of EAMxx with its data and persistent process state.

    first_procs: processes that run before mac_aero_mic inside the physics group, as
    (name, callable(state, dt, clock) -> (fields, status)); the SCM driver passes
    iop_forcing here, as DP-EAMxx lists it first.
    """

    def __init__(self, config: SuiteConfig, grid: Grid, run_t0: TimeStamp, ncol: int,
                 first_procs: tuple = ()):
        self.config = config
        self.grid = grid
        self.run_t0 = run_t0
        self.first_procs = tuple(first_procs)
        dt, dev = config.dtype, config.device
        self.double_procs = (DOUBLE_ONLY if config.double_only_in_float64 and dt != torch.float64
                             else frozenset())
        self.ddev = double_device(dev) if self.double_procs else dev
        ddt = torch.float64 if self.double_procs else dt
        self.p3_tables = (load_p3_tables(config.p3_tables_dir, dtype=dt, device=dev)
                          if config.p3_tables_dir else None)
        self.rrtmgp_data = (cast(load_rrtmgp(*config.rrtmgp_files, config.rrtmgp.gas_names), ddt, self.ddev)
                            if config.rrtmgp_files else None)
        self.spa = None
        if config.spa_data_file:
            lat, lon = config.spa_target
            self.spa = SPA(load_spa_columns(config.spa_data_file, lat, lon, ncol, dtype=dt, device=dev), run_t0)
        self.npbl = npbl_from_hybrid(grid.hyam, grid.hybm)
        self.zm_setup = zm_setup(config.zm, grid.hyai, grid.hybi, dtype=ddt, device=self.ddev) if config.zm else None
        names = ([n for n, _ in self.first_procs] + (["zm"] if config.zm else [])
                 + list(MAC_AERO_MIC) + ["rrtmgp"])
        self.clocks = {n: Clock(run_t0.copy(), run_t0.copy()) for n in names}
        self.num_steps = 0

    def set_time(self, ts: TimeStamp) -> None:
        """Put every process clock at ts (the start of the next step), as after a restart."""
        for c in self.clocks.values():
            c.start, c.end = ts.copy(), ts.copy()
        self.num_steps = ts.num_steps

    # ------------------------------------------------------------------ adapters
    def run_zm(self, s, dt, clock):
        s, st_pre = zm_invariant_checks(s)
        out = zm_run(s, dt, self.zm_setup, clock.start.num_steps == 0)
        out.pop("_internal", None)
        out.setdefault("qv", s["qv"])      # pre-condition repairs reach the next process
        out.setdefault("T_mid", s["T_mid"])
        out, st_inv = zm_invariant_checks(out)
        out, st_post = zm_postconditions(out)
        return out, {**{"pre_" + k: v for k, v in st_pre.items()}, **st_inv, **st_post}

    def run_tms(self, s, dt, clock):
        out, _ = tms(s["horiz_winds"], s["T_mid"], s["p_mid"], s["pseudo_density"], s["qv"],
                     s["sgh30"], s["landfrac"])
        return out, {}

    def run_shoc(self, s, dt, clock):
        tracers = tuple(n for n in SHOC_TRACERS if n in s)
        out = shoc_run(s, dt, self.grid.cell_length, self.npbl, self.config.shoc, apply_tms=True,
                       tracer_names=tracers)
        out.pop("_internal", None)
        return shoc_postconditions(out)

    def run_cld_fraction(self, s, dt, clock):
        out = cld_fraction(s["qi"], s["cldfrac_liq"])
        status = {}
        for n in ("cldfrac_ice", "cldfrac_tot", "cldfrac_ice_for_analysis", "cldfrac_tot_for_analysis"):
            out[n], status[n] = within_interval(out[n], 0.0, 1.0, False)
        return out, status

    def run_spa(self, s, dt, clock):
        return self.spa.run(clock.end, s["p_mid"])

    def run_p3(self, s, dt, clock):
        c = self.config
        s, st_pre = p3_invariant_checks(s)
        out = p3_run(s, dt, self.p3_tables, c.p3, c.p3_predict_nc, c.p3_prescribed_ccn)
        out.pop("_internal", None)
        # pre-condition repairs (qv) are part of the state the next process sees
        out.setdefault("qv", s["qv"])
        out, st_inv = p3_invariant_checks(out)
        out, st_post = p3_postconditions(out)
        return out, {**{"pre_" + k: v for k, v in st_pre.items()}, **st_inv, **st_post}

    def run_rrtmgp(self, s, dt, clock):
        s, st_pre = rrtmgp_invariant_checks(s)
        out = rrtmgp_run(s, dt, self.rrtmgp_data, self.config.rrtmgp, clock.start,
                         clock.start.num_steps, self.grid.lat, self.grid.lon)
        out.pop("_internal", None)
        out, st_post = rrtmgp_invariant_checks(out)
        return out, {"pre_T_mid": st_pre["T_mid"], **st_post}

    # ------------------------------------------------------------------ running
    def _run_process(self, name, fn, state, dt, nsub=1):
        """AtmosphereProcess::run for one process: sub-cycles with their time stamps."""
        clock = self.clocks[name]
        dt_sub = dt / nsub
        if name in self.double_procs:
            fn = self._in_double(fn)
        status = {}
        for _ in range(nsub):
            clock.start = clock.end.copy()
            clock.end = clock.end + dt_sub
            out, status = fn(state, dt_sub, clock)
            state = {**state, **out}
        return state, status

    def _in_double(self, fn):
        """fn run on float64 inputs on the double device, its outputs back in the state's
        precision and device."""
        dt, dev = self.config.dtype, self.config.device

        def run(state, dt_sub, clock):
            out, status = fn(to_precision(state, torch.float64, self.ddev), dt_sub, clock)
            return to_precision(out, dt, dev), to_precision(status, dt, dev)
        return run

    def step(self, state: dict, dt: float) -> dict:
        """One physics step. Returns dict(state, tendencies, status, per_process,
        per_subcycle): per_process[name] is the state after that process's last call,
        per_subcycle[name] the list of states after each mac_aero_mic sub-cycle."""
        before = state
        status, per_process = {}, {}
        per_subcycle = {n: [] for n in MAC_AERO_MIC}
        for name, fn in self.first_procs:
            state, status[name] = self._run_process(name, fn, state, dt)
            per_process[name] = state
        if self.zm_setup is not None:
            state, status["zm"] = self._run_process("zm", self.run_zm, state, dt)
            per_process["zm"] = state
        nsub = self.config.mac_aero_mic_subcycles
        dt_sub = dt / nsub
        adapters = {"tms": self.run_tms, "shoc": self.run_shoc,
                    "cld_fraction": self.run_cld_fraction, "spa": self.run_spa, "p3": self.run_p3}
        for _ in range(nsub):   # the group sub-cycles; each member runs once per sub-cycle
            for name in MAC_AERO_MIC:
                state, status[name] = self._run_process(name, adapters[name], state, dt_sub)
                per_process[name] = state
                per_subcycle[name].append(state)
        state, status["rrtmgp"] = self._run_process("rrtmgp", self.run_rrtmgp, state, dt)
        per_process["rrtmgp"] = state
        self.num_steps += 1
        tend = {}
        for k in ("T_mid", "horiz_winds") + ADVECTED_TRACERS:
            if k in before and k in state:
                tend[k] = (state[k] - before[k]) / dt
        return dict(state=state, tendencies=tend, status=status, per_process=per_process,
                    per_subcycle=per_subcycle)


def physics_step(suite: Suite, state: dict, dt: float) -> dict:
    """The public entry point: Suite.step."""
    return suite.step(state, dt)
