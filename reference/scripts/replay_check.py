#!/usr/bin/env python3
"""Replay every dumped call of an EAMxx process through the port and report differences.

  python reference/scripts/replay_check.py cld_fraction reference/runs/dumps_*  [--rtol 0 --atol 0]

Run in the project env (needs torch and eamxx_torch). Reads the raw dumps
directly, so it covers all dumped columns, steps, ranks and cases, not only the
small fixtures committed to tests/data.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from read_dump import read_dump  # noqa: E402

from eamxx_torch.testing import compare  # noqa: E402

# Working precision of the replay (--dtype); float32 replays dumps of the
# single-precision build (the dump hook widens its floats to double, exactly).
DTYPE = torch.float64


def run_cld_fraction(pre: dict) -> dict:
    from eamxx_torch.cld_fraction import cld_fraction
    t = lambda k: torch.as_tensor(pre[k], dtype=DTYPE)  # noqa: E731
    return cld_fraction(t("qi"), t("cldfrac_liq"))


def case_settings(run_dir: Path) -> dict:
    """Read the few settings a replay needs from the run's input.yaml.

    DP runs have IOP settings; standalone runs (make_standalone_case.py) do not,
    and store the hybrid coefficients in their IC file.
    """
    txt = (run_dir / "input.yaml").read_text()

    def g(key):
        m = re.search(rf"^\s*{key}:\s*(\S+)", txt, re.M)
        return m.group(1) if m else None

    out = {"lat": g("target_latitude"), "lon": g("target_longitude"),
           "run_t0": g("run_t0"), "spa_data_file": g("spa_data_file"),
           "vcoord_file": g("vertical_coordinate_filename") or g("filename"),
           "extra_shoc_diags": g("extra_shoc_diags") == "true"}
    out["lat"] = out["lat"] and float(out["lat"])
    out["lon"] = out["lon"] and float(out["lon"])
    out["cell_length"] = shoc_cell_length(run_dir)
    out["p3"] = p3_settings(run_dir)
    out["rrtmgp"] = rrtmgp_settings(run_dir)
    out["iop"] = iop_settings(run_dir)
    out["zm"] = zm_settings(run_dir)
    return out


# SHOC cell length (dx = dy) is grid geometry, not a field, so it is not in the
# dumps. Patch 0004 logs it; runs made before that patch use the value logged by
# the instrumented build for the same planar domain (runs/check_cell_length).
_CELL_LENGTH_BY_DOMAIN = {(5, 5, 50000.0, 50000.0): 3333.333333333333}


def shoc_cell_length(run_dir: Path) -> float:
    for log in sorted(run_dir.glob("atm.log*")) + sorted(run_dir.glob("run.log")):
        m = re.search(r"\[ref_dump\] shoc cell_length\[0\] = (\S+)", log.read_text(errors="replace"))
        if m:
            return float(m.group(1))
    if not (run_dir / "namelist.nl").exists():
        return None  # not a dycore run; only shoc needs this
    nl = (run_dir / "namelist.nl").read_text()
    g = lambda key: float(re.search(rf"^\s*{key}\s*=\s*(\S+)", nl, re.M).group(1))  # noqa: E731
    return _CELL_LENGTH_BY_DOMAIN[(int(g("se_ne_x")), int(g("se_ne_y")), g("se_lx"), g("se_ly"))]


_SPA_CACHE: dict = {}


def run_spa(pre: dict, meta: dict, settings: dict, with_checks: bool) -> dict:
    from eamxx_torch.spa import SPA, load_spa_columns
    from eamxx_torch.timestamp import TimeStamp
    p_mid = torch.as_tensor(pre["p_mid"], dtype=DTYPE)
    key = (settings["spa_data_file"], settings["lat"], settings["lon"], p_mid.shape[0])
    if key not in _SPA_CACHE:
        _SPA_CACHE[key] = load_spa_columns(settings["spa_data_file"], settings["lat"],
                                           settings["lon"], p_mid.shape[0])
    spa = SPA(_SPA_CACHE[key], TimeStamp.from_string(settings["run_t0"]))
    # The dump's "post" time is end_of_step_ts (what SPA::run_impl uses).
    out, status = spa.run(TimeStamp.from_string(meta["time_end"]), p_mid)
    if not with_checks:
        from eamxx_torch import data_interpolation as di  # noqa: F401
        out = spa.last_raw
    return out


def run_tms(pre: dict) -> dict:
    from eamxx_torch.tms import tms
    t = lambda k: torch.as_tensor(pre[k], dtype=DTYPE)  # noqa: E731
    out, _ = tms(t("horiz_winds"), t("T_mid"), t("p_mid"), t("pseudo_density"), t("qv"),
                 t("sgh30"), t("landfrac"))
    return out


_NPBL_CACHE: dict = {}
SHOC_TRACERS = ("bm", "nc", "ni", "nr", "qi", "qm", "qr")


def run_shoc(pre: dict, meta: dict, settings: dict, with_checks: bool) -> dict:
    import netCDF4
    from eamxx_torch.shoc import SHOCRuntime, npbl_from_hybrid, postconditions, shoc_run
    vf = settings["vcoord_file"]
    if vf not in _NPBL_CACHE:
        with netCDF4.Dataset(vf) as d:
            _NPBL_CACHE[vf] = npbl_from_hybrid(np.asarray(d["hyam"][:]), np.asarray(d["hybm"][:]))
    state = {k: torch.as_tensor(v, dtype=DTYPE) for k, v in pre.items()}
    ncol = state["T_mid"].shape[0]
    cell = torch.full((ncol,), settings["cell_length"], dtype=DTYPE)
    tracers = tuple(n for n in SHOC_TRACERS if n in state)
    rt = SHOCRuntime(extra_diags=settings["extra_shoc_diags"])
    # EAMxx sets apply_tms when TMS is in the process list; SHOC then requires its drag field
    out = shoc_run(state, float(meta["dt"]), cell, _NPBL_CACHE[vf], rt,
                   apply_tms="surf_drag_coeff_tms" in state,
                   tracer_names=tracers)
    out.pop("_internal")
    if with_checks:
        out, _ = postconditions(out)
    return out


P3_TABLES_CACHE: dict = {}


def p3_settings(run_dir: Path) -> dict:
    """The p3 sublist of the run's input.yaml, as P3Runtime fields plus the two flags."""
    from eamxx_torch.p3 import P3Runtime
    txt = (run_dir / "input.yaml").read_text()
    blk = re.search(r"^( +)p3:\n(.*?)(?=^\1\S)", txt, re.M | re.S)
    if blk is None:
        return None  # this run has no p3 process
    body = blk.group(2)
    vals = dict(re.findall(r"^\s+([a-z_0-9]+):\s+(\S+)\s*$", body, re.M))
    fields = {f.name: str(f.type) for f in P3Runtime.__dataclass_fields__.values()}
    kw = {}
    for k, v in vals.items():
        if k in fields:
            kw[k] = (v == "true") if "bool" in fields[k] else float(v)
    return {"runtime": P3Runtime(**kw),
            "predictNc": vals.get("do_predict_nc", "true") == "true",
            "prescribedCCN": vals.get("do_prescribed_ccn", "true") == "true",
            "tables_dir": str(Path(re.search(r"tables:\s*\[([^,\]]+)", body).group(1)).parent)}


def run_p3(pre: dict, meta: dict, settings: dict, with_checks: bool) -> dict:
    from eamxx_torch.p3 import invariant_checks, load_p3_tables, p3_run, postconditions
    ps = settings["p3"]
    d = ps["tables_dir"]
    if (d, DTYPE) not in P3_TABLES_CACHE:
        P3_TABLES_CACHE[(d, DTYPE)] = load_p3_tables(d, dtype=DTYPE)
    state = {k: torch.as_tensor(v, dtype=DTYPE) for k, v in pre.items()}
    if with_checks:
        state, _ = invariant_checks(state)
    out = p3_run(state, float(meta["dt"]), P3_TABLES_CACHE[(d, DTYPE)], ps["runtime"],
                 ps["predictNc"], ps["prescribedCCN"])
    out.pop("_internal")
    if with_checks:
        out, _ = invariant_checks(out)
        out, _ = postconditions(out)
    return out


RRTMGP_DATA_CACHE: dict = {}


def rrtmgp_settings(run_dir: Path) -> dict:
    """The rrtmgp sublist of the run's input.yaml: RRTMGPRuntime plus the data files."""
    from eamxx_torch.rrtmgp import RRTMGPRuntime
    txt = (run_dir / "input.yaml").read_text()
    blk = re.search(r"^( +)rrtmgp:\n", txt, re.M)
    if blk is None:
        return None
    lines = []
    for ln in txt[blk.end():].splitlines():
        if ln.strip() and len(ln) - len(ln.lstrip()) <= len(blk.group(1)):
            break
        lines.append(ln)
    body = "\n".join(lines)
    vals = dict(re.findall(r"^\s+([a-z_0-9]+):\s+(.+?)\s*$", body, re.M))
    kw = {}
    for f in RRTMGPRuntime.__dataclass_fields__.values():
        if f.name not in vals:
            continue
        v, t = vals[f.name], str(f.type)
        if f.name == "active_gases":
            kw[f.name] = tuple(x.strip() for x in v.strip("[]").split(","))
        elif "bool" in t:
            kw[f.name] = v == "true"
        elif "int" in t:
            kw[f.name] = int(v)
        else:
            kw[f.name] = float(v)
    files = tuple(vals[k] for k in ("rrtmgp_coefficients_file_sw", "rrtmgp_coefficients_file_lw",
                                    "rrtmgp_cloud_optics_file_sw", "rrtmgp_cloud_optics_file_lw"))
    out = {"runtime": RRTMGPRuntime(**kw), "files": files, "lat": None, "lon": None}
    ic = run_dir / "ic.nc"
    if ic.exists():  # point grid: one latitude/longitude per column, from the IC file
        import netCDF4
        with netCDF4.Dataset(ic) as d:
            out["lat"], out["lon"] = np.asarray(d["lat"][:]), np.asarray(d["lon"][:])
    return out


def rrtmgp_data(rs: dict):
    from eamxx_torch.rrtmgp import load_rrtmgp
    key = (rs["files"], tuple(rs["runtime"].gas_names))
    if key not in RRTMGP_DATA_CACHE:
        RRTMGP_DATA_CACHE[key] = load_rrtmgp(*rs["files"], rs["runtime"].gas_names)
    return RRTMGP_DATA_CACHE[key]


def run_rrtmgp(pre: dict, meta: dict, settings: dict, with_checks: bool) -> dict:
    from eamxx_torch.rrtmgp import invariant_checks, rrtmgp_run
    from eamxx_torch.timestamp import TimeStamp
    rs = settings["rrtmgp"]
    state = {k: torch.as_tensor(v, dtype=DTYPE) for k, v in pre.items()}
    ncol = state["T_mid"].shape[0]
    if settings["lat"] is not None:   # DP/IOP runs: the IOP target point for every column
        lat = torch.full((ncol,), settings["lat"], dtype=DTYPE)
        lon = torch.full((ncol,), settings["lon"], dtype=DTYPE)
    else:
        lat = torch.as_tensor(rs["lat"][:ncol], dtype=DTYPE)
        lon = torch.as_tensor(rs["lon"][:ncol], dtype=DTYPE)
    if with_checks:
        state, _ = invariant_checks(state)
    # the dump's step is the end-of-step count; run_impl sees the start-of-step one
    out = rrtmgp_run(state, float(meta["dt"]), rrtmgp_data(rs), rs["runtime"],
                     TimeStamp.from_string(meta["time_start"]), int(meta["step"]) - 1, lat, lon)
    out.pop("_internal", None)
    if with_checks:
        out, _ = invariant_checks(out)
    return out


def iop_settings(run_dir: Path) -> dict:
    """iop_options, run_t0, dt and the grid size of a DP run (None for non-IOP runs)."""
    from eamxx_torch.scm import IOPParams
    txt = (run_dir / "input.yaml").read_text()
    blk = re.search(r"^iop_options:\n((?:[ ]+.*\n)+)", txt, re.M)
    if blk is None:
        return None
    vals = dict(re.findall(r"^\s+([a-z_0-9]+):\s+(\S+)\s*$", blk.group(1), re.M))
    kw = {}
    for f in IOPParams.__dataclass_fields__.values():
        if f.name in vals:
            v = vals[f.name]
            kw[f.name] = (v == "true") if "bool" in str(f.type) else (
                v if f.name == "iop_file" else float(v))
    g = lambda key: re.search(rf"^\s*{key}:\s*(\S+)", txt, re.M).group(1)  # noqa: E731
    out = {"params": IOPParams(**kw), "run_t0": g("run_t0"), "dt": float(g("time_step")),
           "num_global_cols": None}
    nl = run_dir / "namelist.nl"
    if nl.exists():
        t = nl.read_text()
        ne = [int(re.search(rf"^\s*{k}\s*=\s*(\S+)", t, re.M).group(1)) for k in ("se_ne_x", "se_ne_y")]
        out["num_global_cols"] = ne[0] * ne[1] * 4   # pg2: 4 physics columns per element
    return out


_HYBRID_CACHE: dict = {}


def load_hybrid_cached(vf):
    from eamxx_torch.scm import load_hybrid
    if vf not in _HYBRID_CACHE:
        _HYBRID_CACHE[vf] = load_hybrid(vf)
    return _HYBRID_CACHE[vf]


def _iop_manager(settings: dict, step: int):
    """A data manager in the state the run had at the start of this (1-based) step."""
    from eamxx_torch.scm import IOPDataManager, load_hybrid
    from eamxx_torch.timestamp import TimeStamp
    io = settings["iop"]
    vf = settings["vcoord_file"]
    if vf not in _HYBRID_CACHE:
        _HYBRID_CACHE[vf] = load_hybrid(vf)
    hyb = _HYBRID_CACHE[vf]
    t0 = TimeStamp.from_string(io["run_t0"])
    m = IOPDataManager(io["params"], t0, hyb["hyam"].shape[0], hyb["hyam"], hyb["hybm"])
    m.advance(t0, io["dt"], step)
    return m, hyb


IOP_TRACERS = ("qv", "bm", "nc", "ni", "nr", "qc", "qi", "qm", "qr", "tke")


def run_iop_forcing(pre: dict, meta: dict, settings: dict, with_checks: bool) -> dict:
    from eamxx_torch.scm import iop_forcing_postconditions, iop_forcing_run
    io = settings["iop"]
    p = io["params"]
    state = {k: torch.as_tensor(v, dtype=DTYPE) for k, v in pre.items()}
    ncol = state["T_mid"].shape[0]
    ncols_global = None
    partial = False
    if p.iop_nudge_tq or p.iop_nudge_uv:
        # The nudging means run over the whole domain: only replayable when the dump holds
        # every column (single-rank runs dumped with EAMXX_REF_DUMP_NCOLS >= ncol). With
        # fewer columns, the nudged fields are left out of the comparison.
        ncols_global = ncol
        partial = ncol != io["num_global_cols"]
    m, hyb = _iop_manager(settings, int(meta["step"]))
    tracers = tuple(k for k in IOP_TRACERS if k in state)
    out = iop_forcing_run(state, float(meta["dt"]), m, hyb["hyam"], hyb["hybm"], tracers,
                          num_global_cols=ncols_global)
    if with_checks:
        out, _ = iop_forcing_postconditions(out)
    if partial:
        drop = (["horiz_winds"] if p.iop_nudge_uv else []) + (["T_mid", "qv"] if p.iop_nudge_tq else [])
        out = {k: v for k, v in out.items() if k not in drop}
    return out


def run_sc_import(pre: dict, meta: dict, settings: dict, with_checks: bool) -> dict:
    from eamxx_torch.scm import overwrite_iop_imports
    from eamxx_torch.scm.iop import IOP_SURFACE_IMPORTS
    state = {k: torch.as_tensor(v, dtype=DTYPE) for k, v in pre.items()}
    if not settings["iop"]["params"].iop_srf_prop:
        return {}
    m, _ = _iop_manager(settings, int(meta["step"]))
    out = overwrite_iop_imports(state, m)
    return {k: out[k] for k in IOP_SURFACE_IMPORTS if k in state}


def zm_settings(run_dir: Path) -> dict:
    """The zm sublist of the run's input.yaml as ZMRuntime (None if the run has no zm)."""
    from eamxx_torch.zm import ZMRuntime
    txt = (run_dir / "input.yaml").read_text()
    blk = re.search(r"^( +)zm:\n((?:\1 +.*\n)+)", txt, re.M)
    if blk is None:
        return None
    vals = dict(re.findall(r"^\s+([a-z_0-9]+):\s+(\S+)\s*$", blk.group(2), re.M))
    kw = {}
    for f in ZMRuntime.__dataclass_fields__.values():
        if f.name in vals:
            v, t = vals[f.name], str(f.type)
            kw[f.name] = (v == "true") if "bool" in t else int(v) if "int" in t else float(v)
    return ZMRuntime(**kw)


_ZM_SETUP_CACHE: dict = {}


def run_zm(pre: dict, meta: dict, settings: dict, with_checks: bool) -> dict:
    from eamxx_torch.zm import invariant_checks, postconditions, zm_run, zm_setup
    rt = settings["zm"]
    vf = settings["vcoord_file"]
    key = (rt, vf)
    if key not in _ZM_SETUP_CACHE:
        hyb = load_hybrid_cached(vf)
        _ZM_SETUP_CACHE[key] = zm_setup(rt, hyb["hyai"], hyb["hybi"])
    state = {k: torch.as_tensor(v, dtype=DTYPE) for k, v in pre.items()}
    if with_checks:
        state, _ = invariant_checks(state)
    # run_impl's first-step test: the start-of-step count, one less than the dump's step
    out = zm_run(state, float(meta["dt"]), _ZM_SETUP_CACHE[key], int(meta["step"]) - 1 == 0)
    nonconv = out.pop("_internal")["nonconverged"]
    assert not bool(nonconv.any()), "invert_entropy did not converge"
    if with_checks:
        out, _ = invariant_checks(out)
        out, _ = postconditions(out)
    return out


PORTS = {"cld_fraction": run_cld_fraction, "spa": run_spa, "tms": run_tms, "shoc": run_shoc,
         "p3": run_p3, "rrtmgp": run_rrtmgp, "iop_forcing": run_iop_forcing,
         "SurfaceCouplingImporter": run_sc_import, "zm": run_zm}
NEEDS_CONTEXT = {"spa", "shoc", "p3", "rrtmgp", "iop_forcing", "SurfaceCouplingImporter", "zm"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("proc", choices=sorted(PORTS))
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("--rtol", type=float, default=0.0)
    ap.add_argument("--atol", type=float, default=0.0)
    ap.add_argument("--math", choices=["torch", "libm"], default="libm",
                    help="transcendental-function backend (libm = same library as the reference)")
    ap.add_argument("--when", choices=["run", "chk"], default="run",
                    help="run: compare around run_impl (pre->post); chk: around the "
                         "property checks too (prechk->postchk)")
    ap.add_argument("--dtype", choices=["float64", "float32"], default="float64",
                    help="port precision (float32 for dumps of the single-precision build)")
    a = ap.parse_args()
    global DTYPE
    DTYPE = getattr(torch, a.dtype)
    # The calls are small; with the machine busy, torch's thread pool waits on
    # scheduling (a 5 s replay took 79 min of wall time with 14 threads).
    torch.set_num_threads(1)

    # Same guard as tests/conftest.py: `scalar / tensor` is reciprocal-then-multiply
    # in PyTorch, so port code must use eamxx_torch.ekat.rdiv.
    def _raise(self, other):
        raise AssertionError("scalar / tensor is inexact in PyTorch; use eamxx_torch.ekat.rdiv")
    torch.Tensor.__rtruediv__ = _raise

    from eamxx_torch import mathfn
    mathfn_ctx = mathfn.use_backend(a.math)
    mathfn_ctx.__enter__()
    ncalls = nvals = 0
    worst: dict[str, tuple[float, float, str]] = {}
    failures = []
    for run in a.run_dirs:
        variant, _, case = run.name.removeprefix("dumps_").partition("_")
        w_in, w_out = ("pre", "post") if a.when == "run" else ("prechk", "postchk")
        settings = case_settings(run) if a.proc in NEEDS_CONTEXT else None
        for pj in sorted((run / "dump" / a.proc).glob(f"*_{w_in}.json")):
            stem = pj.with_suffix("")
            mpre, pre = read_dump(stem)
            mpost, post = read_dump(str(stem)[: -len(w_in)] + w_out)
            meta = {"time_end": mpost["time"], "dt": mpost["dt"], "step": mpre["step"]}
            # start-of-step time: the "pre" dump's stamp (prechk carries end-of-step)
            run_pre = stem.parent / re.sub(r"_c-1_(r\d+)_prechk$", r"_c0_\1_pre", stem.name)
            if run_pre.with_suffix(".json").exists():
                meta["time_start"] = json.loads(run_pre.with_suffix(".json").read_text())["time"]
            if a.proc in NEEDS_CONTEXT:
                out = PORTS[a.proc](pre, meta, settings, with_checks=(a.when == "chk"))
            else:
                out = PORTS[a.proc](pre)
            ncalls += 1
            for name, val in out.items():
                d = compare(name, val, post[name], rtol=a.rtol, atol=a.atol)
                nvals += post[name].size
                w = worst.get(name, (0.0, 0.0, ""))
                if d.max_abs >= w[0]:
                    worst[name] = (d.max_abs, d.max_rel, f"{case}/{stem.name}")
                if not d.ok():
                    failures.append((case, stem.name, d))

    print(f"{a.proc}: {ncalls} calls, {nvals} output values compared "
          f"(rtol={a.rtol}, atol={a.atol}, math={a.math})")
    print(f"{'field':28s} {'max |port-ref|':>15s} {'max rel':>10s}  worst call")
    for name, (ma, mr, where) in sorted(worst.items()):
        print(f"{name:28s} {ma:15.3e} {mr:10.3e}  {where}")
    if failures:
        print(f"MISMATCH in {len(failures)} field-calls; first: {failures[0]}")
        sys.exit(1)
    print("MATCH: every value within tolerance" + (" (bitwise)" if a.rtol == a.atol == 0 else ""))


if __name__ == "__main__":
    main()
