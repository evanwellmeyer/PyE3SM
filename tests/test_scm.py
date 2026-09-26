"""IOP forcing, surface imports, pressures, the suite and the single-column driver.

Reference: DP-EAMxx strict-build dumps, E3SM 80fe0a71 (see docs/status.md).
  * iop_forcing: BOMEX (subsidence), ARM97 day 5 (3D forcing, a time-varying
    surface pressure, so levels outside the file's range keep older values),
    and CASS with every one of its 100 columns (subsidence, Coriolis, and nudging
    of T, q and winds toward the domain mean).
  * SurfaceCouplingImporter: the IOP surface fluxes and temperature.
  * HOMME's pressures handed to physics.
  * One whole physics step through the suite (iop_forcing, tms, shoc,
    cld_fraction, spa, p3, rrtmgp), starting from the state HOMME handed to
    physics: DYCOMS-RF01 step 1, a radiation step in daylight with cloud.
All bitwise with the libm math backend (replay_check.py and suite_step_check.py
cover every dumped call: 130 iop_forcing calls, 50 whole steps).

Multi-step runs of the single-column driver are compared with EAMxx's own physics run
on one column by the single-column reference driver (reference/eamxx_ext/scm_driver,
scm_eamxx): same initial state, same host choices, trajectories bitwise with the libm
math backend (reference/scripts/scm_compare.py covers the full runs; the fixtures
here hold the first steps). Also checked: independent columns, and gradients.

These tests need E3SM input data (IOP files, tables, SPA and IC files); they skip
when it is absent.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from eamxx_torch import mathfn
from eamxx_torch.p3 import P3Runtime
from eamxx_torch.rrtmgp import RRTMGPRuntime
from eamxx_torch.scm import (IOPDataManager, IOPParams, iop_forcing_postconditions,
                             iop_forcing_run, load_hybrid, overwrite_iop_imports)
from eamxx_torch.scm.driver import SCM, ic_from_files, update_pressure
from eamxx_torch.shoc import SHOCRuntime
from eamxx_torch.suite import Grid, Suite, SuiteConfig
from eamxx_torch.testing import compare
from eamxx_torch.timestamp import TimeStamp

DATA = Path(__file__).parent / "data" / "scm"
ROOT = Path(__file__).resolve().parents[1] / "reference" / "inputdata" / "atm"
IOP_DIR = ROOT / "cam" / "scam" / "iop"
VCOORD = ROOT / "scream" / "init" / "vertical_coordinates_L128_20220927.nc"
HAVE_IOP = (IOP_DIR / "CASS_iopfile_4scam.nc").exists() and VCOORD.exists()
needs_iop = pytest.mark.skipif(not HAVE_IOP, reason="IOP files not present")
TRACERS = ("qv", "bm", "nc", "ni", "nr", "qc", "qi", "qm", "qr", "tke")


def load(name):
    with np.load(DATA / name, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        arrays = {k: torch.as_tensor(z[k]) for k in z.files if k != "meta"}
    sub = lambda p: {k[len(p) + 1:]: v for k, v in arrays.items() if k.startswith(p + "/")}  # noqa: E731
    return meta, sub


def manager(meta):
    p = dict(meta["iop"])
    p["iop_file"] = str(IOP_DIR / p["iop_file"])
    hyb = load_hybrid(VCOORD)
    t0 = TimeStamp.from_string(meta["run_t0"])
    m = IOPDataManager(IOPParams(**p), t0, hyb["hyam"].shape[0], hyb["hyam"], hyb["hybm"])
    m.advance(t0, meta["dt"], meta["step"])
    return m, hyb


@needs_iop
@pytest.mark.reference
@pytest.mark.parametrize("name", ["iop_bomex_s108_r0.npz", "iop_arm97_7p5d_s4752_r1.npz",
                                  "iop_cass_allcols_s217_r0.npz"])
def test_iop_forcing_bitwise(name):
    meta, sub = load(name)
    pre, post = sub("pre"), sub("post")
    m, hyb = manager(meta)
    p = m.params
    ncol = pre["T_mid"].shape[0]
    nudging = p.iop_nudge_tq or p.iop_nudge_uv
    all_cols = ncol == meta["num_global_cols"]
    with mathfn.use_backend("libm"):
        out = iop_forcing_run(pre, float(meta["dt"]), m, hyb["hyam"], hyb["hybm"],
                              tuple(k for k in TRACERS if k in pre),
                              num_global_cols=ncol if nudging else None)
    skip = set()
    if nudging and not all_cols:   # the domain means need every column
        skip = ({"horiz_winds"} if p.iop_nudge_uv else set()) | ({"T_mid", "qv"} if p.iop_nudge_tq else set())
    checked = 0
    for k, ref in post.items():
        if k in skip:
            continue
        d = compare(k, out[k], ref.numpy())
        assert d.ok(), d
        checked += 1
    assert checked >= 3
    if name.startswith("iop_cass"):
        assert p.iop_nudge_tq and p.iop_nudge_uv and p.iop_coriolis and p.iop_dosubsidence and all_cols


@needs_iop
@pytest.mark.reference
@pytest.mark.parametrize("name", ["import_cass_s54.npz", "import_arm97_7p5d_s4752.npz"])
def test_surface_import_bitwise(name):
    meta, sub = load(name)
    post = sub("post")
    m, _ = manager(meta)
    state = {k: torch.zeros_like(v) for k, v in post.items()}
    with mathfn.use_backend("libm"):
        out = overwrite_iop_imports(state, m)
    for k, ref in post.items():
        d = compare(k, out[k], ref.numpy())
        assert d.ok(), d


@needs_iop
def test_pressure_update_matches_homme():
    _, sub = load("pressure_arm97_7p5d_s4752.npz")
    post = sub("post")
    ptop = 100000.0 * float(load_hybrid(VCOORD)["hyai"][0])
    out = update_pressure(post["pseudo_density"], post["qv"], ptop)
    for k, v in out.items():
        d = compare(k, v, post[k].numpy())
        assert d.ok(), d


@needs_iop
def test_iop_time_index_matches_the_literal_loop():
    meta, _ = load("iop_arm97_7p5d_s4752_r1.npz")
    m, _ = manager(dict(meta, step=0))
    t0 = TimeStamp.from_string(meta["run_t0"])
    for k in range(0, 6480, 97):
        ts = t0 + k * 100.0
        idx = -1
        times = m.iop_file_times_in_sec
        for t in range(len(times) - 1):
            if m.iop_file_begin_time + times[t] <= ts < m.iop_file_begin_time + times[t + 1]:
                idx = t
        assert m.get_iop_file_time_idx(ts) == idx


def _suite_inputs():
    meta, sub = load("suite_dycomsrf01_s1_r0.npz")
    files = {"p3": ROOT / "scream" / "tables", "spa": ROOT / "scream" / "init" / meta["spa_data_file"]}
    return meta, sub, files


def _have_suite_data():
    if not HAVE_IOP:
        return False
    meta, _, files = _suite_inputs()
    return (files["p3"] / "p3_lookup_table_1.dat-v4.1.1").exists() and files["spa"].exists() \
        and all((ROOT / "scream" / "init" / f).exists() for f in meta["rrtmgp"]["files"])


needs_suite = pytest.mark.skipif(not _have_suite_data(), reason="suite input data not present")


def _config(meta, files):
    rt = dict(meta["rrtmgp"]["runtime"])
    rt["active_gases"] = tuple(rt["active_gases"])
    return SuiteConfig(shoc=SHOCRuntime(extra_diags=meta["extra_shoc_diags"]),
                       p3=P3Runtime(**meta["p3"]["runtime"]), p3_predict_nc=meta["p3"]["predictNc"],
                       p3_prescribed_ccn=meta["p3"]["prescribedCCN"], rrtmgp=RRTMGPRuntime(**rt),
                       p3_tables_dir=str(files["p3"]),
                       rrtmgp_files=tuple(str(ROOT / "scream" / "init" / f) for f in meta["rrtmgp"]["files"]),
                       spa_data_file=str(files["spa"]), spa_target=(meta["lat"], meta["lon"]))


@needs_suite
@pytest.mark.reference
def test_one_physics_step_through_the_suite_bitwise():
    meta, sub, files = _suite_inputs()
    start, end = sub("start"), sub("end")
    ncol = start["T_mid"].shape[0]
    m, hyb = manager(meta)
    f64 = torch.float64
    grid = Grid(lat=torch.full((ncol,), meta["lat"], dtype=f64),
                lon=torch.full((ncol,), meta["lon"], dtype=f64),
                cell_length=torch.full((ncol,), meta["cell_length"], dtype=f64), **hyb)

    def run_iop(s, dt, clock):
        m.read_iop_file_data(clock.start)
        out = iop_forcing_run(s, dt, m, hyb["hyam"], hyb["hybm"], tuple(k for k in TRACERS if k in s))
        return iop_forcing_postconditions(out)

    t0 = TimeStamp.from_string(meta["run_t0"])
    suite = Suite(_config(meta, files), grid, t0, ncol, first_procs=(("iop_forcing", run_iop),))
    with mathfn.use_backend("libm"):
        r = suite.step({k: v.to(f64) for k, v in start.items()}, float(meta["dt"]))
    got = r["state"]
    assert float(end["SW_flux_dn"][:, 0].min()) > 0 and float(start["qc"].max()) > 0  # day, cloud
    for k, ref in end.items():
        d = compare(k, got[k], ref.numpy())
        assert d.ok(), d


def _scm(ncol=1):
    meta, _, files = _suite_inputs()
    p = dict(meta["iop"], iop_file=str(IOP_DIR / "BOMEX_iopfile_4scam.nc"), target_latitude=15.0,
             target_longitude=300.0, iop_nudge_tq=False, iop_nudge_uv=False, iop_coriolis=False,
             iop_dosubsidence=True, iop_srf_prop=True)
    t0 = TimeStamp.from_string("1969-06-25-00000")
    cfg = _config(meta, files)
    cfg.spa_target = None
    scm = SCM(IOPParams(**p), cfg, t0, load_hybrid(VCOORD), ncol=ncol, dt=100.0)
    ic = ic_from_files(str(ROOT / "scream" / "init" / "screami_ne30np4L128_20260813.nc"),
                       str(ROOT / "cam" / "topo" / "USGS-gtopo30_ne30np4pg2_x6t-SGH.c20210614.nc"),
                       15.0, 300.0)
    ic = {k: np.repeat(v, ncol, axis=0) for k, v in ic.items()}
    return scm, scm.initial_state(ic)


needs_scm = pytest.mark.skipif(
    not (_have_suite_data() and (ROOT / "scream" / "init" / "screami_ne30np4L128_20260813.nc").exists()),
    reason="IC files not present")


@needs_scm
def test_scm_runs_bomex_with_independent_columns():
    scm, s = _scm(ncol=2)
    for _ in range(6):
        r = scm.step(s)
        s = r.state
        for proc, st in r.status.items():
            assert all(int(v) != 2 for v in st.values()), (proc, st)   # no check fails
    for k, v in s.items():
        if isinstance(v, torch.Tensor) and v.dtype.is_floating_point and v.dim() >= 1:
            assert torch.isfinite(v).all(), k
            assert torch.equal(v[0], v[1]), k     # identical inputs, independent columns
    assert float(s["tke"].max()) > 0.0004 and float(s["pbl_height"][0]) > 0


@needs_scm
def test_scm_gradients_are_finite():
    """Through two steps of the whole model (docs/differentiability.md). Masked-out
    branches get safe inputs, so no 0*inf reaches the gradient."""
    scm, s = _scm(ncol=1)
    s = dict(s)
    s["T_mid"] = s["T_mid"].clone().requires_grad_(True)
    t0 = s["T_mid"]
    for _ in range(2):
        s = scm.step(s).state
    loss = s["T_mid"].sum() + s["qv"].sum() * 1e3
    loss.backward()
    assert torch.isfinite(t0.grad).all() and (t0.grad != 0).any()


# ------------------------------------------------------------------ trajectories vs EAMxx
TRAJ = sorted(DATA.glob("scm_*.npz"))


def _scm_from_fixture(meta):
    from eamxx_torch.zm import ZMRuntime
    p = dict(meta["iop"])
    p["iop_file"] = str(IOP_DIR / p["iop_file"])
    rt = dict(meta["rrtmgp"]["runtime"])
    rt["active_gases"] = tuple(rt["active_gases"])
    cfg = SuiteConfig(shoc=SHOCRuntime(extra_diags=meta["extra_shoc_diags"]),
                      p3=P3Runtime(**meta["p3"]["runtime"]), p3_predict_nc=meta["p3"]["predictNc"],
                      p3_prescribed_ccn=meta["p3"]["prescribedCCN"], rrtmgp=RRTMGPRuntime(**rt),
                      p3_tables_dir=str(ROOT / "scream" / "tables"),
                      rrtmgp_files=tuple(str(ROOT / "scream" / "init" / f) for f in meta["rrtmgp"]["files"]),
                      spa_data_file=str(ROOT / "scream" / "init" / meta["spa_data_file"]),
                      spa_target=(p["target_latitude"], p["target_longitude"]),
                      zm=ZMRuntime(**meta["zm"]) if meta["zm"] else None,
                      mac_aero_mic_subcycles=meta["mac_aero_mic_subcycles"])
    return SCM(IOPParams(**p), cfg, TimeStamp.from_string(meta["run_t0"]), load_hybrid(VCOORD),
               ncol=1, cell_length=meta["cell_length"], dt=float(meta["dt"]))


@needs_suite
@pytest.mark.reference
@pytest.mark.parametrize("path", TRAJ, ids=lambda p: p.stem)
def test_scm_trajectory_matches_eamxx(path):
    meta, sub = load(path.name)
    scm = _scm_from_fixture(meta)
    s = {k: v.to(torch.float64) for k, v in sub("init").items()}
    ref = sub("ref")
    with mathfn.use_backend("libm"):
        for n in range(meta["nsteps"]):
            s = scm.step(s).state
            for k, r in ref.items():
                got = s[k].reshape(r[n].shape)
                assert torch.equal(got, r[n]), (n + 1, k)
