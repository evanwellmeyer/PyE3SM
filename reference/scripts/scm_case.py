"""The port's single-column model for a scm_eamxx run directory (make_scm_case.py).

Settings come from the run's input.yaml, parsed as replay_check.py parses DP runs.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import replay_check as rc  # noqa: E402

DATA = HERE.parent / "inputdata" / "atm"
VCOORD = DATA / "scream/init/vertical_coordinates_L128_20220927.nc"
IC_FILE = DATA / "scream/init/screami_ne30np4L128_20260813.nc"
TOPO_FILE = DATA / "cam/topo/USGS-gtopo30_ne30np4pg2_x6t-SGH.c20210614.nc"


def mac_subcycles(txt: str) -> int:
    m = re.search(r"^ +mac_aero_mic:\n(?:.*\n)*? +number_of_subcycles: (\d+)", txt, re.M)
    return int(m.group(1)) if m else 1


SMOOTH_TKE_TOL = 1e-14   # docs/differentiability.md: about 50 ulp of the TKE floor


def port_scm(run: Path, ncol: int = 1, smooth: bool = False, seeds_free: bool | None = None,
             tke_tol: float | None = None, dtype=None, device: str = "cpu", p3_imm_f64: bool = False):
    """(SCM, initial state) for the run directory.

    smooth=True turns on the two gradient options (not EAMxx behavior): McICA seeds that do
    not depend on the state, and SHOC's energy-fixer level test with a tolerance.
    seeds_free / tke_tol set the two options separately (they override `smooth`).
    """
    import dataclasses
    import torch
    from eamxx_torch.scm import load_hybrid
    from eamxx_torch.scm.driver import SCM, ic_from_files
    from eamxx_torch.shoc import SHOCRuntime
    from eamxx_torch.suite import SuiteConfig
    from eamxx_torch.timestamp import TimeStamp
    txt = (run / "input.yaml").read_text()
    st = rc.case_settings(run)
    io, ps, rs = st["iop"], st["p3"], st["rrtmgp"]
    lat, lon = io["params"].target_latitude, io["params"].target_longitude
    shoc_rt = SHOCRuntime(extra_diags=st["extra_shoc_diags"],
                          energy_fixer_tke_tol=(tke_tol if tke_tol is not None
                                                else SMOOTH_TKE_TOL if smooth else 0.0))
    free = seeds_free if seeds_free is not None else smooth
    rrtmgp_rt = dataclasses.replace(rs["runtime"], mcica_seed_from_pressure=not free)
    p3_rt = dataclasses.replace(ps["runtime"], immersion_freezing_in_float64=p3_imm_f64)
    cfg = SuiteConfig(shoc=shoc_rt, p3=p3_rt,
                      p3_predict_nc=ps["predictNc"], p3_prescribed_ccn=ps["prescribedCCN"],
                      rrtmgp=rrtmgp_rt, p3_tables_dir=ps["tables_dir"], rrtmgp_files=rs["files"],
                      spa_data_file=st["spa_data_file"], spa_target=(lat, lon), zm=st["zm"],
                      mac_aero_mic_subcycles=mac_subcycles(txt),
                      dtype=dtype if dtype is not None else torch.float64, device=device)
    dx_short = float(re.search(r"^\s*dx_short:\s*(\S+)", txt, re.M).group(1))
    hyb = load_hybrid(VCOORD)
    t0 = TimeStamp.from_string(io["run_t0"])
    scm = SCM(io["params"], cfg, t0, hyb, ncol=ncol, cell_length=dx_short * 1000, dt=io["dt"])
    ic = ic_from_files(str(IC_FILE), str(TOPO_FILE), lat, lon)
    if ncol > 1:
        import numpy as np
        ic = {k: np.repeat(v, ncol, axis=0) for k, v in ic.items()}
    return scm, scm.initial_state(ic)
