#!/usr/bin/env python3
"""Run one whole physics step through the port's suite and compare every process's output.

  python reference/scripts/suite_step_check.py RUN_DIR STEP [RANK]

The starting state is what HOMME handed to physics at that step: for each field,
its value in the "prechk" dump of the first physics process that has it
(iop_forcing, zm if on, tms, shoc, cld_fraction, spa, p3, rrtmgp). The port then
runs the whole group, passing fields from process to process itself, and each
process's state after its checks is compared with that process's "postchk" dump.
A wrong hand-off between processes shows up as a mismatch.

With the mac_aero_mic group sub-cycled (EAMxx-LR: 12 sub-cycles), its members
count their own sub-steps: model step S is member sub-steps 12*(S-1)+1 .. 12*S,
and every one of them is compared.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import replay_check as rc  # noqa: E402
from read_dump import read_dump  # noqa: E402

from eamxx_torch import mathfn  # noqa: E402
from eamxx_torch.scm import iop_forcing_postconditions, iop_forcing_run  # noqa: E402
from eamxx_torch.suite import Grid, Suite, SuiteConfig  # noqa: E402
from eamxx_torch.shoc import SHOCRuntime  # noqa: E402
from eamxx_torch.testing import compare  # noqa: E402
from eamxx_torch.timestamp import TimeStamp  # noqa: E402

TEND = re.compile(r"^(iop_forcing|shoc|p3|rrtmgp|homme)_\w+_tend$")


def is_proc_tend(name: str) -> bool:
    """A tendency the base class computes (compute_tendencies), e.g. p3_T_mid_tend."""
    return TEND.match(name) is not None


MEMBERS = ("tms", "shoc", "cld_fraction", "spa", "p3")


def mac_subcycles(run: Path) -> int:
    txt = (run / "input.yaml").read_text()
    m = re.search(r"^ +mac_aero_mic:\n(?:.*\n)*? +number_of_subcycles: (\d+)", txt, re.M)
    return int(m.group(1)) if m else 1


def suite_for_run(run: Path, ncol: int, step: int):
    st = rc.case_settings(run)
    io = st["iop"]
    hyb = rc.load_hybrid_cached(st["vcoord_file"])
    ps, rs = st["p3"], st["rrtmgp"]
    cfg = SuiteConfig(shoc=SHOCRuntime(extra_diags=st["extra_shoc_diags"]), p3=ps["runtime"],
                      p3_predict_nc=ps["predictNc"], p3_prescribed_ccn=ps["prescribedCCN"],
                      rrtmgp=rs["runtime"], p3_tables_dir=ps["tables_dir"], rrtmgp_files=rs["files"],
                      spa_data_file=st["spa_data_file"], spa_target=(st["lat"], st["lon"]),
                      zm=st["zm"], mac_aero_mic_subcycles=mac_subcycles(run))
    f64 = torch.float64
    grid = Grid(lat=torch.full((ncol,), st["lat"], dtype=f64), lon=torch.full((ncol,), st["lon"], dtype=f64),
                cell_length=torch.full((ncol,), st["cell_length"], dtype=f64), **hyb)
    t0 = TimeStamp.from_string(io["run_t0"])
    iop, _ = rc._iop_manager(st, step)
    p = io["params"]
    nudging = p.iop_nudge_tq or p.iop_nudge_uv
    partial = nudging and ncol != io["num_global_cols"]

    def run_iop(s, dt, clock):
        iop.read_iop_file_data(clock.start)
        tracers = tuple(k for k in rc.IOP_TRACERS if k in s)
        out = iop_forcing_run(s, dt, iop, hyb["hyam"], hyb["hybm"], tracers,
                              num_global_cols=ncol if nudging else None)
        return iop_forcing_postconditions(out)

    # Nudging relaxes toward the mean over every column; with fewer columns dumped, the
    # check starts after iop_forcing (validated on all-column runs, docs/status.md)
    first = () if partial else (("iop_forcing", run_iop),)
    suite = Suite(cfg, grid, t0, ncol, first_procs=first)
    ts = t0.copy()
    for _ in range(step - 1):
        ts = ts + io["dt"]
    suite.set_time(ts)
    return suite, io["dt"], partial


def main() -> None:
    run, step = Path(sys.argv[1]), int(sys.argv[2])
    rank = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    torch.set_num_threads(1)
    nsub = mac_subcycles(run)
    has_zm = (run / "dump" / "zm").exists()
    first = ("iop_forcing",) + (("zm",) if has_zm else ())
    sub_steps = [nsub * (step - 1) + 1 + i for i in range(nsub)]

    def load(proc, s, when):
        return read_dump(str(run / "dump" / proc / f"s{s}_c-1_r{rank}_") + when)[1]

    ncol = load("iop_forcing", step, "prechk")["T_mid"].shape[0]
    suite, dt, partial = suite_for_run(run, ncol, step)
    s0 = {}
    if partial:   # start from iop_forcing's output
        first = first[1:]
        s0 = {k: torch.as_tensor(v, dtype=torch.float64)
              for k, v in load("iop_forcing", step, "postchk").items()}
        print("  (domain-mean nudging with a partial dump: starting after iop_forcing)")
    starts = [(p, step) for p in first] + [(p, sub_steps[0]) for p in MEMBERS] + [("rrtmgp", step)]
    for proc, s in starts:
        for k, v in load(proc, s, "prechk").items():
            s0.setdefault(k, torch.as_tensor(v, dtype=torch.float64))
    with mathfn.use_backend("libm"):
        r = suite.step(s0, dt)
    checks = [(p, step, r["per_process"][p]) for p in first]
    for i, s in enumerate(sub_steps):
        checks += [(p, s, r["per_subcycle"][p][i]) for p in MEMBERS]
    checks.append(("rrtmgp", step, r["per_process"]["rrtmgp"]))
    nbad = nvals = 0
    for proc, s, got in checks:
        for k, ref in load(proc, s, "postchk").items():
            if is_proc_tend(k) or k not in got:   # tendencies are computed by the base class
                continue
            d = compare(k, got[k], ref)
            nvals += ref.size
            if not d.ok():
                nbad += 1
                print(f"  {proc:12s} s{s:<6d} {k:28s} max|diff| {d.max_abs:.3e}")
    print(f"{run.name} step {step} rank {rank}: {nvals} values after {len(checks)} process calls, "
          f"{'bitwise' if nbad == 0 else f'{nbad} fields differ'}")
    sys.exit(1 if nbad else 0)


if __name__ == "__main__":
    main()
