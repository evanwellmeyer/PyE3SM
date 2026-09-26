#!/usr/bin/env python3
"""Run the port's single-column model alongside a scm_eamxx reference run and compare
every output field after every step.

  python reference/scripts/scm_compare.py RUN_DIR [--steps N] [--math libm|torch]

RUN_DIR is made by make_scm_case.py and run with scm_eamxx (reference/eamxx_ext/
scm_driver). The port starts from the same initial state (make_scm_case.py writes the
port's initial state into the reference's IC file). Reports, per field, the largest
difference over the run and the first step where the port and EAMxx differ.
"""
from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def load_reference(run: Path) -> dict:
    import netCDF4
    f = sorted(glob.glob(str(run / "scm_*.INSTANT.*.nc")))[0]
    with netCDF4.Dataset(f) as d:
        out = {}
        for k, v in d.variables.items():
            if k == "time" or "ncol" not in v.dimensions:
                continue
            a = np.asarray(v[:], dtype=np.float64)
            dims = v.dimensions
            if k == "horiz_winds" or k == "surf_mom_flux":   # (time, ncol, dim2[, lev])
                pass
            out[k] = a
    return out


def port_value(k: str, state: dict):
    v = state.get(k)
    if v is None:
        return None
    return v.detach().cpu().numpy().astype(np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--math", choices=["libm", "torch"], default="libm")
    a = ap.parse_args()
    torch.set_num_threads(1)
    import scm_case
    from eamxx_torch import mathfn
    ref = load_reference(a.run)
    nsnap = next(iter(ref.values())).shape[0]
    nsteps = min(a.steps or nsnap - 1, nsnap - 1)
    ctx = mathfn.use_backend(a.math)
    ctx.__enter__()
    scm, s = scm_case.port_scm(a.run)
    states = [s]
    t0 = time.time()
    for _ in range(nsteps):
        r = scm.step(s)
        s = r.state
        bad = {p: st for p, st in r.status.items() if any(int(x) == 2 for x in st.values())}
        if bad:
            print("  port: a property check failed:", bad)
        states.append(s)
    elapsed = time.time() - t0

    rows = []
    first_any = None
    init_diff = []
    for k in sorted(ref):
        worst, first = 0.0, None
        compared = 0
        for n in range(nsteps + 1):
            p = port_value(k, states[n])
            if p is None:
                continue
            r_ = ref[k][n]
            if r_.shape != p.shape:
                p = p.reshape(r_.shape)
            if n == 0:
                # the t=0 snapshot: fields EAMxx recomputes before any use (surface imports
                # at initialization, fields without an IC) may differ; reported apart
                if not np.array_equal(p, r_):
                    init_diff.append(k)
                continue
            compared += 1
            d = np.abs(p - r_)
            m = float(np.nanmax(d)) if d.size else 0.0
            if not np.array_equal(p, r_) and first is None:
                first = n
            worst = max(worst, m)
        if compared:
            scale = float(np.nanmax(np.abs(ref[k][1:nsteps + 1]))) if nsteps else 1.0
            rows.append((k, worst, worst / scale if scale else 0.0, first))
            if first is not None and (first_any is None or first < first_any[1]):
                first_any = (k, first)
    print(f"{a.run.name}: {nsteps} steps, port {elapsed:.1f} s ({a.math} math); "
          f"{len(rows)} fields compared at every step")
    print(f"{'field':24s} {'max |port-ref|':>15s} {'rel to max':>11s}  first differing step")
    for k, w, rel, first in rows:
        print(f"{k:24s} {w:15.3e} {rel:11.2e}  {'-' if first is None else first}")
    if init_diff:
        print("t=0 snapshot only (recomputed before use):", ", ".join(init_diff))
    if first_any is None:
        print("MATCH: bitwise at every step")
    else:
        print(f"DIFFER: first at step {first_any[1]} ({first_any[0]})")
        sys.exit(1)


if __name__ == "__main__":
    main()
