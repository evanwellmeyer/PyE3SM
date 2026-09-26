#!/usr/bin/env python3
"""Run EAMxx's FieldWithinIntervalCheck on synthetic fields (check, then repair if repairable).

Writes tests/data/property_checks/interval_edge_cases.npz. Each case i has
pre/f{i}, post/f{i}, and meta["cases"][i] = {lb, ub, can_repair, lb_rep, ub_rep, result}.
Requires the `interval_check_replay` target of the strict reference build.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXE = (HERE.parent / "build/eamxx-strict/tests/multi-process/dynamics_physics"
       "/ref_extra/unit_replay/interval_check_replay")
EPS = sys.float_info.epsilon
SMAX = sys.float_info.max


def main() -> None:
    nan, inf = np.nan, np.inf
    rng = np.random.default_rng(7)
    base = rng.random((3, 17))
    cases = []

    def add(vals, lb, ub, can_repair, lbr=-SMAX, ubr=SMAX):
        cases.append((np.asarray(vals, dtype=np.float64), lb, ub, can_repair, lbr, ubr))

    nccn = (0.0, 1e11, True, -1e11 * EPS, SMAX)       # SPA nccn check
    aero = (0.0, 1.0, True, -SMAX, SMAX)               # SPA aero_* checks
    strict01 = (0.0, 1.0, False, -SMAX, SMAX)          # cld_fraction-style, not repairable

    for lb, ub, cr, lbr, ubr in (nccn, aero, strict01):
        scale = 1e9 if ub > 1 else 1.0
        f = base * scale
        add(f, lb, ub, cr, lbr, ubr)                               # pass
        g = f.copy(); g[0, 3] = -0.0; g[1, 4] = lb; g[2, 5] = ub
        add(g, lb, ub, cr, lbr, ubr)                               # pass, with -0.0 and exact bounds
        g = f.copy(); g[1, 2] = -1e-7; g[0, 0] = -0.0
        add(g, lb, ub, cr, lbr, ubr)                               # slightly negative
        g = f.copy(); g[2, 7] = -1.0
        add(g, lb, ub, cr, lbr, ubr)                               # strongly negative
        g = f.copy(); g[0, 9] = ub * 1.5; g[1, 1] = -0.0
        add(g, lb, ub, cr, lbr, ubr)                               # above ub
        g = f.copy(); g[1, 5] = nan; g[2, 2] = ub * 2
        add(g, lb, ub, cr, lbr, ubr)                               # NaN present + above ub
        g = f.copy(); g[1, 5] = nan
        add(g, lb, ub, cr, lbr, ubr)                               # NaN only
        g = f.copy(); g[0, 1] = inf
        add(g, lb, ub, cr, lbr, ubr)                               # +inf
        g = f.copy(); g[0, 1] = -inf
        add(g, lb, ub, cr, lbr, ubr)                               # -inf

    with tempfile.TemporaryDirectory() as td:
        fin, fout = Path(td) / "in.bin", Path(td) / "out.bin"
        buf = [float(len(cases))]
        for v, lb, ub, cr, lbr, ubr in cases:
            buf += [v.shape[0], v.shape[1], lb, ub, float(cr), lbr, ubr] + v.ravel().tolist()
        np.array(buf, dtype="<f8").tofile(fin)
        r = subprocess.run([str(EXE)], env=dict(os.environ, REPLAY_IN=str(fin), REPLAY_OUT=str(fout)),
                           capture_output=True, text=True, cwd=td)
        if "All tests passed" not in r.stdout:
            raise SystemExit(r.stdout + r.stderr)
        out = np.fromfile(fout, dtype="<f8")

    arrays, meta_cases, off = {}, [], 0
    for i, (v, lb, ub, cr, lbr, ubr) in enumerate(cases):
        code = int(out[off]); off += 1
        post = out[off: off + v.size].reshape(v.shape); off += v.size
        arrays[f"pre/f{i}"], arrays[f"post/f{i}"] = v, post
        meta_cases.append({"lb": lb, "ub": ub, "can_repair": cr, "lb_rep": lbr, "ub_rep": ubr,
                           "result": code})
    meta = {"proc": "FieldWithinIntervalCheck", "case": "edge_cases", "cases": meta_cases,
            "source": "reference/eamxx_ext/unit_replay/interval_check_replay.cpp (E3SM 80fe0a71)"}
    dest = ROOT / "tests/data/property_checks/interval_edge_cases.npz"
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, meta=np.array(json.dumps(meta)), **arrays)
    print(f"wrote {dest}: {len(cases)} cases, results {[c['result'] for c in meta_cases]}")


if __name__ == "__main__":
    main()
