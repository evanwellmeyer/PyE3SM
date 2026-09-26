#!/usr/bin/env python3
"""Run EAMxx's CldFractionFunctions::main on synthetic edge cases.

Writes tests/data/cld_fraction/edge_cases.npz in the fixture format of
eamxx_torch.testing (pre/<field>, post/<field>, meta). Requires the
`cld_fraction_replay` target of the reference build.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXE = (HERE.parent / "build/eamxx-release/tests/multi-process/dynamics_physics"
       "/ref_extra/unit_replay/cld_fraction_replay")
THR, THR4 = 1e-12, 1e-5  # CldFraction::create_requests defaults


def main() -> None:
    inf, nan = np.inf, np.nan
    qi_vals = [0.0, -0.0, -1e-9, np.nextafter(THR, 0), THR, np.nextafter(THR, 1),
               1e-9, np.nextafter(THR4, 0), THR4, np.nextafter(THR4, 1), 1e-3,
               nan, inf, -inf]
    liq_vals = [0.0, -0.0, 1e-300, 0.25, 0.5, np.nextafter(1.0, 0), 1.0, nan, 2.0, -1.0,
                inf, -inf, 0.999, 1e-6, 0.0, 0.5, 1.0]  # 17 levels: not a multiple of 16
    rng = np.random.default_rng(20260918)
    qi = np.array(qi_vals)[:, None] * np.ones(len(liq_vals))[None, :]
    liq = np.ones(len(qi_vals))[:, None] * np.array(liq_vals)[None, :]
    # Add random columns spanning the thresholds.
    nr = 6
    rq = 10.0 ** rng.uniform(-15, -2, size=(nr, len(liq_vals)))
    rq[rng.random(rq.shape) < 0.3] = 0.0
    rl = rng.random((nr, len(liq_vals)))
    qi = np.vstack([qi, rq])
    liq = np.vstack([liq, rl])
    nj, nk = qi.shape

    with tempfile.TemporaryDirectory() as td:
        fin, fout = Path(td) / "in.bin", Path(td) / "out.bin"
        np.concatenate([[nj, nk, THR, THR4], qi.ravel(), liq.ravel()]).astype("<f8").tofile(fin)
        env = dict(os.environ, REPLAY_IN=str(fin), REPLAY_OUT=str(fout))
        r = subprocess.run([str(EXE)], env=env, capture_output=True, text=True, cwd=td)
        if "All tests passed" not in r.stdout:
            raise SystemExit(r.stdout + r.stderr)
        out = np.fromfile(fout, dtype="<f8").reshape(4, nj, nk)

    names = ["cldfrac_ice", "cldfrac_tot", "cldfrac_ice_for_analysis", "cldfrac_tot_for_analysis"]
    meta = {
        "proc": "cld_fraction", "case": "edge_cases",
        "params": {"ice_cloud_threshold": THR, "ice_cloud_for_analysis_threshold": THR4},
        "source": "reference/eamxx_ext/unit_replay/cld_fraction_replay.cpp "
                  "(CldFractionFunctions::main, E3SM 80fe0a71)",
    }
    arrays = {"pre/qi": qi, "pre/cldfrac_liq": liq}
    arrays.update({f"post/{n}": out[i] for i, n in enumerate(names)})
    dest = ROOT / "tests/data/cld_fraction/edge_cases.npz"
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, meta=np.array(json.dumps(meta)), **arrays)
    print(f"wrote {dest} ({nj} columns x {nk} levels)")


if __name__ == "__main__":
    main()
