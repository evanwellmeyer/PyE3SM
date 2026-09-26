#!/usr/bin/env python3
"""Run EAMxx's TMS computation (run_impl sequence) on synthetic land columns.

Model runs of the standalone reference have landfrac = 0 (a coupler import the
DP driver never sets), so TMS output is always zero there. This builds columns
from real dumped profiles and varies landfrac, sgh30, near-surface shear and
stability, then runs reference/eamxx_ext/unit_replay/tms_replay (strict build).
Writes tests/data/tms/edge_cases.npz.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from read_dump import read_dump  # noqa: E402

EXE = (HERE.parent / "build/eamxx-strict/tests/multi-process/dynamics_physics"
       "/ref_extra/unit_replay/tms_replay")


def main() -> None:
    # Base profiles: a few dumped TMS inputs (any strict dump dir works).
    stems = sorted(glob.glob(str(HERE.parent / "runs/dumps_strict*_arm97/dump/tms/*_pre.json")))[:2]
    stems += sorted(glob.glob(str(HERE.parent / "runs/dumps_strict*_bomex/dump/tms/*_pre.json")))[:1]
    if not stems:
        raise SystemExit("no TMS dumps found")
    bases = []
    for s in stems:
        _, d = read_dump(s[:-5])
        for i in range(d["T_mid"].shape[0]):
            bases.append({k: d[k][i].copy() for k in ("horiz_winds", "T_mid", "p_mid", "pseudo_density", "qv")})

    cols = []
    for bi, b in enumerate(bases[:6]):
        for sgh in (0.0, np.nextafter(1.0, 0), 1.0, 5.0, 10.2582525, 2000.0):
            for lf in (1.0, 0.37, 0.0):
                cols.append(dict(b, sgh30=sgh, landfrac=lf))
        # dv2 below dv2min: identical winds in the lowest two levels
        c = {k: v.copy() for k, v in b.items()}
        c["horiz_winds"][:, -2] = c["horiz_winds"][:, -1]
        cols.append(dict(c, sgh30=50.0, landfrac=1.0))
        # strongly stable (ri > 1): warm the second-lowest level
        c = {k: v.copy() for k, v in b.items()}
        c["T_mid"][-2] += 15.0
        cols.append(dict(c, sgh30=50.0, landfrac=1.0))
        # unstable (ri < 0): cool the second-lowest level
        c = {k: v.copy() for k, v in b.items()}
        c["T_mid"][-2] -= 5.0
        cols.append(dict(c, sgh30=50.0, landfrac=1.0))
        # Richardson number swept across the 0..1 ramp: 1 m/s shear, theta difference dT
        for dT in (-1.0, 0.0, 0.05, 0.2, 0.4, 0.6, 0.9, 1.5):
            c = {k: v.copy() for k, v in b.items()}
            ex = (c["p_mid"] / 100000.0) ** (287.042 / 1004.64)
            c["horiz_winds"][0, -2] = c["horiz_winds"][0, -1] + 1.0
            c["horiz_winds"][1, -2] = c["horiz_winds"][1, -1]
            c["T_mid"][-2] = (c["T_mid"][-1] * ex[-1] + dT) / ex[-2]
            cols.append(dict(c, sgh30=30.0, landfrac=1.0))
        # calm: zero wind at the lowest level
        c = {k: v.copy() for k, v in b.items()}
        c["horiz_winds"][:, -1] = 0.0
        cols.append(dict(c, sgh30=50.0, landfrac=1.0))

    ncol, nlev = len(cols), cols[0]["T_mid"].shape[0]
    buf = [float(ncol), float(nlev)]
    for c in cols:
        buf += c["horiz_winds"][0].tolist() + c["horiz_winds"][1].tolist()
        for k in ("T_mid", "p_mid", "pseudo_density", "qv"):
            buf += c[k].tolist()
        buf += [c["sgh30"], c["landfrac"]]
    with tempfile.TemporaryDirectory() as td:
        fin, fout = Path(td) / "in.bin", Path(td) / "out.bin"
        np.array(buf, dtype="<f8").tofile(fin)
        r = subprocess.run([str(EXE)], env=dict(os.environ, REPLAY_IN=str(fin), REPLAY_OUT=str(fout)),
                           capture_output=True, text=True, cwd=td)
        if "All tests passed" not in r.stdout:
            raise SystemExit(r.stdout + r.stderr)
        out = np.fromfile(fout, dtype="<f8").reshape(ncol, 3 + 2 * nlev)

    arrays = {
        "pre/horiz_winds": np.stack([c["horiz_winds"] for c in cols]),
        "pre/sgh30": np.array([c["sgh30"] for c in cols]),
        "pre/landfrac": np.array([c["landfrac"] for c in cols]),
        "post/surf_drag_coeff_tms": out[:, 0],
        "post/wind_stress_tms": out[:, 1:3],
        "post/exner": out[:, 3:3 + nlev],
        "post/z_mid": out[:, 3 + nlev:],
    }
    for k in ("T_mid", "p_mid", "pseudo_density", "qv"):
        arrays[f"pre/{k}"] = np.stack([c[k] for c in cols])
    meta = {"proc": "tms", "case": "edge_cases", "source":
            "reference/eamxx_ext/unit_replay/tms_replay.cpp, strict build (E3SM 80fe0a71); "
            "base profiles from " + ", ".join(Path(s).parent.parent.parent.name for s in stems)}
    dest = ROOT / "tests/data/tms/edge_cases.npz"
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, meta=np.array(json.dumps(meta)), **arrays)
    nz = int((out[:, 0] != 0).sum())
    print(f"wrote {dest}: {ncol} columns, {nz} with nonzero drag")


if __name__ == "__main__":
    main()
