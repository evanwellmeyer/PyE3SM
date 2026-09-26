#!/usr/bin/env python3
"""Compare the port's ocean surface flux (eamxx_torch.surface_flux) with E3SM's own
shr_flux_atmOcn, run through the Fortran harness (build_flux_harness.sh).

  python reference/scripts/flux_check.py [--save tests/data/surface_flux/cases.npz]

Inputs: random atmosphere/ocean states over the range an ocean column sees, plus
edge cases (zero temperature difference, calm and zero wind, |z/L| clipped at 10 on
both sides, ocean currents, masked points). The strict harness (-ffp-contract=off)
is the bitwise target; the release harness (-O, FMA allowed) is reported.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REF = Path(__file__).resolve().parents[1]
INPUTS = ("zbot", "ubot", "vbot", "thbot", "qbot", "rbot", "tbot", "us", "vs", "ts", "mask")
OUTPUTS = ("sen", "lat", "lwup", "evap", "taux", "tauy", "tref", "qref", "duu10n", "u10res",
           "ustar", "re", "ssq")
MINWIND = 0.5


def make_inputs(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    n = 3000
    ts = rng.uniform(271.0, 305.0, n)
    d = dict(
        zbot=rng.uniform(5.0, 80.0, n),
        ubot=rng.uniform(-25.0, 25.0, n),
        vbot=rng.uniform(-25.0, 25.0, n),
        thbot=ts + rng.uniform(-12.0, 8.0, n),
        qbot=rng.uniform(1e-4, 2.5e-2, n),
        rbot=rng.uniform(1.0, 1.35, n),
        us=np.where(rng.random(n) < 0.3, rng.uniform(-1.0, 1.0, n), 0.0),
        vs=np.where(rng.random(n) < 0.3, rng.uniform(-1.0, 1.0, n), 0.0),
        ts=ts, mask=np.ones(n))
    d["tbot"] = d["thbot"] - 0.0098 * d["zbot"]
    # edge cases
    e = 40
    sl = slice(0, e)
    d["thbot"][0:5] = d["ts"][0:5]                          # delt = 0 exactly
    d["ubot"][5:10] = 0.0; d["vbot"][5:10] = 0.0            # no wind at all
    d["us"][5:10] = 0.0; d["vs"][5:10] = 0.0
    d["ubot"][10:15] = 0.3; d["vbot"][10:15] = 0.1          # calm: below the minimum wind
    d["thbot"][15:20] = d["ts"][15:20] + 25.0               # very stable, weak wind -> hol = 10
    d["ubot"][15:20] = 0.6; d["vbot"][15:20] = 0.0
    d["thbot"][20:25] = d["ts"][20:25] - 20.0               # very unstable, weak wind -> hol = -10
    d["ubot"][20:25] = 0.6; d["vbot"][20:25] = 0.0
    d["zbot"][25:30] = 0.5                                  # very low first level
    d["mask"][30:35] = 0.0                                  # not ocean
    d["qbot"][35:40] = 0.0                                  # dry air
    d["tbot"][sl] = d["thbot"][sl] - 0.0098 * d["zbot"][sl]
    return d


def run_harness(inp: dict, variant: str) -> dict:
    exe = REF / "build" / "flux_harness" / variant / "flux_harness"
    n = len(inp["ts"])
    with tempfile.TemporaryDirectory() as t:
        fi, fo = Path(t) / "in.bin", Path(t) / "out.bin"
        np.concatenate([[float(n), MINWIND]] + [inp[k].astype(np.float64) for k in INPUTS]).tofile(fi)
        subprocess.run([str(exe), str(fi), str(fo)], check=True)
        raw = np.fromfile(fo, dtype=np.float64)
    return {k: raw[i * n:(i + 1) * n] for i, k in enumerate(OUTPUTS)}


def run_port(inp: dict) -> dict:
    from eamxx_torch.surface_flux import shr_flux_atmocn
    t = {k: torch.as_tensor(v, dtype=torch.float64) for k, v in inp.items()}
    out = shr_flux_atmocn(t["zbot"], t["ubot"], t["vbot"], t["thbot"], t["qbot"], t["rbot"],
                          t["tbot"], t["us"], t["vs"], t["ts"], mask=t["mask"] != 0, minwind=MINWIND)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", type=Path)
    a = ap.parse_args()
    torch.set_num_threads(1)
    from eamxx_torch import mathfn
    inp = make_inputs()
    ref = {v: run_harness(inp, v) for v in ("strict", "release")}
    ok = True
    for backend in ("libm", "torch"):
        with mathfn.use_backend(backend):
            out = run_port(inp)
        print(f"port ({backend} math) vs strict harness:")
        for k in OUTPUTS:
            p, r = out[k].numpy(), ref["strict"][k]
            nbad = int((p != r).sum())
            scale = np.abs(r[inp["mask"] != 0]).max()
            nw = np.abs(p - r).max() / scale
            print(f"  {k:8s} differing {nbad:5d}/{len(r)}  normwise {nw:.2e}")
            if backend == "libm" and nbad:
                ok = False
    print("release vs strict harness (FMA contraction):")
    for k in OUTPUTS:
        r, s = ref["release"][k], ref["strict"][k]
        m = inp["mask"] != 0
        print(f"  {k:8s} normwise {np.abs(r - s)[m].max() / np.abs(s[m]).max():.2e}")
    it = out["n_iter"].numpy()
    print("iterations per point:", {int(v): int((it == v).sum()) for v in np.unique(it)})
    if a.save:
        a.save.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(a.save, **{"in/" + k: v for k, v in inp.items()},
                            **{"out/" + k: v for k, v in ref["strict"].items()},
                            minwind=np.array(MINWIND))
        print("saved", a.save)
    print("MATCH (bitwise, libm math)" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
