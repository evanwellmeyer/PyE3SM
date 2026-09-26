#!/usr/bin/env python3
"""Find float64 leaks when the port runs in float32.

  python reference/scripts/dtype_check.py [scheme ...]

Runs each scheme on a committed fixture (tests/data) with every floating input cast to
float32, recording every torch operation that creates float64 from no float64 input (the origins
of float64 in the computation). PyTorch promotes
float32 (op) float64 to float64 silently, so one float64 constant or table makes the rest
of an expression double. Prints the port's source lines where float64 results appear,
with counts. Schemes: cld_fraction, shoc, p3, zm, rrtmgp, scm (two single-column steps).
"""
from __future__ import annotations

import collections
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))


class Record(torch.overrides.TorchFunctionMode):
    def __init__(self):
        super().__init__()
        self.sites = collections.Counter()

    def __torch_function__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        outs = out if isinstance(out, (tuple, list)) else (out,)
        def flat(xs):
            for x in xs:
                if isinstance(x, (list, tuple)):
                    yield from flat(x)
                else:
                    yield x
        ins64 = any(isinstance(x, torch.Tensor) and x.dtype == torch.float64
                    for x in flat(list(args) + list((kwargs or {}).values())))
        if not ins64 and any(isinstance(o, torch.Tensor) and o.dtype == torch.float64 for o in outs):
            for fr in reversed(traceback.extract_stack(limit=10)):
                if "eamxx_torch" in fr.filename and "dtype_check" not in fr.filename:
                    self.sites[f"{Path(fr.filename).parent.name}/{Path(fr.filename).name}:{fr.lineno}: {fr.line.strip()[:90]}"] += 1
                    break
        return out


def f32(pre: dict) -> dict:
    return {k: (torch.as_tensor(v).to(torch.float32) if isinstance(v, np.ndarray) and v.dtype.kind == "f"
                else torch.as_tensor(v) if isinstance(v, np.ndarray) else v) for k, v in pre.items()}


def run(scheme: str):
    from eamxx_torch.testing import load_call
    if scheme == "cld_fraction":
        from eamxx_torch.cld_fraction import cld_fraction
        import test_cld_fraction as T
        c = load_call(sorted((ROOT / "tests/data/cld_fraction").glob("release_*.npz"))[0])
        s = f32(c.pre)
        return cld_fraction(s["qi"], s["cldfrac_liq"])
    if scheme == "shoc":
        import test_shoc as T
        from eamxx_torch.shoc import SHOCRuntime, shoc_run
        c = load_call(T.FIXTURES[0])
        s = f32(c.pre)
        cell = torch.full((s["T_mid"].shape[0],), c.meta["cell_length"], dtype=torch.float32)
        return shoc_run(s, float(c.meta["dt"]), cell, T.NPBL, SHOCRuntime(extra_diags=True), apply_tms=True,
                        tracer_names=tuple(n for n in T.TRACERS if n in s))
    if scheme == "p3":
        import test_p3 as T
        from eamxx_torch.p3 import P3Runtime, p3_run
        c = load_call(T.FIXTURES[0])
        m = c.meta["p3"]
        from eamxx_torch.p3 import load_p3_tables
        return p3_run(f32(c.pre), float(c.meta["dt"]), load_p3_tables(T.TABLES_DIR, dtype=torch.float32),
                      P3Runtime(**m["runtime"]),
                      m["predictNc"], m["prescribedCCN"])
    if scheme == "zm":
        import test_zm as T
        c = load_call(T.FIXTURES[0])
        return T._run(c, state=f32(c.pre))
    if scheme == "rrtmgp":
        import test_rrtmgp as T
        from eamxx_torch.rrtmgp import RRTMGPRuntime, rrtmgp_run
        from eamxx_torch.timestamp import TimeStamp
        c = load_call([p for p in T.FIXTURES if "arm97" in p.name][0])
        m = c.meta["rrtmgp"]
        T._run(c)
        rt = dict(m["runtime"], active_gases=tuple(m["runtime"]["active_gases"]))
        return rrtmgp_run(f32(c.pre), float(c.meta["dt"]), T._DATA_CACHE[tuple(m["files"])], RRTMGPRuntime(**rt),
                          TimeStamp.from_string(c.meta["time_start"]), c.meta["step"] - 1,
                          torch.as_tensor(m["lat"], dtype=torch.float32), torch.as_tensor(m["lon"], dtype=torch.float32))
    if scheme == "scm":
        import test_scm as T
        orig = T._config

        def cfg32(meta, files):
            c = orig(meta, files)
            c.dtype = torch.float32
            return c
        T._config = cfg32
        try:
            scm, s = T._scm(ncol=1)
        finally:
            T._config = orig
        for _ in range(2):
            s = scm.step(s).state
        return s
    raise KeyError(scheme)


def main() -> None:
    torch.set_num_threads(1)
    schemes = sys.argv[1:] or ["cld_fraction", "shoc", "p3", "zm", "rrtmgp", "scm"]
    for sch in schemes:
        rec = Record()
        try:
            with rec:
                out = run(sch)
            outs = out if isinstance(out, dict) else {"out": out}
            dt = collections.Counter(str(v.dtype) for v in outs.values() if isinstance(v, torch.Tensor) and v.is_floating_point())
            print(f"== {sch}: output dtypes {dict(dt)}; float64 results at {len(rec.sites)} source lines")
        except Exception as e:
            print(f"== {sch}: FAILED in float32: {e!r}")
            print("   " + traceback.format_exc().strip().splitlines()[-3])
        for site, n in rec.sites.most_common(25):
            print(f"   {n:6d}  {site}")


if __name__ == "__main__":
    main()
