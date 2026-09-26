#!/usr/bin/env python3
"""First operation whose result differs between the CPU and a GPU, on identical inputs.

  python reference/scripts/device_diff.py SCHEME [--case RUN_DIR] [--step N] [--rtol 1e-3]

Captures the inputs of SCHEME (a suite process: shoc, p3, rrtmgp, zm, ...) at step N of a
single-column run in float32 on the CPU, runs the process on those same inputs on the CPU
and on the GPU while recording every torch operation's floating result in order, and
prints the first operations where the two differ by more than rounding (relative to the
result's size), or where one is finite and the other not, with the port's source line.
"""
from __future__ import annotations

import argparse
import copy
import sys
import traceback
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import scm_case  # noqa: E402


class Rec(torch.overrides.TorchFunctionMode):
    def __init__(self):
        super().__init__()
        self.log = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if isinstance(out, torch.Tensor) and out.is_floating_point() and out.numel():
            site = "?"
            for fr in reversed(traceback.extract_stack(limit=10)):
                if "eamxx_torch" in fr.filename:
                    site = f"{fr.filename.split('eamxx_torch/')[-1]}:{fr.lineno}: {fr.line.strip()[:100]}"
                    break
            self.log.append((site, getattr(func, "__name__", str(func)), out.detach().to("cpu").double()))
        return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scheme")
    ap.add_argument("--case", type=Path, default=Path("reference/runs/val_bomex"))
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--rtol", type=float, default=1e-3)
    ap.add_argument("--show", type=int, default=5)
    a = ap.parse_args()
    gpu = "cuda" if torch.cuda.is_available() else "mps"
    from eamxx_torch.suite import Suite
    captured = {}
    orig = Suite._run_process

    def rec(self, name, fn, state, dt, nsub=1):
        if name == a.scheme and "call" not in captured:
            captured["call"] = (fn, dict(state), dt / nsub, copy.deepcopy(self.clocks[name]), self)
        return orig(self, name, fn, state, dt, nsub)
    scm, s = scm_case.port_scm(a.case, dtype=torch.float32)
    with torch.no_grad():
        for _ in range(a.step - 1):
            s = scm.step(s).state
        Suite._run_process = rec
        scm.step(s)
        Suite._run_process = orig
    fn, st, dt, clk, suite = captured["call"]
    logs = {}
    for dev in ("cpu", gpu):
        from eamxx_torch.precision import cast
        # move the suite's data and the inputs to the device
        for attr in ("p3_tables", "rrtmgp_data", "zm_setup"):
            if getattr(suite, attr, None) is not None:
                setattr(suite, attr, cast(getattr(suite, attr), torch.float32, dev))
        if suite.spa is not None:
            suite.spa.data = cast(suite.spa.data, torch.float32, dev)
        x = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in st.items()}
        with torch.no_grad(), Rec() as r:
            fn(x, dt, copy.deepcopy(clk))
        logs[dev] = r.log
    shown = 0
    n = min(len(logs["cpu"]), len(logs[gpu]))
    print(f"{a.scheme} at step {a.step}: {len(logs['cpu'])} operations on cpu, {len(logs[gpu])} on {gpu}")
    for i in range(n):
        (s1, f1, t1), (s2, f2, t2) = logs["cpu"][i], logs[gpu][i]
        if s1 != s2 or t1.shape != t2.shape:
            print(f"  op {i}: the operation sequences diverge ({s1} vs {s2})")
            break
        fin1, fin2 = torch.isfinite(t1), torch.isfinite(t2)
        scale = float(t1[fin1].abs().max()) if fin1.any() else 1.0
        both = fin1 & fin2
        d = float((t1[both] - t2[both]).abs().max()) / max(scale, 1e-30) if both.any() else 0.0
        if not torch.equal(fin1, fin2) or d > a.rtol:
            k = int((~torch.eq(fin1, fin2) | ((t1 - t2).abs() > a.rtol * scale)).nonzero()[0][-1]) if t1.dim() else 0
            idx = tuple(int(j) for j in (~torch.eq(fin1, fin2) | ((t1 - t2).nan_to_num().abs() > a.rtol * scale)).nonzero()[0])
            print(f"  op {i} {f1}: {s1}\n      finite cpu/{gpu}: {int(fin1.sum())}/{int(fin2.sum())} of {t1.numel()}; "
                  f"max rel diff {d:.2e}; e.g. at {idx}: cpu {float(t1[idx]):.6g} {gpu} {float(t2[idx]):.6g}")
            shown += 1
            if shown >= a.show:
                break
    if shown == 0:
        print(f"  no operation differs by more than rtol={a.rtol}")


if __name__ == "__main__":
    main()
