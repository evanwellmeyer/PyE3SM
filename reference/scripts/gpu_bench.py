#!/usr/bin/env python3
"""Batched single-column runs on CPU and GPU: speed, memory, and CPU-vs-GPU agreement.

  python reference/scripts/gpu_bench.py RUN_DIR [--ncols 1 64 1024] [--steps 6]
  python reference/scripts/gpu_bench.py RUN_DIR --agree [--nsteps 36]

RUN_DIR is a make_scm_case.py directory. Every column starts from the same state (the
columns are independent, so this measures throughput). Timings exclude two warm-up
steps. Devices: cpu float64, cpu float32, and the GPU (mps or cuda) in float32; cuda also
in float64. --agree runs one column for N steps on the CPU and on the GPU in float32, and on the
CPU in float64, and reports the GPU-CPU differences next to the float32-float64 ones.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import scm_case  # noqa: E402


def gpu() -> str | None:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return None


def sync(dev: str) -> None:
    if dev == "cuda":
        torch.cuda.synchronize()
    elif dev == "mps":
        torch.mps.synchronize()


def mem_gb(dev: str) -> float:
    if dev == "cuda":
        return torch.cuda.max_memory_allocated() / 2**30
    if dev == "mps":
        return torch.mps.driver_allocated_memory() / 2**30
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30


def bench(run: Path, ncols, steps: int) -> None:
    g = gpu()
    configs = [("cpu", torch.float64), ("cpu", torch.float32)]
    if g:
        configs.append((g, torch.float32))
        if g == "cuda":
            configs.append((g, torch.float64))
    print(f"{'device':6s} {'dtype':8s} {'columns':>8s} {'s/step':>9s} {'columns*steps/s':>16s} {'memory GB':>10s}")
    for dev, dt in configs:
        for n in ncols:
            with torch.no_grad():
                scm, s = scm_case.port_scm(run, ncol=n, dtype=dt, device=dev)
                for _ in range(2):
                    s = scm.step(s).state
                sync(dev)
                t = time.time()
                for _ in range(steps):
                    s = scm.step(s).state
                sync(dev)
                dtm = (time.time() - t) / steps
            print(f"{dev:6s} {str(dt)[6:]:8s} {n:8d} {dtm:9.3f} {n / dtm:16.1f} {mem_gb(dev):10.2f}", flush=True)
            del scm, s
            if dev == "mps":
                torch.mps.empty_cache()


def agree(run: Path, nsteps: int) -> None:
    g = gpu()
    fields = ("T_mid", "qv", "qc", "qr", "qi", "tke", "cldfrac_tot", "horiz_winds", "precip_liq_surf_mass")
    runs = {}
    for dev, dt in (("cpu", torch.float32), (g, torch.float32), ("cpu", torch.float64)):
        with torch.no_grad():
            scm, s = scm_case.port_scm(run, dtype=dt, device=dev)
            traj = []
            for _ in range(nsteps):
                s = scm.step(s).state
                traj.append({k: s[k].detach().cpu().double() for k in fields if k in s})
        runs[(dev, dt)] = traj
    c32, g32, c64 = runs[("cpu", torch.float32)], runs[(g, torch.float32)], runs[("cpu", torch.float64)]
    print(f"{run.name}: normwise relative difference ||x - y|| / ||y||; "
          f"{g} f32 vs cpu f32, and (yardstick) cpu f32 vs cpu f64")
    for n in sorted({1, 2, 6, nsteps // 2, nsteps}):
        for label, (x, y) in ((f"{g}-cpu ", (g32, c32)), ("f32-f64 ", (c32, c64))):
            a, b = y[n - 1], x[n - 1]
            diffs = {k: float((b[k] - a[k]).norm() / a[k].norm()) if float(a[k].norm()) > 0 else float(b[k].norm())
                     for k in a}
            print(f"  step {n:4d} {label}: " + ", ".join(f"{k} {v:.1e}" for k, v in diffs.items()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--ncols", type=int, nargs="+", default=[1, 64, 1024])
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--agree", action="store_true")
    ap.add_argument("--nsteps", type=int, default=36)
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (0: torch default)")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    if a.agree:
        agree(a.run, a.nsteps)
    else:
        bench(a.run, a.ncols, a.steps)


if __name__ == "__main__":
    main()
