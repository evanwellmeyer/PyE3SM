#!/usr/bin/env python3
"""Gradients through the whole single-column model over N steps.

  python reference/scripts/grad_horizon.py RUN_DIR --steps 1 3 10 36 [--eps 1e-6]

RUN_DIR is a make_scm_case.py directory (settings as in the reference runs). For each N,
from the initial state: L = sum(v_T * T_mid(N)) / |T| + sum(v_q * qv(N)) / |qv| with fixed
random v; the gradient of L with respect to the initial T_mid and qv (autograd, PyTorch's
math) is compared with a central finite difference along a random relative direction.
Reports whether the gradient is finite, the relative AD-FD difference, the gradient norm
and memory/time. A non-finite gradient is traced to its forward line (anomaly mode).
"""
from __future__ import annotations

import argparse
import re
import resource
import sys
import time
import warnings
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


CKPT = False
OPTS: dict = {}


def run(scm, s0, n, x):
    s = dict(s0)
    s["T_mid"], s["qv"] = x
    return scm.run(s, n, checkpoint=CKPT)


def per_process(run_dir: Path, nsteps: int) -> None:
    """In-situ check of each process of step `nsteps` (and of the driver before the suite):
    autograd vs central differences for v.(T_mid, qv) out with respect to all float inputs."""
    import copy
    import scm_case
    from eamxx_torch.suite import Suite
    calls = []
    orig = Suite._run_process

    def rec(self, name, fn, state, dt, nsub=1):
        calls.append((name, fn, dict(state), dt, nsub, copy.deepcopy(self.clocks[name])))
        return orig(self, name, fn, state, dt, nsub)
    scm, s = scm_case.port_scm(run_dir)
    with torch.no_grad():
        for _ in range(nsteps - 1):
            s = scm.step(s).state
        Suite._run_process = rec
        try:
            scm.step(s)
        finally:
            Suite._run_process = orig
    gen = torch.Generator().manual_seed(3)
    for name, fn, st, dt, nsub, clk in calls:
        keys = [k for k, v in st.items() if isinstance(v, torch.Tensor) and v.dtype == torch.float64 and v.dim() >= 1]
        d = {k: torch.randn(st[k].shape, generator=gen, dtype=torch.float64) * st[k].abs() for k in keys}

        def f(xs):
            c = copy.deepcopy(clk)
            out, _ = fn({**st, **xs}, dt / nsub, c)
            return out
        x = {k: st[k].clone().requires_grad_(True) for k in keys}
        out = f(x)
        outs = [k for k in ("T_mid", "qv", "qc", "qr", "qi", "tke", "cldfrac_tot", "nccn", "surf_sens_flux")
                if k in out and isinstance(out[k], torch.Tensor) and out[k].requires_grad]
        if not outs:
            print(f"  {name}: no differentiable outputs among those checked")
            continue
        v = {k: torch.randn(out[k].shape, generator=gen, dtype=torch.float64) for k in outs}
        nrm = {k: max(float(out[k].detach().abs().sum()), 1.0) for k in outs}   # fixed normalization
        L = sum((v[k] * out[k]).sum() / nrm[k] for k in outs)
        g = torch.autograd.grad(L, [x[k] for k in keys], allow_unused=True)
        rows = []
        for k, gk in zip(keys, g):
            if gk is None or not float(d[k].abs().sum()):
                continue
            A = float((gk * d[k]).sum())
            fds = []
            for e in (1e-5, 1e-7):
                with torch.no_grad():
                    op, om = f({k: st[k] + e * d[k]}), f({k: st[k] - e * d[k]})
                    Lp = sum(float((v[o] * op[o]).sum()) / nrm[o] for o in outs)
                    Lm = sum(float((v[o] * om[o]).sum()) / nrm[o] for o in outs)
                fds.append((Lp - Lm) / (2 * e))
            rel = min(abs(A - F) / max(abs(A), abs(F), 1e-12) for F in fds)
            if rel > 1e-2 and max(abs(A), *map(abs, fds)) > 1e-9:
                rows.append(f"      d/d {k:22s} AD {A:+.4e}  FD {fds[0]:+.4e} {fds[1]:+.4e}")
        print(f"  {name}: {len(keys)} inputs, outputs {outs}: " + ("all agree" if not rows else f"{len(rows)} disagree"))
        for r in rows[:8]:
            print(r)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--steps", type=int, nargs="+", default=[1, 3, 10, 36])
    ap.add_argument("--eps", type=float, nargs="+", default=[1e-5, 1e-7])
    ap.add_argument("--smooth", action="store_true", help="turn on the gradient options (scm_case.port_scm)")
    ap.add_argument("--only", choices=["seeds", "tke"], help="with --smooth: only one of the two options")
    ap.add_argument("--checkpoint", action="store_true", help="recompute each step in the backward pass")
    ap.add_argument("--per-process", type=int, default=0, metavar="STEP",
                    help="instead: check each process of step STEP in situ")
    a = ap.parse_args()
    torch.set_num_threads(1)
    global CKPT, OPTS
    CKPT = a.checkpoint
    OPTS = dict(smooth=a.smooth)
    if a.smooth and a.only == "seeds":
        OPTS = dict(seeds_free=True, tke_tol=0.0)
    elif a.smooth and a.only == "tke":
        OPTS = dict(seeds_free=False, tke_tol=1e-14)
    if a.per_process:
        per_process(a.run, a.per_process)
        return
    import scm_case
    gen = torch.Generator().manual_seed(7)
    for n in a.steps:
        scm, s0 = scm_case.port_scm(a.run, **OPTS)
        T0, q0 = s0["T_mid"].detach(), s0["qv"].detach()
        dT = torch.randn(T0.shape, generator=gen, dtype=torch.float64) * T0
        dq = torch.randn(q0.shape, generator=gen, dtype=torch.float64) * q0
        vT = torch.randn(T0.shape, generator=gen, dtype=torch.float64)
        vq = torch.randn(q0.shape, generator=gen, dtype=torch.float64)

        with torch.no_grad():
            sref = run(scm, s0, n, (T0, q0))
            nT, nq = float(sref["T_mid"].abs().sum()), float(sref["qv"].abs().sum())   # fixed

        def loss(s):
            return (vT * s["T_mid"]).sum() / nT + (vq * s["qv"]).sum() / nq
        scm, s0 = scm_case.port_scm(a.run, **OPTS)
        t = time.time()
        x = (T0.clone().requires_grad_(True), q0.clone().requires_grad_(True))
        L = loss(run(scm, s0, n, x))
        L.backward()
        gT, gq = x[0].grad, x[1].grad
        t_ad = time.time() - t
        mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30   # macOS: bytes
        finite = bool(torch.isfinite(gT).all() and torch.isfinite(gq).all())
        line = f"N={n:4d}: gradient {'finite' if finite else 'NOT FINITE'}"
        if finite:
            ad = float((gT * dT).sum() + (gq * dq).sum())
            fds = []
            for e in a.eps:
                with torch.no_grad():
                    scm, s0 = scm_case.port_scm(a.run, **OPTS)
                    Lp = float(loss(run(scm, s0, n, (T0 + e * dT, q0 + e * dq))))
                    scm, s0 = scm_case.port_scm(a.run, **OPTS)
                    Lm = float(loss(run(scm, s0, n, (T0 - e * dT, q0 - e * dq))))
                fds.append((Lp - Lm) / (2 * e))
            rel = min(abs(ad - f) / max(abs(ad), abs(f), 1e-300) for f in fds)
            line += (f"; AD {ad:+.5e}, FD " + ", ".join(f"{f:+.5e}" for f in fds)
                     + f" (best rel. diff {rel:.1e}); |dL/dT0| {float(gT.norm()):.3e}, |dL/dq0| {float(gq.norm()):.3e}")
        line += f"; {t_ad:.0f} s, peak memory {mem:.1f} GB"
        print(line, flush=True)
        if not finite:
            scm, s0 = scm_case.port_scm(a.run, **OPTS)
            x = (T0.clone().requires_grad_(True), q0.clone().requires_grad_(True))
            with warnings.catch_warnings(record=True) as w, torch.autograd.detect_anomaly(check_nan=True):
                warnings.simplefilter("always")
                try:
                    loss(run(scm, s0, n, x)).backward()
                    print("   anomaly mode: no NaN (infinite gradients only)")
                except RuntimeError as err:
                    print("   " + str(err).splitlines()[0])
            tr = [str(m.message) for m in w if "Traceback of forward call" in str(m.message)]
            if tr:
                for f, ln, fn, src in re.findall(r'File "([^"]*eamxx_torch[^"]*)", line (\d+), in (\w+)\n\s*(.*)', tr[-1])[-3:]:
                    print(f"   {f.split('eamxx_torch/')[-1]}:{ln} {fn}: {src.strip()}")
            break


if __name__ == "__main__":
    main()
