#!/usr/bin/env python3
"""Gradient report for each ported scheme (plan section 9, differentiability).

  python reference/scripts/grad_check.py PROC RUN_DIR STEP [--rank R] [--cols N] [--out JSON]

Takes one dumped call of PROC (inputs exactly as EAMxx saw them, via replay_check.py),
runs the port forward in float64 with PyTorch's own math (the autograd-capable backend),
and checks the gradient of every output field with respect to every input field:

  AD   = < d(v.y)/dx_k , d_k >        autograd, one backward pass per output field y
  FD   = (L(x + e d_k) - L(x - e d_k)) / 2e   central differences, e = 1e-4, 1e-6, 1e-8
  FD+, FD-                               one-sided differences at e = 1e-8

v is a fixed random vector per output; d_k is a random relative perturbation of input k,
zero where x_k = 0 (so inputs never leave their domain, e.g. negative mixing ratios).
Each (output, input) pair is classified:

  ok        AD agrees with FD (0.1%, plus rounding noise)
  ok~       AD agrees with FD at e = 1e-4 only (3%): noise at smaller steps
  none      neither depends on the input (|AD|, |FD| below 1e-10 of the output's size)
  nonfinite AD is NaN or infinite
  jump      FD grows like 1/e: a discontinuity (threshold, level pick, substep count)
            lies within the perturbation
  kink      AD equals a one-sided difference: a non-differentiable point (clip at its
            bound, max/min tie); autograd returns one side's slope
  missing   AD is zero but the output changes smoothly with the input
  mismatch  none of the above

For a non-finite gradient the forward line that produced it is found with autograd's
anomaly mode.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
import warnings
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import replay_check as rc  # noqa: E402
from read_dump import read_dump  # noqa: E402

EPS = (1e-4, 1e-6, 1e-8)
_CMP = {torch.lt, torch.gt, torch.le, torch.ge, torch.eq, torch.ne,
        torch.Tensor.__lt__, torch.Tensor.__gt__, torch.Tensor.__le__, torch.Tensor.__ge__,
        torch.Tensor.__eq__, torch.Tensor.__ne__,
        # operations that turn real values into indices or picks
        torch.trunc, torch.floor, torch.ceil, torch.round, torch.Tensor.trunc, torch.Tensor.floor,
        torch.Tensor.ceil, torch.Tensor.round, torch.searchsorted, torch.argmax, torch.argmin,
        torch.Tensor.argmax, torch.Tensor.argmin, torch.Tensor.long, torch.Tensor.int,
        torch.remainder, torch.fmod, torch.Tensor.__mod__, torch.Tensor.__floordiv__}


class RecordComparisons(torch.overrides.TorchFunctionMode):
    """Records the result of every elementwise comparison with the port's source line."""

    def __init__(self):
        super().__init__()
        self.log = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        rec = func in _CMP or (func is torch.Tensor.to and isinstance(out, torch.Tensor)
                               and not out.is_floating_point() and out.dtype != torch.bool)
        if rec and isinstance(out, torch.Tensor):
            site = "?"
            for fr in reversed(traceback.extract_stack(limit=12)):
                if "eamxx_torch" in fr.filename and not fr.filename.endswith(("ekat.py", "mathfn.py")):
                    site = f"{Path(fr.filename).parent.name}/{Path(fr.filename).name}:{fr.lineno}: {fr.line.strip()}"
                    break
            self.log.append((site, out.detach().clone()))
        return out


def first_flips(run_fn, x_lo, x_hi, n=3):
    """Comparisons whose results differ between two nearby inputs (the first n)."""
    logs = []
    for xx in (x_lo, x_hi):
        with torch.no_grad(), RecordComparisons() as rec:
            run_fn(xx)
        logs.append(rec.log)
    flips = []
    for (s1, t1), (s2, t2) in zip(*logs):
        if s1 != s2:
            flips.append(f"(control flow differs at {s1})")
            break
        if t1.shape == t2.shape and not torch.equal(t1, t2) and not any(f.startswith(s1) for f in flips):
            flips.append(f"{s1}  [{int((t1 != t2).sum())} element(s)]")
            if len(flips) >= n:
                break
    if len(logs[0]) != len(logs[1]) and len(flips) < n:
        flips.append(f"(number of comparisons differs: {len(logs[0])} vs {len(logs[1])}: a data-dependent loop count)")
    return flips


def load_call(run: Path, proc: str, step: int, rank: int):
    stem = run / "dump" / proc / f"s{step}_c0_r{rank}_pre"
    mpre, pre = read_dump(stem)
    mpost, _ = read_dump(str(stem)[:-3] + "post")
    meta = {"time_end": mpost["time"], "dt": mpost["dt"], "step": mpre["step"], "time_start": mpre["time"]}
    settings = rc.case_settings(run) if proc in rc.NEEDS_CONTEXT else None
    return pre, meta, settings


def forward(proc, x: dict, meta, settings):
    f = rc.PORTS[proc]
    if proc in rc.NEEDS_CONTEXT:
        return f(x, meta, settings, with_checks=False)
    return f(x)


def float_outputs(out: dict) -> dict:
    return {k: v for k, v in out.items()
            if isinstance(v, torch.Tensor) and v.dtype == torch.float64 and v.dim() >= 1}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("proc", choices=sorted(rc.PORTS))
    ap.add_argument("run", type=Path)
    ap.add_argument("step", type=int)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--cols", type=int, default=0, help="keep only the first N columns (0: all)")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--nan-only", action="store_true",
                    help="only find the forward line of the first non-finite gradient (fast)")
    ap.add_argument("--locate", type=int, default=0,
                    help="for up to N flagged pairs, find the element and the comparison that breaks agreement")
    a = ap.parse_args()
    torch.set_num_threads(1)
    gen = torch.Generator().manual_seed(1234)
    pre, meta, settings = load_call(a.run, a.proc, a.step, a.rank)
    if a.cols:
        ncol = next(v.shape[0] for v in pre.values() if isinstance(v, np.ndarray) and v.ndim >= 1)
        pre = {k: (v[: a.cols] if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == ncol else v)
               for k, v in pre.items()}
    base = {k: torch.as_tensor(v, dtype=torch.float64) if isinstance(v, np.ndarray) and v.dtype.kind == "f"
            else v for k, v in pre.items()}
    inputs = [k for k, v in base.items() if isinstance(v, torch.Tensor) and v.dim() >= 1]
    x = {k: (v.clone().requires_grad_(True) if k in inputs else v) for k, v in base.items()}

    if a.nan_only:
        with warnings.catch_warnings(record=True) as w, torch.autograd.detect_anomaly(check_nan=True):
            warnings.simplefilter("always")
            o = float_outputs(forward(a.proc, x, meta, settings))
            loss = sum((torch.randn(t.shape, generator=gen, dtype=torch.float64) * t).sum()
                       / max(float(t.detach().abs().sum()), 1.0) for t in o.values() if t.requires_grad)
            try:
                loss.backward()
                bad = [k for k in inputs if x[k].grad is not None and not torch.isfinite(x[k].grad).all()]
                print("no NaN in the backward pass" + (f"; infinite gradients for {bad}" if bad else "; all gradients finite"))
                return
            except RuntimeError as e:
                print(str(e).splitlines()[0])
        trace = [str(m.message) for m in w if "Traceback of forward call" in str(m.message)]
        for f, n, fn, src in (re.findall(r'File "([^"]*eamxx_torch[^"]*)", line (\d+), in (\w+)\n\s*(.*)', trace[-1])[-4:] if trace else []):
            print(f"  {Path(f).parent.name}/{Path(f).name}:{n} {fn}: {src.strip()}")
        return

    # ---- forward with graph, one backward per output field
    report = dict(proc=a.proc, run=a.run.name, step=a.step, rank=a.rank, error=None, pairs=[], nonfinite_at=[])
    try:
        out = float_outputs(forward(a.proc, x, meta, settings))
    except Exception as e:  # e.g. an in-place op on a leaf that needs grad
        report["error"] = f"forward with requires_grad failed: {e!r}"
        print(report["error"])
        traceback.print_exc()
        return
    outs = [k for k, v in out.items() if v.requires_grad]
    report["constant_outputs"] = [k for k in out if k not in outs]
    v = {k: torch.randn(out[k].shape, generator=gen, dtype=torch.float64) for k in outs}
    d = {k: torch.randn(base[k].shape, generator=gen, dtype=torch.float64) * base[k].abs() for k in inputs}
    scale = {k: float((v[k] * out[k].detach()).abs().sum()) for k in outs}
    ad = {}
    for y in outs:
        g = torch.autograd.grad((v[y] * out[y]).sum(), [x[k] for k in inputs], retain_graph=True,
                                allow_unused=True)
        ad[y] = {k: (float((gk * d[k]).sum()) if gk is not None else 0.0) for k, gk in zip(inputs, g)}
    del out

    # ---- where a gradient is non-finite: anomaly mode names the forward op
    bad = sorted({y for y in outs for k in inputs if not np.isfinite(ad[y][k])})
    for y in bad[:5]:
        xx = {k: (base[k].clone().requires_grad_(True) if k in inputs else base[k]) for k in base}
        with warnings.catch_warnings(record=True) as w, torch.autograd.detect_anomaly(check_nan=True):
            warnings.simplefilter("always")
            o = float_outputs(forward(a.proc, xx, meta, settings))
            try:
                (v[y] * o[y]).sum().backward()
                msg = "no NaN raised (infinite values only)"
            except RuntimeError as e:
                msg = str(e).splitlines()[0]
        trace = [str(m.message) for m in w if "Traceback of forward call" in str(m.message)]
        lines = re.findall(r'File "([^"]*eamxx_torch[^"]*)", line (\d+), in (\w+)\n\s*(.*)', trace[-1]) if trace else []
        report["nonfinite_at"].append(dict(output=y, error=msg,
                                           forward=[f"{Path(f).name}:{n} {fn}: {src.strip()}" for f, n, fn, src in lines[-3:]]))

    # ---- finite differences, one input at a time
    with torch.no_grad():
        L0 = {y: float((v[y] * t).sum()) for y, t in float_outputs(forward(a.proc, base, meta, settings)).items() if y in v}
        fd = {k: {} for k in inputs}
        for k in inputs:
            for e in EPS:
                res = []
                for sgn in (1.0, -1.0):
                    xp = dict(base)
                    xp[k] = base[k] + sgn * e * d[k]
                    o = float_outputs(forward(a.proc, xp, meta, settings))
                    res.append({y: float((v[y] * o[y]).sum()) for y in outs})
                fd[k][e] = res
    counts = {}
    for y in outs:
        S = scale[y] or 1.0
        for k in inputs:
            A = ad[y][k] / S                                        # normalized by the output's size
            c = {e: (fd[k][e][0][y] - fd[k][e][1][y]) / (2 * e) / S for e in EPS}
            one = {e: ((fd[k][e][0][y] - L0[y]) / e / S, (L0[y] - fd[k][e][1][y]) / e / S) for e in EPS[1:]}
            noise = {e: 3e-14 / e for e in EPS}                      # rounding error of a difference
            tol = lambda e, f: 1e-3 * max(abs(A), abs(f)) + noise[e]  # noqa: E731
            if not np.isfinite(A):
                cls = "nonfinite"
            elif any(abs(A - c[e]) <= tol(e, c[e]) for e in EPS):
                cls = "none" if abs(A) < 1e-12 and all(abs(c[e]) <= noise[e] + 1e-12 for e in EPS) else "ok"
            elif abs(A - c[1e-4]) <= 3e-2 * max(abs(A), abs(c[1e-4])):
                cls = "ok~"      # agrees at the largest step only: truncation or rounding noise
            elif abs(c[1e-8]) > 30 * abs(c[1e-6]) + 100 * noise[1e-8] or abs(c[1e-6]) > 30 * abs(c[1e-4]) + 100 * noise[1e-6]:
                cls = "jump"
            elif any(abs(A - f) <= tol(e, f) for e in EPS[1:] for f in one[e]):
                cls = "kink"
            elif abs(A) < 1e-12:
                cls = "missing"
            else:
                cls = "mismatch"
            counts[cls] = counts.get(cls, 0) + 1
            if cls not in ("ok", "ok~", "none"):
                report["pairs"].append(dict(output=y, input=k, cls=cls, ad=A, fd={str(e): c[e] for e in EPS},
                                            fdp=one[1e-8][0], fdm=one[1e-8][1]))
    report["counts"] = counts

    # ---- locate: bisect the perturbation to one element, then find the comparison that flips
    report["located"] = []
    run_fn = lambda xx: forward(a.proc, xx, meta, settings)  # noqa: E731
    flagged = [p for p in report["pairs"] if p["cls"] in ("jump", "kink", "mismatch", "missing")]
    seen = set()
    for p in flagged:
        if len(report["located"]) >= a.locate:
            break
        y, k = p["output"], p["input"]
        if (k,) in seen:
            continue
        xx = {kk: (base[kk].clone().requires_grad_(True) if kk == k else base[kk]) for kk in base}
        g = torch.autograd.grad((v[y] * float_outputs(run_fn(xx))[y]).sum(), xx[k])[0].detach()
        S = scale[y] or 1.0

        def bad(mask):
            dm = d[k] * mask
            with torch.no_grad():
                Lp = float((v[y] * float_outputs(run_fn({**base, k: base[k] + e * dm}))[y]).sum())
                Lm = float((v[y] * float_outputs(run_fn({**base, k: base[k] - e * dm}))[y]).sum())
            A, F = float((g * dm).sum()) / S, (Lp - Lm) / (2 * e) / S
            return abs(A - F) > 1e-3 * max(abs(A), abs(F)) + 3e-14 / e
        idx = torch.nonzero(d[k].reshape(-1) != 0).reshape(-1)
        for e in (1e-7, 1e-5, 1e-4):          # the smallest step at which the pair disagrees
            if bad(torch.ones_like(d[k])):
                break
        else:
            continue
        while idx.numel() > 1:
            half = idx[: idx.numel() // 2]
            m = torch.zeros(d[k].numel(), dtype=torch.float64)
            m[half] = 1
            if bad(m.reshape(d[k].shape)):
                idx = half
                continue
            rest = idx[idx.numel() // 2:]
            m = torch.zeros(d[k].numel(), dtype=torch.float64)
            m[rest] = 1
            if bad(m.reshape(d[k].shape)):
                idx = rest
                continue
            break
        if idx.numel() != 1:
            report["located"].append(dict(output=y, input=k, element=None, note="only jointly"))
            continue
        i = int(idx[0])
        pos = np.unravel_index(i, tuple(d[k].shape))
        m = torch.zeros(d[k].numel(), dtype=torch.float64)
        m[i] = 1
        dm = d[k] * m.reshape(d[k].shape)
        flips = first_flips(run_fn, {**base, k: base[k] - e * dm}, {**base, k: base[k] + e * dm})
        seen.add((k,))
        report["located"].append(dict(output=y, input=k, cls=p["cls"], element=[int(t) for t in pos],
                                      value=float(base[k].reshape(-1)[i]), flips=flips))
        print(f"  locate {p['cls']:8s} d {y} / d {k}{list(map(int, pos))} (value {float(base[k].reshape(-1)[i]):.6g}):")
        for f in flips or ["(no comparison flips: a smooth-looking function with a wrong or missing gradient)"]:
            print(f"      {f}")
    report["n_inputs"], report["n_outputs"] = len(inputs), len(outs)
    print(f"{a.proc} {a.run.name} s{a.step} r{a.rank}: {len(inputs)} inputs x {len(outs)} outputs: "
          + ", ".join(f"{k} {n}" for k, n in sorted(counts.items())))
    for nf in report["nonfinite_at"]:
        print(f"  non-finite d({nf['output']}): {nf['error']}; forward: {' <- '.join(nf['forward'][::-1])}")
    for p in report["pairs"][:40]:
        print(f"  {p['cls']:9s} d {p['output']:24s} / d {p['input']:22s} AD {p['ad']:+.3e} "
              f"FD {p['fd']['0.0001']:+.3e} {p['fd']['1e-06']:+.3e} {p['fd']['1e-08']:+.3e} "
              f"FD+ {p['fdp']:+.3e} FD- {p['fdm']:+.3e}")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
