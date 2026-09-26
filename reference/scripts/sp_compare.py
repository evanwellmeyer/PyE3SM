#!/usr/bin/env python3
"""The port in float32 against EAMxx's single-precision build, with a yardstick.

  python reference/scripts/sp_compare.py PROC RUN_DIR... [--math torch|libm]

For every dumped call of PROC in runs of the single-precision build (strict-sp), the
port runs the same inputs twice: in float32 (as the reference) and in float64. Per
output field, over all calls:

  e32  = ||port_float32 - ref|| / ||ref||      the port's float32 fidelity
  e64  = ||port_float64 - ref|| / ||ref||      how far single precision itself moves the
                                               answer (the reference computed in float)
  ratio = e32 / e64

A float32 port that follows the reference's float arithmetic has e32 << e64. Norms are
over all values of the field in all calls (2-norm).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import replay_check as rc  # noqa: E402
from read_dump import read_dump  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("proc")
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--math", choices=["torch", "libm"], default="torch")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    torch.set_num_threads(1)
    from eamxx_torch import mathfn
    mathfn.use_backend(a.math).__enter__()
    acc = {}
    ncalls = 0
    for run in a.runs:
        settings = rc.case_settings(run) if a.proc in rc.NEEDS_CONTEXT else None
        for pj in sorted((run / "dump" / a.proc).glob("*_pre.json")):
            stem = pj.with_suffix("")
            mpre, pre = read_dump(stem)
            mpost, post = read_dump(str(stem)[:-3] + "post")
            meta = {"time_end": mpost["time"], "dt": mpost["dt"], "step": mpre["step"], "time_start": mpre["time"]}
            outs = {}
            for dt in (torch.float32, torch.float64):
                rc.DTYPE = dt
                if a.proc in rc.NEEDS_CONTEXT:
                    o = rc.PORTS[a.proc](pre, meta, settings, with_checks=False)
                else:
                    o = rc.PORTS[a.proc](pre)
                outs[dt] = {k: v.detach().double().numpy() for k, v in o.items()
                            if isinstance(v, torch.Tensor) and v.is_floating_point()}
            ncalls += 1
            for k, ref in post.items():
                if k not in outs[torch.float32]:
                    continue
                r = np.asarray(ref, dtype=np.float64)
                p32 = outs[torch.float32][k].reshape(r.shape)
                p64 = outs[torch.float64][k].reshape(r.shape)
                ok = np.isfinite(r)
                s = acc.setdefault(k, [0.0, 0.0, 0.0, 0, 0])
                s[0] += float(np.sum((p32[ok] - r[ok]) ** 2))
                s[1] += float(np.sum((p64[ok] - r[ok]) ** 2))
                s[2] += float(np.sum(r[ok] ** 2))
                s[3] += int(np.sum(p32[ok] == r[ok]))
                s[4] += int(ok.sum())
    rows = []
    print(f"{a.proc}: {ncalls} calls (math={a.math})")
    print(f"{'field':26s} {'e32 (port f32)':>15s} {'e64 (precision)':>16s} {'e32/e64':>9s} {'bitwise':>9s}")
    for k in sorted(acc):
        d32, d64, nr, eq, n = acc[k]
        if nr == 0:
            continue
        e32, e64 = np.sqrt(d32 / nr), np.sqrt(d64 / nr)
        ratio = e32 / e64 if e64 > 0 else (0.0 if e32 == 0 else np.inf)
        rows.append(dict(field=k, e32=e32, e64=e64, ratio=ratio, bitwise_frac=eq / max(n, 1)))
        print(f"{k:26s} {e32:15.2e} {e64:16.2e} {ratio:9.2g} {eq / max(n, 1):9.1%}")
    worst = max((r["ratio"] for r in rows if np.isfinite(r["ratio"])), default=0)
    print(f"largest e32/e64: {worst:.2g}")
    if a.json:
        a.json.write_text(json.dumps(dict(proc=a.proc, calls=ncalls, math=a.math, rows=rows), indent=1))


if __name__ == "__main__":
    main()
