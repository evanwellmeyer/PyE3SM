#!/usr/bin/env python3
"""Summary figure of the validation runs (docs/validation.md).

  python reference/scripts/plot_validation.py [OUT.png]

Reads reference/runs/validation/val_*_libm.npz (validate_obs.py run) and the IOP files.
Needs numpy and matplotlib only (not torch).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import validate_obs as vo  # noqa: E402

COL = {"standard": "tab:blue", "lr": "tab:red"}
LAB = {"standard": "100 s (DP-EAMxx settings)", "lr": "1800 s + ZM (EAMxx-LR)"}


def load(case, cfg):
    f = vo.OUT_DIR / f"val_{case}{'_lr' if cfg == 'lr' else ''}_libm.npz"
    return vo.Run(f) if f.exists() else None


def daily(x, dt):
    n = int(round(86400 / dt))
    m = len(x) // n
    return x[: m * n].reshape(m, n).mean(1)


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parents[1] / "docs" / "figures" / "validation.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(2, 3, figsize=(13, 8))

    a = ax[0, 0]   # BOMEX cloud fraction, hours 3-6
    for cfg in COL:
        r = load("bomex", cfg)
        if r:
            a.plot(r.pmean("cldfrac_tot", (3, 6)), r.pmean("z_mid", (3, 6)), color=COL[cfg], label=LAB[cfg])
    a.add_patch(plt.Rectangle((0.05, 500), 0.02, 200, color="k", alpha=0.25, label="LES maximum (B2, B3)"))
    a.set(ylim=(0, 2500), xlabel="cloud fraction", ylabel="height [m]", title="BOMEX, hours 3-6")
    a.legend(fontsize=7)

    a = ax[0, 1]   # RICO total cloud cover, hourly
    for cfg in COL:
        r = load("rico", cfg)
        if r:
            a.plot(np.arange(24) + 0.5, r.hourly("cldtot"), color=COL[cfg])
    a.axhspan(0.1, 0.3, xmin=20 / 24, color="k", alpha=0.25, label="observed range (R1)")
    a.axhline(0.17, color="k", lw=0.8, ls="--", label="lidar 0.17")
    a.set(xlim=(0, 24), ylim=(0, 1), xlabel="hour", ylabel="total cloud cover", title="RICO")
    a.legend(fontsize=7)

    a = ax[0, 2]   # DYCOMS-RF01 cloud fraction, hours 2-4
    for cfg in COL:
        r = load("dycomsrf01", cfg)
        if r:
            a.plot(r.pmean("cldfrac_tot", (2, 4)), r.pmean("z_mid", (2, 4)), color=COL[cfg])
    a.axhline(840, color="k", ls="--", lw=0.8, label="observed top, base")
    a.axhline(600, color="k", ls="--", lw=0.8)
    a.set(ylim=(0, 1200), xlabel="cloud fraction", ylabel="height [m]", title="DYCOMS-RF01, hours 2-4")
    a.legend(fontsize=7)

    a = ax[1, 0]   # CASS hourly cloud cover vs local time
    for cfg in COL:
        r = load("cass", cfg)
        if r:
            hv = r.hourly("cldtot")
            a.plot(6 + np.arange(len(hv)) + 0.5, hv, color=COL[cfg])
    a.add_patch(plt.Rectangle((13, 0.29), 1, 0.08, color="k", alpha=0.25, label="observed peak (C1)"))
    a.set(xlim=(6, 20), ylim=(0, 0.6), xlabel="local standard time [h]", ylabel="total cloud cover",
          title="CASS")
    a.legend(fontsize=7)

    for a, case, title in ((ax[1, 1], "arm97", "ARM97, 23 Jun - 1 Jul 1997"),
                           (ax[1, 2], "gateiii", "GATE III, 30 Aug - 19 Sep 1974")):
        obs_done = False
        for cfg in COL:
            r = load(case, cfg)
            if not r:
                continue
            p = daily(r.s["precip"], r.dt) * 86400.0
            a.plot(np.arange(len(p)) + 0.5, p, color=COL[cfg], marker="o", ms=3)
            if not obs_done:
                o = vo.iop_obs(r)
                days = np.arange(len(p) + 1) * 86400.0
                po = [vo.obs_integral(o["t"], o["prec"], days[i], days[i + 1]) for i in range(len(p))]
                a.plot(np.arange(len(p)) + 0.5, po, color="k", marker="s", ms=3, label="observed (IOP)")
                obs_done = True
        a.set(xlabel="day", ylabel="precipitation [mm/day]", title=title)
        a.legend(fontsize=7)
    fig.suptitle("EAMxx physics as a single column (port = EAMxx bitwise) vs LES and observations")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
