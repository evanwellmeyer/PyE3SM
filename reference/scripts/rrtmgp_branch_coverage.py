#!/usr/bin/env python3
"""Count how often each RRTMGP branch/limiter is taken when replaying reference dumps.

  python reference/scripts/rrtmgp_branch_coverage.py RUN_DIR [RUN_DIR ...]

A branch never taken in any replayed call is not validated by the replay. The
port matches the reference bitwise on these calls, so its masks are the
reference's.
"""
from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import replay_check as rc  # noqa: E402
from read_dump import read_dump  # noqa: E402

from eamxx_torch import mathfn  # noqa: E402
from eamxx_torch.rrtmgp import cloud_optics as CO  # noqa: E402
from eamxx_torch.rrtmgp import gas_optics as GO  # noqa: E402
from eamxx_torch.rrtmgp import interface as IF  # noqa: E402
from eamxx_torch.rrtmgp import mcica as MC  # noqa: E402
from eamxx_torch.rrtmgp import rte as RT  # noqa: E402

COV: dict[str, list[int]] = defaultdict(lambda: [0, 0])


def rec(name, mask):
    mask = torch.as_tensor(mask)
    COV[name][0] += int(mask.sum())
    COV[name][1] += mask.numel()


def wrap(mod, fname, probe, alias_mods=()):
    orig = getattr(mod, fname)

    def w(*a, **k):
        out = orig(*a, **k)
        probe(a, k, out)
        return out
    setattr(mod, fname, w)
    for m in alias_mods:
        setattr(m, fname, w)


def install():
    def run_probe(a, k, out):
        rec("run: radiation updated this step", torch.tensor(IF.radiation_do(a[3].rad_frequency, a[5])))
    wrap(IF, "rrtmgp_run", run_probe)
    import eamxx_torch.rrtmgp as pkg
    pkg.rrtmgp_run = IF.rrtmgp_run
    wrap(IF, "rrtmgp_sw", lambda a, k, o: rec("sw: daytime column (mu0 > 0)", a[8] > 0))
    wrap(IF, "rrtmgp_main", lambda a, k, o: (
        rec("clouds: liquid present (lwp > 0)", a[9] > 0),
        rec("clouds: ice present (iwp > 0)", a[10] > 0),
        rec("clouds: partial cloud fraction (0 < cf < 1)", (a[13] > 0) & (a[13] < 1)),
        rec("clouds: rel below table", (a[9] > 0) & (a[11] < 2.5)),
        rec("clouds: rel above table", (a[9] > 0) & (a[11] > 21.5)),
        rec("clouds: rei below table", (a[10] > 0) & (a[12] < 10.)),
        rec("clouds: rei above table", (a[10] > 0) & (a[12] > 180.)),
        rec("gas: T_lay below 160 K (limited)", a[2] < 160.),
        rec("gas: T_lay above 355 K (limited)", a[2] > 355.)),
        alias_mods=())
    wrap(GO, "interpolation", lambda a, k, o: (
        rec("gas: upper atmosphere layer (not tropo)", ~o[2]),
        rec("gas: jpress at its limits", (o[1] == 1) | (o[1] == a[0].npres - 1)),
        rec("gas: jtemp at its limits", (o[0] == 0) | (o[0] == a[0].ntemp - 2))),
        alias_mods=(GO,))
    wrap(MC, "get_subcolumn_mask", lambda a, k, o: rec("mcica: cloudy subcolumn", o),
         alias_mods=(MC,))
    wrap(MC, "get_subsampled_clouds", lambda a, k, o: rec(
        "mcica: cloud fraction but no optical depth (cldfrac_rad set to 0)",
        (a[3] > 0) & ~(a[0] > 0).any(dim=-1)), alias_mods=(IF,))
    wrap(RT, "delta_scale_2str", lambda a, k, o: rec("optics: delta-scaled (tau > eps)",
                                                   a[0] > 3 * 2.2250738585072014e-308),
         alias_mods=(IF,))
    orig_two = RT.sw_two_stream

    def two(mu0, tau, w0, g):
        out = orig_two(mu0, tau, w0, g)
        gamma1 = (8. - w0 * (5. + 3. * g)) * .25
        gamma2 = 3. * (w0 * (1. - g)) * .25
        kk = torch.sqrt(torch.fmax((gamma1 - gamma2) * (gamma1 + gamma2), torch.tensor(1e-12)))
        rec("sw: k limited at sqrt(1e-12)", (gamma1 - gamma2) * (gamma1 + gamma2) < 1e-12)
        km = kk * mu0[:, None, None]
        rec("sw: |1 - (k mu0)^2| < eps (denominator replaced)", torch.abs(1. - km * km) < RT.EPS)
        return out
    RT.sw_two_stream = two
    orig_lw = RT.lw_solver_noscat

    def lw(D, weight, tau, *rest):
        rec("lw: tau below sqrt(eps) (series source)", tau * D <= math.sqrt(RT.EPS))
        return orig_lw(D, weight, tau, *rest)
    RT.lw_solver_noscat = lw
    orig_cloud = IF.compute_aerocom_cloudtop

    def aero(tmid, pmid, p_del, z_del, qc, qi, rel, rei, cf, nc):
        rec("aerocom: layer counted as cloud top candidate",
            ((qc + qi) > 0)[:, 1:] & (cf > 0.001)[:, 1:])
        return orig_cloud(tmid, pmid, p_del, z_del, qc, qi, rel, rei, cf, nc)
    IF.compute_aerocom_cloudtop = aero


def install_orbit():
    from eamxx_torch import orbit as OR
    orig = OR.shr_orb_avg_cosz

    def avg(jday, lat, lon, declin, dt_avg):
        out = orig(jday, lat, lon, declin, dt_avg)
        piover2, twopi, pi = OR.pi / 2.0, OR.pi * 2.0, OR.pi
        cos_h = -(torch.tan(lat) * math.tan(declin))
        h = torch.where(cos_h <= -1, torch.tensor(pi, dtype=lat.dtype),
                        torch.where(cos_h >= 1, torch.tensor(0., dtype=lat.dtype),
                                    torch.acos(cos_h.clamp(-1, 1))))
        t1 = (jday - int(jday)) * twopi + lon - pi
        t1 = torch.where(t1 >= pi, t1 - twopi, torch.where(t1 < -pi, t1 + twopi, t1))
        dt = dt_avg / 86400.0 * twopi
        t2 = t1 + dt
        rec("zenith: polar day (cos_h <= -1)", cos_h <= -1)
        rec("zenith: polar night (cos_h >= 1)", cos_h >= 1)
        c1 = (t2 >= pi) & (t1 <= pi) & (pi - h <= dt)
        rec("zenith: window spans a night shorter than dt (case 1)", c1)
        rec("zenith: case 1 with sunlight on both sides", c1 & (out > 0))
        rec("zenith: case 2 (t1 <= -pi)", ~c1 & (t2 >= -pi) & (t1 <= -pi) & (pi - h <= dt))
        rec("zenith: window crosses midnight (t2 > pi)", ~c1 & (t2 > pi))
        rec("zenith: whole window in darkness (cosz = 0)", out == 0)
        return out
    OR.shr_orb_avg_cosz = avg


def main() -> None:
    install_orbit()
    install()
    ncalls = 0
    torch.set_num_threads(1)
    with mathfn.use_backend("libm"):
        for run in map(Path, sys.argv[1:]):
            st = rc.case_settings(run)
            for pj in sorted((run / "dump" / "rrtmgp").glob("*_c0_*_pre.json")):
                mpre, pre = read_dump(str(pj.with_suffix("")))
                meta = {"dt": mpre["dt"], "step": mpre["step"], "time_start": mpre["time"]}
                rc.run_rrtmgp(pre, meta, st, False)
                ncalls += 1
    print(f"RRTMGP branch coverage over {ncalls} calls (count taken / count evaluated):")
    for k, (t, n) in COV.items():
        print(f"  {k:72s} {t:10d} / {n}{'   <-- never taken' if t == 0 else ''}")


if __name__ == "__main__":
    main()
