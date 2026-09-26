#!/usr/bin/env python3
"""Count how often each P3 branch/limiter is taken when replaying reference dumps.

  python reference/scripts/p3_branch_coverage.py RUN_DIR [RUN_DIR ...]

A branch never taken in any replayed call is not validated by the replay. The
probes wrap port functions; the port matches the reference bitwise on these
calls, so its masks are the reference's.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import replay_check as rc  # noqa: E402
from read_dump import read_dump  # noqa: E402

from eamxx_torch import constants as C  # noqa: E402
from eamxx_torch import mathfn  # noqa: E402
from eamxx_torch.p3 import functions as F  # noqa: E402
from eamxx_torch.p3 import main as M  # noqa: E402
from eamxx_torch.p3 import sedimentation as S  # noqa: E402
from eamxx_torch.p3 import tables as T  # noqa: E402

COV: dict[str, list[int]] = defaultdict(lambda: [0, 0])


def rec(name, mask):
    COV[name][0] += int(mask.sum())
    COV[name][1] += mask.numel()


def wrap(mod, fname, probe, attr=None):
    orig = getattr(mod, fname)

    def w(*a, **k):
        out = orig(*a, **k)
        probe(a, k, out)
        return out
    setattr(mod, attr or fname, w)
    return orig


def install():
    # part1 / column-level skips
    o1 = M.p3_main_part1

    def p1(*a, **k):
        out = o1(*a, **k)
        rec("part1: nucleation possible (column)", out["nucleationPossible"])
        rec("part1: hydrometeors present (column)", out["hydrometeorsPresent"])
        rec("part1: qc below qsmall -> evaporate", a[19] < C.QSMALL)
        rec("part1: qi small and warm -> rain", (a[21] >= C.QSMALL) & (a[21] < 1.e-8))
        return out
    M.p3_main_part1 = p1

    o2 = M.p3_main_part2

    def p2(ro, predictNc, ccn, dt, inv_dt, tables, st, dgn):
        skip_all = ((st["qc"] < C.QSMALL) & (st["qr"] < C.QSMALL) & (st["qi"] < C.QSMALL)
                    & (st["T_atm"] < C.T_zerodegc) & (st["qv_supersat_i"] < -0.05))
        rec("part2: level skipped entirely", skip_all)
        rec("part2: microphysics active (qc|qr|qi in-cloud)",
            ~skip_all & ((st["qc_incld"] >= C.QSMALL) | (st["qr_incld"] >= C.QSMALL)
                         | (st["qi_incld"] >= C.QSMALL)))
        rec("part2: ice present in-cloud", st["qi_incld"] >= C.QSMALL)
        rec("part2: rain present in-cloud", st["qr_incld"] >= C.QSMALL)
        return o2(ro, predictNc, ccn, dt, inv_dt, tables, st, dgn)
    M.p3_main_part2 = p2

    wrap(T, "get_cloud_dsd2", lambda a, k, o: (
        rec("dsd2: cloud lambda at a limit",
            ((o[3] == (o[1] + 1) * 2.5e4) | (o[3] == (o[1] + 1) * 1.e6)) & (a[0] >= C.QSMALL)),
        rec("dsd2: nc raised to nsmall", (a[1] < C.NSMALL) & (a[0] >= C.QSMALL))))
    M.get_cloud_dsd2 = T.get_cloud_dsd2
    S.get_cloud_dsd2 = T.get_cloud_dsd2
    wrap(T, "get_rain_dsd2", lambda a, k, o: rec(
        "dsd2: rain lambda at a limit",
        ((o[2] == (o[1] + 1.) * 500) | (o[2] == (o[1] + 1.) * 1.e5)) & (a[0] >= C.QSMALL)))
    M.get_rain_dsd2 = T.get_rain_dsd2
    S.get_rain_dsd2 = T.get_rain_dsd2
    wrap(F, "ice_cldliq_wet_growth", lambda a, k, o: rec("ice: wet growth", o[0]))
    wrap(F, "ice_nucleation", lambda a, k, o: rec(
        "ice: nucleation active", (a[0] < C.Tmelt - 15.0) & (a[4] >= 0.05)))
    wrap(F, "ice_melting", lambda a, k, o: rec(
        "ice: melting", (a[11] >= C.QSMALL) & (a[1] > C.Tmelt)))
    wrap(F, "ice_deposition_sublimation", lambda a, k, o: (
        rec("ice: sublimation", o[1] > 0), rec("ice: vapour deposition", o[0] > 0),
        rec("ice: Bergeron", o[3] > 0)))
    wrap(F, "evaporate_rain", lambda a, k, o: rec("rain: evaporation", o[0] > 0))
    wrap(F, "cloud_water_autoconversion", lambda a, k, o: rec("warm rain: autoconversion", o[0] > 0))
    wrap(F, "cloud_rain_accretion", lambda a, k, o: rec("warm rain: accretion", o[0] > 0))
    wrap(F, "rain_self_collection", lambda a, k, o: rec("rain: self collection", o > 0))
    wrap(F, "cldliq_immersion_freezing", lambda a, k, o: rec("liquid: immersion freezing", o[0] > 0))
    wrap(F, "rain_immersion_freezing", lambda a, k, o: rec("rain: immersion freezing", o[0] > 0))
    wrap(F, "ice_cldliq_collection", lambda a, k, o: rec("ice: collects cloud water", o[0] > 0))
    wrap(F, "ice_rain_collection", lambda a, k, o: rec("ice: collects rain", o[0] > 0))
    wrap(F, "ice_self_collection", lambda a, k, o: rec("ice: self collection", o > 0))
    wrap(F, "cloud_water_conservation", lambda a, k, o: rec(
        "conservation: cloud water limited",
        (o["qc2qr_autoconv_tend"] != a[2]["qc2qr_autoconv_tend"])))
    wrap(F, "rain_water_conservation", lambda a, k, o: rec(
        "conservation: rain water limited", o[0] != a[6]))
    wrap(F, "ice_water_conservation", lambda a, k, o: rec(
        "conservation: ice water limited", o[0] != a[12]))
    wrap(F, "ice_supersat_conservation", lambda a, k, o: rec(
        "conservation: ice supersaturation limited", o[0] != a[0]))
    wrap(F, "prevent_liq_supersaturation", lambda a, k, o: rec(
        "conservation: liquid supersaturation prevented", o[1] != a[7]))
    wrap(F, "impose_max_total_ni", lambda a, k, o: rec("ice: max total ni imposed", o != a[0]))
    wrap(F, "calc_rime_density", lambda a, k, o: rec(
        "ice: rime density computed", (a[7] >= C.QSMALL) & (a[0] < C.T_zerodegc)))
    wrap(S, "calc_bulk_rho_rime", lambda a, k, o: rec("ice: bulk rime density", a[2] >= C.BSMALL))
    wrap(S, "homogeneous_freezing", lambda a, k, o: rec(
        "homogeneous freezing (T < 233 K, liquid)",
        (a[0] < C.T_homogfrz) & ((a[2] >= C.QSMALL) | (a[4] >= C.QSMALL))))
    for name, idx in (("cloud", 0), ("rain", 0), ("ice", 0)):
        pass
    for fn in ("cloud_sedimentation", "rain_sedimentation", "ice_sedimentation"):
        o = getattr(S, fn)

        def mk(o, fn):
            def f(*a, **k):
                out = o(*a, **k)
                rec(f"sedimentation: {fn.split('_')[0]} active (column)", ~out[-1] & _present(out))
                return out
            return f

        def _present(out):
            return torch.ones_like(out[-1])
        setattr(S, fn, mk(o, fn))


def main() -> None:
    install()
    ncalls = 0
    from eamxx_torch.p3 import load_p3_tables, p3_run
    cache = {}
    with mathfn.use_backend("libm"):
        for run in map(Path, sys.argv[1:]):
            st = rc.case_settings(run)
            ps = st["p3"]
            tb = cache.setdefault(ps["tables_dir"], load_p3_tables(ps["tables_dir"]))
            for pj in sorted((run / "dump" / "p3").glob("*_c0_*_pre.json")):
                stem = str(pj.with_suffix(""))[:-3]
                _, pre = read_dump(stem + "pre")
                mpost, _ = read_dump(stem + "post")
                state = {k: torch.as_tensor(v, dtype=torch.float64) for k, v in pre.items()}
                p3_run(state, float(mpost["dt"]), tb, ps["runtime"], ps["predictNc"],
                       ps["prescribedCCN"])
                ncalls += 1
    print(f"P3 branch coverage over {ncalls} calls (count taken / count evaluated):")
    for k, (t, n) in COV.items():
        print(f"  {k:52s} {t:9d} / {n}{'   <-- never taken' if t == 0 else ''}")


if __name__ == "__main__":
    main()
