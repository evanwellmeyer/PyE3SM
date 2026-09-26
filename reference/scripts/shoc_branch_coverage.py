#!/usr/bin/env python3
"""Count how often each SHOC branch/limiter is taken when replaying reference dumps.

  python reference/scripts/shoc_branch_coverage.py RUN_DIR [RUN_DIR ...]

A branch that is never taken in any replayed call is not validated by the
replay. The probes wrap port functions (the port matches the reference
bitwise on these calls, so its masks are the reference's).
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import replay_check as rc  # noqa: E402
from read_dump import read_dump  # noqa: E402

from eamxx_torch import mathfn  # noqa: E402
from eamxx_torch.shoc import functions as F  # noqa: E402
from eamxx_torch.shoc import interface as I  # noqa: E402
from eamxx_torch.shoc.constants import SC  # noqa: E402

COV: dict[str, list[int]] = defaultdict(lambda: [0, 0])


def rec(name, mask):
    COV[name][0] += int(mask.sum())
    COV[name][1] += mask.numel()


def wrap(mod, fname, probe):
    orig = getattr(mod, fname)

    def w(*a, **k):
        out = orig(*a, **k)
        probe(a, out)
        return out
    setattr(mod, fname, w)


def install():
    wrap(I, "preprocess", lambda a, o: (
        rec("preprocess: surface stress small (|tau|<1e-12)",
            (a[8][:, 0].abs() < 1e-12) & (a[8][:, 1].abs() < 1e-12)),))
    wrap(I, "apply_turbulent_mountain_stress", lambda a, o: rec("apply_tms: drag coefficient > 0", a[2] > 0))
    wrap(F, "shoc_diag_obklen", lambda a, o: (rec("obklen: kbfs >= 0", o[1] >= 0),
                                              rec("obklen: ustar at ustar_min", o[0] == SC.ustar_min)))
    wrap(F, "pblintd_height", lambda a, o: rec("pblintd_height: Ri >= 0.3 found", a[9] & ~o[2]))
    wrap(F, "pblintd_surf_temp", lambda a, o: rec("pblintd_surf_temp: kbfs > 0", o[2]))
    wrap(F, "check_length_scale_shoc_length", lambda a, o: (
        rec("length: at minlen", o == SC.minlen),
        rec("length: at sqrt(dx*dy)", o == torch.sqrt(a[0] * a[1])[:, None]),
        rec("length: at maxlen (before check)", a[2] == SC.maxlen)))
    wrap(F, "adv_sgs_tke", lambda a, o: (rec("tke: at mintke", o[0] == SC.mintke),
                                         rec("tke: at maxtke", o[0] == SC.maxtke)))
    wrap(F, "isotropic_ts", lambda a, o: (rec("isotropy: brunt <= 0", a[7] <= 0),
                                          rec("isotropy: at maxiso", o == SC.maxiso)))
    wrap(F, "eddy_diffusivities", lambda a, o: rec(
        "eddy_diff: stable-PBL fallback (T_sfc < 182 K)",
        (a[3] < (a[2] + 200)[:, None]) & (a[4][:, -1:] < 182)))
    wrap(F, "update_prognostics_implicit", lambda a, o: rec(
        "implicit: surface stress > 0 (ksrf above ksrfmin)", (a[8] != 0) | (a[9] != 0)))
    wrap(F, "shoc_diag_second_moments_srf", lambda a, o: rec("second moments: wthl_sfc >= 0 (wstar > 0)", o[1] > 0))
    wrap(F, "clipping_diag_third_shoc_moments", lambda a, o: rec("w3: clipped", o != a[1]))
    wrap(F, "shoc_assumed_pdf_vv_parameters", lambda a, o: (
        rec("pdf: w_sec > w_tol_sqd", a[1] > a[3]), rec("pdf: a clipped to [0.01, 0.99]", (o[5] == 0.01) | (o[5] == 0.99))))
    wrap(F, "shoc_assumed_pdf_compute_s", lambda a, o: (
        rec("pdf: std_s small", ~(o[1] > F._STD_S_SMALL)), rec("pdf: qn <= 0 reset", o[2] == 0),
        rec("pdf: partial cloud 0<C<1", (o[3] > 0) & (o[3] < 1))))
    wrap(F, "shoc_assumed_pdf", lambda a, o: rec("pdf: Tl1 <= 100 K clipped", o["tl1_small"]))
    wrap(F, "shoc_energy_fixer", lambda a, o: None)
    wrap(I, "postprocess", lambda a, o: (
        rec("post: inv_qc_relvar computed", (a[1] != 0) & (a[4] != 0)),
        rec("post: inv_qc_relvar at 10", o["inv_qc_relvar"] == 10),
        rec("post: inv_qc_relvar at 0.001", o["inv_qc_relvar"] == 0.001)))


def main() -> None:
    install()
    ncalls = 0
    with mathfn.use_backend("libm"):
        for run in map(Path, sys.argv[1:]):
            st = rc.case_settings(run)
            for pj in sorted((run / "dump" / "shoc").glob("*_c0_*_pre.json")):
                stem = str(pj.with_suffix(""))[:-3]
                _, pre = read_dump(stem + "pre")
                mpost, _ = read_dump(stem + "post")
                rc.run_shoc(pre, {"dt": mpost["dt"], "time_end": mpost["time"]}, st, False)
                ncalls += 1
    print(f"SHOC branch coverage over {ncalls} calls (count taken / count evaluated):")
    for k, (t, n) in COV.items():
        print(f"  {k:52s} {t:8d} / {n}{'   <-- never taken' if t == 0 else ''}")


if __name__ == "__main__":
    main()
