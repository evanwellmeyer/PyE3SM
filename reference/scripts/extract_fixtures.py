#!/usr/bin/env python3
"""Convert raw per-process dumps into compact .npz replay fixtures.

  python reference/scripts/extract_fixtures.py PROC OUTDIR RUN_DIR [RUN_DIR ...]
      [--steps 1,108] [--ranks 0] [--fields a,b,c] [--cols N]

Each RUN_DIR is a reference/runs/dumps_<case> directory. One .npz is written per
process call: <OUTDIR>/<case>_s<step>_c<subcycle>_r<rank>.npz, holding
"pre/<field>", "post/<field>" (float64) and "meta" (JSON). Only numpy needed.
--cols N keeps N columns of each call (for zm: up to N/2 active and the rest
inactive, from zm_activity), which is exact for column-independent schemes (ZM's
replay on column subsets is bitwise, see docs/status.md).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from read_dump import read_dump  # noqa: E402


RRTMGP_INPUTS = {"p_mid", "p_int", "pseudo_density", "T_mid", "qv", "qc", "nc", "qi",
                 "cldfrac_tot", "eff_radius_qc", "eff_radius_qi", "surf_lw_flux_up",
                 "sfc_alb_dir_vis", "sfc_alb_dir_nir", "sfc_alb_dif_vis", "sfc_alb_dif_nir",
                 "o3_volume_mix_ratio", "aero_tau_sw", "aero_ssa_sw", "aero_g_sw", "aero_tau_lw",
                 "rad_heating_pdel"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("proc")
    ap.add_argument("outdir", type=Path)
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("--steps", help="comma list; default all dumped")
    ap.add_argument("--ranks", help="comma list; default all dumped")
    ap.add_argument("--fields", help="comma list; default all fields of the call")
    ap.add_argument("--cols", type=int, help="keep this many columns per call")
    a = ap.parse_args()
    steps = set(map(int, a.steps.split(","))) if a.steps else None
    ranks = set(map(int, a.ranks.split(","))) if a.ranks else None
    fields = set(a.fields.split(",")) if a.fields else None
    a.outdir.mkdir(parents=True, exist_ok=True)

    n = 0
    for run in a.run_dirs:
        variant, _, case = run.name.removeprefix("dumps_").partition("_")
        for pre_json in sorted((run / "dump" / a.proc).glob("*_pre.json")):
            m = re.match(r"s(\d+)_c(-?\d+)_r(\d+)_pre", pre_json.stem)
            step, sub, rank = map(int, m.groups())
            if steps is not None and step not in steps:
                continue
            if ranks is not None and rank not in ranks:
                continue
            stem = pre_json.with_suffix("")
            mpre, pre = read_dump(stem)
            mpost, post = read_dump(str(stem).replace("_pre", "_post"))
            keep = [f for f in mpre["fields"] if fields is None or f["name"] in fields]
            meta = {
                "proc": a.proc, "case": case, "variant": variant, "step": step, "subcycle": sub, "rank": rank,
                "dt": mpre["dt"], "time_start": mpre["time"], "time_end": mpost["time"],
                "fields": {f["name"]: {"units": f["units"], "dims": f["dims"]} for f in keep},
                "source": f"{run.name}/dump/{a.proc}/{stem.name}",
            }
            if a.proc == "p3":
                # P3 needs its runtime options; the lookup tables stay in reference/inputdata
                from replay_check import p3_settings
                ps = p3_settings(run)
                meta["p3"] = {"runtime": dataclasses.asdict(ps["runtime"]),
                              "predictNc": ps["predictNc"], "prescribedCCN": ps["prescribedCCN"]}
            if a.proc == "rrtmgp":
                # Runtime options and each column's latitude/longitude (grid geometry, not a
                # field). The coefficient files stay in reference/inputdata.
                from replay_check import case_settings
                st = case_settings(run)
                rs = st["rrtmgp"]
                ncol = pre["T_mid"].shape[0]
                if st["lat"] is not None:
                    lat, lon = [st["lat"]] * ncol, [st["lon"]] * ncol
                else:
                    lat, lon = rs["lat"][:ncol].tolist(), rs["lon"][:ncol].tolist()
                meta["rrtmgp"] = {"runtime": dataclasses.asdict(rs["runtime"]),
                                  "files": [Path(f).name for f in rs["files"]],
                                  "lat": lat, "lon": lon}
            if a.proc == "zm":
                # Runtime options and the hybrid coefficients (limcnv comes from them)
                from replay_check import case_settings, load_hybrid_cached
                st = case_settings(run)
                hyb = load_hybrid_cached(st["vcoord_file"])
                meta["zm"] = {"runtime": dataclasses.asdict(st["zm"]),
                              "hyai": hyb["hyai"].tolist(), "hybi": hyb["hybi"].tolist()}
            if a.cols:
                ncol = pre["T_mid"].shape[0]
                if a.proc == "zm":
                    act = np.nonzero(post["zm_activity"] > 0)[0]
                    ina = np.nonzero(post["zm_activity"] <= 0)[0]
                    na = min(len(act), (a.cols + 1) // 2)
                    cols = np.sort(np.concatenate([act[:na], ina[:a.cols - na]]))
                else:
                    cols = np.arange(min(a.cols, ncol))
                meta["columns"] = cols.tolist()
                pre = {k: v[cols] for k, v in pre.items()}
                post = {k: v[cols] for k, v in post.items()}
            if a.proc == "shoc":
                # Grid geometry SHOC needs that is not a field (see replay_check.shoc_cell_length)
                from replay_check import shoc_cell_length
                meta["cell_length"] = shoc_cell_length(run)
            pre_keep, post_keep = keep, keep
            if a.proc == "rrtmgp":
                # Only what a replay reads (inputs) and compares (outputs): halves the size
                pre_keep = [f for f in keep if f["name"] in RRTMGP_INPUTS]
                post_keep = [f for f in keep
                             if f["name"] not in RRTMGP_INPUTS - {"T_mid", "rad_heating_pdel"}]
            arrays = {f"pre/{f['name']}": pre[f["name"]] for f in pre_keep}
            arrays.update({f"post/{f['name']}": post[f["name"]] for f in post_keep})
            out = a.outdir / f"{variant}_{case}_s{step}_c{sub}_r{rank}.npz"
            np.savez_compressed(out, meta=np.array(json.dumps(meta)), **arrays)
            n += 1
    print(f"wrote {n} fixtures to {a.outdir}")


if __name__ == "__main__":
    main()
