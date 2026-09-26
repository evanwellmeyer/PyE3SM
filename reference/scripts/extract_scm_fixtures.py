#!/usr/bin/env python3
"""Write the small fixtures tests/test_scm.py uses.

  python reference/scripts/extract_scm_fixtures.py [OUTDIR]   (default tests/data/scm)

Fixtures (float64 .npz with a JSON "meta"):
  iop_<case>_s<step>_r<rank>   iop_forcing pre/post (the CASS one holds all 100 columns,
                               only the fields nudging needs, so the domain means can be
                               recomputed)
  import_<case>_s<step>        SurfaceCouplingImporter post (the four IOP fields)
  pressure_<case>_s<step>      HOMME's post-process pressures
  suite_<case>_s<step>_r<rank> the state HOMME handed to physics ("start/") and the state
                               at the end of the physics step ("end/": each field as the
                               last process that writes it left it, after its checks)
The IOP files and data tables stay in reference/inputdata.
"""
from __future__ import annotations

import dataclasses
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import replay_check as rc  # noqa: E402
from read_dump import read_dump  # noqa: E402

RUNS = Path(__file__).resolve().parents[1] / "runs"
TEND = re.compile(r"^(iop_forcing|shoc|p3|rrtmgp|homme)_\w+_tend$")


def is_proc_tend(name: str) -> bool:
    """A tendency the base class computes (compute_tendencies), e.g. p3_T_mid_tend."""
    return TEND.match(name) is not None


ORDER = ("iop_forcing", "tms", "shoc", "cld_fraction", "spa", "p3", "rrtmgp")


def iop_meta(run: Path, step: int) -> dict:
    st = rc.case_settings(run)
    io = st["iop"]
    return {"iop": dataclasses.asdict(io["params"]) | {"iop_file": Path(io["params"].iop_file).name},
            "run_t0": io["run_t0"], "dt": io["dt"], "num_global_cols": io["num_global_cols"],
            "step": step, "case": run.name}


def save(path: Path, meta: dict, arrays: dict) -> None:
    np.savez_compressed(path, meta=np.array(json.dumps(meta)), **arrays)
    print(f"{path.name}: {path.stat().st_size / 1e3:.0f} kB")


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tests/data/scm")
    out.mkdir(parents=True, exist_ok=True)
    for run, step, rank, fields in (
            ("dumps_strict_bomex", 108, 0, None),
            ("dumps_strict_arm97_7p5d", 4752, 1, None),
            ("dumps_strict_cass_allcols", 217, 0, ("T_mid", "qv", "horiz_winds", "ps"))):
        d = RUNS / run / "dump" / "iop_forcing"
        _, pre = read_dump(d / f"s{step}_c0_r{rank}_pre", fields)
        _, post = read_dump(d / f"s{step}_c0_r{rank}_post", fields)
        post = {k: v for k, v in post.items() if k != "ps" and not is_proc_tend(k)}
        save(out / f"iop_{run.removeprefix('dumps_strict_')}_s{step}_r{rank}.npz",
             iop_meta(RUNS / run, step),
             {**{"pre/" + k: v for k, v in pre.items() if not is_proc_tend(k)},
              **{"post/" + k: v for k, v in post.items()}})
    for run, step in (("dumps_strict_cass", 54), ("dumps_strict_arm97_7p5d", 4752)):
        _, post = read_dump(RUNS / run / "dump" / "SurfaceCouplingImporter" / f"s{step}_c0_r0_post",
                            ("surf_evap", "surf_sens_flux", "surf_radiative_T", "surf_lw_flux_up"))
        save(out / f"import_{run.removeprefix('dumps_strict_')}_s{step}.npz", iop_meta(RUNS / run, step),
             {"post/" + k: v for k, v in post.items()})
    names = ["pseudo_density", "qv", "p_int", "p_mid", "pseudo_density_dry", "p_dry_int", "p_dry_mid"]
    _, post = read_dump(RUNS / "dumps_strict_arm97_7p5d" / "dump" / "homme" / "s4752_c0_r0_post", names)
    save(out / "pressure_arm97_7p5d_s4752.npz", {"ptop_hyai0": True}, {"post/" + k: v for k, v in post.items()})

    run, step, rank = RUNS / "dumps_strict_dycomsrf01", 1, 0
    start, end = {}, {}
    for proc in ORDER:
        stem = run / "dump" / proc / f"s{step}_c-1_r{rank}_"
        _, a = read_dump(str(stem) + "prechk")
        _, b = read_dump(str(stem) + "postchk")
        for k, v in a.items():
            if not is_proc_tend(k):
                start.setdefault(k, v)
        for k, v in b.items():
            if not is_proc_tend(k):
                end[k] = v          # later processes overwrite: the state at the end of the step
    st = rc.case_settings(run)
    meta = iop_meta(run, step) | {
        "p3": {"runtime": dataclasses.asdict(st["p3"]["runtime"]), "predictNc": st["p3"]["predictNc"],
               "prescribedCCN": st["p3"]["prescribedCCN"]},
        "rrtmgp": {"runtime": dataclasses.asdict(st["rrtmgp"]["runtime"]),
                   "files": [Path(f).name for f in st["rrtmgp"]["files"]]},
        "spa_data_file": Path(st["spa_data_file"]).name, "cell_length": st["cell_length"],
        "lat": st["lat"], "lon": st["lon"], "extra_shoc_diags": st["extra_shoc_diags"]}
    save(out / f"suite_dycomsrf01_s{step}_r{rank}.npz", meta,
         {**{"start/" + k: v for k, v in start.items()}, **{"end/" + k: v for k, v in end.items()}})


TRAJ_FIELDS = ("T_mid", "qv", "qc", "qr", "qi", "horiz_winds", "tke", "cldfrac_tot",
               "precip_liq_surf_mass", "surf_sens_flux", "surf_evap", "surf_mom_flux",
               "SW_flux_dn", "LW_flux_up", "zm_prec", "zm_activity")


def trajectory_fixture(run: Path, nsteps: int, out: Path) -> None:
    """scm_<case>: the port's initial state and settings for a scm_eamxx run, and EAMxx's
    trajectory of a few fields for the first nsteps steps (make_scm_case.py, scm_eamxx)."""
    import glob
    import netCDF4
    import scm_case
    import torch  # noqa: F401
    scm, state = scm_case.port_scm(run)
    cfg = scm.suite.config
    st = rc.case_settings(run)
    io = st["iop"]
    meta = {"case": run.name, "run_t0": io["run_t0"], "dt": io["dt"], "nsteps": nsteps,
            "iop": dataclasses.asdict(io["params"]) | {"iop_file": Path(io["params"].iop_file).name},
            "cell_length": float(scm.grid.cell_length[0]),
            "p3": {"runtime": dataclasses.asdict(cfg.p3), "predictNc": cfg.p3_predict_nc,
                   "prescribedCCN": cfg.p3_prescribed_ccn},
            "rrtmgp": {"runtime": dataclasses.asdict(cfg.rrtmgp),
                       "files": [Path(f).name for f in cfg.rrtmgp_files]},
            "spa_data_file": Path(cfg.spa_data_file).name, "extra_shoc_diags": cfg.shoc.extra_diags,
            "zm": dataclasses.asdict(cfg.zm) if cfg.zm else None,
            "mac_aero_mic_subcycles": cfg.mac_aero_mic_subcycles}
    f = sorted(glob.glob(str(run / "scm_*.INSTANT.*.nc")))[0]
    arrays = {"init/" + k: v.numpy() for k, v in state.items()}
    with netCDF4.Dataset(f) as d:
        for k in TRAJ_FIELDS:
            if k in d.variables:
                arrays["ref/" + k] = np.asarray(d[k][1:nsteps + 1], dtype=np.float64)
    save(out / f"scm_{run.name.removeprefix('scm_')}.npz", meta, arrays)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "trajectory":
        # extract_scm_fixtures.py trajectory RUN_DIR NSTEPS [OUTDIR]
        trajectory_fixture(Path(sys.argv[2]), int(sys.argv[3]),
                           Path(sys.argv[4]) if len(sys.argv) > 4 else Path("tests/data/scm"))
    else:
        main()
