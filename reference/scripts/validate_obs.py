#!/usr/bin/env python3
"""Suite-level comparison with observations and LES (docs/validation.md).

  python reference/scripts/validate_obs.py run RUN_DIR [--math libm|torch] [--out NPZ]
  python reference/scripts/validate_obs.py score [NPZ ...] [--json OUT]
  python reference/scripts/validate_obs.py dp CASE DP_HORIZ_AVG_FILE     (context: DP-EAMxx)

run    Runs the port's single-column model for a run directory made by make_scm_case.py
       and saves the diagnostics the criteria need (per-step scalars, hourly-mean and
       hourly-snapshot profiles). If the directory holds EAMxx's own output (scm_eamxx),
       every output field is compared after every output step while the port runs.
score  Applies reference/scripts/validation_criteria.py to saved runs. The observations of
       ARM97 and GATE III come from the IOP files.
"""
from __future__ import annotations

import argparse
import datetime as dt_
import glob
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REF = HERE.parent
OUT_DIR = REF / "runs" / "validation"
IOP_DIR = REF / "inputdata" / "atm" / "cam" / "scam" / "iop"

PROFILES = ("T_mid", "qv", "qc", "qr", "qi", "cldfrac_tot", "thetal", "qt", "z_mid", "p_mid",
            "pseudo_density", "u", "v", "tke")
SNAPSHOTS = ("thetal", "qt", "z_mid", "pseudo_density")
LV_PREC = 28.5          # W m-2 per mm day-1 (van Zanten et al. 2011, Table 1)
LV = 2.501e6            # J kg-1, for the latent heat flux from surf_evap (EAMxx LatVap)
TZ = {"cass": -6.0}     # local standard time = UTC + TZ [h]


# ------------------------------------------------------------------------------ run
def load_reference(run: Path):
    import netCDF4
    files = sorted(glob.glob(str(run / "scm_*.INSTANT.*.nc")))
    if not files:
        return None, 0
    out = {}
    with netCDF4.Dataset(files[0]) as d:
        for k, v in d.variables.items():
            if k != "time" and "ncol" in v.dimensions:
                out[k] = np.asarray(v[:], dtype=np.float64)
    every = int(re.search(r"frequency: (\d+)", (run / "output.yaml").read_text()).group(1))
    return out, every


def diagnostics(s: dict, dt: float) -> tuple[dict, dict]:
    """Scalars and profiles of one state (column 0)."""
    import torch
    from eamxx_torch import physics_functions as pf
    from eamxx_torch import constants as C
    T, p, dp, qv = s["T_mid"], s["p_mid"], s["pseudo_density"], s["qv"]
    zero = torch.zeros_like(T)
    s = {**{k: zero for k in ("cldfrac_tot", "tke")}, **s}    # the initial state has no cloud fraction yet
    dz = pf.calculate_dz(dp, p, T, qv)
    z_mid = pf.calculate_z_mid(pf.calculate_z_int(dz))
    theta = pf.calculate_theta_from_T(T, p)
    prof = dict(T_mid=T, qv=qv, qc=s["qc"], qr=s["qr"], qi=s["qi"], cldfrac_tot=s["cldfrac_tot"],
                thetal=pf.calculate_thetal_from_theta(theta, T, s["qc"]), qt=qv + s["qc"] + s["qr"],
                z_mid=z_mid, p_mid=p, pseudo_density=dp, u=s["horiz_winds"][:, 0],
                v=s["horiz_winds"][:, 1], tke=s["tke"])
    prof = {k: v[0].detach().cpu().numpy().astype(np.float64) for k, v in prof.items()}
    g = C.gravit
    precip = (s["precip_liq_surf_mass"] + s["precip_ice_surf_mass"]) / dt       # kg m-2 s-1
    sc = dict(precip=precip, lwp=(s["qc"] * dp).sum(-1) / g, iwp=(s["qi"] * dp).sum(-1) / g,
              pw=(qv * dp).sum(-1) / g, shf=s["surf_sens_flux"], lhf=s["surf_evap"] * LV,
              pbl_height=s["pbl_height"],
              cldtot=s.get("cldtot", torch.full_like(s["pbl_height"], float("nan"))),
              cldlow=s.get("cldlow", torch.full_like(s["pbl_height"], float("nan"))))
    sc["zm_prec"] = s.get("zm_prec", torch.full_like(s["pbl_height"], float("nan")))
    sc = {k: float(v.reshape(-1)[0]) for k, v in sc.items()}
    return sc, prof


def cmd_run(a) -> None:
    import torch
    torch.set_num_threads(1)
    import scm_case
    from eamxx_torch import mathfn
    from scm_compare import port_value
    run = a.run.resolve()
    txt = (run / "input.yaml").read_text()
    nsteps = int(re.search(r"number_of_steps: (\d+)", txt).group(1))
    run_t0 = re.search(r"run_t0: (\S+)", txt).group(1)
    ref, every = load_reference(run) if (a.math == "libm" and not a.smooth and a.dtype == "float64") else (None, 0)
    ctx = mathfn.use_backend(a.math)
    ctx.__enter__()
    scm, s = scm_case.port_scm(run, smooth=a.smooth, dtype=getattr(torch, a.dtype), device=a.device,
                               p3_imm_f64=a.p3_imm_f64)
    if a.mcica_seed is not None:
        import torch as _t
        s = dict(s, mcica_seed=_t.full((s["T_mid"].shape[0],), a.mcica_seed, dtype=_t.int64))
    dt = float(scm.dt)
    nh = int(round(nsteps * dt / 3600.0))
    sc0, p0 = diagnostics(s, dt)
    scal = {k: np.full(nsteps, np.nan) for k in sc0}
    hsum = {k: np.zeros((nh, v.size)) for k, v in p0.items() if k in PROFILES}
    hcnt = np.zeros(nh)
    snap = {k: np.full((nh + 1, p0[k].size), np.nan) for k in SNAPSHOTS}
    for k in SNAPSHOTS:
        snap[k][0] = p0[k]
    worst, first, compared, bad_checks = {}, None, 0, 0
    t0 = time.time()
    for n in range(1, nsteps + 1):
        r = scm.step(s)
        s = r.state
        bad_checks += sum(int(x) == 2 for st in r.status.values() for x in st.values())
        sc, pr = diagnostics(s, dt)
        for k, v in sc.items():
            scal[k][n - 1] = v
        tend = n * dt
        h = int(np.ceil(tend / 3600.0 - 1e-9)) - 1
        for k in hsum:
            hsum[k][h] += pr[k]
        hcnt[h] += 1
        if abs(tend / 3600.0 - round(tend / 3600.0)) < 1e-9:
            for k in SNAPSHOTS:
                snap[k][int(round(tend / 3600.0))] = pr[k]
        if ref is not None and n % every == 0 and n // every < next(iter(ref.values())).shape[0]:
            j = n // every
            compared += 1
            for k, rv in ref.items():
                pv = port_value(k, s)
                if pv is None:
                    continue
                rj = rv[j]
                pv = pv.reshape(rj.shape)
                if not np.array_equal(pv, rj):
                    worst[k] = max(worst.get(k, 0.0), float(np.nanmax(np.abs(pv - rj))))
                    if first is None:
                        first = (n, k)
    elapsed = time.time() - t0
    hmean = {k: v / hcnt[:, None] for k, v in hsum.items()}
    tag = ('_smooth' if a.smooth else '') + ('_f32' if a.dtype == 'float32' else '') + \
        ('_immf64' if a.p3_imm_f64 else '') + \
        ('' if a.device == 'cpu' else '_' + a.device)
    out = a.out or OUT_DIR / f"{run.name}_{a.math}{tag}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    # (zm_prec is NaN throughout when ZM is off: a placeholder, not a failure)
    nonfinite = sorted(k for k, v in list(scal.items()) + list(hmean.items())
                       if not np.all(np.isfinite(np.asarray(v, dtype=np.float64)))
                       and not (k == "zm_prec" and np.all(np.isnan(np.asarray(v, dtype=np.float64)))))
    meta = dict(run=run.name, math=a.math, dt=dt, nsteps=nsteps, run_t0=run_t0,
                lr=bool(re.search(r"\bzm\b", txt)), iop_file=re.search(r"iop_file: (\S+)", txt).group(1),
                bad_checks=bad_checks, elapsed=elapsed, nonfinite=nonfinite,
                ref_compared=compared, ref_first_diff=first, ref_worst=worst)
    np.savez_compressed(out, meta=json.dumps(meta),
                        **{f"s_{k}": v for k, v in scal.items()},
                        **{f"h_{k}": v for k, v in hmean.items()},
                        **{f"n_{k}": v for k, v in snap.items()})
    msg = (f"{run.name} ({a.math}): {nsteps} steps in {elapsed:.0f} s; failed property checks: {bad_checks}; "
           f"non-finite outputs: {len(nonfinite)}" + (f" ({', '.join(nonfinite[:6])}...)" if nonfinite else ""))
    if ref is not None:
        msg += (f"; EAMxx compared at {compared} output steps: "
                + ("MATCH bitwise" if first is None else f"DIFFER first at step {first[0]} ({first[1]})"))
    print(msg)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------- score
class Run:
    def __init__(self, path: Path):
        z = np.load(path, allow_pickle=False)
        self.meta = json.loads(str(z["meta"]))
        self.s = {k[2:]: z[k] for k in z.files if k.startswith("s_")}
        self.h = {k[2:]: z[k] for k in z.files if k.startswith("h_")}
        self.n = {k[2:]: z[k] for k in z.files if k.startswith("n_")}
        self.dt = self.meta["dt"]
        self.tend = self.dt * np.arange(1, self.meta["nsteps"] + 1)      # end of each step [s]
        d, sec = self.meta["run_t0"].rsplit("-", 1)
        self.t0 = dt_.datetime.fromisoformat(d) + dt_.timedelta(seconds=int(sec))

    def steps(self, w):
        return (self.tend > w[0] * 3600.0 + 1e-6) & (self.tend <= w[1] * 3600.0 + 1e-6)

    def smean(self, k, w):
        return float(np.nanmean(self.s[k][self.steps(w)]))

    def pmean(self, k, w):
        return self.h[k][int(w[0]):int(w[1])].mean(axis=0)

    def hourly(self, k):
        nh = self.h["T_mid"].shape[0]
        return np.array([np.nanmean(self.s[k][self.steps((i, i + 1))]) for i in range(nh)])


def to_hours(run: Run, case: str, w):
    """Window in hours after the start; ("lst", a, b) is local standard time."""
    if isinstance(w, tuple) and w[0] == "lst":
        start_lst = run.t0.hour + run.t0.minute / 60.0 + TZ[case]
        return (w[1] - start_lst, w[2] - start_lst)
    return w


def level_top_base(z, cf, thr):
    idx = np.nonzero(cf >= thr)[0]
    if idx.size == 0:
        return float("nan"), float("nan")
    return float(z[idx].max()), float(z[idx].min())


def iop_obs(run: Run):
    import netCDF4
    with netCDF4.Dataset(IOP_DIR / Path(run.meta["iop_file"]).name) as d:
        bdate = dt_.datetime.strptime(str(int(d["bdate"][...])), "%Y%m%d")
        off = (run.t0 - bdate).total_seconds()
        t = np.asarray(d["tsec"][:], dtype=np.float64) - off               # s after run start
        lev = np.asarray(d["lev"][:], dtype=np.float64)                     # Pa
        T = np.ma.filled(d["T"][:, :, 0, 0].astype(np.float64), np.nan)
        q = np.ma.filled(d["q"][:, :, 0, 0].astype(np.float64), np.nan)
        ps = np.ma.filled(d["Ps"][:, 0, 0].astype(np.float64), np.nan)
        prec = np.ma.filled(d["Prec"][:, 0, 0].astype(np.float64), np.nan)
        unit = d["Prec"].units
        extra = {}
        for k in ("totcld", "prew"):
            if k in d.variables:
                extra[k] = np.ma.filled(d[k][:, 0, 0].astype(np.float64), np.nan)
    prec = prec * {"mm/s": 1.0, "m/s": 1000.0}[unit]                        # kg m-2 s-1
    return dict(t=t, lev=lev, T=T, q=q, ps=ps, prec=prec, **extra)


def obs_hourly(obs, i, key):
    m = (obs["t"] > i * 3600.0 - 1e-6) & (obs["t"] <= (i + 1) * 3600.0 + 1e-6)
    return np.nanmean(obs[key][m], axis=0) if m.any() else None


def obs_integral(t, rate, a, b):
    """Integral of a piecewise linear rate over [a, b] seconds."""
    tt = np.linspace(a, b, int((b - a) / 60.0) + 1)
    return float(np.trapezoid(np.interp(tt, t, rate), tt))


def obs_quantities(run: Run, w):
    obs = iop_obs(run)
    hrs = range(int(w[0]), int(w[1]))
    dT, dq, npw, apw = [], [], [], []
    for i in hrs:
        To, qo, pso = obs_hourly(obs, i, "T"), obs_hourly(obs, i, "q"), obs_hourly(obs, i, "ps")
        if To is None:
            continue
        pm = run.h["p_mid"][i]
        ok = (obs["lev"] <= pso) & (obs["lev"] >= 1.0e4) & (obs["lev"] <= pm[-1]) & np.isfinite(To) & np.isfinite(qo)
        lp = np.log(pm)
        Tm = np.interp(np.log(obs["lev"]), lp, run.h["T_mid"][i])
        qm = np.interp(np.log(obs["lev"]), lp, run.h["qv"][i])
        dT.append(np.where(ok, Tm - To, np.nan))
        dq.append(np.where(ok, (qm - qo) * 1e3, np.nan))
        # observed precipitable water from the IOP profile, surface to top
        lev, qq = obs["lev"], qo
        good = (lev <= pso) & np.isfinite(qq)
        pl = np.concatenate([[pso], lev[good][::-1]]) if lev[0] < lev[-1] else np.concatenate([[pso], lev[good]])
        ql = np.concatenate([[qq[good][-1] if lev[0] < lev[-1] else qq[good][0]],
                             qq[good][::-1] if lev[0] < lev[-1] else qq[good]])
        npw.append(float(-np.trapezoid(ql, pl) / 9.80616))
        apw.append(float(np.nanmean(run.s["pw"][run.steps((i, i + 1))])))
    dT, dq = np.nanmean(np.array(dT), axis=0), np.nanmean(np.array(dq), axis=0)
    a, b = w[0] * 3600.0, w[1] * 3600.0
    p_obs = obs_integral(obs["t"], obs["prec"], a, b)
    p_mod = float(np.sum(run.s["precip"][run.steps(w)]) * run.dt)
    out = dict(t_bias_max=float(np.nanmax(np.abs(dT))), q_bias_max=float(np.nanmax(np.abs(dq))),
               precip_rel=(p_mod - p_obs) / p_obs, precip_model_mm=p_mod, precip_obs_mm=p_obs,
               pw_mae=float(np.mean(np.abs(np.array(apw) - np.array(npw)))),
               t_bias_profile=dT, q_bias_profile=dq, lev=obs["lev"])
    if "totcld" in obs:
        m = (obs["t"] > a) & (obs["t"] <= b)
        out["cldtot_obs"] = float(np.nanmean(obs["totcld"][m])) / 100.0
    return out


def quantity(run: Run, case: str, q: str, w):
    if w == "run":
        w = (0, run.h["T_mid"].shape[0])
    w = to_hours(run, case, w)
    if q in ("cldtot", "lwp", "precip", "precip_wm2"):
        v = run.smean({"precip_wm2": "precip"}.get(q, q), w)
        return {"lwp": v * 1e3, "precip": v * 86400.0, "precip_wm2": v * 86400.0 * LV_PREC}.get(q, v)
    if q == "shf_lhf":
        return (run.smean("shf", w), run.smean("lhf", w))
    if q in ("cf_max", "z_cf_max"):
        cf, z = run.pmean("cldfrac_tot", w), run.pmean("z_mid", w)
        return float(cf.max()) if q == "cf_max" else float(z[np.argmax(cf)])
    if q.startswith("z_top_cf") or q.startswith("z_base_cf"):
        thr = {"50": 0.5, "01": 0.01}[q[-2:]]
        top, base = level_top_base(run.pmean("z_mid", w), run.pmean("cldfrac_tot", w), thr)
        return top if q.startswith("z_top") else base
    if q in ("rms_dthl", "rms_dqt"):
        k, scale = ("thetal", 1.0) if q == "rms_dthl" else ("qt", 1e3)
        i0, i1 = int(w[0]), int(w[1])
        tend = (run.n[k][i1] - run.n[k][i0]) * scale / ((i1 - i0) / 24.0)
        z, dp = run.n["z_mid"][i1], run.n["pseudo_density"][i1]
        m = z <= 2500.0
        return float(np.sqrt(np.sum(tend[m] ** 2 * dp[m]) / np.sum(dp[m])))
    if q in ("cldtot_hmax", "lwp_hmax", "cldtot_hmax_time"):
        hv = run.hourly("cldtot" if q.startswith("cldtot") else "lwp")
        i = int(np.nanargmax(hv))
        if q == "cldtot_hmax_time":
            return run.t0.hour + TZ.get(case, 0.0) + i + 0.5
        return float(hv[i]) * (1e3 if q == "lwp_hmax" else 1.0)
    if q == "pbl_height":
        return tuple(run.smean("pbl_height", (h - 0.5, h + 0.5)) for h in w)
    raise KeyError(q)


def fmt(v):
    if isinstance(v, tuple):
        return ", ".join(fmt(x) for x in v)
    if isinstance(v, float):
        return f"{v:.3g}"
    return str(v)


def cmd_score(a) -> None:
    import validation_criteria as vc
    crit_file = HERE / "validation_criteria.py"
    sha = hashlib.sha256(crit_file.read_bytes()).hexdigest()
    files = [Path(f) for f in (a.files or sorted(glob.glob(str(OUT_DIR / "val_*.npz"))))]
    runs = {}
    for f in files:
        r = Run(f)
        runs[(r.meta["run"], r.meta["math"])] = r
    print(f"criteria: {crit_file.name} sha256 {sha[:16]}...")
    results = []
    for case, c in vc.CRITERIA.items():
        for cfg in vc.CONFIGS:
            name = f"val_{case}" + ("_lr" if cfg == "lr" else "")
            for math in ("libm", "torch"):
                r = runs.get((name, math))
                if r is None:
                    continue
                obs = None
                for cid, q, w, lo, hi, src in c["checks"]:
                    if q in ("precip_rel", "t_bias_max", "q_bias_max"):
                        obs = obs or obs_quantities(r, to_hours(r, case, w))
                        v = obs[q]
                    else:
                        v = quantity(r, case, q, w)
                    ok = bool(lo <= v <= hi) if np.isfinite(v) else False
                    results.append(dict(case=case, config=cfg, math=math, id=cid, quantity=q,
                                        value=v, lo=lo, hi=hi, passed=ok))
                for cid, q, w, ctx in c["reported"]:
                    if q in ("pw_mae",):
                        obs = obs or obs_quantities(r, to_hours(r, case, w))
                        v = obs[q]
                    else:
                        v = quantity(r, case, q, w)
                    if q == "cldtot" and obs is not None and "cldtot_obs" in obs:
                        v = (v, obs["cldtot_obs"])
                    results.append(dict(case=case, config=cfg, math=math, id=cid, quantity=q,
                                        value=v, passed=None))
                if obs is not None:
                    results.append(dict(case=case, config=cfg, math=math, id="obs", quantity="precip_mm",
                                        value=(obs["precip_model_mm"], obs["precip_obs_mm"]), passed=None))
    # table: one row per (id, config), libm and torch side by side
    print(f"{'id':5s} {'config':8s} {'quantity':17s} {'band':>17s} {'libm':>14s} {'torch':>14s}  verdict")
    keys = []
    for rr in results:
        k = (rr["case"], rr["id"], rr["config"])
        if k not in keys:
            keys.append(k)
    for k in keys:
        rows = {rr["math"]: rr for rr in results if (rr["case"], rr["id"], rr["config"]) == k}
        any_ = next(iter(rows.values()))
        band = f"{any_['lo']:.3g}..{any_['hi']:.3g}" if "lo" in any_ else "reported"
        vals = {m: fmt(rows[m]["value"]) if m in rows else "-" for m in ("libm", "torch")}
        verd = [rows[m]["passed"] for m in ("libm", "torch") if m in rows]
        if any_["passed"] is None:
            v = ""
        elif all(verd):
            v = "PASS"
        elif not any(verd):
            v = "FAIL"
        else:
            v = "MARGINAL"
        print(f"{k[1]:5s} {k[2]:8s} {any_['quantity']:17s} {band:>17s} {vals['libm']:>14s} {vals['torch']:>14s}  {v}")
    if a.json:
        def clean(o):
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (np.floating,)):
                return float(o)
            return o
        Path(a.json).write_text(json.dumps(dict(criteria_sha256=sha, results=results), default=clean, indent=1))
        print(f"wrote {a.json}")


def cmd_dp(a) -> None:
    """Context only: the same profile quantities from a DP-EAMxx run's 15-minute horizontal
    averages (the configuration with the dycore). No cldtot in that output."""
    import netCDF4
    import validation_criteria as vc
    c = vc.CRITERIA[a.case]
    with netCDF4.Dataset(a.file) as d:
        t_end = np.asarray(d["time_bnds"][:, 1], dtype=np.float64) * 24.0     # hours
        g = lambda k: np.asarray(d[k + "_horiz_avg"][:], dtype=np.float64)  # noqa: E731
        z, cf, lwp = g("z_mid"), g("cldfrac_tot"), g("LiqWaterPath")
        thl, qt, p = g("LiqPotentialTemperature"), g("qv") + g("qc") + g("qr"), g("p_mid")
    print(f"DP-EAMxx {Path(a.file).parent.name} ({a.case}), context only")
    for cid, q, w, *rest in c["checks"] + c["reported"]:
        m = (t_end > w[0] + 1e-6) & (t_end <= w[1] + 1e-6) if isinstance(w, tuple) and w[0] != "lst" else None
        if m is None:
            continue
        zc, cfm = z[m].mean(0), cf[m].mean(0)
        if q == "cf_max":
            v = cfm.max()
        elif q == "z_cf_max":
            v = zc[np.argmax(cfm)]
        elif q == "lwp":
            v = lwp[m].mean() * 1e3
        elif q.startswith("z_top_cf") or q.startswith("z_base_cf"):
            top, base = level_top_base(zc, cfm, {"50": 0.5, "01": 0.01}[q[-2:]])
            v = top if q.startswith("z_top") else base
        elif q in ("rms_dthl", "rms_dqt"):
            x, sc = (thl, 1.0) if q == "rms_dthl" else (qt, 1e3)
            i0, i1 = np.argmin(abs(t_end - w[0])), np.argmin(abs(t_end - w[1]))
            tend = (x[i1] - x[i0]) * sc / ((t_end[i1] - t_end[i0]) / 24.0)
            wgt = np.abs(np.gradient(p[i1]))
            k = z[i1] <= 2500.0
            v = np.sqrt(np.sum(tend[k] ** 2 * wgt[k]) / np.sum(wgt[k]))
        else:
            continue
        print(f"  {cid:5s} {q:12s} {v:10.3g}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    dpp = sub.add_parser("dp")
    dpp.add_argument("case")
    dpp.add_argument("file")
    r = sub.add_parser("run")
    r.add_argument("run", type=Path)
    r.add_argument("--math", choices=["libm", "torch"], default="libm")
    r.add_argument("--out", type=Path)
    r.add_argument("--smooth", action="store_true", help="gradient options on (not EAMxx)")
    r.add_argument("--mcica-seed", type=int, default=None, help="with --smooth: McICA column seed")
    r.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    r.add_argument("--device", default="cpu")
    r.add_argument("--p3-imm-f64", action="store_true",
                   help="float32: P3 immersion freezing in float64 (option, not EAMxx)")
    s = sub.add_parser("score")
    s.add_argument("files", nargs="*")
    s.add_argument("--json")
    a = ap.parse_args()
    {"run": cmd_run, "score": cmd_score, "dp": cmd_dp}[a.cmd](a)


if __name__ == "__main__":
    main()
