"""SHOC port vs EAMxx SHOC (process run_impl), E3SM 80fe0a71.

Reference: DP-EAMxx L128 runs of the strict build (-ffp-contract=off, pack
size 1), dumped around SHOCMacrophysics::run_impl (see docs/plan.md). The
fixtures are 6 calls chosen to cover shallow cumulus (BOMEX), stratocumulus
(DYCOMS-II RF01), a cold start and day/night continental boundary layers
(ARM97), the dx cap on the mixing length, the minimum mixing length, and
inv_qc_relvar at its clip. The DP cases never have surface stress or TMS drag
(ocean-like, prescribed heat/moisture fluxes only), so one more fixture comes
from a standalone tms+shoc reference run on 8 ARM97 columns with synthetic
stress, land fraction and a <182 K surface layer
(reference/scripts/make_standalone_case.py shoc_tms). The full set (143
calls) is replayed by reference/scripts/replay_check.py; branch coverage is
reported by reference/scripts/shoc_branch_coverage.py.

Tolerances:
  * libm math backend: bitwise.
  * torch math backend: torch's exp/log/pow/erf/tanh differ from libm by ~1 ulp.
    Normwise (max|port-ref| / max|ref| per field) errors are <= 2.2e-12 over
    all 140 calls. They are largest where SHOC subtracts nearly equal numbers:
    um/vm_pert_diff are differences of two implicit wind solves. The bound
    used is 1e-9.
    Exception: inv_qc_relvar = clip(qc^2/qc2, 0.001, 10), or 1 where qc2 == 0.
    In fully cloudy cells the variance qc2 = a*(s1*ql1 + ...) - ql^2 cancels
    to roundoff (qc2/qc^2 ~ 1e-16 in the reference itself), so a 1-ulp change
    flips inv_qc_relvar between 1 and 10. Such cells, identified by
    qc2 <= 64*eps*qc^2 in either backend, are excluded for that field only.
Budgets: column water and energy change by exactly the surface fluxes, and the
other tracers are conserved, to roundoff of the 128-term column sums.
"""
from pathlib import Path

import numpy as np
import pytest
import torch

from eamxx_torch import budgets, mathfn
from eamxx_torch.shoc import SHOCRuntime, npbl_from_hybrid, postconditions, shoc_run
from eamxx_torch.testing import compare, list_calls, load_call

DATA = Path(__file__).parent / "data" / "shoc"
FIXTURES = [p for p in list_calls(DATA) if p.name != "l128_hybrid.npz"]
TRACERS = ("bm", "nc", "ni", "nr", "qi", "qm", "qr")
EPS = float(np.finfo(np.float64).eps)


def _npbl():
    z = np.load(DATA / "l128_hybrid.npz")
    return npbl_from_hybrid(z["hyam"], z["hybm"])


NPBL = _npbl()


def _run(call):
    state = {k: torch.as_tensor(v, dtype=torch.float64) for k, v in call.pre.items()}
    ncol = state["T_mid"].shape[0]
    # dx = dy, grid geometry logged by the instrumented reference build (patch 0004)
    cell = torch.full((ncol,), call.meta["cell_length"], dtype=torch.float64)
    out = shoc_run(state, float(call.meta["dt"]), cell, NPBL, SHOCRuntime(extra_diags=True),
                   apply_tms=True, tracer_names=tuple(n for n in TRACERS if n in state))
    internal = out.pop("_internal")
    return out, internal


@pytest.mark.parametrize("step", [1, 3])
def test_synthetic_fixture_exercises_stress_and_tms(step):
    call = load_call(DATA / f"standalone_strict_shoc_tms_s{step}_c0_r0.npz")
    tau = call.pre["surf_mom_flux"]
    small = (np.abs(tau[:, 0]) < 1e-12) & (np.abs(tau[:, 1]) < 1e-12)
    assert small.any() and (~small).sum() >= 5
    assert (call.pre["surf_drag_coeff_tms"] > 0).sum() >= 5
    if step == 1:
        # SHOC's stable-PBL fallback mixes this layer above 182 K within one step.
        assert (call.pre["T_mid"][:, -1] < 182).any()


def test_npbl_l128():
    # 56 of the 128 reference-pressure midpoints are at or below 400 hPa.
    assert NPBL == 56


@pytest.mark.reference
@pytest.mark.parametrize("path", FIXTURES, ids=[p.stem for p in FIXTURES])
def test_replay_libm_bitwise(path):
    call = load_call(path)
    with mathfn.use_backend("libm"):
        out, _ = _run(call)
    assert set(out) <= set(call.post)
    for name, val in out.items():
        d = compare(name, val, call.post[name])
        assert d.ok(), d
    checked, status = postconditions(out)
    assert all(int(s) == 0 for s in status.values())


@pytest.mark.reference
@pytest.mark.parametrize("path", FIXTURES, ids=[p.stem for p in FIXTURES])
def test_replay_torch_normwise(path):
    call = load_call(path)
    with mathfn.use_backend("libm"):
        _, ref_int = _run(call)
    out, tor_int = _run(call)
    for name, val in out.items():
        ref = call.post[name]
        port = val.numpy()
        if name == "inv_qc_relvar":
            qc = call.post["qc"]
            floor = 64 * EPS * qc * qc
            noise = (ref_int["main"]["shoc_ql2"].numpy() <= floor) | (tor_int["main"]["shoc_ql2"].numpy() <= floor)
            port = np.where(noise, ref, port)
        scale = np.abs(ref).max()
        err = np.abs(port - ref).max()
        assert err <= 1e-9 * scale, (name, err, scale)


@pytest.mark.reference
@pytest.mark.parametrize("path", FIXTURES, ids=[p.stem for p in FIXTURES])
def test_budgets(path):
    call = load_call(path)
    out, internal = _run(call)
    t = call.pre_t
    dt = float(call.meta["dt"])
    pd = t("pseudo_density")
    nlev = pd.shape[-1]
    tol = nlev * EPS  # bound on relative roundoff of a sequential nlev-term sum
    # Water: d(column qv+qc) = surf_evap * dt
    w0 = budgets.column_integral(t("qv") + t("qc"), pd)
    w1 = budgets.column_integral(out["qv"] + out["qc"], pd)
    resid = w1 - w0 - t("surf_evap") * dt
    assert (resid.abs() <= tol * budgets.column_integral(t("qv").abs() + t("qc").abs(), pd)).all(), resid
    # Energy: d(TE) = (surf_sens_flux + (Lv+Li)*surf_evap) * dt  (SHOC's energy fixer)
    z_mid = internal["pre"]["z_mid"]
    hw0, hw1 = t("horiz_winds"), out["horiz_winds"]
    e0 = budgets.column_total_energy(t("T_mid"), z_mid, t("phis"), hw0[:, 0], hw0[:, 1], t("qv"), t("qc"), pd)
    e1 = budgets.column_total_energy(out["T_mid"], z_mid, t("phis"), hw1[:, 0], hw1[:, 1], out["qv"], out["qc"], pd)
    flux = t("surf_sens_flux") + (2501000.0 + 333700.0) * t("surf_evap")
    resid = e1 - e0 - flux * dt
    assert (resid.abs() <= tol * budgets.column_energy_scale(t("T_mid"), z_mid, pd)).all(), resid
    # Other turbulence-advected tracers: zero surface flux, conserved
    for n in (n for n in TRACERS if n in call.pre):
        m0 = budgets.column_integral(t(n), pd)
        m1 = budgets.column_integral(out[n], pd)
        assert ((m1 - m0).abs() <= tol * budgets.column_integral(t(n).abs(), pd)).all(), n
