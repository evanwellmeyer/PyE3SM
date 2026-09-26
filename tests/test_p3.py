"""P3 port vs EAMxx P3 (process run_impl), E3SM 80fe0a71.

Reference: DP-EAMxx L128 runs of the strict build (-ffp-contract=off, pack
size 1), dumped around P3Microphysics::run_impl (see docs/plan.md). The
fixtures cover shallow cumulus (BOMEX), stratocumulus (DYCOMS-II RF01),
continental convection with ice (ARM97 7.5 days), CASS, and one standalone
run built to reach states the DP cases never do: supercooled liquid (which
freezes homogeneously), droplet number below nsmall, heavy subliming ice,
rain evaporating into near-saturated air and very high cloud water
(reference/scripts/make_standalone_case.py p3). The standalone fixture also
has extra_p3_diags on, so the 15 extra process-rate diagnostics are compared.

These tests need P3's lookup tables, which live in the (git-ignored)
reference input data; they skip when it is absent.

Tolerances:
  * libm math backend: bitwise (163 calls, 1.9M values, via replay_check.py).
  * torch math backend: normwise error (max|port-ref| / max|ref| per field)
    <= 5.5e-14 over all calls; the bound used here is 1e-11.
Conservation uses EAMxx's own definitions (mass_and_energy_conservation_check):
column water changes by the surface precipitation, and total energy by the
latent heat carried out by the liquid part of it.
"""
from pathlib import Path

import numpy as np
import pytest
import torch

from eamxx_torch import mathfn
from eamxx_torch.p3 import P3Runtime, invariant_checks, load_p3_tables, p3_run, postconditions
from eamxx_torch.p3 import sedimentation as sed
from eamxx_torch.testing import compare, list_calls, load_call

DATA = Path(__file__).parent / "data" / "p3"
FIXTURES = list_calls(DATA)
TABLES_DIR = Path(__file__).resolve().parents[1] / "reference/inputdata/atm/scream/tables"
HAVE_TABLES = (TABLES_DIR / "p3_lookup_table_1.dat-v4.1.1").exists()
needs_tables = pytest.mark.skipif(not HAVE_TABLES, reason="P3 lookup tables not present")
G, CP, LV, LI, RHO_H2O = 9.80616, 1004.64, 2501000.0, 333700.0, 1000.0

_TABLES = None


def tables():
    global _TABLES
    if _TABLES is None:
        _TABLES = load_p3_tables(TABLES_DIR)
    return _TABLES


def _run(call, **kw):
    meta = call.meta["p3"]
    ro = P3Runtime(**meta["runtime"])
    state = {k: torch.as_tensor(v, dtype=torch.float64) for k, v in call.pre.items()}
    out = p3_run(state, float(call.meta["dt"]), tables(), ro, meta["predictNc"],
                 meta["prescribedCCN"], **kw)
    internal = out.pop("_internal")
    return out, internal, ro


@needs_tables
@pytest.mark.reference
@pytest.mark.parametrize("path", FIXTURES, ids=[p.stem for p in FIXTURES])
def test_replay_libm_bitwise(path):
    call = load_call(path)
    with mathfn.use_backend("libm"):
        out, _, _ = _run(call)
    assert set(out) <= set(call.post)
    for name, val in out.items():
        d = compare(name, val, call.post[name])
        assert d.ok(), d


@needs_tables
@pytest.mark.reference
@pytest.mark.parametrize("path", FIXTURES, ids=[p.stem for p in FIXTURES])
def test_replay_torch_normwise(path):
    call = load_call(path)
    out, _, _ = _run(call)
    for name, val in out.items():
        ref = call.post[name]
        scale = np.abs(ref).max()
        err = np.abs(val.numpy() - ref).max()
        assert err <= 1e-11 * max(scale, 1e-300), (name, err, scale)


@needs_tables
@pytest.mark.reference
@pytest.mark.parametrize("path", FIXTURES, ids=[p.stem for p in FIXTURES])
def test_conservation(path):
    """EAMxx's own mass and energy definitions, with P3's boundary fluxes."""
    call = load_call(path)
    with mathfn.use_backend("libm"):
        out, _, _ = _run(call)
    t = call.pre_t
    dt = float(call.meta["dt"])
    pd = t("pseudo_density")
    nlev = pd.shape[-1]
    tol = nlev * float(np.finfo(np.float64).eps)

    def water(q):
        return ((q["qv"] + q["qc"] + q["qi"] + q["qr"]) * pd / G).sum(dim=-1)

    def energy(q):
        # P3 changes no winds and no surface geopotential, so those terms cancel
        return ((q["T_mid"] * CP + (LV + LI) * q["qv"] + LI * (q["qc"] + q["qr"])) * pd / G).sum(dim=-1)

    pre = {k: t(k) for k in ("qv", "qc", "qi", "qr", "T_mid")}
    post = {k: out[k] for k in ("qv", "qc", "qi", "qr", "T_mid")}
    liq = (out["precip_liq_surf_mass"] - t("precip_liq_surf_mass")) / (RHO_H2O * dt)
    ice = (out["precip_ice_surf_mass"] - t("precip_ice_surf_mass")) / (RHO_H2O * dt)
    w0, w1 = water(pre), water(post)
    resid = w1 - w0 + (liq + ice) * RHO_H2O * dt
    assert (resid.abs() <= tol * w0).all(), resid
    e0, e1 = energy(pre), energy(post)
    resid = e1 - e0 + liq * RHO_H2O * LI * dt
    assert (resid.abs() <= tol * e0).all(), resid


@needs_tables
@pytest.mark.reference
@pytest.mark.parametrize("path", FIXTURES, ids=[p.stem for p in FIXTURES])
def test_substep_cap_not_reached(path):
    call = load_call(path)
    with mathfn.use_backend("libm"):
        _, internal, _ = _run(call)
    assert not bool(internal["main"]["hit_substep_cap"].any())


@needs_tables
@pytest.mark.reference
def test_early_exit_does_not_change_results():
    """The substep loop's early exit is an optimisation, not a change."""
    call = load_call(DATA / "strict_arm97_7p5d_s4752_c0_r1.npz")
    with mathfn.use_backend("libm"):
        out_fast, _, _ = _run(call)
        sed.EARLY_EXIT = False
        try:
            out_full, _, _ = _run(call)
        finally:
            sed.EARLY_EXIT = True
    for name in out_fast:
        d = compare(name, out_fast[name], out_full[name].numpy())
        assert d.ok(), d


@needs_tables
@pytest.mark.reference
@pytest.mark.parametrize("path", FIXTURES, ids=[p.stem for p in FIXTURES])
def test_property_checks_never_fail(path):
    """No check fails. The repairable qv check does fire (the reference repairs it too)."""
    call = load_call(path)
    with mathfn.use_backend("libm"):
        out, _, _ = _run(call)
    checked, status = invariant_checks(out)
    checked, status2 = postconditions(checked)
    for name, st in list(status.items()) + list(status2.items()):
        assert int(st) != 2, (path.name, name)  # 2 = FAIL, 1 = REPAIRED, 0 = PASS


def test_synthetic_fixture_reaches_rare_states():
    first = load_call(DATA / "standalone_strict_p3_s1_c0_r0.npz")
    assert first.meta["p3"]["runtime"]["extra_p3_diags"] is True
    # supercooled liquid, which freezes homogeneously below 233.15 K (step 1 only:
    # the freezing removes it, so later steps no longer hold any)
    assert ((first.pre["T_mid"] < 233.15) & (first.pre["qc"] > 1e-14)).any()
    # droplet number below nsmall with cloud present
    assert ((first.pre["nc"] < 1e-16) & (first.pre["qc"] > 1e-14)).any()
    # a column with much more cloud water than the sinks can take
    assert first.pre["qc"].max() > 0.01
