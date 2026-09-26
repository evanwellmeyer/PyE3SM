"""Gradients (differentiability phase, docs/differentiability.md).

- gradsafe: the gradient-safe forms give exactly the plain values (including -0, NaN and
  negative inputs) and gradient 0 only where the plain form would give 0*inf = NaN.
- On the committed SHOC, P3 and ZM fixtures (inputs exactly as EAMxx saw them), the
  gradient of a random linear functional of every output with respect to every float
  input is finite (PyTorch math, the autograd-capable backend).
The single-column model over two steps is tested in test_scm.py.
"""
from pathlib import Path

import numpy as np
import pytest
import torch

from eamxx_torch.gradsafe import nonsingular, sqrt0
from eamxx_torch.testing import list_calls, load_call

import test_p3
import test_shoc
import test_zm


def test_sqrt0_values_and_gradient():
    x = torch.tensor([4.0, 0.0, -0.0, 1e-300, -1.0, float("nan"), float("inf")], dtype=torch.float64)
    ref = torch.sqrt(x)
    got = sqrt0(x)
    assert torch.equal(torch.isnan(got), torch.isnan(ref))
    ok = ~torch.isnan(ref)
    assert torch.equal(got[ok], ref[ok])
    assert torch.equal(torch.signbit(got[ok]), torch.signbit(ref[ok]))   # -0 stays -0
    xg = torch.tensor([4.0, 0.0], dtype=torch.float64, requires_grad=True)
    sqrt0(xg).sum().backward()
    assert xg.grad.tolist() == [0.25, 0.0]


def test_nonsingular_values_and_gradient():
    a = torch.tensor([1.0, 0.0, 2.0], dtype=torch.float64, requires_grad=True)
    b = torch.tensor([2.0, 0.0, 0.0], dtype=torch.float64, requires_grad=True)
    bad = b == 0
    got = nonsingular(bad, lambda p, q: p / q, a, b)
    with torch.no_grad():
        ref = a / b
    assert torch.equal(torch.isnan(got), torch.isnan(ref))
    ok = ~torch.isnan(ref)
    assert torch.equal(got[ok], ref[ok])            # 0/0 stays NaN, 2/0 stays inf
    got[0].backward(retain_graph=True)
    assert a.grad.tolist() == [0.5, 0.0, 0.0] and b.grad.tolist() == [-0.25, 0.0, 0.0]
    a.grad = b.grad = None
    (got * torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)).sum().backward()
    assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()   # no 0*inf


def _finite_gradients(run, call, cols=None):
    pre = call.pre
    if cols:
        n = pre["T_mid"].shape[0]
        pre = {k: (v[:cols] if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n else v)
               for k, v in pre.items()}
    x = {k: torch.as_tensor(v, dtype=torch.float64).clone().requires_grad_(True)
         if isinstance(v, np.ndarray) and v.dtype.kind == "f" and v.ndim >= 1 else v for k, v in pre.items()}
    out = run(x)
    gen = torch.Generator().manual_seed(0)
    loss = sum((torch.randn(t.shape, generator=gen, dtype=torch.float64) * t).sum()
               / max(float(t.detach().abs().sum()), 1.0)
               for t in out.values() if isinstance(t, torch.Tensor) and t.requires_grad)
    leaves = [v for v in x.values() if isinstance(v, torch.Tensor) and v.requires_grad]
    grads = torch.autograd.grad(loss, leaves, allow_unused=True)
    bad = [k for k, g in zip([k for k, v in x.items() if isinstance(v, torch.Tensor) and v.requires_grad], grads)
           if g is not None and not torch.isfinite(g).all()]
    assert not bad, f"non-finite gradients for {bad}"


def _shoc(x):
    call = _shoc.call
    from eamxx_torch.shoc import SHOCRuntime, shoc_run
    ncol = x["T_mid"].shape[0]
    cell = torch.full((ncol,), call.meta["cell_length"], dtype=torch.float64)
    out = shoc_run(x, float(call.meta["dt"]), cell, test_shoc.NPBL, SHOCRuntime(extra_diags=True),
                   apply_tms=True, tracer_names=tuple(n for n in test_shoc.TRACERS if n in x))
    out.pop("_internal")
    return out


@pytest.mark.parametrize("path", test_shoc.FIXTURES, ids=[p.stem for p in test_shoc.FIXTURES])
def test_shoc_gradients_finite(path):
    _shoc.call = load_call(path)
    _finite_gradients(_shoc, _shoc.call)


def _p3(x):
    from eamxx_torch.p3 import P3Runtime, p3_run
    call = _p3.call
    meta = call.meta["p3"]
    out = p3_run(x, float(call.meta["dt"]), test_p3.tables(), P3Runtime(**meta["runtime"]),
                 meta["predictNc"], meta["prescribedCCN"])
    out.pop("_internal")
    return out


@pytest.mark.skipif(not test_p3.HAVE_TABLES, reason="P3 lookup tables not present")
@pytest.mark.parametrize("path", test_p3.FIXTURES, ids=[p.stem for p in test_p3.FIXTURES])
def test_p3_gradients_finite(path):
    _p3.call = load_call(path)
    _finite_gradients(_p3, _p3.call)


def _zm(x):
    return test_zm._run(_zm.call, state=x)


@pytest.mark.parametrize("path", test_zm.FIXTURES, ids=[p.stem for p in test_zm.FIXTURES])
def test_zm_gradients_finite(path):
    _zm.call = load_call(path)
    _finite_gradients(_zm, _zm.call)


# ------------------------------------------------------------ options for gradients
def test_shoc_energy_fixer_tolerance():
    """tke_tol = 0 is EAMxx's exact test; 1e-14 treats a one-ulp TKE as 'at the floor'."""
    from eamxx_torch.shoc.constants import SC
    from eamxx_torch.shoc.functions import shoc_energy_fixer
    f64 = torch.float64
    nlev = 10
    zt = torch.linspace(5000, 50, nlev, dtype=f64)[None]
    zi = torch.linspace(5100, 0, nlev + 1, dtype=f64)[None]
    pint = torch.linspace(50000, 100000, nlev + 1, dtype=f64)[None]
    one = torch.ones(1, dtype=f64)
    tke = torch.full((1, nlev), SC.mintke, dtype=f64)
    tke[:, 6:] = 0.1
    tke_ulp = tke.clone()
    tke_ulp[:, 2] = SC.mintke * (1 + 2.2e-16)
    args = lambda t: (100.0, 1, zt, zi, one * 3e7, one, one * 30, one, one * 3e7 + 50, one, one * 30,  # noqa: E731
                      one, one * 0.01, one * 1e-5, torch.ones(1, nlev, dtype=f64), t, pint,
                      torch.full((1, nlev), 3e5, dtype=f64))
    exact = shoc_energy_fixer(*args(tke))
    assert not torch.equal(shoc_energy_fixer(*args(tke_ulp)), exact)          # EAMxx: the ulp moves shoctop
    assert torch.equal(shoc_energy_fixer(*args(tke_ulp), tke_tol=1e-14), exact)


@pytest.mark.skipif(not __import__("test_rrtmgp").HAVE_DATA, reason="RRTMGP coefficient files not present")
def test_mcica_state_free_seeds_remove_the_pressure_jump():
    import dataclasses
    import test_rrtmgp as TR
    from eamxx_torch.rrtmgp import rrtmgp_run
    from eamxx_torch.timestamp import TimeStamp
    call = load_call(next(p for p in TR.FIXTURES if "arm97" in p.name))
    meta = call.meta["rrtmgp"]
    rt = dict(meta["runtime"], active_gases=tuple(meta["runtime"]["active_gases"]))
    TR._run(call)                                           # loads the coefficient data
    data = TR._DATA_CACHE[tuple(meta["files"])]

    def fluxes(from_pressure, dp):
        ro = dataclasses.replace(TR.RRTMGPRuntime(**rt), mcica_seed_from_pressure=from_pressure)
        st = {k: torch.as_tensor(v, dtype=torch.float64) for k, v in call.pre.items()}
        st["p_mid"] = st["p_mid"].clone()
        st["p_mid"][:, -2:] *= 1 + dp
        st["cldfrac_tot"] = st["cldfrac_tot"].clone()        # a partly cloudy layer
        st["cldfrac_tot"][:, 100:110] = 0.5
        st["qc"] = st["qc"].clone()
        st["qc"][:, 100:110] = 2e-4
        out = rrtmgp_run(st, float(call.meta["dt"]), data, ro, TimeStamp.from_string(call.meta["time_start"]),
                         call.meta["step"] - 1, torch.as_tensor(meta["lat"], dtype=torch.float64),
                         torch.as_tensor(meta["lon"], dtype=torch.float64))
        return torch.cat([out["SW_flux_dn"].flatten(), out["LW_flux_dn"].flatten()])
    for fp in (True, False):
        jump = float((fluxes(fp, 1e-12) - fluxes(fp, 0.0)).abs().max())
        if fp:
            assert jump > 1e-3          # EAMxx: the seeds come from the pressure digits
        else:
            assert jump < 1e-6          # state-free seeds: a smooth response


def test_checkpointed_run_matches_plain_run():
    import test_scm as TS
    if not (TS._have_suite_data() and (TS.ROOT / "scream" / "init" / "screami_ne30np4L128_20260813.nc").exists()):
        pytest.skip("IC files not present")
    res = []
    for ck in (False, True):
        scm, s = TS._scm(ncol=1)
        T = s["T_mid"].clone().requires_grad_(True)
        out = scm.run(dict(s, T_mid=T), 4, checkpoint=ck)
        loss = out["T_mid"].sum() + 1e3 * out["qv"].sum()
        g, = torch.autograd.grad(loss, T)
        res.append((out["T_mid"].detach(), g, scm.time.num_steps, scm.step(out).state["T_mid"].detach()))
    assert torch.equal(res[0][0], res[1][0]) and torch.equal(res[0][1], res[1][1])
    assert res[0][2] == res[1][2] and torch.equal(res[0][3], res[1][3])   # the host state continues
