"""ZM port vs EAMxx ZM (process run_impl), E3SM 80fe0a71.

Reference: DP-EAMxx-LR runs (ZM on, 1800 s, 12 mac_aero_mic sub-cycles) of the
strict build, one MPI rank, every one of the 100 columns dumped around
ZMDeepConvection::run_impl (reference/scripts/make_dumps.sh with LR=1). The
fixtures keep a few columns of each call, active and inactive: ARM97 step 1 (first
step, no DCAPE), 19 (8 of 100 columns active), 80 (all rain evaporates before the
surface), 209 (28 mm/day), 240 (MCSP on); BOMEX step 12; DYCOMS-RF01 step 2 (no
convection).

Replay on a subset of a call's columns equals the full call bitwise: the two values
EAMxx reduces over all columns (ktm, kbm) only bound loops that are zero outside
each column's cloud.

Tolerances:
  * libm math backend: bitwise (21 calls, 7.3M values, via replay_check.py).
  * torch math backend: normwise error (max|port-ref| / max|ref| per field) <= 1.7e-12,
    except the surface rain where all of it evaporates on the way down (ARM97 step 80):
    the reference's own value is a residue of 1.5e-12 of the column's rain flux, and the
    torch-mode difference is 1e-12 of that flux. The bound here is 1e-10, with the
    surface rain scaled by 1e-10 m/s (0.009 mm/day).
Conservation: column water changes by minus the surface precipitation, and
cp*T + Lv*qv - Li*qi + KE (liquid-water reference) is unchanged, to roundoff.
"""
from pathlib import Path

import numpy as np
import pytest
import torch

from eamxx_torch import mathfn
from eamxx_torch.testing import compare, list_calls, load_call
from eamxx_torch.zm import ZMRuntime, invariant_checks, set_limcnv, zm_run, zm_setup
from eamxx_torch.zm import functions as FN

DATA = Path(__file__).parent / "data" / "zm"
FIXTURES = list_calls(DATA)
G, CP, LV, LI = 9.80616, 1004.64, 2501000.0, 333700.0
_SETUP = {}


def _setup(meta):
    key = tuple(sorted(meta["runtime"].items()))
    if key not in _SETUP:
        _SETUP[key] = zm_setup(ZMRuntime(**meta["runtime"]), meta["hyai"], meta["hybi"])
    return _SETUP[key]


def _run(call, state=None):
    s = state or {k: torch.as_tensor(v, dtype=torch.float64) for k, v in call.pre.items()}
    out = zm_run(s, float(call.meta["dt"]), _setup(call.meta["zm"]), call.meta["step"] == 1)
    internal = out.pop("_internal")
    assert not bool(internal["nonconverged"].any())
    return out


def test_fixtures_present():
    assert len(FIXTURES) == 7


@pytest.mark.reference
@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_zm_bitwise_libm(path):
    call = load_call(path)
    with mathfn.use_backend("libm"):
        out = _run(call)
    for k, v in out.items():
        d = compare(k, v, call.post[k])
        assert d.ok(), d


@pytest.mark.reference
@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_zm_torch_math(path):
    call = load_call(path)
    out = _run(call)
    dt = float(call.meta["dt"])
    floor = {"zm_prec": 1e-10, "precip_liq_surf_mass": 1e-10 * 1000 * dt}
    for k, v in out.items():
        ref = call.post[k]
        scale = max(float(np.abs(ref).max()), floor.get(k, 0.0))
        err = float(np.abs(v.numpy() - ref).max())
        assert err <= 1e-10 * scale or err == 0.0, (k, err, scale)


def test_limcnv_l128():
    call = load_call(FIXTURES[0])
    m = call.meta["zm"]
    assert set_limcnv(m["runtime"]["upper_limit_pref"], m["hyai"], m["hybi"]) == 19  # EAMxx log


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_zm_conserves_water_and_energy(path):
    call = load_call(path)
    s = {k: torch.as_tensor(v, dtype=torch.float64) for k, v in call.pre.items()}
    out = _run(call, s)
    dp = s["pseudo_density"]

    def water(d):
        return ((d["qv"] + d["qc"] + d["qi"]) * dp).sum(-1) / G

    def energy(d):
        u, v = d["horiz_winds"][:, 0], d["horiz_winds"][:, 1]
        return ((CP * d["T_mid"] + LV * d["qv"] - LI * d["qi"] + 0.5 * (u * u + v * v)) * dp).sum(-1) / G

    new = {**s, **out}
    precip = (out["precip_liq_surf_mass"] - s["precip_liq_surf_mass"]
              + out["precip_ice_surf_mass"] - s["precip_ice_surf_mass"])
    assert torch.all(out["precip_ice_surf_mass"] == s["precip_ice_surf_mass"])  # no surface snow here
    dw = water(new) - water(s) + precip
    assert float(torch.abs(dw / water(s)).max()) < 1e-14
    de = energy(new) - energy(s)
    assert float(torch.abs(de / (CP * s["T_mid"] * dp).sum(-1) * G).max()) < 1e-14


def test_columns_are_independent():
    """Bitwise with libm. In torch-math mode torch.pow can round an element differently
    depending on the batch it is in (vectorized vs scalar path; docs/plan.md, section 4),
    so there a column alone agrees with the batch to roundoff only."""
    call = load_call(DATA / "strict_arm97_lr_zm_s19_c0_r0.npz")
    for backend in ("libm", "torch"):
        with mathfn.use_backend(backend):
            full = _run(call)
            for i in range(call.pre["T_mid"].shape[0]):
                one = {k: torch.as_tensor(v[i:i + 1], dtype=torch.float64) for k, v in call.pre.items()}
                out = _run(call, one)
                for k, v in out.items():
                    if backend == "libm":
                        assert torch.equal(v[0], full[k][i]), (i, k)
                    else:
                        scale = float(torch.abs(full[k][i]).max()) or 1.0
                        assert float(torch.abs(v[0] - full[k][i]).max()) <= 1e-10 * scale, (i, k)


def test_invert_entropy_early_exit_is_exact():
    torch.manual_seed(0)
    t = 200 + 100 * torch.rand(64, dtype=torch.float64)
    p = 200 + 800 * torch.rand(64, dtype=torch.float64)
    qt = 1e-4 + 2e-2 * torch.rand(64, dtype=torch.float64)
    s = FN.entropy(t + 3 * torch.randn(64, dtype=torch.float64), p, qt)
    a = FN.invert_entropy(s, p, qt, t, early_exit=True)
    b = FN.invert_entropy(s, p, qt, t, early_exit=False)
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    assert not bool(a[2].any())


def test_checks_pass_on_fixtures():
    for path in FIXTURES:
        call = load_call(path)
        s = {k: torch.as_tensor(v, dtype=torch.float64) for k, v in call.pre.items()}
        _, st = invariant_checks(s)
        assert all(int(v) == 0 for v in st.values())
