"""RRTMGP port vs EAMxx RRTMGPRadiation::run_impl, E3SM 80fe0a71.

Reference: the strict build (-ffp-contract=off), dumped around run_impl.
Fixtures: CASS daytime with shallow liquid cloud; ARM97 night with deep ice
cloud; a BOMEX step without a radiation update (only T_mid changes); and two
standalone runs built to reach states the DP cases never do
(reference/scripts/make_standalone_case.py rrtmgp / rrtmgp_alt): temperatures
outside the gas-optics tables, effective radii outside the cloud-optics tables,
an optically very thin top layer, polar day and night, a night shorter than the
zenith-angle averaging window, |lat| > 45, fixed orbit and solar constant, the
extra clean/clear-sky calls, a fixed zenith angle, no McICA, no aerosol.

These tests need RRTMGP's coefficient files, which live in the (git-ignored)
reference input data; they skip when it is absent.

Tolerances:
  * libm math backend: bitwise (268 calls, 1.94M values, via replay_check.py).
  * torch math backend: normwise error (max|port-ref| / max|ref| per field)
    <= 1.7e-10 over all calls; the bound used here is 1e-9. Two places amplify
    torch's 1-ulp differences from libm, in the reference's arithmetic too: the
    averaged zenith angle cancels near sunrise and sunset (mu0 = 0.006 gives
    3e-13 relative in mu0, 1e-10 in the surface direct beam), and heating rates
    are differences of nearly equal fluxes.
"""
from pathlib import Path

import numpy as np
import pytest
import torch

from eamxx_torch import constants as C
from eamxx_torch import mathfn
from eamxx_torch.rrtmgp import RRTMGPRuntime, invariant_checks, load_rrtmgp, radiation_do, rrtmgp_run
from eamxx_torch.rrtmgp.mcica import Random
from eamxx_torch.testing import compare, list_calls, load_call
from eamxx_torch.timestamp import TimeStamp

DATA = Path(__file__).parent / "data" / "rrtmgp"
FIXTURES = list_calls(DATA)
INPUT = Path(__file__).resolve().parents[1] / "reference/inputdata/atm/scream/init"
HAVE_DATA = (INPUT / "rrtmgp-data-sw-g112-210809.nc").exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="RRTMGP coefficient files not present")

_DATA_CACHE = {}


def _run(call):
    meta = call.meta["rrtmgp"]
    rt = dict(meta["runtime"])
    rt["active_gases"] = tuple(rt["active_gases"])
    ro = RRTMGPRuntime(**rt)
    key = tuple(meta["files"])
    if key not in _DATA_CACHE:
        _DATA_CACHE[key] = load_rrtmgp(*[INPUT / f for f in meta["files"]], ro.gas_names)
    state = {k: torch.as_tensor(v, dtype=torch.float64) for k, v in call.pre.items()}
    lat = torch.as_tensor(meta["lat"], dtype=torch.float64)
    lon = torch.as_tensor(meta["lon"], dtype=torch.float64)
    out = rrtmgp_run(state, float(call.meta["dt"]), _DATA_CACHE[key], ro,
                     TimeStamp.from_string(call.meta["time_start"]), call.meta["step"] - 1, lat, lon)
    internal = out.pop("_internal", None)
    return out, internal, ro


@needs_data
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


@needs_data
@pytest.mark.reference
@pytest.mark.parametrize("path", FIXTURES, ids=[p.stem for p in FIXTURES])
def test_replay_torch_normwise(path):
    call = load_call(path)
    out, _, _ = _run(call)
    for name, val in out.items():
        ref = call.post[name]
        scale = np.abs(ref).max()
        err = np.abs(val.numpy() - ref).max()
        assert err <= 1e-9 * max(scale, 1e-300), (name, err, scale)


@needs_data
@pytest.mark.reference
@pytest.mark.parametrize("path", FIXTURES, ids=[p.stem for p in FIXTURES])
def test_energy_closure(path):
    """Column heating equals the net radiative flux into the column (EAMxx's heat_flux).

    sum_k rad_heating_pdel*cp/g = (fsnt - fsns) - (flnt - flns), and T_mid moves by
    rad_heating*dt. rad_heating_pdel = pdel*heating, heating = convergence*g/(cp*pdel).
    """
    call = load_call(path)
    with mathfn.use_backend("libm"):
        out, _, ro = _run(call)
    if not radiation_do(ro.rad_frequency, call.meta["step"] - 1):
        dT = out["T_mid"] - call.pre_t("T_mid")
        expect = call.pre_t("rad_heating_pdel") / call.pre_t("pseudo_density") * call.meta["dt"]
        assert torch.allclose(dT, expect, rtol=0, atol=1e-12)
        return
    heat = (out["rad_heating_pdel"] * C.Cpair / C.gravit).sum(dim=-1)
    sw_up, sw_dn = out["SW_flux_up"], out["SW_flux_dn"]
    lw_up, lw_dn = out["LW_flux_up"], out["LW_flux_dn"]
    fsnt, fsns = sw_dn[:, 0] - sw_up[:, 0], sw_dn[:, -1] - sw_up[:, -1]
    flnt, flns = lw_up[:, 0] - lw_dn[:, 0], lw_up[:, -1] - lw_dn[:, -1]
    net = (fsnt - fsns) - (flnt - flns)
    scale = torch.maximum(sw_dn.abs().amax(dim=-1), lw_up.abs().amax(dim=-1))
    nlev = sw_up.shape[-1]
    assert ((heat - net).abs() <= 4 * nlev * float(np.finfo(np.float64).eps) * scale).all()
    dT = out["T_mid"] - call.pre_t("T_mid")
    heating = out["rad_heating_pdel"] / call.pre_t("pseudo_density")
    assert torch.allclose(dT, heating * call.meta["dt"], rtol=0, atol=1e-12)


@needs_data
@pytest.mark.reference
@pytest.mark.parametrize("path", FIXTURES, ids=[p.stem for p in FIXTURES])
def test_property_check_passes(path):
    call = load_call(path)
    with mathfn.use_backend("libm"):
        out, _, _ = _run(call)
    for fields in ({"T_mid": call.pre_t("T_mid")}, out):   # checked before and after run_impl
        _, status = invariant_checks(fields)
        assert int(status["T_mid"]) == 0


def _random_reference(seed: int, n: int) -> list[float]:
    """conv::Random with Python integers (mod 2^64), independent of the torch version."""
    M = (1 << 64) - 1

    def rot(x, k):
        return ((x << k) | (x >> (64 - k))) & M

    a, b, c, d = 0xf1ea5eed, seed, seed, seed
    out = []
    for i in range(20 + n):
        e = (a - rot(b, 7)) & M
        a = b ^ rot(c, 13)
        b = (c + rot(d, 37)) & M
        c = (d + e) & M
        d = (e + a) & M
        if i >= 20:
            out.append(float(d) / float(M))
    return out


def test_mcica_random_matches_uint64_arithmetic():
    seeds = [0, 1, 7, 123456789, 999999999, 2**31 - 1, 2**40 + 17]
    r = Random(torch.tensor(seeds))
    got = torch.stack([r.genFP() for _ in range(5)], dim=1)
    for i, s in enumerate(seeds):
        ref = _random_reference(s, 5)
        assert got[i].tolist() == ref, (s, got[i].tolist(), ref)


def test_synthetic_fixture_reaches_rare_states():
    call = load_call(DATA / "standalone_strict_rrtmgp_s1_c0_r0.npz")
    pre, meta = call.pre, call.meta["rrtmgp"]
    assert (pre["T_mid"] < 160.0).any() and (pre["T_mid"] > 355.0).any()   # gas-optics table limits
    assert ((pre["qi"] > 0) & (pre["eff_radius_qi"] > 180.0)).any()        # ice radius above table
    assert (np.diff(pre["p_int"], axis=1) < 1e-3).any()                   # optically thin layer
    lat = np.array(meta["lat"])
    assert (lat > 70).any() and (lat < -70).any() and (np.abs(lat) > 45).sum() >= 3
    assert meta["runtime"]["extra_clnsky_diag"] and meta["runtime"]["extra_clnclrsky_diag"]
    alt = load_call(DATA / "standalone_strict_rrtmgp_alt_s4_c0_r0.npz").meta["rrtmgp"]["runtime"]
    assert not alt["do_subcol_sampling"] and not alt["do_aerosol_rad"]
    assert alt["fixed_solar_zenith_angle"] > 0
