"""Replay tests for eamxx_torch.cld_fraction against saved EAMxx reference data.

Reference: E3SM 80fe0a71, physics/cld_fraction (see docs/plan.md).
Tolerance: bitwise. The scheme is comparisons and max() only, with no
floating-point arithmetic, so any difference at all is a translation error.
"""
from pathlib import Path

import numpy as np
import pytest
import torch

from eamxx_torch.cld_fraction import CldFractionParams, check_output, cld_fraction
from eamxx_torch.testing import compare, list_calls, load_call

DATA = Path(__file__).parent / "data" / "cld_fraction"
OUTPUTS = ["cldfrac_ice", "cldfrac_tot", "cldfrac_ice_for_analysis", "cldfrac_tot_for_analysis"]
DUMP_FIXTURES = [p for p in list_calls(DATA) if p.name != "edge_cases.npz"]


def _run(call, params=CldFractionParams()):
    return cld_fraction(call.pre_t("qi"), call.pre_t("cldfrac_liq"), params)


def test_edge_cases_bitwise():
    call = load_call(DATA / "edge_cases.npz")
    p = call.meta["params"]
    out = _run(call, CldFractionParams(p["ice_cloud_threshold"], p["ice_cloud_for_analysis_threshold"]))
    for name in OUTPUTS:
        d = compare(name, out[name], call.post[name])
        assert d.ok(), d


@pytest.mark.reference
@pytest.mark.skipif(not DUMP_FIXTURES, reason="no dump fixtures")
@pytest.mark.parametrize("path", DUMP_FIXTURES, ids=[p.stem for p in DUMP_FIXTURES])
def test_replay_dump_bitwise(path):
    call = load_call(path)
    out = _run(call)
    for name in OUTPUTS:
        d = compare(name, out[name], call.post[name])
        assert d.ok(), d
    # Inputs are not modified by the scheme.
    for name in ("qi", "cldfrac_liq"):
        assert np.array_equal(call.pre[name], call.post[name], equal_nan=True)


def test_batched_member_dimension():
    """(ncol, nmember, nlev) gives the same answer as each (ncol, nlev) slice."""
    g = torch.Generator().manual_seed(0)
    qi = 10.0 ** (-15 + 13 * torch.rand(5, 3, 128, generator=g, dtype=torch.float64))
    liq = torch.rand(5, 3, 128, generator=g, dtype=torch.float64)
    out = cld_fraction(qi, liq)
    for m in range(3):
        o = cld_fraction(qi[:, m], liq[:, m])
        for name in OUTPUTS:
            assert torch.equal(out[name][:, m], o[name])
    assert bool(check_output(out))


def test_float32_runs():
    qi = torch.tensor([[0.0, 2e-12, 2e-5]], dtype=torch.float32)
    liq = torch.tensor([[0.3, 0.3, 0.3]], dtype=torch.float32)
    out = cld_fraction(qi, liq)
    assert out["cldfrac_tot"].dtype == torch.float32
    assert out["cldfrac_tot"].tolist() == [[pytest.approx(0.3), 1.0, 1.0]]


def test_torch_compile_fullgraph():
    """No graph breaks: no .item(), no Python branching on tensor values."""
    f = torch.compile(cld_fraction, fullgraph=True, backend="eager")
    qi = torch.tensor([[0.0, 2e-12, 2e-5, float("nan")]], dtype=torch.float64)
    liq = torch.tensor([[0.3, 0.3, 0.3, float("nan")]], dtype=torch.float64)
    a, b = f(qi, liq), cld_fraction(qi, liq)
    for name in OUTPUTS:
        assert torch.equal(a[name].nan_to_num(-7.0), b[name].nan_to_num(-7.0))
