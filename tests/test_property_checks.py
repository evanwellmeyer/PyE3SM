"""FieldWithinIntervalCheck port vs EAMxx's own check/repair on synthetic fields.

Reference: reference/eamxx_ext/unit_replay/interval_check_replay.cpp run on the
strict build (E3SM 80fe0a71). Tolerance: bitwise (comparisons and min/max only).
"""
from pathlib import Path

import pytest
import torch

from eamxx_torch.property_checks import within_interval
from eamxx_torch.testing import compare, load_call

CALL = load_call(Path(__file__).parent / "data" / "property_checks" / "interval_edge_cases.npz")
CASES = CALL.meta["cases"]


@pytest.mark.parametrize("i", range(len(CASES)))
def test_interval_check_matches_reference(i):
    c = CASES[i]
    f = torch.as_tensor(CALL.pre[f"f{i}"])
    out, status = within_interval(f, c["lb"], c["ub"], c["can_repair"], c["lb_rep"], c["ub_rep"])
    assert int(status) == c["result"]
    d = compare(f"f{i}", out, CALL.post[f"f{i}"])
    assert d.ok(), d
