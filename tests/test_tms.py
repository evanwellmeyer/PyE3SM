"""TMS port vs EAMxx's TMS run_impl sequence on synthetic land columns.

Reference: reference/eamxx_ext/unit_replay/tms_replay.cpp, strict build
(-ffp-contract=off, pack size 1), E3SM 80fe0a71.
  - libm math backend: bitwise.
  - torch math backend: torch.pow differs from libm pow by up to 1 ulp in exner
    (rtol 4.5e-16 = 2 ulp). Downstream, the Richardson number takes the
    difference of two nearly equal potential temperatures and then 1 - ri, which
    amplifies that 1 ulp to ~1e-12 relative in the drag (observed max 1.05e-12).
    Outputs are therefore compared at rtol 1e-11.
"""
from pathlib import Path

import pytest

from eamxx_torch import mathfn
from eamxx_torch.testing import compare, load_call
from eamxx_torch.tms import tms

CALL = load_call(Path(__file__).parent / "data" / "tms" / "edge_cases.npz")


def _run():
    t = CALL.pre_t
    return tms(t("horiz_winds"), t("T_mid"), t("p_mid"), t("pseudo_density"), t("qv"),
               t("sgh30"), t("landfrac"))


def test_nonzero_drag_is_exercised():
    assert (CALL.post["surf_drag_coeff_tms"] != 0).sum() >= 30


@pytest.mark.parametrize("backend,rtol_aux,rtol_out",
                         [("libm", 0.0, 0.0), ("torch", 4.5e-16, 1e-11)])
def test_tms_edge_cases(backend, rtol_aux, rtol_out):
    with mathfn.use_backend(backend):
        out, aux = _run()
    for name in ("surf_drag_coeff_tms", "wind_stress_tms"):
        d = compare(name, out[name], CALL.post[name], rtol=rtol_out)
        assert d.ok(), d
    for name in ("exner", "z_mid"):
        d = compare(name, aux[name], CALL.post[name], rtol=rtol_aux)
        assert d.ok(), d
