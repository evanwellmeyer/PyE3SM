"""Running the port in float32 and on a GPU (docs/float32_gpu.md).

EAMxx's single-precision build has float P3, SHOC, cld_fraction and SPA; RRTMGP, ZM, TMS
and IOP forcing are built only in double. The suite follows that by default: with dtype
float32 those processes run in float64 (on the CPU when the device has no float64).
"""
from __future__ import annotations

import pytest
import torch

import test_scm as TS
from eamxx_torch.precision import cast, double_device, to_precision

HAVE_MPS = torch.backends.mps.is_available()


def _scm32(double_only_in_float64=True, device="cpu"):
    orig = TS._config

    def cfg32(meta, files):
        c = orig(meta, files)
        c.dtype, c.device, c.double_only_in_float64 = torch.float32, device, double_only_in_float64
        return c
    TS._config = cfg32
    try:
        return TS._scm(ncol=1)
    finally:
        TS._config = orig


def test_cast_converts_precision_and_device_in_either_order():
    x = {"a": torch.ones(3, dtype=torch.float64), "i": torch.arange(3), "s": "x"}
    y = to_precision(x, torch.float32)
    assert y["a"].dtype == torch.float32 and y["i"].dtype == torch.int64 and y["s"] == "x"
    assert cast(x, torch.float32) is cast(x, torch.float32)       # cached
    assert double_device("mps") == "cpu" and double_device("cpu") == "cpu"
    if HAVE_MPS:
        m = to_precision(x, torch.float32, "mps")
        assert m["a"].device.type == "mps"
        back = to_precision(m, torch.float64, "cpu")               # MPS has no float64: move first
        assert back["a"].dtype == torch.float64 and back["a"].device.type == "cpu"


def test_float32_rrtmgp_overflows_as_the_reference_formula_does():
    """The minor-gas scaling scaling * col_gas * vmr_fact * dry_fact (mo_gas_optics_kernels.h,
    evaluated left to right) exceeds float32's range; in float64 it does not."""
    col = torch.tensor([1.0e22], dtype=torch.float32)
    other = torch.tensor([3.0e21], dtype=torch.float32)
    assert not torch.isfinite(col * 3.4 * other).all()
    assert torch.isfinite(col.double() * 3.4 * other.double()).all()


@TS.needs_scm
def test_float32_scm_stays_finite_with_double_only_processes_in_float64():
    scm, s = _scm32()
    assert s["T_mid"].dtype == torch.float32
    with torch.no_grad():
        for _ in range(3):
            r = scm.step(s)
            s = r.state
    for k, v in s.items():
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            assert v.dtype == torch.float32, k
            assert torch.isfinite(v).all(), k


@TS.needs_scm
def test_float32_scm_all_in_float32_gets_nan_longwave():
    """With double_only_in_float64=False RRTMGP runs in float32: the reference's formula
    overflows and the longwave fluxes are NaN. Kept to document why the default is True."""
    scm, s = _scm32(double_only_in_float64=False)
    with torch.no_grad():
        s = scm.step(s).state
    assert not torch.isfinite(s["LW_flux_up"]).all()


@TS.needs_scm
@pytest.mark.skipif(not HAVE_MPS, reason="no MPS device")
def test_float32_scm_on_mps_stays_finite():
    scm, s = _scm32(device="mps")
    with torch.no_grad():
        for _ in range(2):
            s = scm.step(s).state
    assert s["T_mid"].device.type == "mps"
    assert torch.isfinite(s["T_mid"].cpu()).all() and torch.isfinite(s["LW_flux_up"].cpu()).all()


def test_p3_immersion_freezing_overflows_in_float32_unless_the_option_is_on():
    """Cloud liquid at 190 K: EAMxx's float expression overflows; the option (off by default)
    evaluates it in float64 and rounds, giving the float64 result rounded to float32."""
    from eamxx_torch.p3 import P3Runtime
    from eamxx_torch.p3.functions import cldliq_immersion_freezing
    args = dict(T_atm=190.0, lamc=2.0e5, mu_c=10.0, cdist1=1.0e8, qc_incld=1.0e-6, inv_qc_relvar=1.0,
                qc2qi_hetero_freeze_tend=0.0, nc2ni_immers_freeze_tend=0.0)

    def run(dt, ro):
        return cldliq_immersion_freezing(**{k: torch.tensor([[v]], dtype=dt) for k, v in args.items()}, ro=ro)
    ref = run(torch.float64, P3Runtime())
    off = run(torch.float32, P3Runtime())
    on = run(torch.float32, P3Runtime(immersion_freezing_in_float64=True))
    assert torch.isfinite(ref[0]).all() and not torch.isfinite(off[0]).all()
    assert on[0].dtype == torch.float32
    for a, b in zip(on, ref):
        assert torch.equal(a, b.float())
    same = run(torch.float64, P3Runtime(immersion_freezing_in_float64=True))
    assert all(torch.equal(a, b) for a, b in zip(same, ref))       # no effect in float64
