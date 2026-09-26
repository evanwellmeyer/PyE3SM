"""The comparison helper must not let a mismatch through."""
import numpy as np

from eamxx_torch.testing import compare

inf, nan = np.inf, np.nan


def test_bitwise_equal_special_values():
    a = np.array([0.0, 1.0, inf, -inf, nan])
    assert compare("x", a.copy(), a).ok()


def test_detects_mismatches():
    ref = np.array([1.0, inf, nan, 0.0, inf])
    for bad in ([1.0 + 1e-15, inf, nan, 0.0, inf],   # tiny difference
                [1.0, -inf, nan, 0.0, inf],          # inf sign
                [1.0, inf, 0.0, 0.0, inf],           # NaN lost
                [1.0, inf, nan, -0.0, inf],          # sign of zero
                [1.0, inf, nan, 0.0, 1e308]):        # inf vs finite
        assert not compare("x", np.array(bad), ref).ok(), bad


def test_tolerance_mode():
    ref = np.array([1.0, 100.0, inf])
    assert compare("x", np.array([1.0 + 1e-13, 100.0 - 1e-11, inf]), ref, rtol=1e-12).ok()
    assert not compare("x", np.array([1.0 + 1e-11, 100.0, inf]), ref, rtol=1e-12).ok()
