"""Ocean surface fluxes vs E3SM's shr_flux_atmOcn (scheme 0), E3SM 80fe0a71.

Reference: share/util/shr_flux_mod.F90 compiled unmodified into a small Fortran
harness (reference/eamxx_ext/flux_harness, reference/scripts/build_flux_harness.sh)
and called as the MCT coupler calls it for EAMxx: explicit stress, no gustiness, no
waves, at most 2 iterations, minimum wind 0.5 m/s. Built with -O
-ffp-contract=off; E3SM's own -O build gives identical results on these inputs.

Fixture (reference/scripts/flux_check.py): 3000 random ocean-surface states plus
edge cases: zero temperature difference, zero and calm wind, |z/L| clipped at 10
on both sides, ocean currents, a 0.5 m first level, dry air, masked (non-ocean)
points.

Tolerances: libm math backend bitwise; torch math backend normwise <= 2.2e-16
(bound 1e-14 here).
"""
from pathlib import Path

import numpy as np
import pytest
import torch

from eamxx_torch import mathfn
from eamxx_torch.surface_flux import SHR_CONST_SPVAL, shr_flux_atmocn

DATA = Path(__file__).parent / "data" / "surface_flux" / "cases.npz"


def _load():
    with np.load(DATA) as z:
        inp = {k[3:]: torch.as_tensor(z[k]) for k in z.files if k.startswith("in/")}
        ref = {k[4:]: z[k] for k in z.files if k.startswith("out/")}
        minwind = float(z["minwind"])
    return inp, ref, minwind


def _run(inp, minwind):
    t = inp
    return shr_flux_atmocn(t["zbot"], t["ubot"], t["vbot"], t["thbot"], t["qbot"], t["rbot"],
                           t["tbot"], t["us"], t["vs"], t["ts"], mask=t["mask"] != 0,
                           minwind=minwind)


@pytest.mark.reference
def test_bitwise_libm():
    inp, ref, minwind = _load()
    with mathfn.use_backend("libm"):
        out = _run(inp, minwind)
    for k, r in ref.items():
        assert np.array_equal(out[k].numpy(), r), k


@pytest.mark.reference
def test_torch_math():
    inp, ref, minwind = _load()
    out = _run(inp, minwind)
    m = inp["mask"].numpy() != 0
    for k, r in ref.items():
        err = np.abs(out[k].numpy() - r).max() / np.abs(r[m]).max()
        assert err <= 1e-14, (k, err)


def test_branches_covered_and_masked():
    inp, ref, minwind = _load()
    out = _run(inp, minwind)
    m = inp["mask"] != 0
    assert int(((out["hol"] == 10) & m).sum()) > 0 and int(((out["hol"] == -10) & m).sum()) > 0
    assert torch.all(out["sen"][~m] == SHR_CONST_SPVAL)
    # a colder ocean than air gives downward sensible heat (fluxes positive downward)
    warm_air = m & (inp["thbot"] > inp["ts"] + 1)
    assert torch.all(out["sen"][warm_air] > 0)


def test_batched_shapes():
    inp, ref, minwind = _load()
    two = {k: v[:600].reshape(100, 6) for k, v in inp.items()}
    out = _run(two, minwind)
    flat = _run({k: v[:600] for k, v in inp.items()}, minwind)
    for k in ref:
        assert torch.equal(out[k].reshape(-1), flat[k]), k
