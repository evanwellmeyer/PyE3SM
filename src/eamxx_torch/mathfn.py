"""Transcendental functions used by the port, with a switchable backend.

Why this exists. The reference C++ calls the system libm (std::pow, std::exp,
...). torch's CPU kernels use their own vectorized implementations, which can
differ from libm by about one unit in the last place (measured: torch.pow
differs from libm pow in ~1.4% of values, by 1 ulp). That is roundoff, but it
hides whether the translation itself is exact.

So ported code calls these wrappers instead of torch.* directly:
  - backend "torch" (default): plain torch ops. Fast, GPU, autograd,
    torch.compile. This is what production uses.
  - backend "libm" (tests only): evaluates with numpy on CPU, which calls the
    same libm as the reference. Float64 CPU tensors only, no autograd.
Tests replay each scheme under "libm" and require bitwise agreement (the
translation is exact), and under "torch" with an ulp-level tolerance.

sqrt is not wrapped: IEEE requires it to be correctly rounded everywhere.
"""
from __future__ import annotations

import contextlib

import numpy as np
import torch

_BACKEND = "torch"


def backend() -> str:
    return _BACKEND


@contextlib.contextmanager
def use_backend(name: str):
    """Temporarily switch backend ("torch" or "libm")."""
    global _BACKEND
    if name not in ("torch", "libm"):
        raise ValueError(name)
    old, _BACKEND = _BACKEND, name
    try:
        yield
    finally:
        _BACKEND = old


def _np(x):
    if isinstance(x, torch.Tensor):
        if x.requires_grad:
            raise RuntimeError("libm math backend does not support autograd")
        if x.device.type != "cpu":
            raise RuntimeError("libm math backend is CPU only")
        return x.numpy()
    return x


def _libm(fn, *args):
    ref = next(a for a in args if isinstance(a, torch.Tensor))
    out = fn(*[_np(a) for a in args])
    return torch.as_tensor(np.asarray(out), dtype=ref.dtype, device=ref.device)


def pow(x, y):  # noqa: A001
    # numpy's power is not libm pow: for an exponent of 2 it multiplies (exact),
    # while libm pow can be 1 ulp away (measured: 0.13% of values). P3 calls
    # pow(lamc, 2), so the libm backend has to go through libm itself.
    if _BACKEND == "libm":
        return _libm(_libm_fn2("pow"), x, y)
    return torch.pow(x, y)


def exp10(x):
    """pow(10, x) as the reference build computes it.

    clang rewrites pow(10.0, x) to __exp10(x) on Darwin (LLVM SimplifyLibCalls),
    and the two differ by 1 ulp for some arguments, so the libm backend calls
    __exp10 directly.
    """
    if _BACKEND == "libm":
        return _libm(_libm_fn("__exp10"), x)
    return torch.pow(10.0, x)


def exp(x):
    return _libm(np.exp, x) if _BACKEND == "libm" else torch.exp(x)


def log(x):
    return _libm(np.log, x) if _BACKEND == "libm" else torch.log(x)


def log10(x):
    return _libm(np.log10, x) if _BACKEND == "libm" else torch.log10(x)


def expm1(x):
    return _libm(np.expm1, x) if _BACKEND == "libm" else torch.expm1(x)


def log1p(x):
    return _libm(np.log1p, x) if _BACKEND == "libm" else torch.log1p(x)


def cbrt(x):
    if _BACKEND == "libm":
        return _libm(np.cbrt, x)
    # torch has no cbrt; std::cbrt(x) keeps the sign of x.
    return torch.sign(x) * torch.pow(torch.abs(x), 1.0 / 3.0)


def tanh(x):
    # numpy's tanh is its own SIMD code, not libm (29k/400k mismatches measured)
    return _libm(_libm_fn("tanh"), x) if _BACKEND == "libm" else torch.tanh(x)


_LIBM = None


def _libm_fn2(name):
    """A two-argument libm function via ctypes."""
    global _LIBM
    import ctypes
    import ctypes.util
    if _LIBM is None:
        _LIBM = ctypes.CDLL(ctypes.util.find_library("m"))
    f = getattr(_LIBM, name)
    f.restype, f.argtypes = ctypes.c_double, [ctypes.c_double, ctypes.c_double]
    return np.vectorize(f, otypes=[np.float64])


def _libm_fn(name):
    """A libm function via ctypes (CPython's math.gamma is not libm tgamma)."""
    global _LIBM
    import ctypes
    import ctypes.util
    if _LIBM is None:
        _LIBM = ctypes.CDLL(ctypes.util.find_library("m"))
    f = getattr(_LIBM, name)
    f.restype, f.argtypes = ctypes.c_double, [ctypes.c_double]
    return np.vectorize(f, otypes=[np.float64])


def erf(x):
    return _libm(_libm_fn("erf"), x) if _BACKEND == "libm" else torch.erf(x)


# Trigonometric functions (the zenith angle). The libm backend calls libm itself.
def sin(x):
    return _libm(_libm_fn("sin"), x) if _BACKEND == "libm" else torch.sin(x)


def cos(x):
    return _libm(_libm_fn("cos"), x) if _BACKEND == "libm" else torch.cos(x)


def tan(x):
    return _libm(_libm_fn("tan"), x) if _BACKEND == "libm" else torch.tan(x)


def acos(x):
    return _libm(_libm_fn("acos"), x) if _BACKEND == "libm" else torch.acos(x)


def atan(x):
    return _libm(_libm_fn("atan"), x) if _BACKEND == "libm" else torch.atan(x)


def tgamma(x):
    if _BACKEND == "libm":
        return _libm(_libm_fn("tgamma"), x)
    return torch.exp(torch.lgamma(x)) * torch.where(
        (x < 0) & (torch.remainder(torch.floor(x), 2) == 1), -1.0, 1.0).to(x.dtype)
