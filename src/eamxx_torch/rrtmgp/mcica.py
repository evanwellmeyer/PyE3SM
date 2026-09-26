"""McICA subcolumn sampling of clouds, with EAMxx's random numbers.

Translation of, E3SM 80fe0a71:
  - components/eamxx/src/physics/rrtmgp/eamxx_rrtmgp_interface.hpp
      get_subcolumn_mask (maximum-random overlap), get_subsampled_clouds (SW and LW)
  - rrtmgp external cpp/rrtmgp_conversion.h   conv::Random (copied from YAKL_random.h)

conv::Random works on unsigned 64-bit integers. torch has no unsigned 64-bit
shifts, so the port holds the same bits in int64: addition, subtraction, xor,
or and left shift wrap identically; the logical right shift is an arithmetic
shift followed by a mask. The unsigned-to-double conversion rounds once, as the
hardware does (hi*2^32 + lo, both exact).
"""
from __future__ import annotations

import torch

_MASK32 = (1 << 32) - 1


def _lsr(x: torch.Tensor, s: int) -> torch.Tensor:
    """Logical right shift of the uint64 held in x."""
    return (x >> s) & ((1 << (64 - s)) - 1)


def _rot(x: torch.Tensor, k: int) -> torch.Tensor:
    return (x << k) | _lsr(x, 64 - k)


class Random:
    """conv::Random: state (a, b, c, d) of uint64 held as int64 tensors."""

    def __init__(self, seed: torch.Tensor):
        seed = seed.to(torch.int64)
        self.a = torch.full_like(seed, 0xf1ea5eed)
        self.b = seed.clone()
        self.c = seed.clone()
        self.d = seed.clone()
        for _ in range(20):
            self.gen()

    def gen(self) -> torch.Tensor:
        e = self.a - _rot(self.b, 7)
        self.a = self.b ^ _rot(self.c, 13)
        self.b = self.c + _rot(self.d, 37)
        self.c = self.d + e
        self.d = e + self.a
        return self.d

    def genFP(self, dtype=torch.float64) -> torch.Tensor:
        """static_cast<T>(gen()) / static_cast<T>(UINT64_MAX)."""
        v = self.gen()
        hi = _lsr(v, 32).to(torch.float64)
        lo = (v & _MASK32).to(torch.float64)
        x = hi * 4294967296.0 + lo
        return (x / 18446744073709551615.0).to(dtype)


def get_subcolumn_mask(cldf: torch.Tensor, seeds: torch.Tensor, ngpt: int) -> torch.Tensor:
    """Maximum-random overlap (overlap_option 1). cldf (ncol, nlay), seeds (ncol,) int.

    Returns the subcolumn mask (ncol, nlay, ngpt), bool.
    """
    ncol, nlay = cldf.shape
    ilay = torch.arange(nlay, device=cldf.device)[:, None]
    igpt = torch.arange(ngpt, device=cldf.device)[None, :]
    # conv::Random rand(seeds(icol) + ilay*ngpt + igpt): int arithmetic, then to uint64
    # The generator's uint64 arithmetic and its conversion to a real run on the CPU (the
    # conversion rounds in double, as the hardware does; devices like MPS have no float64);
    # the random numbers then go to cldf's device.
    s = seeds.to("cpu")[:, None, None].to(torch.int64) + (ilay.cpu() * ngpt + igpt.cpu())[None]
    cldx = Random(s).genFP(cldf.dtype).to(cldf.device)
    layers = [cldx[:, 0, :]]
    for k in range(1, nlay):
        prev = layers[-1]
        one_m = 1.0 - cldf[:, k - 1, None]
        layers.append(torch.where(prev > one_m, prev, cldx[:, k, :] * one_m))
    cldx = torch.stack(layers, dim=1)
    return cldx > (1.0 - cldf)[..., None]


def column_seeds(p_lay: torch.Tensor, ilay: int) -> torch.Tensor:
    """seeds(icol) = 1e9 * (p_lay(icol,ilay) - int(p_lay(icol,ilay))), stored as int."""
    p = p_lay[:, ilay]
    frac = p - p.to(torch.int64).to(p.dtype)
    return (1e9 * frac).to(torch.int64)


def get_subsampled_clouds(tau, ssa, g, cld, p_lay, gpt2band, seed_layer: int, seeds=None):
    """get_subsampled_clouds: band cloud optics -> one sampled cloud state per g-point.

    tau/ssa/g: (ncol, nlay, nbnd) band optics (ssa and g None for the 1-scalar LW form).
    seed_layer: nlay-1 for SW, nlay-2 for LW.
    seeds: optional (ncol,) int64 seeds replacing EAMxx's pressure-digit seeds (the
      mcica_seed_from_pressure = False option; see RRTMGPRuntime).
    Returns (tau, ssa, g) by g-point (ssa, g None for LW).
    """
    ngpt = gpt2band.shape[0]
    # radiative cloud fraction: cloud fraction only where some band has optical depth
    has_tau = (tau > 0).any(dim=-1)
    cldfrac_rad = torch.where(has_tau, cld, torch.zeros_like(cld))
    if seeds is None:
        seeds = column_seeds(p_lay, seed_layer)
    mask = get_subcolumn_mask(cldfrac_rad, seeds, ngpt)
    gb = gpt2band.to(tau.device)
    zero = torch.zeros((), dtype=tau.dtype, device=tau.device)
    out_tau = torch.where(mask, tau[..., gb], zero)
    if ssa is None:
        return out_tau, None, None
    return out_tau, torch.where(mask, ssa[..., gb], zero), torch.where(mask, g[..., gb], zero)
