"""Column water and energy budgets for checking conservation of physics steps.

Not a translation of EAMxx code: these are the diagnostics the per-scheme
workflow uses to check that a port step conserves what the reference conserves.
Definitions follow the SHOC energy fixer (shoc_energy_fixer_impl.hpp), which is
the energy EAMxx's physics conserves: total energy per unit area

    TE = sum_k [ cp*T + g*z + phis + (u^2+v^2)/2 + (Lv+Li)*qv + Li*ql ] * dp_k / g

with the liquid-water reference state (ice has zero latent energy).
"""
from __future__ import annotations

import torch

from . import constants as C


def column_integral(x: torch.Tensor, pseudo_density: torch.Tensor) -> torch.Tensor:
    """sum_k x_k * dp_k / g over the last dimension."""
    return (x * pseudo_density / C.gravit).sum(dim=-1)


def column_total_energy(T_mid, z_mid, phis, u, v, qv, ql, pseudo_density) -> torch.Tensor:
    """Total energy per unit area [J/m2] (see module docstring)."""
    e = (C.CP * T_mid + C.gravit * z_mid + phis[..., None] + 0.5 * (u * u + v * v)
         + (C.LatVap + C.LatIce) * qv + C.LatIce * ql)
    return column_integral(e, pseudo_density)


def column_energy_scale(T_mid, z_mid, pseudo_density) -> torch.Tensor:
    """Magnitude of the dominant energy terms, sum |cp*T| + |g*z|, for relative residuals."""
    return column_integral(torch.abs(C.CP * T_mid) + torch.abs(C.gravit * z_mid), pseudo_density)
