"""Common physics functions and column operations of EAMxx.

Translation of, at E3SM 80fe0a71:
  components/eamxx/src/share/physics/eamxx_common_physics_functions_impl.hpp
      (PhysicsFunctions<DeviceT>)
  components/eamxx/src/share/util/eamxx_column_ops.hpp
      ColumnOps::column_scan (pack size 1 branch), compute_midpoint_values
      (adj_avg), compute_midpoint_delta (diff)

All functions take (..., nlev) tensors, level 0 at the model top.

Summation order follows EAMxx built with SCREAM_PACK_SIZE=1 (its GPU layout,
and our validation reference): scans are plain sequential sums. EAMxx CPU
builds with packs of 16 sum in a different order, so they differ from this
at roundoff level only.
"""
from __future__ import annotations

import torch

from . import constants as C
from . import mathfn


# ---------------------------------------------------------------- pointwise
def exner_function(pressure: torch.Tensor) -> torch.Tensor:
    """pow(pressure/p0, rd*inv_cp)."""
    return mathfn.pow(pressure / C.P0, C.RD * C.INV_CP)


def calculate_theta_from_T(temperature, pressure):
    return temperature / exner_function(pressure)


def calculate_T_from_theta(theta, pressure):
    return theta * exner_function(pressure)


def calculate_thetal_from_theta(theta, temperature, qc):
    """theta - (theta/temperature) * (LatVap/Cpair) * qc."""
    return theta - (theta / temperature) * (C.LatVap / C.Cpair) * qc


_C1_VIRT = -C.ONE + C.ONE / C.ep_2  # constexpr c1 = -one + one/ep_2


def calculate_virtual_temperature(temperature, qv):
    """temperature * (one + c1*qv)."""
    return temperature * (C.ONE + _C1_VIRT * qv)


def calculate_temperature_from_virtual_temperature(T_virtual, qv):
    return T_virtual / (C.ONE + _C1_VIRT * qv)


def calculate_dse(temperature, z, surf_geopotential):
    """cp*temperature + g*z + surf_geopotential."""
    return C.CP * temperature + C.gravit * z + surf_geopotential


def calculate_temperature_from_dse(dse, z, surf_geopotential):
    return (dse - C.gravit * z - surf_geopotential) / C.CP


def calculate_density(pseudo_density, dz):
    """pseudo_density/dz/g."""
    return pseudo_density / dz / C.gravit


def calculate_vertical_velocity(omega, density):
    """-omega/(density*g)."""
    return -omega / (density * C.gravit)


def calculate_wetmmr_from_drymmr(drymmr, qv_dry):
    return drymmr / (1 + qv_dry)


def calculate_drymmr_from_wetmmr(wetmmr, qv_wet):
    return wetmmr / (1 - qv_wet)


def calculate_wetmmr_from_drymmr_dp_based(drymmr, pseudo_density, pseudo_density_dry):
    return drymmr * pseudo_density_dry / pseudo_density


def calculate_drymmr_from_wetmmr_dp_based(wetmmr, pseudo_density, pseudo_density_dry):
    return wetmmr * pseudo_density / pseudo_density_dry


_RD_OVER_G = C.RD / C.gravit


def calculate_dz(pseudo_density, p_mid, T_mid, qv):
    """(rd/g)*pseudo_density*T_virtual / p_mid."""
    T_virtual = calculate_virtual_temperature(T_mid, qv)
    return _RD_OVER_G * pseudo_density * T_virtual / p_mid


# ---------------------------------------------------------------- column ops
def column_scan_from_bottom(dx_m: torch.Tensor, s0: float | torch.Tensor = 0.0) -> torch.Tensor:
    """ColumnOps::column_scan<FromTop=false>, pack size 1.

    x_i(nlev) = s0; accumulator (starts at 0) += dx_m(k) for k = nlev-1 .. 0,
    x_i(k) = s0 + accumulator. Returns (..., nlev+1).
    torch.cumsum is a sequential loop on CPU, so the rounding matches exactly.
    """
    acc = torch.cumsum(torch.flip(dx_m, dims=(-1,)), dim=-1).flip(dims=(-1,))
    s0 = torch.as_tensor(s0, dtype=dx_m.dtype, device=dx_m.device)
    x_top = s0 + acc
    x_bot = s0.expand(dx_m.shape[:-1]).unsqueeze(-1) if s0.ndim else torch.full_like(dx_m[..., :1], float(s0))
    return torch.cat([x_top, x_bot], dim=-1)


def column_scan_from_top(dx_m: torch.Tensor, s0: float | torch.Tensor = 0.0) -> torch.Tensor:
    """ColumnOps::column_scan<FromTop=true>, pack size 1.

    x_i(0) = s0; accumulator += dx_m(k) for k = 0 .. nlev-1; x_i(k+1) = s0 + accumulator.
    """
    acc = torch.cumsum(dx_m, dim=-1)
    s0 = torch.as_tensor(s0, dtype=dx_m.dtype, device=dx_m.device)
    x_rest = s0 + acc
    x_top = s0.expand(dx_m.shape[:-1]).unsqueeze(-1) if s0.ndim else torch.full_like(dx_m[..., :1], float(s0))
    return torch.cat([x_top, x_rest], dim=-1)


def compute_midpoint_values(x_i: torch.Tensor) -> torch.Tensor:
    """adj_avg, pack size 1: x_m(k) = (x_i(k) + x_i(k+1)) / 2."""
    return (x_i[..., :-1] + x_i[..., 1:]) / 2


def compute_midpoint_delta(x_i: torch.Tensor) -> torch.Tensor:
    """diff: dx_m(k) = x_i(k+1) - x_i(k)."""
    return x_i[..., 1:] - x_i[..., :-1]


def calculate_z_int(dz: torch.Tensor, z_surf: float = 0.0) -> torch.Tensor:
    """PhysicsFunctions::calculate_z_int: column_scan<FromTop=false>(dz, z_surf)."""
    return column_scan_from_bottom(dz, z_surf)


def calculate_z_mid(z_int: torch.Tensor) -> torch.Tensor:
    """PhysicsFunctions::calculate_z_mid: compute_midpoint_values(z_int)."""
    return compute_midpoint_values(z_int)
