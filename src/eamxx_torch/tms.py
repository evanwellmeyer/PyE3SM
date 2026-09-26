"""Turbulent mountain stress (EAMxx process ``tms``).

Translation of, at E3SM 80fe0a71:
  components/eamxx/src/physics/tms/eamxx_tms_process_interface.cpp
      TurbulentMountainStress::run_impl (preprocessing)
  components/eamxx/src/physics/tms/impl/compute_tms_impl.hpp
      Functions::compute_tms

Inputs (level 0 = model top):
  horiz_winds (ncol, 2, nlev), T_mid, p_mid, pseudo_density, qv (ncol, nlev),
  sgh30, landfrac (ncol,)
Outputs: surf_drag_coeff_tms (ncol,), wind_stress_tms (ncol, 2).

Branching: columns with horo < horomin get zero stress. Both branches are
computed and selected with torch.where. In the zero branch, z0oro is replaced
by 1 before the log so that no inf/NaN is produced there (this keeps gradients
finite; the selected values are unchanged).
"""
from __future__ import annotations

import torch

from . import constants as C
from . import mathfn
from . import physics_functions as PF
from .ekat import max_, min_, rdiv

# compute_tms local constants
HOROMIN = 1.0   # Minimum value of subgrid orographic height for mountain stress [m]
Z0MAX = 100.0   # Maximum value of z_0 for orography [m]
DV2MIN = 0.01   # Minimum shear squared [m2/s2]


def compute_tms(horiz_wind, t_mid, p_mid, exner, z_mid, sgh, landfrac):
    """Functions::compute_tms. Returns (ksrf, tau_tms)."""
    u = horiz_wind[:, 0]
    v = horiz_wind[:, 1]
    horo = C.orocnst * sgh
    active = ~(horo < HOROMIN)

    z0oro = min_(C.z0fac * horo, Z0MAX)
    z0oro = torch.where(active, z0oro, torch.ones_like(z0oro))
    tmp = rdiv(C.Karman, mathfn.log((z_mid[:, -1] + z0oro) / z0oro))
    cd = tmp * tmp

    kb, kt = -1, -2
    tmp_u = u[:, kt] - u[:, kb]
    tmp_v = v[:, kt] - v[:, kb]
    dv2 = max_(tmp_u * tmp_u + tmp_v * tmp_v, DV2MIN)

    ri = (2 * C.gravit * (t_mid[:, kt] * exner[:, kt] - t_mid[:, kb] * exner[:, kb])
          * (z_mid[:, kt] - z_mid[:, kb])
          / ((t_mid[:, kt] * exner[:, kt] + t_mid[:, kb] * exner[:, kb]) * dv2))
    stabfri = max_(0.0, min_(1.0, 1.0 - ri))
    cd = cd * stabfri

    rho = p_mid[:, -1] / (C.Rair * t_mid[:, -1])
    vmag = torch.sqrt(u[:, -1] * u[:, -1] + v[:, -1] * v[:, -1])
    ksrf = rho * cd * vmag * landfrac
    tau_x = -ksrf * u[:, -1]
    tau_y = -ksrf * v[:, -1]

    zero = torch.zeros_like(ksrf)
    ksrf = torch.where(active, ksrf, zero)
    tau = torch.stack([torch.where(active, tau_x, zero), torch.where(active, tau_y, zero)], dim=-1)
    return ksrf, tau


def tms(horiz_winds, T_mid, p_mid, pseudo_density, qv, sgh30, landfrac):
    """TurbulentMountainStress::run_impl. Returns the two computed fields."""
    exner = PF.exner_function(p_mid)
    dz = PF.calculate_dz(pseudo_density, p_mid, T_mid, qv)
    z_int = PF.calculate_z_int(dz, 0.0)
    z_mid = PF.calculate_z_mid(z_int)
    ksrf, tau = compute_tms(horiz_winds, T_mid, p_mid, exner, z_mid, sgh30, landfrac)
    return {"surf_drag_coeff_tms": ksrf, "wind_stress_tms": tau}, {"exner": exner, "z_mid": z_mid}
