"""Atmosphere-ocean surface fluxes: E3SM's coupler bulk formula, scheme 0 (Large and Pond).

Translation of share/util/shr_flux_mod.F90 at E3SM 80fe0a71: shr_flux_atmOcn with
ocn_surface_flux_scheme = 0, as the MCT coupler calls it for an EAMxx atmosphere
(driver-mct/main/seq_flux_mct.F90): explicit flux integration (no wsresp/tau_est), no
gustiness from the atmosphere (ATM_SUPPLIES_GUSTINESS = FALSE for SCREAM), no wave
coupling, no cold-air-outbreak modification, flux_convergence = 0 and at most 2
iterations (flux_max_iteration = -1 -> 2 for explicit), minimum wind 0.5 m/s. The
water-isotope fluxes are not ported (EAMxx does not use them).

EAMxx itself computes no surface fluxes (docs/plan.md, section 1). This scheme is
what E3SM's coupler would give an EAMxx column over open ocean. All inputs are
(ncol,) tensors. Fluxes follow the coupler convention: positive downward.

Constants are E3SM's shr_const_mod values (loc_* defaults), evaluated with the same
operations as the Fortran parameters. gfortran expands ts**4 into two squarings.
"""
from __future__ import annotations

import math

import torch

from . import mathfn
from .ekat import max_, min_, rdiv

# shr_const_mod.F90
SHR_CONST_G = 9.80616
SHR_CONST_STEBOL = 5.670374419e-8
SHR_CONST_BOLTZ = 1.38065e-23
SHR_CONST_AVOGAD = 6.02214e26
SHR_CONST_RGAS = SHR_CONST_AVOGAD * SHR_CONST_BOLTZ
SHR_CONST_MWDAIR = 28.966
SHR_CONST_MWWV = 18.016
SHR_CONST_RDAIR = SHR_CONST_RGAS / SHR_CONST_MWDAIR
SHR_CONST_RWV = SHR_CONST_RGAS / SHR_CONST_MWWV
SHR_CONST_ZVIR = (SHR_CONST_RWV / SHR_CONST_RDAIR) - 1.0
SHR_CONST_KARMAN = 0.4
SHR_CONST_CPDAIR = 1.00464e3
SHR_CONST_CPWV = 1.810e3
SHR_CONST_CPVIR = (SHR_CONST_CPWV / SHR_CONST_CPDAIR) - 1.0
SHR_CONST_LATVAP = 2.501e6
SHR_CONST_SPVAL = 1.0e30

ZREF = 10.0     # reference height (m)
ZTREF = 2.0     # reference height for air T (m)
DTAUMIN = 0.01  # stress convergence limit (Pa); only with implicit stress
MINWIND = 0.5   # seq_flux_atmocn_minwind default (namelist_definition_drv.xml)


def _qsat(Tk):
    """Saturation humidity of air (kg/m^3)."""
    return rdiv(640380.0, mathfn.exp(rdiv(5107.4, Tk)))


def _cdn(Umps):
    """Neutral 10 m drag coefficient, Large et al. (1994)."""
    return rdiv(0.0027, Umps) + 0.000142 + 0.0000764 * Umps


def _psimhu(xd):
    return (mathfn.log((1.0 + xd * (2.0 + xd)) * (1.0 + xd * xd) / 8.0)
            - 2.0 * mathfn.atan(xd) + 1.571)


def _psixhu(xd):
    return 2.0 * mathfn.log((1.0 + xd * xd) / 2.0)


def _sign(a: float, b):
    """Fortran SIGN(a, b) for a > 0: |a| with the sign of b (gfortran keeps -0.0)."""
    return torch.copysign(torch.full_like(b, a), b)


def shr_flux_atmocn(zbot, ubot, vbot, thbot, qbot, rbot, tbot, us, vs, ts, mask=None,
                    minwind: float = MINWIND, flux_con_tol: float = 0.0,
                    flux_con_max_iter: int = 2) -> dict:
    """Bulk atmosphere-ocean fluxes.

    Inputs: zbot atm level height (m), ubot/vbot winds (m/s), thbot potential T (K),
    qbot specific humidity (kg/kg), rbot air density (kg/m^3), tbot T (K), us/vs ocean
    velocity (m/s), ts ocean temperature (K); mask (bool) marks ocean points (others
    get the missing value 1e30, as in the coupler).
    Returns sen, lat, lwup (W/m^2), evap (kg/m^2/s), taux, tauy (N/m^2), tref, qref,
    duu10n, u10res, ustar, re, ssq, and n_iter (iterations done per point).
    """
    karman, g, zvir = SHR_CONST_KARMAN, SHR_CONST_G, SHR_CONST_ZVIR
    al2 = math.log(ZREF / ZTREF)

    # needed quantities
    du, dv = ubot - us, vbot - vs
    wind0 = max_(torch.sqrt(du * du + dv * dv), 0.01)
    vmag = wind0
    vmag = max_(minwind, vmag)
    ssq = 0.98 * _qsat(ts) / rbot
    delt = thbot - ts
    delq = qbot - ssq
    alz = mathfn.log(zbot / ZREF)
    cp = SHR_CONST_CPDAIR * (1.0 + SHR_CONST_CPVIR * ssq)

    # first estimate of Z/L and ustar, tstar and qstar (neutral coefficients)
    stable = 0.5 + _sign(0.5, delt)
    rdn = torch.sqrt(_cdn(vmag))
    rhn = (1.0 - stable) * 0.0327 + stable * 0.018
    ren = torch.full_like(ts, 0.0346)
    ustar = rdn * vmag
    tstar = rhn * delt
    qstar = ren * delq
    ustar_prev = ustar * 2.0
    tau_diff = torch.zeros_like(ts)       # no implicit stress: tau_diff = 0
    wind_adj = wind0
    hol = torch.full_like(ts, SHR_CONST_SPVAL)
    u10n = torch.full_like(ts, SHR_CONST_SPVAL)
    psixh = torch.full_like(ts, SHR_CONST_SPVAL)
    rd = rh = re = torch.full_like(ts, SHR_CONST_SPVAL)
    n_iter = torch.zeros_like(ts, dtype=torch.int64)

    for it in range(flux_con_max_iter):
        run = ((torch.abs((ustar - ustar_prev) / ustar) > flux_con_tol)
               | (torch.abs(tau_diff) > DTAUMIN))
        n_iter = n_iter + run.to(torch.int64)
        ustar_prev = torch.where(run, ustar, ustar_prev)
        # stability and stability functions
        hol_n = karman * g * zbot * (tstar / thbot + qstar / (1.0 / zvir + qbot)) / (ustar * ustar)
        hol_n = torch.copysign(min_(torch.abs(hol_n), 10.0), hol_n)   # SIGN(MIN(|hol|,10), hol)
        stable_n = 0.5 + _sign(0.5, hol_n)
        xsq = max_(torch.sqrt(torch.abs(1.0 - 16.0 * hol_n)), 1.0)
        xqq = torch.sqrt(xsq)
        psimh = -5.0 * hol_n * stable_n + (1.0 - stable_n) * _psimhu(xqq)
        psixh_n = -5.0 * hol_n * stable_n + (1.0 - stable_n) * _psixhu(xqq)
        # shift wind speed using old coefficient; update the 10 m neutral coefficients
        rd_n = rdn / max_(1.0 + rdn / karman * (alz - psimh), 1.e-3)
        u10n_n = vmag * rd_n / rdn
        rdn_n = torch.sqrt(_cdn(u10n_n))
        ren_n = torch.full_like(ts, 0.0346)
        rhn_n = (1.0 - stable_n) * 0.0327 + stable_n * 0.018
        # shift all coefficients to measurement height and stability
        rd_n = rdn_n / max_(1.0 + rdn_n / karman * (alz - psimh), 1.e-3)
        rh_n = rhn_n / (1.0 + rhn_n / karman * (alz - psixh_n))
        re_n = ren_n / (1.0 + ren_n / karman * (alz - psixh_n))
        upd = lambda new, old: torch.where(run, new, old)   # noqa: E731
        hol, stable, psixh, u10n = upd(hol_n, hol), upd(stable_n, stable), upd(psixh_n, psixh), upd(u10n_n, u10n)
        rdn, ren, rhn = upd(rdn_n, rdn), upd(ren_n, ren), upd(rhn_n, rhn)
        rd, rh, re = upd(rd_n, rd), upd(rh_n, rh), upd(re_n, re)
        ustar = upd(rd_n * vmag, ustar)
        tstar = upd(rh_n * delt, tstar)
        qstar = upd(re_n * delq, qstar)

    # fluxes
    tau = rbot * ustar * ustar
    taux = tau * du / vmag * (wind_adj / wind0)
    tauy = tau * dv / vmag * (wind_adj / wind0)
    sen = cp * tau * tstar / ustar
    lat = SHR_CONST_LATVAP * tau * qstar / ustar
    ts2 = ts * ts
    lwup = -SHR_CONST_STEBOL * (ts2 * ts2)
    evap = lat / SHR_CONST_LATVAP

    # diagnostics: 2 m reference T and q, 10 m wind speed squared
    hol_final = hol
    hol = hol * ZTREF / zbot
    xsq = max_(1.0, torch.sqrt(torch.abs(1.0 - 16.0 * hol)))
    xqq = torch.sqrt(xsq)
    psix2 = -5.0 * hol * stable + (1.0 - stable) * _psixhu(xqq)
    fac = (rh / karman) * (alz + al2 - psixh + psix2)
    tref = thbot - delt * fac
    tref = tref - 0.01 * ZTREF
    fac = (re / karman) * (alz + al2 - psixh + psix2)
    qref = qbot - delq * fac
    duu10n = u10n * u10n
    u10res = u10n * wind_adj / vmag

    out = dict(sen=sen, lat=lat, lwup=lwup, evap=evap, taux=taux, tauy=tauy, tref=tref,
               qref=qref, duu10n=duu10n, u10res=u10res, ustar=ustar, re=re, ssq=ssq)
    if mask is not None:
        spval = torch.full_like(ts, SHR_CONST_SPVAL)
        out = {k: torch.where(mask, v, spval) for k, v in out.items()}
    out["n_iter"] = n_iter
    out["hol"] = hol_final     # z/L at zbot (diagnostic, for branch coverage)
    return out
