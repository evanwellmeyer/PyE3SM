"""ZM thermodynamic helpers: saturation, entropy and its inversion.

Translation of, at E3SM 80fe0a71 (components/eamxx/src/physics/zm/impl/):
  zm_inline_functions.hpp     goffgratch_svp_water, qsat_hPa, qsat (table lookup)
  zm_entropy_impl.hpp         entropy
  zm_invert_entropy_impl.hpp  invert_entropy (Brent's method)

All functions work elementwise on tensors of any shape.
"""
from __future__ import annotations

import torch

from .. import constants as PC
from .. import mathfn
from ..ekat import max_, min_, rdiv
from .constants import ZMC

_MAGIC_W = (-7.90298, 5.02808, 1.3816e-7, 11.344, 8.1328e-3, -3.49149, 1013.246)


def goffgratch_svp_water(t):
    """pow(10, ...)*100. clang turns pow(10, x) into __exp10(x) (mathfn.exp10)."""
    m1, m2, m3, m4, m5, m6, m7 = _MAGIC_W
    log10_m7 = mathfn.log10(torch.as_tensor(m7, dtype=t.dtype, device=t.device))
    arg = (m1 * (rdiv(ZMC.tboil, t) - 1)
           + m2 * mathfn.log10(rdiv(ZMC.tboil, t))
           - m3 * (mathfn.exp10(m4 * (1 - t / ZMC.tboil)) - 1)
           + m5 * (mathfn.exp10(m6 * (rdiv(ZMC.tboil, t) - 1)) - 1)
           + log10_m7)
    return mathfn.exp10(arg) * 100


def qsat_hPa(t, p):
    """Returns (es [hPa], qm [kg/kg]); p in hPa."""
    es = goffgratch_svp_water(t)
    p100 = p * 100
    qm = torch.where((p100 - es) <= 0, torch.ones_like(es),
                     PC.ep_2 * es / (p100 - ZMC.omeps * es))
    es = min_(es, p100)
    es = es * 0.01
    return es, qm


def qsat(t, p, estbl: torch.Tensor):
    """Table version (conv_evap). Returns (es [Pa], qs [kg/kg]); p in Pa."""
    t_tmp = max_(min_(t, ZMC.tmax) - ZMC.tmin, 0.0)
    i = t_tmp.to(torch.int64)                   # int(t_tmp), t_tmp >= 0
    weight = t_tmp - torch.trunc(t_tmp)
    es = (1 - weight) * estbl[i] + weight * estbl[i + 1]
    qs = torch.where((p - es) < 0, torch.ones_like(es), PC.ep_2 * es / (p - ZMC.omeps * es))
    es = min_(es, p)
    return es, qs


_CPLIQ_MINUS_CPWV = PC.CpLiq - ZMC.cpwv


def entropy(tk, p, qtot):
    """Entropy per unit mass of dry air (Raymond and Blyth 1992, eq. 1); p in mb."""
    L = PC.LatVap - _CPLIQ_MINUS_CPWV * (tk - PC.Tmelt)
    _est, qst = qsat_hPa(tk, p)
    qv = min_(qtot, qst)
    e = qv * p / (PC.ep_2 + qv)
    return ((PC.Cpair + qtot * PC.CpLiq) * mathfn.log(tk / PC.Tmelt)
            - PC.Rair * mathfn.log((p - e) / ZMC.pref)
            + L * qv / tk
            - qv * PC.RH2O * mathfn.log(qv / qst))


def invert_entropy(s, p, qt, tfg, active=None, early_exit: bool = True):
    """Brent's method for the temperature with entropy s (at p, qt), from first guess tfg.

    Elementwise over tensors. Elements outside `active` do not iterate (they start as
    converged and return tfg). The reference aborts if an element has not converged
    after LOOPMAX+1 iterations; here the third return value marks such elements.
    With early_exit, the loop stops once every element has converged (a converged
    element's state no longer changes, so the result is the same).
    Returns (t, qst, not_converged).
    """
    t = tfg
    a = tfg - 10
    b = tfg + 10
    fa = entropy(a, p, qt) - s
    fb = entropy(b, p, qt) - s
    c = b
    fc = fb
    d = torch.zeros_like(tfg)
    ebr = torch.zeros_like(tfg)
    done = torch.zeros_like(tfg, dtype=torch.bool) if active is None else ~active
    if active is not None:  # inactive elements: keep b = tfg so the final t is finite
        b = torch.where(active, b, tfg)
    for _ in range(ZMC.LOOPMAX + 1):
        run = ~done
        same_sign = ((fb > 0) & (fc > 0)) | ((fb < 0) & (fc < 0))
        m = run & same_sign
        c = torch.where(m, a, c)
        d = torch.where(m, b - a, d)
        fc = torch.where(m, fa, fc)
        ebr = torch.where(m, d, ebr)

        m = run & (torch.abs(fc) < torch.abs(fb))
        a, b, c = torch.where(m, b, a), torch.where(m, c, b), torch.where(m, b, c)
        fa, fb, fc = torch.where(m, fb, fa), torch.where(m, fc, fb), torch.where(m, fb, fc)

        tolerance = 2 * ZMC.tol_eps * torch.abs(b) + ZMC.half * ZMC.tol_coeff
        xm = ZMC.half * (c - b)
        converged = (torch.abs(xm) <= tolerance) | (fb == 0)
        done = done | (run & converged)
        run = ~done
        if early_exit and not bool(run.any()):
            break

        # interpolation (inverse quadratic or secant) or bisection
        use_interp = (torch.abs(ebr) >= tolerance) & (torch.abs(fa) > torch.abs(fb))
        fa_safe = torch.where(use_interp, fa, torch.ones_like(fa))
        fc_safe = torch.where(use_interp & (a != c), fc, torch.ones_like(fc))
        sbr = fb / fa_safe
        pbr_sec = 2 * xm * sbr
        qbr_sec = 1 - sbr
        qbr_q = fa_safe / fc_safe
        rbr = fb / fc_safe
        pbr_q = sbr * (2 * xm * qbr_q * (qbr_q - rbr) - (b - a) * (rbr - 1))
        qbr_q = (qbr_q - 1) * (rbr - 1) * (sbr - 1)
        pbr = torch.where(a == c, pbr_sec, pbr_q)
        qbr = torch.where(a == c, qbr_sec, qbr_q)
        qbr = torch.where(pbr > 0, -qbr, qbr)
        pbr = torch.abs(pbr)
        accept = 2 * pbr < min_(3 * xm * qbr - torch.abs(tolerance * qbr), torch.abs(ebr * qbr))
        qbr_safe = torch.where(use_interp & accept, qbr, torch.ones_like(qbr))
        d_new = torch.where(use_interp & accept, pbr / qbr_safe, xm)
        ebr_new = torch.where(use_interp & accept, d, d_new)

        d = torch.where(run, d_new, d)
        ebr = torch.where(run, ebr_new, ebr)
        a = torch.where(run, b, a)
        fa = torch.where(run, fb, fa)
        step = torch.where(torch.abs(d) > tolerance, d, torch.copysign(tolerance, xm))
        b = torch.where(run, b + step, b)
        fb = torch.where(run, entropy(b, p, qt) - s, fb)

    t = b
    _est, qst = qsat_hPa(t, p)
    return t, qst, ~done
