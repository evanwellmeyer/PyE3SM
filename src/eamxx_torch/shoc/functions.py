"""SHOC subroutines (Simplified Higher-Order Closure).

Line-by-line translation of components/eamxx/src/physics/shoc/impl/*_impl.hpp
at E3SM 80fe0a71. Each function names its source. Tensors are (ncol, nlev) on
midpoints ("zt") or (ncol, nlevi = nlev+1) on interfaces ("zi"), level 0 at the
model top. Column scalars are (ncol,).

Translation conventions (see docs/plan.md, "Translation rules"):
  * The reference is EAMxx built with pack size 1 and -ffp-contract=off, so
    every expression keeps the C++ evaluation order (left-associative).
  * `scalar / tensor` is written rdiv(scalar, tensor) (PyTorch's operator is
    not IEEE division).
  * std::min/std::max semantics via ekat.min_/max_ (argument order matters).
  * Level reductions (ekat view_reduction) are sequential sums: seq_sum.
  * Masked Pack::set(mask, value) becomes torch.where(mask, value, old).
  * Transcendental functions go through mathfn.
"""
from __future__ import annotations

import math
import sys

import torch

from .. import constants as C
from .. import mathfn
from .. import physics_functions as PF
from .. import saturation
from ..ekat import max_, min_, rdiv
from ..gradsafe import sqrt0 as _sqrt0
from .constants import SC


# ---------------------------------------------------------------- helpers
def seq_sum(x: torch.Tensor) -> torch.Tensor:
    """ekat view_reduction over [0, n) with pack size 1: result = 0; result += x(k) in order.

    Adding 0.0 first maps -0.0 to +0.0, which reproduces the C++ running sum
    that starts from the reduction identity +0.0.
    """
    return torch.cumsum(0.0 + x, dim=-1)[..., -1]


def square(x):
    """ekat::square(x) = x*x."""
    return x * x


def cube(x):
    """ekat::cube(x) = x*x*x."""
    return x * x * x


def linear_interp(x1, x2, y1, km1: int, km2: int, minthresh: float):
    """shoc_linear_interp_impl.hpp: Functions::linear_interp.

    x1, y1: (ncol, km1) source; x2: (ncol, km2) target.
    """
    if km1 == km2 + 1:
        # indx = k2+1 (never >= km1); x1 = sx1(indx), x1s = sx1(indx-1)
        x1_k, x1_s = x1[:, 1:], x1[:, :-1]
        y1_k, y1_s = y1[:, 1:], y1[:, :-1]
    elif km2 == km1 + 1:
        # indx = k2; indx<1 -> 1; indx>=km2-1 -> km1-1
        idx = torch.arange(km2, device=x1.device).clamp(min=1)
        idx = torch.where(idx >= km2 - 1, torch.full_like(idx, km1 - 1), idx)
        x1_k, x1_s = x1[:, idx], x1[:, idx - 1]
        y1_k, y1_s = y1[:, idx], y1[:, idx - 1]
    else:
        raise ValueError("Unsupported dimensions for linear interp")
    y2 = y1_s + (y1_k - y1_s) * (x2 - x1_s) / (x1_k - x1_s)
    return torch.where(y2 < minthresh, torch.full_like(y2, minthresh), y2)


# ---------------------------------------------------------------- grid, state
def shoc_grid(zt_grid, zi_grid, pdel):
    """shoc_grid_impl.hpp. Returns dz_zt (nlev), dz_zi (nlevi), rho_zt (nlev)."""
    ggr = C.gravit
    dz_zt = zi_grid[:, :-1] - zi_grid[:, 1:]
    # dz_zi(k) = zt(k-1) - zt(k) for k>=1; dz_zi(0) = 0; dz_zi(nlevi-1) = zt(nlev-1)
    dz_zi_mid = zt_grid[:, :-1] - zt_grid[:, 1:]
    zero = torch.zeros_like(zt_grid[:, :1])
    dz_zi = torch.cat([zero, dz_zi_mid, zt_grid[:, -1:]], dim=-1)
    rho_zt = (1 / ggr) * (pdel / dz_zt)
    return dz_zt, dz_zi, rho_zt


def compute_shoc_vapor(qw, ql):
    """shoc_compute_shoc_vapor_impl.hpp: qv = qw - ql."""
    return qw - ql


def compute_shoc_temperature(thetal, ql, inv_exner):
    """shoc_compute_shoc_temperature_impl.hpp: tabs = thetal/inv_exner + (lcond/cp)*ql."""
    return thetal / inv_exner + (C.LatVap / C.CP) * ql


def check_tke(tke):
    """shoc_check_tke_impl.hpp: tke.set(tke < mintke, mintke)."""
    return torch.where(tke < SC.mintke, torch.full_like(tke, SC.mintke), tke)


def shoc_energy_integrals(host_dse, pdel, rtm, rcm, u_wind, v_wind):
    """shoc_energy_integrals_impl.hpp. Returns (se_int, ke_int, wv_int, wl_int)."""
    ggr = C.gravit
    se_int = seq_sum(host_dse * pdel / ggr)
    ke_int = seq_sum(0.5 * (square(u_wind) + square(v_wind)) * pdel / ggr)
    wv_int = seq_sum((rtm - rcm) * pdel / ggr)
    wl_int = seq_sum(rcm * pdel / ggr)
    return se_int, ke_int, wv_int, wl_int


def shoc_diag_obklen(uw_sfc, vw_sfc, wthl_sfc, wqw_sfc, thl_sfc, cldliq_sfc, qv_sfc):
    """shoc_diag_obklen_impl.hpp. Returns (ustar, kbfs, obklen)."""
    cp, lcond, eps = C.CP, C.LatVap, C.ZVIR
    ggr, vk = C.gravit, C.Karman
    th_sfc = thl_sfc + (lcond / cp) * cldliq_sfc
    thv_sfc = th_sfc * (1 + eps * qv_sfc - cldliq_sfc)
    ustar_val = _sqrt0(uw_sfc * uw_sfc + vw_sfc * vw_sfc)
    ustar = max_(SC.ustar_min, ustar_val)
    kbfs = wthl_sfc + eps * th_sfc * wqw_sfc
    sign_val = torch.where(kbfs >= 0, torch.full_like(kbfs, 1e-10), torch.full_like(kbfs, -1e-10))
    obklen = -thv_sfc * (ustar * ustar * ustar) / (ggr * vk * (kbfs + sign_val))
    return ustar, kbfs, obklen


# ---------------------------------------------------------------- PBL height
def shoc_pblintd_init_pot(thl, ql, q):
    """shoc_pblintd_init_pot_impl.hpp: virtual potential temperature."""
    th = thl + (C.LatVap / C.Cpair) * ql
    return th * (C.ONE + C.ZVIR * q - ql)


def pblintd_height(npbl, z, u, v, ustar, thv, thv_ref, pblh, rino, check):
    """shoc_pblintd_height_impl.hpp. Only columns with check=True are modified.

    Returns (pblh, rino, check).
    """
    nlev = z.shape[-1]
    ggr = C.gravit
    tiny, fac, ricr = 1e-36, 100.0, 0.3
    lower_indx, upper_indx = nlev - npbl, nlev - 1
    if lower_indx >= upper_indx:  # static (npbl is a model constant)
        return pblh, rino, check
    k = torch.arange(nlev, device=z.device)
    in_range = (k < nlev - 1) & (k >= nlev - npbl)
    vvk = max_(tiny, square(u - u[:, -1:]) + square(v - v[:, -1:]) + fac * (ustar * ustar)[:, None])
    # Reference: vvk = 0 outside the range, where rino is not computed. Use 1
    # there so the masked-out division stays finite (no NaN in gradients).
    vvk = torch.where(in_range, vvk, torch.ones_like(vvk))
    rino_new = ggr * (thv - thv_ref[:, None]) * (z - z[:, -1:]) / (thv[:, -1:] * vvk)
    rino_upd = torch.where(in_range, rino_new, rino)
    rino = torch.where(check[:, None], rino_upd, rino)
    # max index k in range with rino(k) >= ricr
    hit = in_range & ~(rino < ricr)
    kk = torch.where(hit, k.expand_as(rino), torch.full_like(rino, -1, dtype=torch.long))
    max_indx = kk.amax(dim=-1)
    found = (max_indx >= 0) & check
    m = max_indx.clamp(min=0)
    z_m = torch.gather(z, 1, m[:, None])[:, 0]
    z_m1 = torch.gather(z, 1, (m + 1).clamp(max=nlev - 1)[:, None])[:, 0]
    r_m = torch.gather(rino, 1, m[:, None])[:, 0]
    r_m1 = torch.gather(rino, 1, (m + 1).clamp(max=nlev - 1)[:, None])[:, 0]
    denom = r_m - r_m1
    denom = torch.where(found, denom, torch.ones_like(denom))
    pblh_new = z_m1 + (ricr - r_m1) / denom * (z_m - z_m1)
    pblh = torch.where(found, pblh_new, pblh)
    check = check & ~found
    return pblh, rino, check


def pblintd_surf_temp(npbl, z, ustar, obklen, kbfs, thv, pblh, check, rino):
    """shoc_pblintd_surf_temp_impl.hpp. Returns (tlv, pblh, check, rino)."""
    nlevi = z.shape[-1] + 1
    fak, betam, sffrac = 8.5, 15.0, 0.1
    binm = betam * sffrac
    pblh = torch.where(check, z[:, nlevi - npbl - 1], pblh)
    check = kbfs > 0
    tlv = thv[:, -1]
    rino_last = torch.where(check, torch.zeros_like(rino[:, -1]), rino[:, -1])
    rino = torch.cat([rino[:, :-1], rino_last[:, None]], dim=-1)
    obk = torch.where(check, obklen, torch.ones_like(obklen))
    phiminv = mathfn.cbrt(1 - binm * pblh / obk)
    tlv = torch.where(check, tlv + kbfs * fak / (ustar * phiminv), tlv)
    return tlv, pblh, check, rino


def pblintd_check_pblh(npbl, z, ustar, check, pblh):
    """shoc_pblintd_check_pblh_impl.hpp."""
    nlevi = z.shape[-1] + 1
    pblh = torch.where(check, z[:, nlevi - npbl - 1], pblh)
    return max_(pblh, 700 * ustar)


def shoc_pblintd_cldcheck(zi, cldn, pblh):
    """shoc_pblintd_cldcheck_impl.hpp: if (cldn >= 0) pblh = max(pblh, zi + 50)."""
    return torch.where(cldn >= 0, max_(pblh, zi + 50.0), pblh)


def pblintd(npbl, z, zi, thl, ql, q, u, v, ustar, obklen, kbfs, cldn):
    """shoc_pblintd_impl.hpp (release build: NDEBUG, no SCREAM_SHORT_TESTS)."""
    nlev = z.shape[-1]
    thv = shoc_pblintd_init_pot(thl, ql, q)
    check = torch.ones_like(ustar, dtype=torch.bool)
    # rino: only rino(nlev-1) is initialized (to 0) in release builds; entries
    # that are never written are never read. Start from zeros.
    rino = torch.zeros_like(z)
    pblh = z[:, nlev - 1]
    pblh, rino, check = pblintd_height(npbl, z, u, v, ustar, thv, thv[:, -1], pblh, rino, check)
    tlv, pblh, check, rino = pblintd_surf_temp(npbl, z, ustar, obklen, kbfs, thv, pblh, check, rino)
    pblh, rino, check = pblintd_height(npbl, z, u, v, ustar, thv, tlv, pblh, rino, check)
    pblh = pblintd_check_pblh(npbl, z, ustar, check, pblh)
    pblh = shoc_pblintd_cldcheck(zi[:, nlev - 1], cldn[:, nlev - 1], pblh)
    return pblh


# ---------------------------------------------------------------- length scale
def compute_brunt_shoc_length(dz_zt, thv, thv_zi):
    """shoc_compute_brunt_shoc_length_impl.hpp."""
    ggr = C.gravit
    return rdiv(ggr, thv) * (thv_zi[:, :-1] - thv_zi[:, 1:]) / dz_zt


def compute_l_inf_shoc_length(zt_grid, dz_zt, tke):
    """shoc_compute_l_inf_shoc_length_impl.hpp."""
    numer = seq_sum(torch.sqrt(tke) * zt_grid * dz_zt)
    denom = seq_sum(torch.sqrt(tke) * dz_zt)
    return 0.1 * (numer / denom)


def compute_shoc_mix_shoc_length(length_fac, tke, brunt, zt_grid, l_inf):
    """shoc_compute_shoc_mix_shoc_length_impl.hpp."""
    maxlen, vk, tscale = SC.maxlen, C.Karman, 400.0
    tkes = torch.sqrt(tke)
    brunt2 = max_(0.0, brunt)
    inner = (rdiv(1.0, tscale * tkes * vk * zt_grid)
             + rdiv(1.0, tscale * tkes * l_inf[:, None])
             + 0.01 * (brunt2 / tke))
    return min_(maxlen, 2.8284 * (torch.sqrt(rdiv(1.0, inner))) / length_fac)


def check_length_scale_shoc_length(dx, dy, shoc_mix):
    """shoc_check_length_scale_shoc_length_impl.hpp."""
    return min_(torch.sqrt(dx * dy)[:, None], max_(SC.minlen, shoc_mix))


def shoc_length(length_fac, dx, dy, zt_grid, zi_grid, dz_zt, tke, thv):
    """shoc_length_impl.hpp. Returns (brunt, shoc_mix)."""
    nlev = zt_grid.shape[-1]
    thv_zi = linear_interp(zt_grid, zi_grid, thv, nlev, nlev + 1, 0.0)
    brunt = compute_brunt_shoc_length(dz_zt, thv, thv_zi)
    l_inf = compute_l_inf_shoc_length(zt_grid, dz_zt, tke)
    shoc_mix = compute_shoc_mix_shoc_length(length_fac, tke, brunt, zt_grid, l_inf)
    shoc_mix = check_length_scale_shoc_length(dx, dy, shoc_mix)
    return brunt, shoc_mix


# ---------------------------------------------------------------- TKE
def integ_column_stability(dz_zt, pres, brunt):
    """shoc_integ_column_stability_impl.hpp."""
    troppres = 80000.0
    return seq_sum(torch.where(pres > troppres, dz_zt * brunt, torch.zeros_like(brunt)))


def compute_shr_prod(dz_zi, u_wind, v_wind):
    """shoc_compute_shr_prod_impl.hpp. Returns sterm on interfaces (nlevi)."""
    Ck_sh = 0.1
    grid_dz = rdiv(1.0, dz_zi[:, 1:-1])
    u_grad = grid_dz * (u_wind[:, :-1] - u_wind[:, 1:])
    v_grad = grid_dz * (v_wind[:, :-1] - v_wind[:, 1:])
    inner = Ck_sh * (u_grad * u_grad + v_grad * v_grad)
    zero = torch.zeros_like(dz_zi[:, :1])
    return torch.cat([zero, inner, zero], dim=-1)


def adv_sgs_tke(dtime, shoc_1p5tke, shoc_mix, wthv_sec, sterm_zt, tk, brunt, tke):
    """shoc_adv_sgs_tke_impl.hpp (1D path, do_3d_turb=false). Returns (tke, a_diss)."""
    ggr, basetemp = C.gravit, C.basetemp
    mintke, maxtke = SC.mintke, SC.maxtke
    Cs, Ck = 0.15, 0.1
    Ce = (Ck * Ck * Ck) / ((Cs * Cs) * (Cs * Cs))
    Ce1 = Ce / 0.7 * 0.19
    Ce2 = Ce / 0.7 * 0.51
    Cee = Ce1 + Ce2
    if shoc_1p5tke:
        a_prod_bu = -tke * brunt
    else:
        a_prod_bu = (ggr / basetemp) * wthv_sec
    tke = max_(0.0, tke)
    a_prod_sh = tk * sterm_zt
    a_diss = rdiv(Cee, shoc_mix) * mathfn.pow(tke, 1.5)
    tke = max_(mintke, tke + dtime * (max_(0.0, a_prod_sh + a_prod_bu) - a_diss))
    tke = min_(tke, maxtke)
    return tke, a_diss


def isotropic_ts(lambda_low, lambda_high, lambda_slope, lambda_thresh, brunt_int, tke, a_diss, brunt):
    """shoc_isotropic_ts_impl.hpp."""
    ggr = C.gravit
    maxiso = 20000.0
    tscale = 2 * tke / a_diss
    lam = lambda_low + ((brunt_int / ggr) - lambda_thresh) * lambda_slope  # (ncol,)
    lam = max_(lambda_low, min_(lambda_high, lam))
    lam = lam[:, None].expand_as(brunt)
    lam = torch.where(brunt <= 0, torch.zeros_like(lam), lam)
    return min_(maxiso, tscale / (1 + lam * brunt * square(tscale)))


def eddy_diffusivities(Ckh, Ckm, pblh, zt_grid, tabs, shoc_mix, sterm_zt, isotropy, tke):
    """shoc_eddy_diffusivities_impl.hpp. Returns (tkh, tk)."""
    tabs_crit, pbl_trans = 182, 200
    Ckh_s = Ckm_s = 0.1
    condition = (zt_grid < (pblh + pbl_trans)[:, None]) & (tabs[:, -1:] < tabs_crit)
    sqrt_sterm = _sqrt0(torch.where(condition, sterm_zt, torch.ones_like(sterm_zt)))   # masked: safe input
    tkh = torch.where(condition, Ckh_s * square(shoc_mix) * sqrt_sterm, Ckh * isotropy * tke)
    tk = torch.where(condition, Ckm_s * square(shoc_mix) * sqrt_sterm, Ckm * isotropy * tke)
    return tkh, tk


def shoc_tke(dtime, lambda_low, lambda_high, lambda_slope, lambda_thresh, Ckh, Ckm, shoc_1p5tke,
             wthv_sec, shoc_mix, dz_zi, dz_zt, pres, tabs, u_wind, v_wind, brunt, zt_grid, zi_grid,
             pblh, tke, tk):
    """shoc_tke_impl.hpp (do_3d_turb=false). Returns (tke, tk, tkh, isotropy, shear_strain3d)."""
    nlev = zt_grid.shape[-1]
    brunt_int = integ_column_stability(dz_zt, pres, brunt)
    sterm = compute_shr_prod(dz_zi, u_wind, v_wind)
    sterm_zt = linear_interp(zi_grid, zt_grid, sterm, nlev + 1, nlev, 0.0)
    shear_strain3d = torch.zeros_like(zt_grid)
    tke, a_diss = adv_sgs_tke(dtime, shoc_1p5tke, shoc_mix, wthv_sec, sterm_zt, tk, brunt, tke)
    isotropy = isotropic_ts(lambda_low, lambda_high, lambda_slope, lambda_thresh, brunt_int, tke, a_diss, brunt)
    tkh, tk = eddy_diffusivities(Ckh, Ckm, pblh, zt_grid, tabs, shoc_mix, sterm_zt, isotropy, tke)
    return tke, tk, tkh, isotropy, shear_strain3d


# ---------------------------------------------------------------- implicit update
def compute_tmpi(dtime, rho_zi, dz_zi):
    """shoc_compute_tmpi_impl.hpp: tmpi(0)=0; tmpi(k) = dtime*(ggr*rho_zi(k))/dz_zi(k)."""
    ggr = C.gravit
    inner = dtime * (ggr * rho_zi[:, 1:]) / dz_zi[:, 1:]
    return torch.cat([torch.zeros_like(rho_zi[:, :1]), inner], dim=-1)


def dp_inverse(rho_zt, dz_zt):
    """shoc_dp_inverse_impl.hpp: rdp_zt = 1/(ggr*rho_zt*dz_zt)."""
    return rdiv(1.0, C.gravit * rho_zt * dz_zt)


def vd_shoc_decomp(kv_term, tmpi, rdp_zt, dtime, flux):
    """shoc_tridiag_solver_impl.hpp: vd_shoc_decomp. Returns (du, dl, d), each (ncol, nlev).

    flux may be a float or a (ncol,) tensor.
    """
    ggr = C.gravit
    nlev = rdp_zt.shape[-1]
    du = -kv_term[:, 1:] * tmpi[:, 1:] * rdp_zt
    dl = -kv_term[:, :-1] * tmpi[:, :-1] * rdp_zt
    k = torch.arange(nlev, device=rdp_zt.device)
    du = torch.where(k == nlev - 1, torch.zeros_like(du), du)
    dl = torch.where(k == 0, torch.zeros_like(dl), dl)
    d = 1 - du - dl
    flux = torch.as_tensor(flux, dtype=rdp_zt.dtype, device=rdp_zt.device)
    flux = flux[:, None] if flux.ndim == 1 else flux
    d = torch.where(k == nlev - 1, d + flux * dtime * ggr * rdp_zt, d)
    return du, dl, d


def thomas(dl, d, du, X):
    """ekat::tridiag::thomas (thomas_a1xm), the CPU non-BFB solver used by vd_shoc_solve.

    dl, d, du: (ncol, nrow); X: (ncol, nrow, nrhs). Returns the solution.
    Sequential in level, exactly as the C++ loops.
    """
    nrow = d.shape[-1]
    dcur = [d[:, 0]]
    xs = [X[:, 0]]
    for i in range(1, nrow):
        dli = dl[:, i] / dcur[i - 1]
        dcur.append(d[:, i] - dli * du[:, i - 1])
        xs.append(X[:, i] - dli[:, None] * xs[i - 1])
    xs[nrow - 1] = xs[nrow - 1] / dcur[nrow - 1][:, None]
    for i in range(nrow - 1, 0, -1):
        xs[i - 1] = (xs[i - 1] - du[:, i - 1][:, None] * xs[i]) / dcur[i - 1][:, None]
    return torch.stack(xs, dim=1)


def update_prognostics_implicit(dtime, dz_zt, dz_zi, rho_zt, zt_grid, zi_grid, tk, tkh,
                                uw_sfc, vw_sfc, wthl_sfc, wqw_sfc, wtracer_sfc,
                                thetal, qw, qtracers, tke, u_wind, v_wind,
                                uw_sfc_pert, vw_sfc_pert, um_pert, vm_pert):
    """shoc_update_prognostics_implicit_impl.hpp.

    qtracers: (ncol, ntrac, nlev); wtracer_sfc: (ncol, ntrac).
    Returns (thetal, qw, qtracers, tke, u_wind, v_wind, um_pert, vm_pert).
    """
    nlev = zt_grid.shape[-1]
    nlevi = nlev + 1
    tkh_zi = linear_interp(zt_grid, zi_grid, tkh, nlev, nlevi, 0.0)
    tk_zi = linear_interp(zt_grid, zi_grid, tk, nlev, nlevi, 0.0)
    rho_zi = linear_interp(zt_grid, zi_grid, rho_zt, nlev, nlevi, 0.0)
    tmpi = compute_tmpi(dtime, rho_zi, dz_zi)
    rdp_zt = dp_inverse(rho_zt, dz_zt)

    # Implicit surface stress (ksrf) and TKE flux (wtke_sfc)
    wsmin, ksrfmin, ustarmin = 1.0, 1e-4, 0.01
    rho = rho_zi[:, nlevi - 1]
    uw, vw = uw_sfc, vw_sfc
    taux, tauy = rho * uw, rho * vw
    u_sfc, v_sfc = u_wind[:, -1], v_wind[:, -1]
    ws = max_(_sqrt0((u_sfc * u_sfc) + v_sfc * v_sfc), wsmin)
    tau = _sqrt0(taux * taux + tauy * tauy)
    ksrf = max_(tau / ws, ksrfmin)
    ustar = max_(_sqrt0(_sqrt0(uw * uw + vw * vw)), ustarmin)
    wtke_sfc = ustar * ustar * ustar
    uw_pert = uw + uw_sfc_pert
    vw_pert = vw + vw_sfc_pert
    taux_pert, tauy_pert = rho * uw_pert, rho * vw_pert
    u_sfc_pert = u_sfc + um_pert[:, -1]
    v_sfc_pert = v_sfc + vm_pert[:, -1]
    ws_pert = max_(_sqrt0((u_sfc_pert * u_sfc_pert) + v_sfc_pert * v_sfc_pert), wsmin)
    tau_pert = _sqrt0(taux_pert * taux_pert + tauy_pert * tauy_pert)
    ksrf_pert = max_(tau_pert / ws_pert, ksrfmin)

    # Explicit surface fluxes into the lowest level
    cmnfac = dtime * (C.gravit * rho_zi[:, nlevi - 1] * rdp_zt[:, nlev - 1])

    def add_bottom(x, inc):
        return torch.cat([x[..., :-1], (x[..., -1] + inc)[..., None]], dim=-1)

    thetal = add_bottom(thetal, cmnfac * wthl_sfc)
    qw = add_bottom(qw, cmnfac * wqw_sfc)
    tke = add_bottom(tke, cmnfac * wtke_sfc)
    qtracers = add_bottom(qtracers, cmnfac[:, None] * wtracer_sfc)

    # Solves
    wind_rhs = torch.stack([u_wind, v_wind], dim=-1)                                 # (ncol, nlev, 2)
    wind_pert_rhs = torch.stack([u_wind + um_pert, v_wind + vm_pert], dim=-1)
    trac_rhs = torch.cat([qtracers.transpose(1, 2),
                          torch.stack([thetal, qw, tke], dim=-1)], dim=-1)          # (ncol, nlev, ntrac+3)

    du, dl, d = vd_shoc_decomp(tk_zi, tmpi, rdp_zt, dtime, ksrf)
    wind = thomas(dl, d, du, wind_rhs)
    du, dl, d = vd_shoc_decomp(tk_zi, tmpi, rdp_zt, dtime, ksrf_pert)
    wind_pert = thomas(dl, d, du, wind_pert_rhs)
    du, dl, d = vd_shoc_decomp(tkh_zi, tmpi, rdp_zt, dtime, 0.0)
    trac = thomas(dl, d, du, trac_rhs)

    u_new, v_new = wind[..., 0], wind[..., 1]
    um_pert = wind_pert[..., 0] - u_new
    vm_pert = wind_pert[..., 1] - v_new
    ntrac = qtracers.shape[1]
    qtracers = trac[..., :ntrac].transpose(1, 2)
    thetal, qw, tke = trac[..., ntrac], trac[..., ntrac + 1], trac[..., ntrac + 2]
    return thetal, qw, qtracers, tke, u_new, v_new, um_pert, vm_pert


# ---------------------------------------------------------------- second moments
def shoc_diag_second_moments_srf(wthl_sfc, uw_sfc, vw_sfc):
    """shoc_diag_second_moments_srf_impl.hpp. Returns (ustar2, wstar)."""
    one, zero, third = C.ONE, C.ZERO, C.THIRD
    ggr, basetemp = C.gravit, C.basetemp
    ustar2 = _sqrt0(uw_sfc * uw_sfc + vw_sfc * vw_sfc)   # |stress|: subgradient 0 at zero stress
    pos = wthl_sfc >= zero
    arg = torch.where(pos, one / basetemp * ggr * wthl_sfc * one, torch.zeros_like(wthl_sfc))
    wstar = torch.where(pos, mathfn.pow(arg, third), torch.zeros_like(wthl_sfc))
    return ustar2, wstar


def shoc_diag_second_moments_lbycond(wthl_sfc, wqw_sfc, uw_sfc, vw_sfc, ustar2, wstar):
    """shoc_diag_second_moments_lbycond_impl.hpp.

    Returns (wthl_sec, wqw_sec, uw_sec, vw_sec, wtke_sec, thl_sec, qw_sec, qwthl_sec).
    """
    ufmin = 0.01
    uf = _sqrt0(ustar2 + 0.3 * wstar * wstar)
    uf = max_(ufmin, uf)
    thl_sec = 0.4 * 1.8 * ((wthl_sfc / uf) * (wthl_sfc / uf))
    qw_sec = 0.4 * 1.8 * ((wqw_sfc / uf) * (wqw_sfc / uf))
    qwthl_sec = 0.2 * 1.8 * (wthl_sfc / uf) * (wqw_sfc / uf)
    max_val = max_(_sqrt0(ustar2), ufmin)
    wtke_sec = max_val * max_val * max_val
    return wthl_sfc, wqw_sfc, uw_sfc, vw_sfc, wtke_sec, thl_sec, qw_sec, qwthl_sec


def calc_shoc_varorcovar(tunefac, isotropy_zi, tkh_zi, dz_zi, invar1, invar2):
    """shoc_calc_shoc_varorcovar_impl.hpp. Interior interfaces k = 1..nlev-1 only."""
    grid_dz = rdiv(1.0, dz_zi[:, 1:-1])
    grid_dz2 = square(grid_dz)
    return (tunefac * (isotropy_zi[:, 1:-1] * tkh_zi[:, 1:-1]) * grid_dz2
            * (invar1[:, :-1] - invar1[:, 1:]) * (invar2[:, :-1] - invar2[:, 1:]))


def calc_shoc_vertflux(tkh_zi, dz_zi, invar):
    """shoc_calc_shoc_vertflux_impl.hpp. Interior interfaces k = 1..nlev-1 only."""
    grid_dz = rdiv(1.0, dz_zi[:, 1:-1])
    return -(tkh_zi[:, 1:-1] * grid_dz * (invar[:, :-1] - invar[:, 1:]))


def diag_second_shoc_moments(thl2tune, qw2tune, qwthl2tune, w2tune, shoc_1p5tke,
                             thetal, qw, u_wind, v_wind, tke, isotropy, tkh, tk, dz_zi,
                             zt_grid, zi_grid, shoc_mix, wthl_sfc, wqw_sfc, uw_sfc, vw_sfc):
    """shoc_diag_second_shoc_moments_impl.hpp + shoc_diag_second_moments_impl.hpp.

    Returns dict with thl_sec, qw_sec, wthl_sec, wqw_sec, qwthl_sec, uw_sec,
    vw_sec, wtke_sec (all nlevi), w_sec (nlev), ustar2, wstar.
    """
    nlev = zt_grid.shape[-1]
    nlevi = nlev + 1
    ustar2, wstar = shoc_diag_second_moments_srf(wthl_sfc, uw_sfc, vw_sfc)
    lb = shoc_diag_second_moments_lbycond(wthl_sfc, wqw_sfc, uw_sfc, vw_sfc, ustar2, wstar)
    lb = dict(zip(("wthl_sec", "wqw_sec", "uw_sec", "vw_sec", "wtke_sec", "thl_sec", "qw_sec", "qwthl_sec"), lb))

    isotropy_zi = linear_interp(zt_grid, zi_grid, isotropy, nlev, nlevi, 0.0)
    tkh_zi = linear_interp(zt_grid, zi_grid, tkh, nlev, nlevi, 0.0)
    tk_zi = linear_interp(zt_grid, zi_grid, tk, nlev, nlevi, 0.0)
    if shoc_1p5tke:
        w_sec = torch.zeros_like(tke)
    else:
        w_sec = w2tune * (2. / 3.) * tke

    zero = torch.zeros_like(zt_grid[:, :1])

    def assemble(interior, name):
        # upper boundary (index 0): zero (ubycond); lower boundary (nlevi-1): lbycond
        return torch.cat([zero, interior, lb[name][:, None]], dim=-1)

    if shoc_1p5tke:
        thl_sec = torch.zeros_like(zi_grid)
        qw_sec = torch.zeros_like(zi_grid)
        qwthl_sec = torch.zeros_like(zi_grid)
    else:
        thl_sec = assemble(calc_shoc_varorcovar(thl2tune, isotropy_zi, tkh_zi, dz_zi, thetal, thetal), "thl_sec")
        qw_sec = assemble(calc_shoc_varorcovar(qw2tune, isotropy_zi, tkh_zi, dz_zi, qw, qw), "qw_sec")
        qwthl_sec = assemble(calc_shoc_varorcovar(qwthl2tune, isotropy_zi, tkh_zi, dz_zi, thetal, qw), "qwthl_sec")
    out = dict(
        thl_sec=thl_sec, qw_sec=qw_sec, qwthl_sec=qwthl_sec,
        wthl_sec=assemble(calc_shoc_vertflux(tkh_zi, dz_zi, thetal), "wthl_sec"),
        wqw_sec=assemble(calc_shoc_vertflux(tkh_zi, dz_zi, qw), "wqw_sec"),
        wtke_sec=assemble(calc_shoc_vertflux(tkh_zi, dz_zi, tke), "wtke_sec"),
        uw_sec=assemble(calc_shoc_vertflux(tk_zi, dz_zi, u_wind), "uw_sec"),
        vw_sec=assemble(calc_shoc_vertflux(tk_zi, dz_zi, v_wind), "vw_sec"),
        w_sec=w_sec, ustar2=ustar2, wstar=wstar,
    )
    return out


# ---------------------------------------------------------------- third moments
def compute_diag_third_shoc_moment(c, shoc_1p5tke, w_sec, thl_sec, wthl_sec, tke, dz_zt, dz_zi,
                                   isotropy_zi, brunt_zi, w_sec_zi, thetal_zi):
    """shoc_compute_diag_third_shoc_moment_impl.hpp. Returns w3 (nlevi)."""
    ggr = C.gravit
    a0 = (0.52 * (1 / (c * c))) / (c - 2)
    a1 = 0.87 / (c * c)
    a2 = 0.5 / c
    a3 = 0.6 / (c * (c - 2))
    a4 = 2.4 / (3 * c + 5)
    a5 = 0.6 / (c * (3 + 5 * c))
    s = slice(1, -1)  # active interfaces k = 1..nlev-1
    if shoc_1p5tke:
        inner = torch.zeros_like(dz_zi[:, s])
    else:
        dz_zt_k, dz_zt_km1 = dz_zt[:, 1:], dz_zt[:, :-1]
        wthl_k, wthl_km1, wthl_kp1 = wthl_sec[:, 1:-1], wthl_sec[:, :-2], wthl_sec[:, 2:]
        thl_km1, thl_kp1 = thl_sec[:, :-2], thl_sec[:, 2:]
        w_sec_km1, w_sec_k = w_sec[:, :-1], w_sec[:, 1:]
        tke_km1, tke_k = tke[:, :-1], tke[:, 1:]
        thedz = rdiv(1.0, dz_zi[:, s])
        thedz2 = rdiv(1.0, dz_zt_k + dz_zt_km1)
        iso = isotropy_zi[:, s]
        isosqrd = square(iso)
        buoy_sgs2 = isosqrd * brunt_zi[:, s]
        bet2 = rdiv(ggr, thetal_zi[:, s])
        wsz = w_sec_zi[:, s]
        thl_sec_diff = thl_km1 - thl_kp1
        wthl_sec_diff = wthl_km1 - wthl_kp1
        wsec_diff = w_sec_km1 - w_sec_k
        tke_diff = tke_km1 - tke_k
        f0 = thedz2 * cube(bet2) * ((iso * iso) * (iso * iso)) * wthl_k * thl_sec_diff
        f1 = thedz2 * square(bet2) * cube(iso) * (wthl_k * wthl_sec_diff + 0.5 * wsz * thl_sec_diff)
        f2 = thedz * bet2 * isosqrd * wthl_k * wsec_diff + 2 * thedz2 * bet2 * isosqrd * wsz * wthl_sec_diff
        f3 = thedz2 * bet2 * isosqrd * wsz * wthl_sec_diff + thedz * bet2 * isosqrd * (wthl_k * tke_diff)
        f4 = thedz * iso * wsz * (wsec_diff + tke_diff)
        f5 = thedz * iso * wsz * wsec_diff
        omega0 = rdiv(a4, 1 - a5 * buoy_sgs2)
        omega1 = omega0 / (2 * c)
        omega2 = omega1 * f3 + (5.0 / 4.0) * omega0 * f4
        x0 = (a2 * buoy_sgs2 * (1 - a3 * buoy_sgs2)) / (1 - (a1 + a3) * buoy_sgs2)
        y0 = (2 * a2 * buoy_sgs2 * x0) / (1 - a3 * buoy_sgs2)
        x1 = (a0 * f0 + a1 * f1 + a2 * (1 - a3 * buoy_sgs2) * f2) / (1 - (a1 + a3) * buoy_sgs2)
        y1 = (2 * a2 * (buoy_sgs2 * x1 + (a0 / a1) * f0 + f1)) / (1 - a3 * buoy_sgs2)
        aa0 = omega0 * x0 + omega1 * y0
        aa1 = omega0 * x1 + omega1 * y1 + omega2
        inner = (aa1 - 1.2 * x1 - 1.5 * f5) / (c - 1.2 * x0 + aa0)
    zero = torch.zeros_like(dz_zi[:, :1])
    return torch.cat([zero, inner, zero], dim=-1)


def clipping_diag_third_shoc_moments(w_sec_zi, w3):
    """shoc_clipping_diag_third_shoc_moments_impl.hpp."""
    w3clip = SC.w3clip
    cond = w3clip * _sqrt0(2 * cube(w_sec_zi))
    w3clipdef = 0.02
    tsign = torch.where(w3 < 0, torch.full_like(w3, -1.0), torch.ones_like(w3))
    return torch.where(tsign * w3 > cond, torch.full_like(w3, w3clipdef), w3)


def diag_third_shoc_moments(c_diag_3rd_mom, shoc_1p5tke, w_sec, thl_sec, wthl_sec, isotropy, brunt,
                            thetal, tke, dz_zt, dz_zi, zt_grid, zi_grid):
    """shoc_diag_third_shoc_moments_impl.hpp. Returns w3 (nlevi)."""
    nlev = zt_grid.shape[-1]
    nlevi = nlev + 1
    isotropy_zi = linear_interp(zt_grid, zi_grid, isotropy, nlev, nlevi, 0.0)
    brunt_zi = linear_interp(zt_grid, zi_grid, brunt, nlev, nlevi, SC.largeneg)
    w_sec_zi = linear_interp(zt_grid, zi_grid, w_sec, nlev, nlevi, (2.0 / 3.0) * SC.mintke)
    thetal_zi = linear_interp(zt_grid, zi_grid, thetal, nlev, nlevi, 0.0)
    w3 = compute_diag_third_shoc_moment(c_diag_3rd_mom, shoc_1p5tke, w_sec, thl_sec, wthl_sec, tke,
                                        dz_zt, dz_zi, isotropy_zi, brunt_zi, w_sec_zi, thetal_zi)
    return clipping_diag_third_shoc_moments(w_sec_zi, w3)


# ---------------------------------------------------------------- assumed PDF
def shoc_assumed_pdf_vv_parameters(w_first, w_sec, w3var, w_tol_sqd):
    """shoc_assumed_pdf_vv_parameters_impl.hpp. Returns (Skew_w, w1_1, w1_2, w2_1, w2_2, a)."""
    condition = w_sec > w_tol_sqd
    tmp_val = 0.4
    one_m_tmp_val = 1 - tmp_val
    sqrtw2t = math.sqrt(1 - tmp_val)
    zero = torch.zeros_like(w_sec)
    Skew_w = torch.where(condition, w3var / torch.sqrt(torch.where(condition, cube(w_sec), torch.ones_like(w_sec))), zero)
    a_c = max_(0.01, min_(0.99, 0.5 * (1 - Skew_w * torch.sqrt(
        rdiv(1.0, 4 * (one_m_tmp_val * one_m_tmp_val * one_m_tmp_val) + square(Skew_w))))))
    a = torch.where(condition, a_c, torch.full_like(w_sec, 0.5))
    w1_1 = torch.where(condition, torch.sqrt((1 - a) / a) * sqrtw2t, w_first)
    w1_2 = torch.where(condition, -1 * torch.sqrt(a / (1 - a)) * sqrtw2t, w_first)
    w2_1 = torch.where(condition, tmp_val * w_sec, zero)
    w2_2 = torch.where(condition, tmp_val * w_sec, zero)
    return Skew_w, w1_1, w1_2, w2_1, w2_2, a


def _plume_second_moments(tmp_val_1, tmp_val_2, a, Skew, sec):
    """Shared expressions of thl_parameters and qw_parameters (same C++ text).

    Where tmp_val_1 == tmp_val_2 (a zero correlation) the C++ divides 0 by 0 and
    max_(0, NaN) returns 0; the port returns that 0 directly, so the values are the same
    and the gradient is 0 instead of NaN (plan section 6, masked branches).
    """
    num1 = (3 * tmp_val_1 * (1 - a * square(tmp_val_2) - (1 - a) * square(tmp_val_1))
            - (Skew - a * cube(tmp_val_2) - (1 - a) * cube(tmp_val_1)))
    num2 = (-3 * tmp_val_2 * (1 - a * square(tmp_val_2) - (1 - a) * square(tmp_val_1))
            + (Skew - a * cube(tmp_val_2) - (1 - a) * cube(tmp_val_1)))
    dt = tmp_val_1 - tmp_val_2
    same = dt == 0
    one, zero = torch.ones_like(dt), torch.zeros_like(dt)
    r1 = torch.where(same, zero, num1 / torch.where(same, one, 3 * a * dt))
    r2 = torch.where(same, zero, num2 / torch.where(same, one, 3 * (1 - a) * dt))
    v1 = min_(100.0, max_(0.0, r1)) * sec
    v2 = min_(100.0, max_(0.0, r2)) * sec
    return v1, v2


def shoc_assumed_pdf_thl_parameters(wthlsec, sqrtw2, sqrtthl, thlsec, thl_first, w1_1, w1_2, Skew_w, a,
                                    thl_tol, w_thresh):
    """shoc_assumed_pdf_thl_parameters_impl.hpp (SC::dothetal_skew is false)."""
    condition = (thlsec > (thl_tol * thl_tol)) & (torch.abs(w1_2 - w1_1) > w_thresh)
    corrtest1 = max_(-1.0, min_(1.0, wthlsec / (sqrtw2 * sqrtthl)))
    # masked-out lanes (w1 is then the grid-mean w, possibly 0): safe input (plan section 6)
    tmp_val_1 = -corrtest1 / torch.where(condition, w1_1, torch.ones_like(w1_1))
    tmp_val_2 = -corrtest1 / torch.where(condition, w1_2, torch.ones_like(w1_2))
    Skew_thl = torch.zeros_like(thlsec)
    t2_1, t2_2 = _plume_second_moments(tmp_val_1, tmp_val_2, a, Skew_thl, thlsec)
    zero = torch.zeros_like(thlsec)
    thl2_1 = torch.where(condition, t2_1, zero)
    thl2_2 = torch.where(condition, t2_2, zero)
    thl1_1 = torch.where(condition, tmp_val_2 * sqrtthl + thl_first, thl_first)
    thl1_2 = torch.where(condition, tmp_val_1 * sqrtthl + thl_first, thl_first)
    one = torch.ones_like(thlsec)   # masked-out entries: safe input
    sqrtthl2_1 = torch.where(condition, _sqrt0(torch.where(condition, thl2_1, one)), zero)
    sqrtthl2_2 = torch.where(condition, _sqrt0(torch.where(condition, thl2_2, one)), zero)
    return thl1_1, thl1_2, thl2_1, thl2_2, sqrtthl2_1, sqrtthl2_2


def shoc_assumed_pdf_qw_parameters(wqwsec, sqrtw2, Skew_w, sqrtqt, qwsec, w1_2, w1_1, qw_first, a,
                                   rt_tol, w_thresh):
    """shoc_assumed_pdf_qw_parameters_impl.hpp."""
    condition = (qwsec > (rt_tol * rt_tol)) & (torch.abs(w1_2 - w1_1) > w_thresh)
    corrtest2 = max_(-1.0, min_(1.0, wqwsec / (sqrtw2 * sqrtqt)))
    # masked-out lanes (w1 is then the grid-mean w, possibly 0): safe input (plan section 6)
    tmp_val_1 = -corrtest2 / torch.where(condition, w1_1, torch.ones_like(w1_1))
    tmp_val_2 = -corrtest2 / torch.where(condition, w1_2, torch.ones_like(w1_2))
    tsign = torch.abs(tmp_val_1 - tmp_val_2)
    Skew_qw = torch.zeros_like(qwsec)
    Skew_qw = torch.where(tsign > 0.4, 1.2 * Skew_w, Skew_qw)
    Skew_qw = torch.where((tsign > 0.2) & (tsign <= 0.4), (((1.2 * Skew_w) / 0.2) * (tsign - 0.2)), Skew_qw)
    q2_1, q2_2 = _plume_second_moments(tmp_val_1, tmp_val_2, a, Skew_qw, qwsec)
    zero = torch.zeros_like(qwsec)
    qw2_1 = torch.where(condition, q2_1, zero)
    qw2_2 = torch.where(condition, q2_2, zero)
    qw1_1 = torch.where(condition, tmp_val_2 * sqrtqt + qw_first, qw_first)
    qw1_2 = torch.where(condition, tmp_val_1 * sqrtqt + qw_first, qw_first)
    one = torch.ones_like(qwsec)   # masked-out entries: safe input
    sqrtqw2_1 = torch.where(condition, _sqrt0(torch.where(condition, qw2_1, one)), zero)
    sqrtqw2_2 = torch.where(condition, _sqrt0(torch.where(condition, qw2_2, one)), zero)
    return qw1_1, qw1_2, qw2_1, qw2_2, sqrtqw2_1, sqrtqw2_2


def shoc_assumed_pdf_tilde_to_real(w_first, sqrtw2, w1):
    """shoc_assumed_pdf_tilde_to_real_impl.hpp: w1 *= sqrtw2; w1 += w_first."""
    w1 = w1 * sqrtw2
    return w1 + w_first


def shoc_assumed_pdf_inplume_correlations(sqrtqw2_1, sqrtthl2_1, a, sqrtqw2_2, sqrtthl2_2, qwthlsec,
                                          qw1_1, qw_first, thl1_1, thl_first, qw1_2, thl1_2):
    """shoc_assumed_pdf_inplume_correlations_impl.hpp."""
    testvar = a * sqrtqw2_1 * sqrtthl2_1 + (1 - a) * sqrtqw2_2 * sqrtthl2_2
    r = max_(-1.0, min_(1.0, (qwthlsec - a * (qw1_1 - qw_first) * (thl1_1 - thl_first)
                              - (1 - a) * (qw1_2 - qw_first) * (thl1_2 - thl_first))
                             / torch.where(testvar != 0, testvar, torch.ones_like(testvar))))   # masked: safe input
    return torch.where(testvar != 0, r, torch.zeros_like(r))


_RAIR_OVER_CP = C.Rair / C.CP


def shoc_assumed_pdf_compute_temperature(thl1, pval):
    """shoc_assumed_pdf_compute_temperature_impl.hpp."""
    return thl1 / (mathfn.pow(rdiv(C.P0, pval), _RAIR_OVER_CP))


def shoc_assumed_pdf_compute_qs(Tl1_1, Tl1_2, pval):
    """shoc_assumed_pdf_compute_qs_impl.hpp. Returns (qs1, beta1, qs2, beta2)."""
    rair, rv, cp, lcond = C.Rair, C.RV, C.CP, C.LatVap
    esval1_1 = saturation.MurphyKoop_svp(Tl1_1, False)
    esval1_2 = saturation.MurphyKoop_svp(Tl1_2, False)
    qs1 = 0.622 * esval1_1 / max_(esval1_1, pval - esval1_1)
    beta1 = (rair / rv) * rdiv(lcond, rair * Tl1_1) * rdiv(lcond, cp * Tl1_1)
    condition = Tl1_1 != Tl1_2
    qs2_c = 0.622 * esval1_2 / max_(esval1_2, pval - esval1_2)
    beta2_c = (rair / rv) * rdiv(lcond, rair * Tl1_2) * rdiv(lcond, cp * Tl1_2)
    qs2 = torch.where(condition, qs2_c, qs1)
    beta2 = torch.where(condition, beta2_c, beta1)
    return qs1, beta1, qs2, beta2


_SQRT2 = math.sqrt(2.0)
_SQRT2PI = math.sqrt(2 * C.Pi)
def _std_s_small(x: torch.Tensor) -> float:
    """sqrt(norm_min_v<Real>) * 100 for the working type."""
    return math.sqrt(torch.finfo(x.dtype).tiny) * 100


def shoc_assumed_pdf_compute_s(qw1, qs, beta, pval, thl2, qw2, sqrtthl2, sqrtqw2, r_qwthl, C_prev, qn_prev):
    """shoc_assumed_pdf_compute_s_impl.hpp. Returns (s, std_s, qn, C).

    C_prev, qn_prev are the caller's values, kept where the C++ leaves them unset
    (default-constructed Packs are 0).
    """
    rair, basepres, cp, lcond = C.Rair, C.P0, C.CP, C.LatVap
    cthl = (((1 + beta * qw1) / square(1 + beta * qs)) * (cp / lcond)
            * beta * qs * mathfn.pow(pval / basepres, (rair / cp)))
    cqt = rdiv(1.0, 1 + beta * qs)
    std_s = _sqrt0(max_(0.0, square(cthl) * thl2 + square(cqt) * qw2
                        - 2 * cthl * sqrtthl2 * cqt * sqrtqw2 * r_qwthl))
    not_small = std_s > _std_s_small(std_s)
    s = qw1 - qs * ((1 + beta * qw1) / (1 + beta * qs))
    # Masked-out entries get std_s = 1, so s/std_s stays finite (no NaN in gradients);
    # selected entries are unchanged.
    std_safe = torch.where(not_small, std_s, torch.ones_like(std_s))
    Cv = torch.where(not_small, 0.5 * (1 + mathfn.erf(s / (_SQRT2 * std_safe))), C_prev)
    Cv = torch.where(~not_small & (s > 0), torch.ones_like(Cv), Cv)
    sc_ok = not_small & (Cv != 0)
    qn = torch.where(sc_ok, s * Cv + (std_safe / _SQRT2PI) * mathfn.exp(-0.5 * square(s / std_safe)),
                     qn_prev)
    qn = torch.where(~not_small & (s > 0), s, qn)
    qn_le_zero = qn <= 0
    Cv = torch.where(qn_le_zero, torch.zeros_like(Cv), Cv)
    qn = torch.where(qn_le_zero, torch.zeros_like(qn), qn)
    return s, std_s, qn, Cv


def shoc_assumed_pdf(thetal, qw, w_field, thl_sec, qw_sec, dtime, extra_diags, wthl_sec, w_sec,
                     wqw_sec, qwthl_sec, w3, pres, zt_grid, zi_grid, shoc_ql):
    """shoc_assumed_pdf_impl.hpp.

    Returns dict: shoc_cond, shoc_evap (if extra_diags, else None), shoc_cldfrac,
    shoc_ql, wqls, wthv_sec, shoc_ql2, plus diagnostics (tl_small counts).
    """
    nlev = zt_grid.shape[-1]
    nlevi = nlev + 1
    thl_tol, rt_tol, w_tol_sqd, w_thresh = 1e-2, 1e-4, 4e-4, 0.0
    largeneg, Tl_min = SC.largeneg, 100.0

    w3_zt = linear_interp(zi_grid, zt_grid, w3, nlevi, nlev, largeneg)
    thl_sec_zt = linear_interp(zi_grid, zt_grid, thl_sec, nlevi, nlev, 0.0)
    wthl_sec_zt = linear_interp(zi_grid, zt_grid, wthl_sec, nlevi, nlev, largeneg)
    qwthl_sec_zt = linear_interp(zi_grid, zt_grid, qwthl_sec, nlevi, nlev, largeneg)
    wqw_sec_zt = linear_interp(zi_grid, zt_grid, wqw_sec, nlevi, nlev, largeneg)
    qw_sec_zt = linear_interp(zi_grid, zt_grid, qw_sec, nlevi, nlev, 0.0)

    # if (k==0) shoc_ql(k)[0] = 0  (pack size 1: level 0)
    shoc_ql = torch.cat([torch.zeros_like(shoc_ql[:, :1]), shoc_ql[:, 1:]], dim=-1)

    pval = pres
    thl_first, w_first, qw_first = thetal, w_field, qw
    w3var, thlsec, qwsec = w3_zt, thl_sec_zt, qw_sec_zt
    qwthlsec, wqwsec, wthlsec, w2sec = qwthl_sec_zt, wqw_sec_zt, wthl_sec_zt, w_sec

    sqrtw2 = _sqrt0(w_sec)
    sqrtthl = max_(thl_tol, _sqrt0(thlsec))
    sqrtqt = max_(rt_tol, _sqrt0(qwsec))

    Skew_w, w1_1, w1_2, w2_1, w2_2, a = shoc_assumed_pdf_vv_parameters(w_first, w2sec, w3var, w_tol_sqd)
    thl1_1, thl1_2, thl2_1, thl2_2, sqrtthl2_1, sqrtthl2_2 = shoc_assumed_pdf_thl_parameters(
        wthlsec, sqrtw2, sqrtthl, thlsec, thl_first, w1_1, w1_2, Skew_w, a, thl_tol, w_thresh)
    qw1_1, qw1_2, qw2_1, qw2_2, sqrtqw2_1, sqrtqw2_2 = shoc_assumed_pdf_qw_parameters(
        wqwsec, sqrtw2, Skew_w, sqrtqt, qwsec, w1_2, w1_1, qw_first, a, rt_tol, w_thresh)
    w1_1 = shoc_assumed_pdf_tilde_to_real(w_first, sqrtw2, w1_1)
    w1_2 = shoc_assumed_pdf_tilde_to_real(w_first, sqrtw2, w1_2)
    r_qwthl_1 = shoc_assumed_pdf_inplume_correlations(sqrtqw2_1, sqrtthl2_1, a, sqrtqw2_2, sqrtthl2_2,
                                                      qwthlsec, qw1_1, qw_first, thl1_1, thl_first,
                                                      qw1_2, thl1_2)
    Tl1_1 = shoc_assumed_pdf_compute_temperature(thl1_1, pval)
    Tl1_2 = shoc_assumed_pdf_compute_temperature(thl1_2, pval)
    small1, small2 = Tl1_1 <= Tl_min, Tl1_2 <= Tl_min
    Tl1_1 = torch.where(small1, torch.full_like(Tl1_1, Tl_min), Tl1_1)
    Tl1_2 = torch.where(small2, torch.full_like(Tl1_2, Tl_min), Tl1_2)

    qs1, beta1, qs2, beta2 = shoc_assumed_pdf_compute_qs(Tl1_1, Tl1_2, pval)
    zero = torch.zeros_like(thetal)
    s1, std_s1, qn1, C1 = shoc_assumed_pdf_compute_s(qw1_1, qs1, beta1, pval, thl2_1, qw2_1,
                                                     sqrtthl2_1, sqrtqw2_1, r_qwthl_1, zero, zero)
    equal = (qw1_1 == qw1_2) & (thl2_1 == thl2_2) & (qs1 == qs2)
    s2c, std_s2c, qn2c, C2c = shoc_assumed_pdf_compute_s(qw1_2, qs2, beta2, pval, thl2_2, qw2_2,
                                                         sqrtthl2_2, sqrtqw2_2, r_qwthl_1, zero, zero)
    std_s2 = torch.where(equal, std_s1, std_s2c)
    s2 = torch.where(equal, s1, s2c)
    C2 = torch.where(equal, C1, C2c)
    qn2 = torch.where(equal, qn1, qn2c)

    ql1 = min_(qn1, qw1_1)
    ql2 = min_(qn2, qw1_2)
    shoc_cldfrac = min_(1.0, a * C1 + (1 - a) * C2)

    shoc_cond = shoc_evap = None
    if extra_diags:
        dum = max_(0.0, a * ql1 + (1 - a) * ql2)
        shoc_cond = max_(0.0, (dum - shoc_ql) / dtime)
        shoc_evap = max_(0.0, (shoc_ql - dum) / dtime)

    shoc_ql = max_(0.0, a * ql1 + (1 - a) * ql2)
    shoc_ql2 = max_(0.0, a * (s1 * ql1 + C1 * square(std_s1))
                    + (1 - a) * (s2 * ql2 + C2 * square(std_s2))
                    - square(shoc_ql))
    wqls = a * ((w1_1 - w_first) * ql1) + (1 - a) * ((w1_2 - w_first) * ql2)

    basepres, rair, rv, cp, lcond, basetemp = C.P0, C.Rair, C.RV, C.CP, C.LatVap, C.basetemp
    epsterm = rair / rv
    wthv_sec = (wthlsec + ((1 - epsterm) / epsterm) * basetemp * wqwsec
                + ((lcond / cp) * mathfn.pow(rdiv(basepres, pval), (rair / cp))
                   - (1 / epsterm) * basetemp) * wqls)
    return dict(shoc_cond=shoc_cond, shoc_evap=shoc_evap, shoc_cldfrac=shoc_cldfrac, shoc_ql=shoc_ql,
                wqls=wqls, wthv_sec=wthv_sec, shoc_ql2=shoc_ql2,
                tl1_small=(small1 | small2), tl1_nan=(torch.isnan(Tl1_1) | torch.isnan(Tl1_2)))


# ---------------------------------------------------------------- energy
def update_host_dse(thlm, shoc_ql, inv_exner, zt_grid, phis):
    """shoc_update_host_dse_impl.hpp."""
    lcond, cp, ggr = C.LatVap, C.CP, C.gravit
    temp = (thlm / inv_exner) + (lcond / cp) * shoc_ql
    return cp * temp + ggr * zt_grid + phis[:, None]


def shoc_energy_fixer(dtime, nadv, zt_grid, zi_grid, se_b, ke_b, wv_b, wl_b, se_a, ke_a, wv_a, wl_a,
                      wthl_sfc, wqw_sfc, rho_zt, tke, pint, host_dse, tke_tol: float = 0.0):
    """shoc_energy_fixer_impl.hpp (pack size 1)."""
    nlev = zt_grid.shape[-1]
    nlevi = nlev + 1
    cp, lcond, lice, mintke, ggr = C.CP, C.LatVap, C.LatIce, SC.mintke, C.gravit
    rho_zi = linear_interp(zt_grid, zi_grid, rho_zt, nlev, nlevi, 0.0)
    hdtime = dtime * nadv
    exner_int = PF.exner_function(pint[:, nlevi - 1])
    shf = wthl_sfc * cp * rho_zi[:, nlevi - 1] * exner_int
    lhf = wqw_sfc * rho_zi[:, nlevi - 1]
    te_a = se_a + ke_a + (lcond + lice) * wv_a + lice * wl_a
    te_b = se_b + ke_b + (lcond + lice) * wv_b + lice * wl_b
    te_b = te_b + (shf + lhf * (lcond + lice)) * hdtime
    # shoctop: min index k in [0, nlev-2) with tke(k) != mintke; nlev-2 if none
    k = torch.arange(nlev - 2, device=tke.device)
    # tke_tol = 0 is the C++ test (tke == mintke); see SHOCRuntime.energy_fixer_tke_tol
    at_floor = (tke[:, : nlev - 2] == mintke) if tke_tol == 0 else \
        (torch.abs(tke[:, : nlev - 2] - mintke) <= tke_tol * mintke)
    cand = torch.where(at_floor, torch.full_like(k, nlev - 2), k.expand(tke.shape[0], -1))
    shoctop = cand.amin(dim=-1)
    pint_top = torch.gather(pint, 1, shoctop[:, None])[:, 0]
    se_dis = (te_a - te_b) / (pint[:, nlevi - 1] - pint_top)
    kk = torch.arange(nlev, device=tke.device)
    active = kk[None, :] >= shoctop[:, None]
    return torch.where(active, host_dse - (se_dis * ggr)[:, None], host_dse)
