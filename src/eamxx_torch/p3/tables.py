"""P3 lookup tables and size-distribution parameters.

Translation of, at E3SM 80fe0a71:
  physics/p3/impl/p3_init_impl.hpp       (table files; read, not recomputed)
  physics/p3/impl/p3_table3_impl.hpp     (lookup, apply_table)
  physics/p3/impl/p3_table_ice_impl.hpp  (lookup_ice, lookup_rain, apply_table_ice,
                                          apply_table_coll)
  physics/p3/impl/p3_dsd2_impl.hpp       (get_cloud_dsd2, get_rain_dsd2,
                                          get_cdistr_logn0r)

The reference build reads its tables from files (write_tables=false), so the
port reads the same files instead of recomputing them:
  tables/p3_lookup_table_1.dat-v4.1.1   ASCII, ice and ice-rain collection
  tables/{mu_r,vn,vm,revap}_table_vals_v2.dat8  raw float64, computed offline
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .. import constants as C
from ..gradsafe import nonsingular
from .. import mathfn
from ..ekat import max_, min_, rdiv

# P3C (p3_functions.hpp)
DENSIZE, RIMSIZE, ISIZE, ICE_TABLE_SIZE = 5, 4, 50, 12
RCOLLSIZE, COLLECT_TABLE_SIZE = 30, 2
IPARAM, DNUSIZE = 3, 16
LOOKUP_TABLE_1A_DUM1_C = 4.135985029041767e+00  # 1/(0.1*log10(261.7))
P3_VERSION = "4.1.1"

DNU_TABLE_VALS = (0.000, -0.557, -0.430, -0.307, -0.186, -0.067, -0.050, -0.167,
                  -0.282, -0.397, -0.512, -0.626, -0.739, -0.853, -0.966, -0.966)


@dataclass(frozen=True)
class P3Tables:
    """P3LookupTables."""
    mu_r_table_vals: torch.Tensor        # (150,)
    vn_table_vals: torch.Tensor          # (300, 10)
    vm_table_vals: torch.Tensor          # (300, 10)
    revap_table_vals: torch.Tensor       # (300, 10)
    ice_table_vals: torch.Tensor         # (5, 4, 50, 12)
    collect_table_vals: torch.Tensor     # (5, 4, 50, 30, 2)
    dnu_table_vals: torch.Tensor         # (16,)


def read_ice_lookup_tables(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """read_ice_lookup_tables: the ASCII table, in the reference's read order."""
    ice = np.zeros((DENSIZE, RIMSIZE, ISIZE, ICE_TABLE_SIZE))
    coll = np.zeros((DENSIZE, RIMSIZE, ISIZE, RCOLLSIZE, COLLECT_TABLE_SIZE))
    with open(path) as f:
        tok = f.read().split()
    pos = 0
    assert tok[0] == "VERSION" and tok[1] == P3_VERSION, "unexpected table version"
    pos = 2
    for jj in range(DENSIZE):
        for ii in range(RIMSIZE):
            for i in range(ISIZE):
                pos += 2  # two leading integers
                j_idx = 0
                for j in range(15):
                    val = float(tok[pos]); pos += 1
                    if j > 1 and j != 10:
                        ice[jj, ii, i, j_idx] = val
                        j_idx += 1
            for i in range(ISIZE):
                for j in range(RCOLLSIZE):
                    pos += 2
                    k_idx = 0
                    for k in range(6):
                        val = float(tok[pos]); pos += 1
                        if k in (3, 4):
                            coll[jj, ii, i, j, k_idx] = np.log10(val)
                            k_idx += 1
    return ice, coll


def load_p3_tables(tables_dir: str | Path, dtype=torch.float64, device="cpu") -> P3Tables:
    """p3_init with write_tables=false: read the ice tables and the computed tables."""
    d = Path(tables_dir)
    ice, coll = read_ice_lookup_tables(d / f"p3_lookup_table_1.dat-v{P3_VERSION}")

    # EAMxx reads *.dat8 in double builds and *.dat4 (computed in float) in single
    # precision; the ice tables are parsed as double and stored as Real (a cast).
    single = dtype == torch.float32
    ext, npdt = ("dat4", np.float32) if single else ("dat8", np.float64)

    def raw(name, shape):
        a = np.fromfile(d / f"{name}_table_vals_v2.{ext}", dtype=npdt)
        assert a.size == int(np.prod(shape)), f"{name}: got {a.size}, expected {shape}"
        return a.reshape(shape)  # Kokkos LayoutRight

    if dtype not in (torch.float32, torch.float64):
        raise ValueError(f"P3 tables exist in float32 and float64, not {dtype}")
    t = lambda a: torch.as_tensor(a).to(dtype).to(device)  # noqa: E731  (precision first: MPS has no float64)
    return P3Tables(
        mu_r_table_vals=t(raw("mu_r", (C.MU_R_TABLE_DIM,))),
        vn_table_vals=t(raw("vn", (C.VTABLE_DIM0, C.VTABLE_DIM1))),
        vm_table_vals=t(raw("vm", (C.VTABLE_DIM0, C.VTABLE_DIM1))),
        revap_table_vals=t(raw("revap", (C.VTABLE_DIM0, C.VTABLE_DIM1))),
        ice_table_vals=t(ice), collect_table_vals=t(coll),
        dnu_table_vals=t(np.array(DNU_TABLE_VALS)),
    )


# ---------------------------------------------------------------- Table3 (rain)
@dataclass
class Table3:
    dumii: torch.Tensor
    dumjj: torch.Tensor
    rdumii: torch.Tensor
    rdumjj: torch.Tensor


def _trunc(x: torch.Tensor) -> torch.Tensor:
    """static_cast<Int>(Pack): truncation toward zero."""
    return torch.trunc(x).to(torch.int64)


def lookup(mu_r: torch.Tensor, lamr: torch.Tensor, context: torch.Tensor | None = None) -> Table3:
    """p3_table3_impl.hpp: lookup. Lanes outside `context` keep dumii=1, rdumii=0."""
    ctx = torch.ones_like(mu_r, dtype=torch.bool) if context is None else context
    # lamr == 0 (no rain) gives dum1 = +inf, which the index clips below turn into a
    # constant; the division is done on a safe denominator there so its gradient is 0,
    # not 0*inf = NaN (plan section 6). Values are unchanged.
    lam0 = lamr == 0
    dum1 = torch.where(lam0, ((mu_r + 1) / lamr).detach(),
                       (mu_r + 1) / torch.where(lam0, torch.ones_like(lamr), lamr))
    dum1_lt = ctx & (dum1 <= 195.e-6)

    inv_dum3_a = 0.1
    rdumii_a = (dum1 * 1.e6 + 5) * inv_dum3_a
    rdumii_a = min_(20.0, max_(1.0, rdumii_a))
    dumii_a = _trunc(rdumii_a).clamp(min=1, max=20)

    inv_dum3_b = C.THIRD * 0.1
    rdumii_b = (dum1 * 1.e+6 - 195) * inv_dum3_b + 20
    rdumii_b = min_(300.0, max_(20.0, rdumii_b))
    dumii_b = _trunc(rdumii_b).clamp(min=20, max=299)

    dum1_gte = ctx & ~dum1_lt
    zero = torch.zeros_like(mu_r)
    one_i = torch.ones_like(dumii_a)
    rdumii = torch.where(dum1_lt, rdumii_a, torch.where(dum1_gte, rdumii_b, zero))
    dumii = torch.where(dum1_lt, dumii_a, torch.where(dum1_gte, dumii_b, one_i))

    rdumjj = mu_r + 1
    rdumjj = min_(10, max_(1, rdumjj))
    dumjj = _trunc(rdumjj).clamp(min=1, max=9)
    rdumjj = torch.where(ctx, rdumjj, zero)
    dumjj = torch.where(ctx, dumjj, one_i)
    return Table3(dumii=dumii, dumjj=dumjj, rdumii=rdumii, rdumjj=rdumjj)


def _index2(table: torch.Tensor, i: torch.Tensor, j: torch.Tensor) -> torch.Tensor:
    """ekat::index(view_2d_table, i, j) per lane."""
    n1 = table.shape[1]
    flat = table.reshape(-1)
    return flat[(i * n1 + j).reshape(-1)].reshape(i.shape)


def apply_table(table: torch.Tensor, t: Table3) -> torch.Tensor:
    """p3_table3_impl.hpp: apply_table (bilinear in the (dumii, dumjj) quad)."""
    rdumii_m_dumii = t.rdumii - t.dumii.to(t.rdumii.dtype)
    t_im1_jm1 = _index2(table, t.dumii - 1, t.dumjj - 1)
    dum1 = t_im1_jm1 + rdumii_m_dumii * (_index2(table, t.dumii, t.dumjj - 1) - t_im1_jm1)
    t_im1_j = _index2(table, t.dumii - 1, t.dumjj)
    dum2 = t_im1_j + rdumii_m_dumii * (_index2(table, t.dumii, t.dumjj) - t_im1_j)
    return dum1 + (t.rdumjj - t.dumjj.to(t.rdumjj.dtype)) * (dum2 - dum1)


# ---------------------------------------------------------------- ice tables
@dataclass
class TableIce:
    dumi: torch.Tensor
    dumjj: torch.Tensor
    dumii: torch.Tensor
    dumzz: torch.Tensor
    dum1: torch.Tensor
    dum4: torch.Tensor
    dum5: torch.Tensor
    dum6: torch.Tensor


@dataclass
class TableRain:
    dumj: torch.Tensor
    dum3: torch.Tensor


def lookup_ice(qi, ni, qm, rhop, context: torch.Tensor | None = None) -> TableIce:
    """p3_table_ice_impl.hpp: lookup_ice."""
    ctx = torch.ones_like(qi, dtype=torch.bool) if context is None else context
    # qi or ni == 0 (no ice): log10 of 0 or 0/0; values kept, gradient 0 there (gradsafe)
    dum1 = nonsingular((qi == 0) | (ni == 0), lambda a, b: (mathfn.log10(a / b) + 18) * LOOKUP_TABLE_1A_DUM1_C - 10, qi, ni)
    dumi = _trunc(dum1)
    dum1 = min_(float(ISIZE), dum1)
    dum1 = max_(1., dum1)
    dumi = dumi.clamp(min=1, max=ISIZE - 1)

    dum4 = nonsingular(qi == 0, lambda a, b: (a / b) * 3 + 1, qm, qi)
    dumii = _trunc(dum4)
    dum4 = min_(float(RIMSIZE), dum4)
    dum4 = max_(1., dum4)
    dumii = dumii.clamp(min=1, max=RIMSIZE - 1)

    rhop_le_650 = ctx & (rhop <= 650)
    rhop_gt_650 = ~rhop_le_650 & ctx
    dum5 = torch.zeros_like(qi)
    dum5 = torch.where(rhop_le_650, (rhop - 50) * 0.005 + 1, dum5)
    dum5 = torch.where(rhop_gt_650, (rhop - 650) * 0.004 + 4, dum5)
    dumjj = _trunc(dum5)
    dum5 = min_(float(DENSIZE), dum5)
    dum5 = max_(1., dum5)
    dumjj = dumjj.clamp(min=1, max=DENSIZE - 1)

    dum6 = torch.full_like(qi, -99.0)
    dumzz = torch.full_like(dumi, -99 - 1)
    return TableIce(dumi=dumi - 1, dumjj=dumjj - 1, dumii=dumii - 1, dumzz=dumzz,
                    dum1=dum1, dum4=dum4, dum5=dum5, dum6=dum6)


def lookup_rain(qr, nr, context: torch.Tensor | None = None) -> TableRain:
    """p3_table_ice_impl.hpp: lookup_rain."""
    ctx = torch.ones_like(qr, dtype=torch.bool) if context is None else context
    gt_small = (qr > C.QSMALL) & (nr > 0.0) & ctx
    lt_small = ~gt_small & ctx
    dumlr = torch.ones_like(qr)
    one = torch.ones_like(qr)   # masked-out lanes: safe input (plan section 6)
    dumlr = torch.where(gt_small, mathfn.cbrt(torch.where(gt_small, qr, one)
                                              / (C.Pi * C.RHO_H2O * torch.where(gt_small, nr, one))), dumlr)
    dum3 = (mathfn.log10(1 * dumlr) + 5) * 10.70415
    dumj = _trunc(dum3)
    dum3 = min_(float(RCOLLSIZE), dum3)
    dum3 = max_(1, dum3)
    dumj = dumj.clamp(min=1, max=RCOLLSIZE - 1)
    dumj = torch.where(lt_small, torch.ones_like(dumj), dumj)
    dum3 = torch.where(lt_small, torch.ones_like(dum3), dum3)
    return TableRain(dumj=dumj - 1, dum3=dum3)


def _index_ice(table: torch.Tensor, jj, ii, i, idx: int) -> torch.Tensor:
    """ice_table_vals(dumjj, dumii, dumi, idx) per lane."""
    n = table.shape
    flat = table.reshape(-1)
    lin = ((jj * n[1] + ii) * n[2] + i) * n[3] + idx
    return flat[lin.reshape(-1)].reshape(jj.shape)


def apply_table_ice(idx: int, ice_table_vals: torch.Tensor, tab: TableIce) -> torch.Tensor:
    """p3_table_ice_impl.hpp: apply_table_ice (trilinear over dumi, dumii, dumjj)."""
    d1 = tab.dum1 - tab.dumi.to(tab.dum1.dtype) - 1
    d4 = tab.dum4 - tab.dumii.to(tab.dum4.dtype) - 1
    d5 = tab.dum5 - tab.dumjj.to(tab.dum5.dtype) - 1
    jj, ii, i = tab.dumjj, tab.dumii, tab.dumi

    def f(jj_, ii_):
        a = _index_ice(ice_table_vals, jj_, ii_, i, idx)
        b = _index_ice(ice_table_vals, jj_, ii_, i + 1, idx)
        return a + d1 * (b - a)

    iproc1 = f(jj, ii)
    gproc1 = f(jj, ii + 1)
    tmp1 = iproc1 + d4 * (gproc1 - iproc1)
    iproc1 = f(jj + 1, ii)
    gproc1 = f(jj + 1, ii + 1)
    tmp2 = iproc1 + d4 * (gproc1 - iproc1)
    return tmp1 + d5 * (tmp2 - tmp1)


def _index_coll(table: torch.Tensor, jj, ii, i, j, idx: int) -> torch.Tensor:
    """collect_table_vals(dumjj, dumii, dumi, dumj, idx) per lane."""
    n = table.shape
    flat = table.reshape(-1)
    lin = (((jj * n[1] + ii) * n[2] + i) * n[3] + j) * n[4] + idx
    return flat[lin.reshape(-1)].reshape(jj.shape)


def apply_table_coll(idx: int, collect_table_vals: torch.Tensor, ti: TableIce,
                     tr: TableRain) -> torch.Tensor:
    """p3_table_ice_impl.hpp: apply_table_coll (quadrilinear, adding the rain size)."""
    d1 = ti.dum1 - ti.dumi.to(ti.dum1.dtype) - 1
    d4 = ti.dum4 - ti.dumii.to(ti.dum4.dtype) - 1
    d5 = ti.dum5 - ti.dumjj.to(ti.dum5.dtype) - 1
    d3 = tr.dum3 - tr.dumj.to(tr.dum3.dtype) - 1
    i, j = ti.dumi, tr.dumj

    def f(jj_, ii_):
        a = _index_coll(collect_table_vals, jj_, ii_, i, j, idx)
        b = _index_coll(collect_table_vals, jj_, ii_, i + 1, j, idx)
        dproc1 = a + d1 * (b - a)
        a2 = _index_coll(collect_table_vals, jj_, ii_, i, j + 1, idx)
        b2 = _index_coll(collect_table_vals, jj_, ii_, i + 1, j + 1, idx)
        dproc2 = a2 + d1 * (b2 - a2)
        return dproc1 + d3 * (dproc2 - dproc1)

    jj, ii = ti.dumjj, ti.dumii
    iproc1 = f(jj, ii)
    gproc1 = f(jj, ii + 1)
    tmp1 = iproc1 + d4 * (gproc1 - iproc1)
    iproc1 = f(jj + 1, ii)
    gproc1 = f(jj + 1, ii + 1)
    tmp2 = iproc1 + d4 * (gproc1 - iproc1)
    return tmp1 + d5 * (tmp2 - tmp1)


# ---------------------------------------------------------------- size distributions
def get_cloud_dsd2(qc, nc, mu_c, rho, nu, dnu, lamc, cdist, cdist1,
                   context: torch.Tensor | None = None):
    """p3_dsd2_impl.hpp: get_cloud_dsd2. Returns (nc, mu_c, nu, lamc, cdist, cdist1).

    lamc, cdist, cdist1, nu and mu_c are zeroed where `context` holds and only
    computed where qc >= qsmall; outside `context` the inputs are kept.
    iparam = 3 in EAMxx, so `nu` stays 0 (the dnu table is for iparam == 1).
    """
    ctx = torch.ones_like(qc, dtype=torch.bool) if context is None else context
    zero = torch.zeros_like(qc)
    lamc = torch.where(ctx, zero, lamc)
    cdist = torch.where(ctx, zero, cdist)
    cdist1 = torch.where(ctx, zero, cdist1)
    nu = torch.where(ctx, zero, nu)
    mu_c = torch.where(ctx, zero, mu_c)
    qc_gt_small = (qc >= C.QSMALL) & ctx

    nc_lim = max_(nc, C.NSMALL)
    nc = torch.where(qc_gt_small, nc_lim, nc)
    mu_c_local = 0.0005714 * (nc * 1.e-6 * rho) + 0.2714
    mu_c_local = rdiv(1.0, mu_c_local * mu_c_local) - 1
    mu_c_local = max_(mu_c_local, 2)
    mu_c_local = min_(mu_c_local, 15)
    mu_c = torch.where(qc_gt_small, mu_c_local, mu_c)
    if IPARAM == 1:  # not the EAMxx default; kept for completeness
        dumi = _trunc(mu_c) - 1
        dnu0 = dnu[dumi.clamp(min=0, max=DNUSIZE - 2)]
        dnu1 = dnu[(dumi + 1).clamp(min=0, max=DNUSIZE - 1)]
        nu = torch.where(qc_gt_small, dnu0 + (dnu1 - dnu0) * (mu_c - dumi.to(qc.dtype) - 1), nu)

    # masked-out lanes (qc below qsmall, possibly 0) get a safe input (plan section 6)
    one = torch.ones_like(qc)
    lamc_v = mathfn.cbrt(torch.where(qc_gt_small, C.CONS1 * nc * (mu_c + 3) * (mu_c + 2) * (mu_c + 1)
                                     / torch.where(qc_gt_small, qc, one), one))
    lamc = torch.where(qc_gt_small, lamc_v, lamc)
    lammin = (mu_c + 1) * 2.5e+4
    lammax = (mu_c + 1) * 1.e+6
    lamc_lt_min = (lamc < lammin) & qc_gt_small
    lamc_gt_max = (lamc > lammax) & qc_gt_small
    min_or_max = lamc_lt_min | lamc_gt_max
    lamc = torch.where(lamc_lt_min, lammin, lamc)
    lamc = torch.where(lamc_gt_max, lammax, lamc)
    nc_adj = 6 * (lamc * lamc * lamc) * qc / (C.Pi * C.RHO_H2O * (mu_c + 3) * (mu_c + 2) * (mu_c + 1))
    nc = torch.where(min_or_max, nc_adj, nc)
    cdist = torch.where(qc_gt_small, nc * (mu_c + 1) / torch.where(qc_gt_small, lamc, torch.ones_like(lamc)), cdist)
    cdist1 = torch.where(qc_gt_small, nc / mathfn.tgamma(mu_c + 1), cdist1)
    return nc, mu_c, nu, lamc, cdist, cdist1


def get_rain_dsd2(qr, nr, mu_r, lamr, constant_mu_rain: float,
                  context: torch.Tensor | None = None):
    """p3_dsd2_impl.hpp: get_rain_dsd2. Returns (nr, mu_r, lamr)."""
    ctx = torch.ones_like(qr, dtype=torch.bool) if context is None else context
    zero = torch.zeros_like(qr)
    lamr = torch.where(ctx, zero, lamr)
    qr_gt_small = (qr >= C.QSMALL) & ctx
    nr_lim = max_(nr, C.NSMALL)
    mu_r = torch.where(qr_gt_small, torch.full_like(qr, constant_mu_rain), mu_r)
    mass_to_d3_factor = C.CONS1 * (mu_r + 3) * (mu_r + 2) * (mu_r + 1)
    # masked-out lanes (qr below qsmall, possibly 0) get a safe input (plan section 6)
    qr_safe = torch.where(qr_gt_small, qr, torch.ones_like(qr))
    lamr = torch.where(qr_gt_small, mathfn.cbrt(mass_to_d3_factor * nr_lim / qr_safe), lamr)
    lammax = (mu_r + 1.) * 1.e+5
    lammin = (mu_r + 1.) * 500
    lt = qr_gt_small & (lamr < lammin)
    gt = qr_gt_small & (lamr > lammax)
    either = lt | gt
    nr = torch.where(qr_gt_small, nr_lim, nr)
    lamr = torch.where(lt, lammin, lamr)
    lamr = torch.where(gt, lammax, lamr)
    nr = torch.where(either, lamr * lamr * lamr * qr / mass_to_d3_factor, nr)
    return nr, mu_r, lamr


def get_cdistr_logn0r(qr, nr, mu_r, lamr, cdistr, logn0r,
                      context: torch.Tensor | None = None):
    """p3_dsd2_impl.hpp: get_cdistr_logn0r. Returns (cdistr, logn0r)."""
    ctx = torch.ones_like(qr, dtype=torch.bool) if context is None else context
    zero = torch.zeros_like(qr)
    cdistr = torch.where(ctx, zero, cdistr)
    logn0r = torch.where(ctx, zero, logn0r)
    qr_gt_small = (qr >= C.QSMALL) & ctx
    cdistr_v = nr / mathfn.tgamma(mu_r + 1)
    cdistr = torch.where(qr_gt_small, cdistr_v, cdistr)
    # masked-out entries get 1 in the logs (translation rule for masked branches)
    one = torch.ones_like(qr)
    logn0r = torch.where(qr_gt_small, mathfn.log10(torch.where(qr_gt_small, cdistr, one))
                         + (mu_r + 1) * mathfn.log10(torch.where(qr_gt_small, lamr, one)), logn0r)
    return cdistr, logn0r
