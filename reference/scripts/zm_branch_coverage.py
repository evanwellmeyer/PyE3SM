#!/usr/bin/env python3
"""Count how often each ZM branch/limiter is taken when replaying reference dumps.

  python reference/scripts/zm_branch_coverage.py RUN_DIR [RUN_DIR ...]

A branch never taken in any replayed call is not validated by the replay. The port
matches the reference bitwise on these calls, so its masks are the reference's.
Conditions are recomputed from each function's inputs and outputs (wrappers), and
for invert_entropy from an instrumented copy of the Brent loop that must return
the port's values exactly.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import replay_check as rc  # noqa: E402
from read_dump import read_dump  # noqa: E402

from eamxx_torch import constants as PC  # noqa: E402
from eamxx_torch import mathfn  # noqa: E402
from eamxx_torch.ekat import min_  # noqa: E402
from eamxx_torch.zm import cape as CA  # noqa: E402
from eamxx_torch.zm import closure as CL  # noqa: E402
from eamxx_torch.zm import cloud as CP  # noqa: E402
from eamxx_torch.zm import evap_mcsp as EM  # noqa: E402
from eamxx_torch.zm import functions as FN  # noqa: E402
from eamxx_torch.zm import interface as IF  # noqa: E402
from eamxx_torch.zm import main as MN  # noqa: E402
from eamxx_torch.zm.constants import ZMC  # noqa: E402
from eamxx_torch.zm.util import gather  # noqa: E402

COV: dict[str, list[int]] = defaultdict(lambda: [0, 0])
CTX: dict = {}   # the active-column mask of the current call (set before cloud properties run)


def rec(name, mask):
    mask = torch.as_tensor(mask)
    COV[name][0] += int(mask.sum())
    COV[name][1] += mask.numel()


def wrap(mod, fname, probe, alias_mods=()):
    orig = getattr(mod, fname)

    def w(*a, **k):
        out = orig(*a, **k)
        probe(a, k, out)
        return out
    setattr(mod, fname, w)
    for m in alias_mods:
        setattr(m, fname, w)


def brent_probe(s, p, qt, tfg, active=None, early_exit=True):
    """invert_entropy with branch counters (same arithmetic as the port)."""
    t_ref, qst_ref, bad_ref = ORIG_INVERT(s, p, qt, tfg, active=active, early_exit=early_exit)
    act = torch.ones_like(tfg, dtype=torch.bool) if active is None else active
    a, b = tfg - 10, tfg + 10
    fa = FN.entropy(a, p, qt) - s
    fb = FN.entropy(b, p, qt) - s
    c, fc = b, fb
    d = torch.zeros_like(tfg)
    ebr = torch.zeros_like(tfg)
    done = ~act
    b = torch.where(act, b, tfg)
    iters = torch.zeros_like(tfg, dtype=torch.int64)
    for _ in range(ZMC.LOOPMAX + 1):
        run = ~done
        m = run & (((fb > 0) & (fc > 0)) | ((fb < 0) & (fc < 0)))
        c, d, fc, ebr = (torch.where(m, a, c), torch.where(m, b - a, d), torch.where(m, fa, fc),
                         torch.where(m, b - a, ebr))
        m = run & (torch.abs(fc) < torch.abs(fb))
        a, b, c = torch.where(m, b, a), torch.where(m, c, b), torch.where(m, b, c)
        fa, fb, fc = torch.where(m, fb, fa), torch.where(m, fc, fb), torch.where(m, fb, fc)
        tol = 2 * ZMC.tol_eps * torch.abs(b) + ZMC.half * ZMC.tol_coeff
        xm = ZMC.half * (c - b)
        conv = (torch.abs(xm) <= tol) | (fb == 0)
        rec("invert_entropy: converged with fb == 0", run & (fb == 0))
        done = done | (run & conv)
        run = ~done
        if not bool(run.any()):
            break
        iters = iters + run.to(torch.int64)
        use = (torch.abs(ebr) >= tol) & (torch.abs(fa) > torch.abs(fb))
        fa_s = torch.where(use, fa, torch.ones_like(fa))
        fc_s = torch.where(use & (a != c), fc, torch.ones_like(fc))
        sbr = fb / fa_s
        qbr_q = fa_s / fc_s
        rbr = fb / fc_s
        pbr = torch.where(a == c, 2 * xm * sbr, sbr * (2 * xm * qbr_q * (qbr_q - rbr) - (b - a) * (rbr - 1)))
        qbr = torch.where(a == c, 1 - sbr, (qbr_q - 1) * (rbr - 1) * (sbr - 1))
        qbr = torch.where(pbr > 0, -qbr, qbr)
        pbr = torch.abs(pbr)
        acc = 2 * pbr < min_(3 * xm * qbr - torch.abs(tol * qbr), torch.abs(ebr * qbr))
        rec("invert_entropy: secant step accepted (a == c)", run & use & acc & (a == c))
        rec("invert_entropy: inverse quadratic step accepted", run & use & acc & (a != c))
        rec("invert_entropy: interpolation rejected -> bisection", run & use & ~acc)
        rec("invert_entropy: bisection (small ebr or |fa| <= |fb|)", run & ~use)
        dn = torch.where(use & acc, pbr / torch.where(use & acc, qbr, torch.ones_like(qbr)), xm)
        ebr = torch.where(run, torch.where(use & acc, d, dn), ebr)
        d = torch.where(run, dn, d)
        a, fa = torch.where(run, b, a), torch.where(run, fb, fa)
        small = torch.abs(d) <= tol
        rec("invert_entropy: step below tolerance -> copysign(tol, xm)", run & small)
        b = torch.where(run, b + torch.where(small, torch.copysign(tol, xm), d), b)
        fb = torch.where(run, FN.entropy(b, p, qt) - s, fb)
    assert torch.equal(b, t_ref), "instrumented Brent loop differs from the port"
    rec("invert_entropy: iterations >= 5", act & (iters >= 5))
    rec("invert_entropy: iterations >= 10", act & (iters >= 10))
    COV["invert_entropy: max iterations"][0] = max(COV["invert_entropy: max iterations"][0],
                                                   int(iters.max()))
    COV["invert_entropy: max iterations"][1] = ZMC.LOOPMAX + 1
    rec("invert_entropy: not converged (reference aborts)", bad_ref)
    return t_ref, qst_ref, bad_ref


ORIG_INVERT = FN.invert_entropy


def install():
    CA.invert_entropy = brent_probe

    def mse_probe(a, k, out):
        rt, pver, num_msg, top_k = a[:4]
        kl, _ = out
        rec("launch level: at the lowest searched level (pver-2)", kl == pver - 1 - rt.mx_bot_lyr_adj)
        rec("launch level: at the upper search limit (ULL)", kl == top_k)
    wrap(CA, "find_mse_max", mse_probe)

    def parcel_probe(a, k, out):
        rt, pver, num_msg, klaunch = a[:4]
        tpert, pblt = a[7], a[8]
        pt, pv, pq, lcl_p, lcl_t, lcl_k, _ = out
        rec("parcel: launch above the PBL top, tpert zeroed (tpert_fix)",
            (klaunch < pblt) & (tpert > 0))
        rec("parcel: tpert applied at launch", ~(klaunch < pblt) & (tpert > 0))
        rec("parcel: LCL found above the launch level", lcl_k < klaunch)
        rec("parcel: LCL pressure below 600 mb (no CAPE)", lcl_p < ZMC.lcl_pressure_threshold)
        lev = torch.arange(pver).unsqueeze(0)
        above = (lev < klaunch.unsqueeze(-1)) & (lev >= num_msg)
        rec("parcel levels: below freezing (freezing entropy terms)", above & (pt <= PC.Tmelt))
    wrap(CA, "compute_dilute_parcel", parcel_probe)

    def capep_probe(a, k, out):
        pver = a[1]
        _, _, _, eql, cape = out
        rec("CAPE: positive", cape > 0)
        rec("CAPE: no negative-buoyancy crossing (eql = pver-1)", eql == pver - 1)
    wrap(CA, "compute_cape_from_parcel", capep_probe)

    def main_probe(a, k, out):
        rt, limcnv, first = a[0], a[1], a[2]
        rec("main: first step (no DCAPE)", torch.full_like(out["active"], first))
        cape_pos = out["cape"] > 0
        rec("main: CAPE > 0 but DCAPE <= 0 (not triggered)", cape_pos & ~out["active"])
        rec("main: active", out["active"])
        final = gather(out["mflx_up"], out["msemax_klev"])   # unit flux at cloud base, scaled
        rec("main (active cols): cloud-base flux limited by 1/(dt*max(mu/dp))",
            out["active"] & (final < CTX["cbmf"]))
    wrap(MN, "zm_conv_main", main_probe, alias_mods=(IF,))

    def interf_probe(a, k, out):
        s_mid, q_mid, msg, pver, active = a
        s_int, q_int = out
        mean_s = 0.5 * (s_mid[:, msg + 1:] + s_mid[:, msg:-1])
        mean_q = 0.5 * (q_mid[:, msg + 1:] + q_mid[:, msg:-1])
        CTX["active"] = active
        act = active.unsqueeze(-1)
        rec("interfaces: s log-linear", act & (s_int[:, msg + 1:] != mean_s))
        rec("interfaces: q log-linear", act & (q_int[:, msg + 1:] != mean_q))
        rec("interfaces: q arithmetic mean", act & (q_int[:, msg + 1:] == mean_q))
    wrap(MN, "_interfaces", interf_probe)

    def ent_probe(a, k, out):
        pver, msg, jb, jt, j0_in = a[:5]
        j0, _, lam, lmax = out
        act = CTX["active"]
        rec("entrainment (active cols): lambda_max > 0", act & (lmax > 0))
        rec("entrainment (active cols): lambda_max == 0 (no updraft)", act & (lmax == 0))
        rec("entrainment (active cols): lambda_max at its upper clip", act & (lmax == ZMC.lambda_limit_max))
        rec("entrainment (active cols): detrainment level moved down", act & (j0 != j0_in))
    wrap(CP, "zm_calc_fractional_entrainment", ent_probe)

    def cloud_probe(a, k, out):
        rt, pver, msg, limcnv = a[:4]
        jb, lel = a[13], a[14]
        jt, jlcl, jd = out["jt"], out["jlcl"], out["jd"]
        act = CTX["active"]
        a2 = act.unsqueeze(-1)
        for name, m in out["_diag"].items():
            rec(f"cloud (active cols): {name}", act & m)
        rec("cloud (active cols): top below the parcel equilibrium level (jt > lel)", act & (jt > lel))
        rec("cloud (active cols): downdraft present (jd < jb)", act & (jd < jb))
        rec("cloud (active cols): no downdraft (jd == jb)", act & (jd == jb))
        lev = torch.arange(pver).unsqueeze(0)
        inside = a2 & (lev >= jt.unsqueeze(-1)) & (lev < jb.unsqueeze(-1))
        rec("cloud levels (active): condensation > 0", inside & (out["cu"] > 0))
        rec("cloud levels (active): rain production > 0", inside & (out["rprd"] > 0))
        rec("cloud levels (active): net rain production < 0 (evaporation wins)", inside & (out["rprd"] < 0))
        rec("cloud levels (active): downdraft evaporation > 0", a2 & (out["evp"] > 0))
    wrap(CP, "zm_cloud_properties", cloud_probe, alias_mods=(MN,))

    def closure_probe(a, k, out):
        act = CTX["active"]
        CTX["cbmf"] = out
        rec("closure (active cols): mass flux > 0", act & (out > 0))
        rec("closure (active cols): mass flux 0 (clipped, or dadt == 0)", act & (out == 0))
    wrap(CL, "zm_closure", closure_probe, alias_mods=(MN,))

    def evap_probe(a, k, out):
        rt, estbl, dt, p_mid, p_del, t_mid, q_mid, prd, cld, prec = a
        fp = out["flxprec"][:, :-1]
        rec("evap levels: rain falling in (flxprec > 0)", fp > 0)
        rec("evap levels: evaporation > 0", out["tend_q"] > 0)
        rec("evap levels: snow melting", (out["flxsnow"][:, :-1] > 0) & (t_mid > PC.Tmelt))
        rec("evap levels: snow falling (flxsnow > 0)", out["flxsnow"][:, :-1] > 0)
        rec("evap levels: mixed-phase snow fraction", (t_mid <= PC.Tmelt) & (t_mid >= PC.Tmelt - 5)
            & (prd != 0))
        rec("evap levels: flux clipped at 0", (out["flxprec"][:, 1:] == 0) & (fp > 0))
        rec("evap: surface snow > 0", out["snow"] > 0)
    wrap(EM, "zm_conv_evap", evap_probe, alias_mods=(IF,))

    def mcsp_probe(a, k, out):
        rt, dt, jctop, pmid = a[:4]
        pver = pmid.shape[-1]
        has = jctop != pver - 1
        sh = torch.abs(out["mcsp_shear"])
        rec("MCSP: on (tendencies applied)", out["mcsp_freq"] > 0)
        rec("MCSP: convection too shallow (< 700 hPa)", has & (out["zm_depth"] < ZMC.MCSP_conv_depth_min))
        rec("MCSP: shear below 3 m/s", has & (sh < ZMC.MCSP_shear_min))
        rec("MCSP: shear at or above 200 m/s", has & (sh >= ZMC.MCSP_shear_max))
    wrap(EM, "zm_conv_mcsp_tend", mcsp_probe, alias_mods=(IF,))

    def tpert_probe(a, k, out):
        rec("tpert: > 0", out > 0)
        rec("tpert: at the 2 K limiter", out == ZMC.tpert_limiter)
    wrap(IF, "calculate_tpert", tpert_probe)


def main():
    torch.set_num_threads(1)
    ctx = mathfn.use_backend("libm")
    ctx.__enter__()
    install()
    for run in map(Path, sys.argv[1:]):
        st = rc.case_settings(run)
        for pj in sorted((run / "dump" / "zm").glob("*_c0_*_pre.json")):
            mpre, pre = read_dump(pj.with_suffix(""))
            rc.run_zm(pre, {"dt": mpre["dt"], "step": mpre["step"]}, st, False)
    w = max(len(k) for k in COV)
    for k in sorted(COV):
        n, tot = COV[k]
        flag = "   <-- never" if n == 0 else ""
        print(f"{k:{w}s} {n:9d} / {tot:9d}{flag}")


if __name__ == "__main__":
    main()
