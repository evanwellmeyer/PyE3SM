"""A minimal single-column driver: IOP data, surface imports, pressures, the physics suite.

This is the host side, separate from the physics. Per step, in DP-EAMxx's order:
  0. the fields in EAMxx's ACCUMULATED group (surface precipitation masses, SHOC's
     wind perturbations) are reset to zero (AtmosphereDriver::run);
  1. surface import: the IOP file's surface fluxes and temperature
     (SurfaceCouplingImporter::overwrite_iop_imports, when iop_srf_prop), then the
     importer's albedo post-conditions;
  2. pressures, as HOMME's post-process gives them to physics
     (HommeDynamics::update_pressure): layer thicknesses from the hybrid grid and the
     surface pressure, interface pressures by a scan from the model top, dry
     thicknesses dp*(1 - qv); and omega from the IOP file;
  2b. without iop_srf_prop: open-ocean surface fluxes as E3SM's coupler would give them
     (ocean_surface_fluxes below) at the IOP surface temperature Tg;
  3. the physics group: iop_forcing, then (zm,) mac_aero_mic and rrtmgp (eamxx_torch.suite).

What replaces the dycore (there is none): the surface pressure stays at the IOP
file's value at the start time, the layer thicknesses follow from it, and the
large-scale vertical velocity seen by SHOC is the IOP omega. These are choices
of this driver, recorded in docs/plan.md; the single-column reference driver
(reference/eamxx_ext/scm_driver) makes the same ones.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch

from .. import constants as C
from .. import mathfn
from .. import physics_functions as PF
from ..property_checks import within_interval
from ..suite import Grid, Suite, SuiteConfig
from ..surface_flux import MINWIND, shr_flux_atmocn
from ..timestamp import TimeStamp
from .iop import (IOPDataManager, IOPParams, iop_forcing_postconditions, iop_forcing_run,
                  overwrite_iop_imports)

# DP-EAMxx initial-condition constants (reference/scripts/make_dp_case.py, from CIME's
# namelist defaults for FIOP-SCREAMv1-DP)
IC_CONSTANTS = dict(surf_evap=0.0, precip_liq_surf_mass=0.0, precip_ice_surf_mass=0.0,
                    cldfrac_liq=0.0, sgs_buoy_flux=0.0, eddy_diff_mom=0.0, eddy_diff_heat=0.0,
                    T_prev_micro_step=0.0, qv_prev_micro_step=0.0, qr=0.0, nr=0.0, qm=0.0,
                    bm=0.0, ni_activated=0.0, nc_nuceat_tend=0.0, tke=0.0, pbl_height=2000.0,
                    sfc_alb_dir_vis=0.0, sfc_alb_dir_nir=0.0, sfc_alb_dif_vis=0.0,
                    sfc_alb_dif_nir=0.0, surf_sens_flux=0.0, surf_lw_flux_up=0.0)
COLUMN_FIELDS = ("surf_evap", "precip_liq_surf_mass", "precip_ice_surf_mass", "pbl_height",
                 "sfc_alb_dir_vis", "sfc_alb_dir_nir", "sfc_alb_dif_vis", "sfc_alb_dif_nir",
                 "surf_sens_flux", "surf_lw_flux_up")


def update_pressure(pseudo_density, qv, ptop: float):
    """HommeDynamics::update_pressure (pack size 1): wet and dry pressures from dp and qv."""
    p_int = PF.column_scan_from_top(pseudo_density, ptop)
    p_mid = PF.compute_midpoint_values(p_int)
    dp_dry = pseudo_density * (1.0 - qv)
    p_dry_int = PF.column_scan_from_top(dp_dry, ptop)
    p_dry_mid = PF.compute_midpoint_values(p_dry_int)
    return dict(p_int=p_int, p_mid=p_mid, pseudo_density_dry=dp_dry, p_dry_int=p_dry_int,
                p_dry_mid=p_dry_mid)


def hybrid_dp(ps, hyai, hybi):
    """Layer thickness of the hybrid grid: (hyai(k+1)-hyai(k))*P0 + (hybi(k+1)-hybi(k))*ps."""
    dai = torch.as_tensor(np.diff(hyai), dtype=ps.dtype, device=ps.device)
    dbi = torch.as_tensor(np.diff(hybi), dtype=ps.dtype, device=ps.device)
    return dai * C.P0 + dbi * ps[:, None]


ACCUMULATED = ("precip_liq_surf_mass", "precip_ice_surf_mass", "um_pert_diff", "vm_pert_diff")


def ocean_surface_fluxes(state: dict, ts, minwind: float = MINWIND) -> dict:
    """Open-ocean surface fluxes for an EAMxx column, as E3SM's coupler would give them.

    1. The lowest-level quantities EAMxx exports (SurfaceCouplingExporter::
       compute_eamxx_exports): height of the lowest midpoint above the surface, winds,
       T, theta with the exner function taken to the lowest interface, qv, density.
    2. shr_flux_atmOcn, scheme 0 (eamxx_torch.surface_flux), ocean at rest at ts.
    3. The coupler merge with an ocean fraction of 1 and the importer's constant
       multiples (-1 on the fluxes): surf_sens_flux = -sen, surf_evap = -evap,
       surf_mom_flux = -(taux, tauy), surf_lw_flux_up = -lwup, surf_radiative_T = ts.
    """
    # E3SM's coupler works in double whatever the atmosphere's precision: the exporter
    # widens the lowest-level fields, and the importer stores the fluxes as Real.
    real, dev = state["T_mid"].dtype, state["T_mid"].device
    # (on the CPU when the state's device has no float64, e.g. MPS)
    cdev = dev if dev.type in ("cpu", "cuda") else torch.device("cpu")
    st = {k: state[k].to(cdev).double() for k in ("T_mid", "qv", "pseudo_density", "p_mid", "p_int", "horiz_winds")}
    T, qv, dp = st["T_mid"], st["qv"], st["pseudo_density"]
    p_mid, p_int, winds = st["p_mid"], st["p_int"], st["horiz_winds"]
    dz = PF.calculate_dz(dp, p_mid, T, qv)
    z_int = PF.calculate_z_int(dz, 0.0)
    z_mid = PF.calculate_z_mid(z_int)
    zbot = z_mid[:, -1]
    ubot, vbot = winds[:, 0, -1], winds[:, 1, -1]
    tbot = T[:, -1]
    thbot = tbot / mathfn.pow(p_mid[:, -1] / p_int[:, -1], C.RD * C.INV_CP)
    qbot = qv[:, -1]
    rbot = PF.calculate_density(dp[:, -1], dz[:, -1])
    ts = torch.as_tensor(ts, dtype=T.dtype, device=T.device).expand_as(tbot)
    zero = torch.zeros_like(tbot)
    f = shr_flux_atmocn(zbot, ubot, vbot, thbot, qbot, rbot, tbot, zero, zero, ts,
                        minwind=minwind)
    ofrac = 1.0
    out = dict(surf_sens_flux=-1.0 * (f["sen"] * ofrac), surf_evap=-1.0 * (f["evap"] * ofrac),
               surf_mom_flux=torch.stack([-1.0 * (f["taux"] * ofrac), -1.0 * (f["tauy"] * ofrac)], -1),
               surf_lw_flux_up=-1.0 * (f["lwup"] * ofrac), surf_radiative_T=ts * ofrac)
    return {k: v.to(real).to(dev) for k, v in out.items()}


def albedo_postconditions(state: dict) -> tuple[dict, dict]:
    """SurfaceCouplingImporter's post-conditions: albedos within [0, 1], repairable."""
    out, status = dict(state), {}
    for n in ("sfc_alb_dir_vis", "sfc_alb_dir_nir", "sfc_alb_dif_vis", "sfc_alb_dif_nir"):
        if n in out:
            out[n], status[n] = within_interval(out[n], 0.0, 1.0, True)
    return out, status


@dataclass
class SCMStep:
    state: dict
    tendencies: dict
    status: dict


class SCM:
    """Single-column (or batched independent columns) EAMxx physics with IOP forcing."""

    def __init__(self, iop_params: IOPParams, suite_config: SuiteConfig, run_t0: TimeStamp,
                 hybrid: dict, ncol: int = 1, cell_length: float = 3333.333333333333,
                 dt: float = 100.0):
        self.iop = IOPDataManager(iop_params, run_t0, hybrid["hyam"].shape[0], hybrid["hyam"],
                                  hybrid["hybm"])
        self.hybrid = hybrid
        self.ncol = ncol
        self.dt = dt
        self.run_t0 = run_t0.copy()
        self.time = run_t0.copy()          # start of the next step
        self.dtype, self.device = suite_config.dtype, torch.device(suite_config.device)
        f64, dev = self.dtype, self.device
        lat = torch.full((ncol,), iop_params.target_latitude, dtype=f64, device=dev)
        lon = torch.full((ncol,), iop_params.target_longitude, dtype=f64, device=dev)
        self.grid = Grid(lat=lat, lon=lon, cell_length=torch.full((ncol,), cell_length, dtype=f64, device=dev),
                         **hybrid)
        if suite_config.spa_target is None:
            suite_config.spa_target = (iop_params.target_latitude, iop_params.target_longitude)
        self.suite = Suite(suite_config, self.grid, run_t0, ncol,
                           first_procs=(("iop_forcing", self._run_iop_forcing),))
        self.ptop = C.P0 * float(hybrid["hyai"][0])
        self.iop.read_iop_file_data(run_t0)   # the driver's read at initialization

    # ------------------------------------------------------------------ initialization
    def initial_state(self, ic: dict) -> dict:
        """ICs: DP-EAMxx's constants, then `ic` (tensors or floats, e.g. from an EAMxx IC
        file column), then the IOP file (set_fields_from_iop_data), then the initial-run
        settings of SHOC and RRTMGP."""
        n, nlev = self.ncol, self.hybrid["hyam"].shape[0]
        f64 = torch.float64

        def full(v, cols_only):
            shape = (n,) if cols_only else (n, nlev)
            return torch.full(shape, float(v), dtype=f64)

        s = {k: full(v, k in COLUMN_FIELDS) for k, v in IC_CONSTANTS.items()}
        s["surf_mom_flux"] = torch.zeros((n, 2), dtype=f64)
        for k, v in ic.items():
            s[k] = torch.as_tensor(v, dtype=f64).expand_as(s[k]).clone() if k in s and not \
                isinstance(v, torch.Tensor) else torch.as_tensor(v, dtype=f64)
        s.setdefault("ps", torch.full((n,), float(self.iop.fields["Ps"]), dtype=f64))
        s = self.iop.set_fields_from_iop_data(s)
        # SHOC initialize_impl, RunType::Initial
        for k, v in (("sgs_buoy_flux", 0.0), ("eddy_diff_mom", 0.0), ("tke", 0.0004),
                     ("cldfrac_liq", 0.0), ("um_pert_diff", 0.0), ("vm_pert_diff", 0.0)):
            s[k] = full(v, False)
        # RRTMGP initialize_impl: n2 and co volume mixing ratios are constants
        ro = self.suite.config.rrtmgp
        s["n2_volume_mix_ratio"] = full(ro.n2vmr, False)
        s["co_volume_mix_ratio"] = full(ro.covmr, False)
        s["rad_heating_pdel"] = s.get("rad_heating_pdel", full(0.0, False))
        if self.suite.config.zm is not None:
            # ZM runs before the processes that compute these; EAMxx starts them at zero
            s.setdefault("cldfrac_tot", full(0.0, False))
            s.setdefault("zm_t_prev", full(0.0, False))
            s.setdefault("zm_q_prev", full(0.0, False))
            s.setdefault("thl_sec", torch.zeros((n, nlev + 1), dtype=f64))
        # built in float64 as read (the IOP data manager works in double); then the
        # working precision and device (SuiteConfig.dtype, .device)
        from ..precision import cast
        return self._pressures(cast(s, self.dtype, self.device))

    def _pressures(self, s: dict, ts: TimeStamp | None = None) -> dict:
        s = dict(s)
        s["pseudo_density"] = hybrid_dp(s["ps"], self.hybrid["hyai"], self.hybrid["hybi"])
        s.update(update_pressure(s["pseudo_density"], s["qv"], self.ptop))
        if self.iop.has_iop_field("omega"):
            if ts is not None:
                self.iop.read_iop_file_data(ts)
            s["omega"] = torch.as_tensor(self.iop.fields["omega"], dtype=torch.float64).to(
                s["T_mid"].dtype).to(s["T_mid"].device).expand(self.ncol, -1).clone()
        else:
            s["omega"] = torch.zeros_like(s["T_mid"])
        return s

    # ------------------------------------------------------------------ processes
    def _run_iop_forcing(self, s, dt, clock):
        self.iop.read_iop_file_data(clock.start)
        tracers = tuple(k for k in ("qv", "bm", "nc", "ni", "nr", "qc", "qi", "qm", "qr", "tke")
                        if k in s)
        out = iop_forcing_run(s, dt, self.iop, self.hybrid["hyam"], self.hybrid["hybm"], tracers,
                              num_global_cols=None)
        return iop_forcing_postconditions(out)

    def step(self, state: dict) -> SCMStep:
        """One model step starting at self.time."""
        s = dict(state)
        status = {}
        for k in ACCUMULATED:                  # AtmosphereDriver::run
            if k in s:
                s[k] = torch.zeros_like(s[k])
        if self.iop.params.iop_srf_prop:       # sc_import
            self.iop.read_iop_file_data(self.time)
            s = overwrite_iop_imports(s, self.iop)
        s, status["sc_import"] = albedo_postconditions(s)
        s = self._pressures(s, self.time)      # what HOMME hands to physics
        if not self.iop.params.iop_srf_prop:   # open ocean at the IOP surface temperature
            self.iop.read_iop_file_data(self.time)
            s.update(ocean_surface_fluxes(s, float(self.iop.fields["Tg"])))
        r = self.suite.step(s, self.dt)
        status.update(r["status"])
        self.time = self.time + self.dt
        return SCMStep(state=r["state"], tendencies=r["tendencies"], status=status)

    # ------------------------------------------------------------------ host state
    def save_host_state(self) -> dict:
        """Everything a step mutates outside the state dict: the time stamp, the process
        clocks and step count, SPA's data-interval state, the IOP data manager's fields
        (which keep history between reads). Small (SPA's monthly data is the largest part)."""
        return copy.deepcopy(dict(
            time=self.time, clocks=self.suite.clocks, num_steps=self.suite.num_steps,
            spa=self.suite.spa, iop={k: v for k, v in self.iop.__dict__.items() if k != "_nc"}))

    def restore_host_state(self, h: dict) -> None:
        h = copy.deepcopy(h)
        self.time = h["time"]
        self.suite.clocks = h["clocks"]
        self.suite.num_steps = h["num_steps"]
        self.suite.spa = h["spa"]
        self.iop.__dict__.update(h["iop"])

    def run(self, state: dict, nsteps: int, checkpoint: bool = False) -> dict:
        """nsteps steps from `state`; returns the final state.

        checkpoint=True: each step runs without keeping its autograd graph; only its input
        state is stored, and the step is recomputed with a graph during the backward pass
        (one step at a time). Memory for gradients through N steps is then one step's graph
        plus N states; forward values and gradients are the same as without it. Gradients
        flow to and from the state tensors (not to tensors captured inside the schemes).
        Each recomputation restores the host state the step started from (clocks, IOP data,
        SPA) and afterwards puts the current host state back.
        """
        s = state
        for _ in range(nsteps):
            if not checkpoint or not torch.is_grad_enabled():
                s = self.step(s).state
                continue
            keys = [k for k, v in s.items() if isinstance(v, torch.Tensor)]
            rest = {k: v for k, v in s.items() if k not in keys}
            box = {}
            outs = _CheckpointedStep.apply(self, self.save_host_state(), keys, rest, box,
                                           *[s[k] for k in keys])
            s = {**box["rest"], **dict(zip(box["keys"], outs))}
        return s


class _CheckpointedStep(torch.autograd.Function):
    """One SCM step: forward without a graph, recomputed with one in backward."""

    @staticmethod
    def forward(ctx, scm, pre, keys, rest, box, *tensors):
        with torch.no_grad():
            out = scm.step({**rest, **dict(zip(keys, tensors))}).state
        okeys = sorted(k for k, v in out.items() if isinstance(v, torch.Tensor))
        box["keys"] = okeys
        box["rest"] = {k: v for k, v in out.items() if not isinstance(v, torch.Tensor)}
        ctx.scm, ctx.pre, ctx.keys, ctx.rest, ctx.okeys = scm, pre, keys, rest, okeys
        ctx.save_for_backward(*tensors)
        res = tuple(out[k] for k in okeys)
        ctx.mark_non_differentiable(*[t for t in res if not t.is_floating_point()])
        return res

    @staticmethod
    def backward(ctx, *grads):
        scm = ctx.scm
        now = scm.save_host_state()
        scm.restore_host_state(ctx.pre)
        try:
            xs = [t.detach().requires_grad_(t.is_floating_point()) for t in ctx.saved_tensors]
            with torch.enable_grad():
                out = scm.step({**ctx.rest, **dict(zip(ctx.keys, xs))}).state
        finally:
            scm.restore_host_state(now)
        pairs = [(out[k], g) for k, g in zip(ctx.okeys, grads)
                 if g is not None and out[k].requires_grad]
        leaves = [x for x in xs if x.requires_grad]
        got = iter(torch.autograd.grad([o for o, _ in pairs], leaves, [g for _, g in pairs],
                                       allow_unused=True) if pairs else [None] * len(leaves))
        gx = [next(got) if x.requires_grad else None for x in xs]
        return (None, None, None, None, None, *gx)

def ic_from_files(ic_file: str, topo_file: str, lat: float, lon: float) -> dict:
    """Initial conditions DP-EAMxx reads from its IC and topography files, at the column
    IOPRemapper picks (smallest |dlat| + |dlon|, first on ties). The IOP file then
    overrides T_mid, qv, the winds and ps (SCM.initial_state)."""
    import netCDF4
    from ..spa import nearest_column_iop
    out = {}
    with netCDF4.Dataset(ic_file) as d:
        i = nearest_column_iop(np.asarray(d["lat"][:]), np.asarray(d["lon"][:]), lat, lon)
        for k in ("T_mid", "qv", "qc", "qi", "nc", "ni", "o3_volume_mix_ratio"):
            out[k] = np.asarray(d[k][0, i, :], dtype=np.float64)[None]
        out["ps"] = np.asarray(d["ps"][0, i], dtype=np.float64).reshape(1)
        out["horiz_winds"] = np.stack([np.asarray(d["U"][i, :], dtype=np.float64),
                                       np.asarray(d["V"][i, :], dtype=np.float64)])[None]
    with netCDF4.Dataset(topo_file) as d:
        j = nearest_column_iop(np.asarray(d["lat"][:]), np.asarray(d["lon"][:]), lat, lon)
        out["phis"] = np.asarray(d["PHIS"][j], dtype=np.float64).reshape(1)
        out["sgh30"] = np.asarray(d["SGH30"][j], dtype=np.float64).reshape(1)
    out["landfrac"] = np.zeros(1)   # a coupler import; zero without a coupler (as in DP runs)
    return out
