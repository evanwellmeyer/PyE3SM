# Status

Updated 25 September 2026.

## Verdict

- **`cld_fraction`: matches the reference, bitwise.**
- **SPA: matches the reference, bitwise.** Includes the switch to the next monthly data interval.
- **TMS: matches the reference, bitwise** (libm math mode).
- **SHOC: matches the reference, bitwise** (libm math mode). 163 calls, 2.9M values.
- **P3: matches the reference, bitwise** (libm math mode). 212 calls, 5.6M values.
- **RRTMGP: matches the reference, bitwise** (libm math mode). 268 calls, 1.94M values.
- **ZM: matches the reference, bitwise** (libm math mode). 21 calls of 100 columns, 7.3M values, including DCAPE, MCSP, downdrafts, rain evaporation and momentum transport.
- **Ocean surface fluxes (E3SM's coupler scheme 0): match the reference, bitwise.** 3000 states through E3SM's own Fortran.
- **IOP forcing: matches the reference, bitwise** (libm math mode). 130 calls, 2.3M values, including nudging over a 100-column domain.
- **Surface import from IOP data, and the pressures HOMME hands to physics: match the reference, bitwise.**
- **The physics suite, one whole step: matches the reference, bitwise**, in both configurations. 50 DP steps (7 processes) and 3 EAMxx-LR steps (ZM, 12 sub-cycles of 5 processes, RRTMGP).
- **Single-column runs: match EAMxx, bitwise, at every step.** The port's single-column model against EAMxx's own physics run on one column by the new reference driver: 9 cases, 5568 steps (BOMEX, DYCOMS-RF01, CASS, ARM97 up to 48 h, RICO and GATE III with ocean fluxes; BOMEX, ARM97 and GATE III with ZM for up to 5 days). Every output field after every step.
- **Property checks (interval, lower bound, repair): match the reference, bitwise.**
- **Gradients: finite everywhere tested; correct where the model is smooth** (docs/differentiability.md). The remaining disagreements with finite differences are discontinuities in EAMxx itself: SHOC's energy-fixer level pick, RRTMGP's McICA seed, P3's table indices, ZM's trigger and level picks. Two options (state-free McICA seeds, SHOC energy-fixer tolerance) and per-step checkpointing are available, off by default; with them on, one-step gradients agree with finite differences to 3e-4 or better in all 8 cases and all 40 validation verdicts are unchanged. A day with ZM needs 1.4 GB with checkpointing.
- **The suite against observations and LES: does not match in most checks.** 13 of 40 pass (docs/validation.md). Stratocumulus (DYCOMS-RF01) matches the observations. Shallow cumulus (BOMEX, RICO, CASS) has too much low cloud. Deep convection (ARM97, GATE III) is further from the observations than E3SM's own doubly periodic runs. These are EAMxx's results: in all 12 validation runs the port matches EAMxx bitwise. EAMxx with its dycore shows the same BOMEX excess.

- **Float32: cld_fraction matches EAMxx's single-precision build bitwise; SHOC and P3 do not match it bitwise** (docs/float32_gpu.md). Most of their fields are far closer to EAMxx float than float precision itself moves them; a few are only as close as float64 is. EAMxx has no float RRTMGP, ZM, TMS or IOP forcing, so float32 runs keep those in float64. The reference's own float formulas overflow in RRTMGP (every call) and in P3's immersion freezing (below about 195 K). The port reproduces both; the P3 one turns 3 of 12 float32 validation runs into NaN. `P3Runtime.immersion_freezing_in_float64` (off by default) evaluates that step in float64; with it, all 12 are finite and 2 of 40 validation verdicts differ from float64 (both marginal). Float32 stays at tolerance level (decided 25 Sep): no bitwise audit against EAMxx's float build.
- **GPU (Apple MPS, float32): runs; agrees with the CPU about as well as float32 agrees with float64.** It is not faster than the CPU here (4.6 vs 3.9 s per step at 1024 columns); RRTMGP in float64 dominates. Float64 on a GPU is untested (needs CUDA).

"Bitwise" means against the `strict` reference build (no FMA contraction, pack size 1), with the port's transcendental functions taken from the same libm. With PyTorch's own functions (the default), results differ by about 1 ulp per call; tolerances are stated per scheme below.

## Matches the reference

### IOP forcing, surface import, pressures (`src/eamxx_torch/scm/iop.py`, `scm/driver.py`)

Source: `share/data_managers/IOPDataManager.cpp`, `physics/iop_forcing/eamxx_iop_forcing_process_interface.cpp`, EKAT's `LinInterp`, `share/field/utils/horiz_contraction.cpp`, `control/atmosphere_surface_coupling_importer.cpp` (`overwrite_iop_imports`), and HOMME's `update_pressure`. E3SM 80fe0a71. Ported: IOP file reading (base date, time index, vertical interpolation to the model levels with the file's surface values or an extrapolated one, 3D forcing, the zero-temperature correction, setting the initial state), subsidence, large-scale T and q tendencies, Coriolis, nudging toward the domain mean, the post-condition checks, the four IOP surface imports.

| Check | Data | Result |
|---|---|---|
| `iop_forcing` replay | BOMEX, DYCOMS-RF01 (subsidence), ARM97 6 h and 7.5 days (3D forcing, surface pressure varying in time), CASS, and single-rank runs of ARM97 and CASS dumped on all 100 columns (nudging of winds, and of T, q and winds with Coriolis and subsidence). 130 calls, 2.3M values | **Bitwise**, around `run_impl` and around the checks |
| `SurfaceCouplingImporter` replay | same runs, 130 calls | **Bitwise** |
| Pressures (`p_int`, `p_mid`, dry versions) vs HOMME's post-process | 170 HOMME calls | **Bitwise** |

Notes:
- The data manager keeps its fields between reads: a read rewrites only the levels inside the file's pressure range (T, q and winds also get their ends filled). Where the surface pressure varies (ARM97), levels near the edge keep older values. The port replays the same sequence of reads, so it keeps the same values.
- The IOP time (`tsec`) is read into an integer, which truncates ARM97's fractional times. ARM97 step 37 starts exactly on an interval boundary and matches bitwise, which confirms truncation.
- Nudging relaxes toward the mean over the whole domain, summed column by column. On 4-column dumps of a 100-column run it cannot be reproduced, so those replays leave out the nudged fields; the all-column runs cover it.

### The physics suite (`src/eamxx_torch/suite.py`)

`Suite.step` runs the physics group as EAMxx does: `iop_forcing` (supplied by the single-column driver), then `zm` in the low-resolution configuration, then `tms`, `shoc`, `cld_fraction`, `spa`, `p3` (sub-cycled 12 times in the low-resolution configuration), then `rrtmgp`. Each process has its own time stamps and runs its pre- and post-condition checks in registration order (an invariant check counts as both).

| Check | Data | Result |
|---|---|---|
| One whole physics step, every process's output after its checks (`reference/scripts/suite_step_check.py`) | BOMEX and DYCOMS-RF01 at 5 steps × 4 ranks; ARM97 and CASS all-column runs at 5 steps each (radiation updates, nudging, Coriolis, daytime cloud). 50 steps | **Bitwise** |
| The same in the EAMxx-LR configuration: ZM, then 12 sub-cycles of the 5 `mac_aero_mic` processes (every sub-cycle compared), then RRTMGP; 62 process calls per step | ARM97 LR, steps 2, 31, 209 (ZM active; step 209 has 28 mm/day). The run nudges winds toward the domain mean and 8 columns were dumped, so these checks start after `iop_forcing` | **Bitwise**, 1.9M values per step |

The starting state is what HOMME handed to physics; everything after that (for example SHOC's `inv_qc_relvar` into P3, SPA's `nccn` into P3, P3's effective radii into RRTMGP, repaired `qv` from one process's checks into the next) is passed by the port.

### Single-column model vs EAMxx (`src/eamxx_torch/scm/driver.py`; `reference/eamxx_ext/scm_driver`)

The port's single-column model is compared with EAMxx's own physics on one column, run by the single-column reference driver (`scm_eamxx`, plan section 4): the same initial state, the same host choices (surface pressure held at its initial value, pressures as HOMME computes them, IOP omega; open-ocean fluxes from E3SM's coupler formula when the IOP file has no surface fluxes). Every output field (42, or 48 with ZM) is compared after every step (`reference/scripts/scm_compare.py`).

| Case | Configuration | Steps | Result |
|---|---|---|---|
| BOMEX | standard, 100 s, IOP fluxes | 216 (6 h) | **Bitwise at every step** |
| DYCOMS-RF01 | standard, daytime radiation | 216 (6 h) | **Bitwise at every step** |
| CASS | standard; nudging of T, q and winds, Coriolis | 324 (9 h) | **Bitwise at every step** |
| ARM97 | standard; time-varying 3D forcing, wind nudging | 1728 (48 h) | **Bitwise at every step** |
| RICO | standard; **ocean fluxes** | 864 (24 h) | **Bitwise at every step** |
| GATE III | standard; **ocean fluxes**, deep convection (7 mm/day) | 1728 (48 h) | **Bitwise at every step** |
| BOMEX | EAMxx-LR: ZM, 1800 s, 12 sub-cycles | 12 (6 h) | **Bitwise at every step** (ZM active every step) |
| ARM97 | EAMxx-LR | 240 (5 days) | **Bitwise at every step** (ZM active in 119 steps, up to 19 mm/day) |
| GATE III | EAMxx-LR, ocean fluxes | 240 (5 days) | **Bitwise at every step** (ZM active in 238 steps, up to 23 mm/day) |

The 12 validation runs of docs/validation.md also match bitwise at every output step (every step, or hourly for the long 100 s runs): both configurations, including CASS with E3SM's scmlib settings, an 8-day ARM97 window and 20 days of GATE III; 27,436 steps.

What the runs cover beyond the per-scheme replays: fields carried from step to step (TKE, P3's moments and memory, radiation between updates, ZM's previous-step state for DCAPE), the time-varying IOP data, nudging, Coriolis, the ocean fluxes, and EAMxx's per-step reset of accumulated fields. RICO and GATE III give plausible surface fluxes (RICO: 7 and 138 W/m², sensible and latent); these runs check agreement with EAMxx, not with observations.

**Torch math mode (the default).** Trajectories start 1 ulp apart and the difference grows, as in any moist-convective run. After 6 h of BOMEX: 0.047 K in T, 8% of the maximum in cloud water and TKE. EAMxx's own production build (FMA, packs of 16) drifts from its strict build by the same amount at every stage (0.047 K and 8% after 6 h; 8.2e-12 vs 9.0e-12 K after 72 steps). So in torch mode the port agrees with EAMxx as well as EAMxx agrees with itself across builds.

Also in the driver now: EAMxx resets its ACCUMULATED fields (surface precipitation masses, SHOC's wind perturbations) at the start of every step, and ZM's inputs that are computed later in the step (`thl_sec`, `cldfrac_tot`, `zm_t_prev`, `zm_q_prev`) start at zero, as in EAMxx.

**Gradients.** See docs/differentiability.md: gradients through the model are finite; masked-branch NaNs are removed with values unchanged.

### ZM (`src/eamxx_torch/zm/`)

Source: `physics/zm/impl/*_impl.hpp`, `zm_functions.hpp`, `eamxx_zm_process_interface.cpp`, E3SM 80fe0a71. Ported: the runtime options and `limcnv` from the hybrid grid, the saturation vapor pressure table, entropy and its inversion (Brent's method), the PBL temperature perturbation from SHOC's theta-l variance, dilute CAPE (launch level with ULL, entraining parcel, freezing, LCL, equilibrium level), DCAPE from the previous step's state, the trigger, cloud properties (fractional entrainment, updraft mass flux and thermodynamics, cloud top, condensation, rain production), downdrafts, the closure, the tendencies, rain evaporation and snow, MCSP, convective momentum transport with the kinetic-energy dissipation heating, detrainment of cloud water and number, and the checks.

| Check | Data | Result |
|---|---|---|
| Replay, DP-EAMxx-LR dumps (strict build, one rank, all 100 columns) | ARM97 at 14 steps over 5 days (first step without DCAPE; 0, 8, 25, 91 and 100 columns active; 28 mm/day; MCSP on and off; rain fully evaporated before the surface), BOMEX at 5 steps (all columns active), DYCOMS-RF01 at 2 steps (none active). 21 calls, 7.3M values | **Bitwise**, around `run_impl` and around the checks |
| Replay on column subsets (3 random 7-column subsets of 6 calls) | ARM97 | **Bitwise**: the per-rank values `ktm`, `kbm` do not change results |
| Conservation | all fixtures | Column water changes by minus the surface precipitation, and cp·T + Lv·qv − Li·qi + KE (liquid-water reference) is unchanged, to 1e-14 of the column totals |
| Torch math mode | all calls | Normwise error ≤ 1.7e-12 per field, except the surface rain at ARM97 step 80 (below) |

**Surface rain that fully evaporates is ill-conditioned in the reference.** At ARM97 step 80 all rain evaporates before reaching the surface; the reference's surface value is a residue of 1.5e-12 of the column's rain flux, and torch mode differs from it by 1e-12 of that flux.

Branch coverage (`reference/scripts/zm_branch_coverage.py`, 21 calls): active and inactive columns, CAPE without a DCAPE trigger, launch at the lowest level and at the ULL limit, tpert applied and zeroed, LCL found, LCL pressure below 600 mb, freezing, no negative-buoyancy crossing, cloud top by buoyancy or weak mass flux and by the MSE crossing, weak mass flux inside the plume, no updraft (`lambda_max = 0`), with and without downdrafts, downdraft evaporation, closure clipped at 0, the cloud-base flux limit, snow falling and melting, evaporation clipped, MCSP on, too shallow and shear too small, Brent's inverse-quadratic, secant, rejected-interpolation and tolerance steps (at most 9 iterations). **Never taken, checked by reading only:** the cloud top one level up after an MSE undershoot, no cloud top found, the detrainment level moved down, snow reaching the surface (summer cases), MCSP shear ≥ 200 m/s, the 2 K tpert limit, an exact root in Brent's method.

Not exercised (off in EAMxx's configuration): `clos_dyn_adj`, `no_deep_pbl`, `num_cin > 1`, `zm_microp`, `old_snow = false`, MCSP q/u/v coefficients; `zm_transport_tracer` is not called by EAMxx and not ported.

Tests: `tests/test_zm.py` (7 fixtures of a few columns, 0.9 MB; no input data needed).

### Ocean surface fluxes (`src/eamxx_torch/surface_flux.py`)

Source: `share/util/shr_flux_mod.F90`, `shr_flux_atmOcn`, scheme 0, on the path E3SM's coupler takes for EAMxx (explicit stress, no gustiness, no waves, 2 iterations, minimum wind 0.5 m/s). EAMxx has no flux scheme of its own.

| Check | Data | Result |
|---|---|---|
| Against E3SM's Fortran, compiled unmodified in a harness (`reference/scripts/flux_check.py`) | 3000 states: stable and unstable, z/L clipped at +10 and −10, zero temperature difference, zero and calm wind, ocean currents, a 0.5 m first level, dry air, masked points | **Bitwise** (13 outputs) |
| Torch math mode | same | Normwise ≤ 2.2e-16 |
| E3SM's own `-O` build vs `-O -ffp-contract=off` | same | Identical |

Never taken: the early stop of the 2-iteration loop (it needs u* unchanged to the last bit) and the 1e-3 floor in the stability shift. Tests: `tests/test_surface_flux.py`.

### RRTMGP (`src/eamxx_torch/rrtmgp/`, `orbit.py`, `trcmix.py`)

Source: `physics/rrtmgp/eamxx_rrtmgp_process_interface.cpp`, `eamxx_rrtmgp_interface.hpp`, `rrtmgp_utils.hpp`; the Kokkos code of the rte-rrtmgp submodule (gas optics, cloud optics, RTE solvers, flux reductions, optical-property arithmetic, `conv::Random`); `share/physics/eamxx_trcmix.cpp`; `share/util/shr_orb_mod.F90`. E3SM 80fe0a71. Ported: coefficient loading as RRTMGP does it (gas list reduction, flavors, minor absorbers), trace-gas profiles, orbit and averaged zenith angle, T at interfaces, cloud mass and effective-radius limits, cloud optics (lookup tables), McICA maximum-random subcolumns with EAMxx's random numbers, SW two-stream and LW no-scattering solvers, clear/clean-sky variants, heating rates, surface broadband fluxes, cloud cover, AeroCom cloud-top diagnostics, the steps without a radiation update, and the property check.

| Check | Data | Result |
|---|---|---|
| Replay, DP-EAMxx dumps | BOMEX, DYCOMS-RF01, CASS, ARM97 7.5 days (new runs dumped on radiation steps; ARM97 includes three local noons under cloud with ice and liquid), plus the earlier ARM97 6 h and 7.5 days, BOMEX, DYCOMS, CASS dumps. 260 calls (104 with a radiation update, 292 daytime columns), 1.80M values | **Bitwise**, around `run_impl` and around the property check |
| Replay, synthetic reference runs | rrtmgp alone on 8 columns: temperatures below 160 K and above 355 K (outside the gas-optics tables), ice radius above 180 µm and liquid radius below/above the tables, a top layer 1e-5 Pa thick, polar day, polar night, a night shorter than the 300 s averaging window, a window across local midnight, |lat| > 45. Two option sets: fixed orbit, fixed solar constant, extra clean/clear-sky calls; and orbit from the date, fixed zenith angle, no McICA, no aerosol. 8 calls, 136k values | **Bitwise** |
| Energy | all fixtures | Column heating equals the net radiative flux into the column (EAMxx's `heat_flux`) to roundoff |
| Torch math mode | all calls | Normwise error ≤ 1.7e-10 per field (see below) |
| vs production build (`release`) | 140 calls | `T_mid` relative ≤ 7e-14; SW fluxes normwise ≤ 4e-11; heating ≤ 2.6e-10 (FMA contraction, amplified as below) |

**Why the torch-mode error is larger here than for other schemes.** Two places amplify torch's 1-ulp differences from libm, in the reference's own arithmetic. The averaged zenith angle cancels near sunrise and sunset: at μ0 = 0.006, a 1-ulp difference in `sin` gives 3e-13 relative in μ0 and 1e-10 in the surface direct beam; with libm's μ0 that error is 1e-16. And heating rates are differences of nearly equal fluxes at adjacent levels.

Branch coverage (`reference/scripts/rrtmgp_branch_coverage.py`, 268 calls): every table clamp, both parts of the atmosphere, McICA cloudy and clear subcolumns (including cloud fraction without optical depth), delta-scaling on and off, the SW `k` floor, the LW thin-layer series, day and night columns, and every branch of the averaged zenith angle, including the night-shorter-than-window case with sunlight on both sides. **Never taken, checked by reading only:** the SW guard for `|1 − (k μ0)²| < ε`, which needs an exact coincidence of optical properties and sun angle.

Not ported (EAMxx does not use them): Pade cloud optics (EAMxx loads the lookup tables), overlap option 0, multiple LW quadrature angles, `col_dry` and `latitude` inputs to gas optics, `top_at_1 = false`.

The ARM97 7.5-day reference run finished all 6480 steps and EAMxx finalized; MPI then reported rank 3 exiting improperly at shutdown. The dumps are complete and readable.

Tests: `tests/test_rrtmgp.py` (5 fixtures, 1.2 MB; skipped when the coefficient files are absent).

### P3 (`src/eamxx_torch/p3/`)

Source: `physics/p3/impl/*_impl.hpp`, `eamxx_p3_process_interface.{hpp,cpp}` and `eamxx_p3_run.cpp`, E3SM 80fe0a71. Ported: the lookup tables (read from the same files the reference reads), pre-processing (wet-to-dry conversion, cloud fractions), `p3_main` parts 1-3, all process rates and conservation limiters, the three sedimentation loops with their Courant substepping, homogeneous freezing, post-processing, and the property checks.

| Check | Data | Result |
|---|---|---|
| Replay, DP-EAMxx dumps | BOMEX, DYCOMS-RF01, ARM97 6 h and 7.5 days, CASS. 160 calls, 1.80M output values | **Bitwise**, around `run_impl` and around the property checks |
| Replay after the masked-branch fix (23 Sep), every strict dump | all of the above plus the radiation-step runs and 39 sub-step calls of the EAMxx-LR configuration (150 s). 212 calls, 5.6M values | **Bitwise** |
| Replay, synthetic reference run | p3 alone on 8 columns built to reach rare states: supercooled liquid, droplet number below `nsmall`, heavy subliming ice, rain evaporating into near-saturated air, very high cloud water. `extra_p3_diags` on, so the 15 extra rate diagnostics are compared too. 3 calls, 114k values | **Bitwise** |
| Conservation (EAMxx's own definitions) | all calls | Column water changes by the surface precipitation and total energy by the latent heat it carries, to 3.5e-16 and 4.2e-16 of the column totals |
| Torch math mode | all calls | Normwise error ≤ 5.5e-14 per field |
| vs production build (`release`) | 140 calls | `T_mid` relative ≤ 4.4e-16; tracers ≤ 7.7e-13 relative (absolute ≤ 2e-17) |
| Sedimentation substeps | all calls | At most 1 (cloud), 36 (rain), 11 (ice) against a cap of 400; the cap is never reached |

**Two differences come from the reference's compiler, not its source.** clang rewrites `pow(x, 2.0)` into `x*x` and `pow(10.0, x)` into `__exp10(x)` (LLVM SimplifyLibCalls). Both differ from libm's `pow` by 1 ulp for a small fraction of arguments, and both show up in P3 (cloud fall speed, ice-rain collection). The port follows the build. This also uncovered a flaw in the port's own test backend: numpy's `power` is not libm's `pow` for an exponent of 2, so the libm backend now calls libm directly. See plan, sections 4 and 10.

Branch coverage (`reference/scripts/p3_branch_coverage.py`, 163 calls): every process rate, limiter and lookup path is exercised except the liquid-supersaturation limiter (`prevent_liq_supersaturation`), which no state I could build reaches — the ice limiter ahead of it keeps vapour below liquid saturation. That one is checked by reading only.

Not exercised (off in this configuration, so not validated): `use_hetfrz_classnuc` (classical nucleation), `use_separate_ice_liq_frac`, `do_predict_nc=false`, `do_prescribed_ccn=false`.

Tests: `tests/test_p3.py` (6 fixtures; skipped when the reference input data is absent, since P3 needs its lookup tables).

### SHOC (`src/eamxx_torch/shoc/`)

Source: `physics/shoc/impl/*_impl.hpp` and `eamxx_shoc_process_interface.{hpp,cpp}`, E3SM 80fe0a71. Ported: preprocess, TMS coupling, `shoc_main` (all subroutines), postprocess, post-condition checks. Also ported with it: saturation (`MurphyKoop_svp`, `polysvp1`, `qv_sat_*`), SHOC constants, `npbl` from the L128 hybrid coefficients (56).

| Check | Data | Result |
|---|---|---|
| Replay, DP-EAMxx dumps | BOMEX, DYCOMS-RF01, ARM97 6 h, ARM97 7.5 days, CASS 9 h. 160 calls, 4 columns each, 128 levels. 2.79M output values. | **Bitwise**, every output field, around `run_impl` and around the property checks |
| Replay, synthetic reference run | `tms`+`shoc` on 8 ARM97 columns with surface stress, land fraction 0/0.5/1, a 175 K surface layer, 400 W/m² heat flux. 3 steps. | **Bitwise** (TMS too) |
| Budgets | all dumps | Column water changes by `surf_evap·dt`, total energy by `(SH + (Lv+Li)·E)·dt`, other tracers conserved. Residuals ≤ 1.4e-15 of the column totals, i.e. roundoff. |
| Torch math mode | same 140 strict calls | Normwise error ≤ 2.2e-12 per field, except `inv_qc_relvar` (see below) |
| vs production builds (`release`: FMA, packs of 16; `strict16`) | 140 and 80 calls | Same picture: `T_mid` relative ≤ 3e-15, other fields roundoff; `inv_qc_relvar` flips as below |

**`inv_qc_relvar` is ill-conditioned in the reference itself.** In fully cloudy cells SHOC's liquid variance `qc2 = a·(s1·ql1 + …) − ql²` cancels to roundoff (`qc2/qc² ≈ 1e-16`). `inv_qc_relvar` is then 10 (clip of `qc²/qc2`) or 1 (if `qc2` is exactly 0). One ulp anywhere upstream flips it. The production build differs from the strict build this way too. P3 reads this field, so P3 comparisons against non-strict builds will show it.

Branch coverage (`reference/scripts/shoc_branch_coverage.py`, 163 calls):
- Covered: both signs of the surface buoyancy flux, PBL height found and not found, min length and dx cap on the mixing length, TKE at its floor, isotropy clip, cold-surface fallback, stress small and not small, TMS drag, `w3` clipping, PDF skewness clip, partial cloud, `qn ≤ 0` reset, `inv_qc_relvar` at both clips.
- **Never taken, checked by reading only:** mixing length at 20 km, TKE at 50 m²/s², plume temperature floor at 100 K, energy fixer with no active level. Each is a one-line clamp or default.

Not ported, off in the standard configuration: 3D turbulence (`do_3d_turbulence_shoc`), `check_flux_state_consistency`, column conservation checks. See plan, section 10.

Tests: `tests/test_shoc.py` (8 fixtures: 6 DP calls, 2 synthetic). Full replay: `python reference/scripts/replay_check.py shoc reference/runs/dumps_strict_* reference/runs/standalone_strict_*`.

### SPA (`src/eamxx_torch/spa.py`, `data_interpolation.py`, `timestamp.py`)

| Check | Data | Result |
|---|---|---|
| Replay | 268 calls: BOMEX, DYCOMS-RF01, ARM97 6 h and 7.5 days, CASS, ARM97 26 days. 8.1M values | **Bitwise**, both math modes, including property checks |
| Monthly interval switch | ARM97 26 days, steps 21616–21621 straddle 15 July 00Z: data slices go from (5,6) to (6,7) | **Bitwise** on both sides |
| vs release build | same | relative ≤ 2.6e-14 |

The repair path of its checks never fires in real data; it is covered by the property-check harness.

### TMS (`src/eamxx_torch/tms.py`)

| Check | Data | Result |
|---|---|---|
| `tms_replay` harness (EAMxx's TMS functions on synthetic land columns) | 180 columns, 48 with nonzero drag | libm mode **bitwise**; torch mode rtol 1e-11 (1 ulp in `exner` amplified by a cancellation in the Richardson number) |
| Standalone `tms`+`shoc` reference run | 8 columns × 3 steps, 18 with nonzero drag | **Bitwise** |
| DP dumps (drag is always 0: no coupler, `landfrac`=0) | 160 calls | **Bitwise** |

### Property checks (`src/eamxx_torch/property_checks.py`)

`FieldWithinIntervalCheck` and `FieldLowerBoundCheck`, including repair: 27 synthetic cases run through EAMxx's own check class: **bitwise** (including −0 → +0 and NaN handling).

### `cld_fraction` (`src/eamxx_torch/cld_fraction.py`)

Bitwise on 160 strict-build DP calls (all five cases), on 140 release-build calls, and on synthetic edge cases run through `CldFractionFunctions::main` (thresholds ±1 ulp, NaN, ±inf, −0). Batched `(ncol, nmember, nlev)` and `torch.compile(fullgraph=True)` pass. No state change, so no budget.

Tests: `pytest tests/` → **201 pass** (float64 still bitwise; 6 new float32/MPS tests in `tests/test_precision.py`).

## Does not match the reference

Nothing so far.

## Checked about the reference itself

| Check | Result |
|---|---|
| EAMxx `field` and `field_header` unit tests (after patch 0003) | Pass |
| DP-EAMxx BOMEX, DYCOMS-RF01, ARM97 (6 h and 7.5 days, 26 days), CASS (9 h) | Complete |
| DP-EAMxx-LR (ZM) BOMEX, DYCOMS-RF01 6 h; ARM97 26 days | Complete. ZM active in 1125 of ARM97's 1248 steps |
| DP-EAMxx-LR CASS | **Aborts at step 2 in EAMxx**: ZM's post-condition `ni ≥ 0` (not repairable) fails on −4e-25, a value ZM received, not one it made |
| Dump hook on vs off | Bitwise identical |
| 1 vs 4 MPI ranks | Max difference 8.7e-19 (output averaging order) |
| TWP-ICE (TWP06) | **Fails at step 1**: no surface temperature in its IOP file. Use GATE III. |
| CASS | IOP data cover only 9.03 h after the case start; a 12 h run stops at step 325. Runs use 9 h. |
| DP surface stress | Always 0 in DP runs (no coupler). SHOC's stress paths are covered by the standalone run instead. |
| McICA seeds in a single column | Frozen: the lowest pressures never change there (surface pressure held, hybrid thicknesses), so EAMxx's pressure-digit seeds are the same on every radiation call (checked in BOMEX and ARM97). DP and global runs vary them. See docs/differentiability.md |
| SHOC energy fixer's `shoctop` | Usually level 0–16: upper levels hold TKE one ulp above the floor, so EAMxx spreads the energy residual over nearly the whole column |
| Single-column reference driver, strict vs production build (BOMEX 6 h) | The builds drift apart from step 1 (FMA contraction, packs of 16): 0.047 K in T after 6 h |
| EAMxx single-precision build (`strict-sp`) | Builds P3, SHOC, cld_fraction, SPA only; no HOMME. Its P3 table writer `p3_tables_setup` writes zeros (host mirror never filled); our `p3_tables_write` writes correct `.dat4` files (its double output matches the official `.dat8` to 4e-15) |
| RRTMGP minor-gas scaling in float | `scaling * col_gas * vmr_fact * dry_fact` overflows float (C++ and Fortran kernels alike). Harmless in EAMxx, which builds RRTMGP only in double |
| P3 immersion freezing in float | Overflows below about 195 K with cloud liquid present, then inf·0 = NaN in cloud-water conservation. `CONS6` is float in the float build, so EAMxx float computes the same |

## Untested

- The ZM and ocean-flux branches listed above as checked by reading only.
- GABLS1 (a cold, stable surface over land or ice: E3SM would use the land or sea-ice flux, which is not ported) and RCE.
- The single-column model's initialization against EAMxx's own (EAMxx builds its IOP initial state only on the dycore's grid; the reference starts from the port's state).
- Float64 on GPU, and CUDA at all: not possible on this Mac (MPS has no float64).
- Float32 RRTMGP, ZM, TMS and IOP forcing against a reference: EAMxx has none.
