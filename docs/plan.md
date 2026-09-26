# Plan: PyTorch port of the E3SM EAMxx physics suite

Written 18 September 2026 at the end of Phase 0. Nothing is ported yet.

## Phase 0 verdict

- **The reference runs.** Unmodified EAMxx physics runs on this Mac in a standalone build. DP-EAMxx BOMEX ran its full 6 hours. It holds the prescribed fluxes and grows a trade-cumulus layer.
- **The reference can be instrumented.** A switchable hook saves every input and output of every scheme, on chosen columns and steps. With the hook on, results are bitwise identical to runs without it.
- **All four scoping decisions were made on 18 September.** See "Decisions for you" at the end.

## 1. What EAMxx runs in its standard configuration

Source: `components/eamxx/cime_config/namelist_defaults_eamxx.xml` and `config_compsets.xml` at the pinned commit.

EAMxx has three relevant configurations. All three use the same 128-level grid.

| Configuration | Compsets | Physics process order | Physics step |
|---|---|---|---|
| **SCREAMv1, standard** (km-scale, production) | `F2010-SCREAMv1` and relatives | `mac_aero_mic = [tms, shoc, cld_fraction, spa, p3]`, then `rrtmgp` | 100 s at ne1024 (3.25 km). No sub-cycling. Radiation every 3 steps. |
| **DP-EAMxx** (doubly periodic, for IOP cases) | `FIOP-SCREAMv1-DP`, `FRCE-SCREAMv1-DP` | `iop_forcing`, then the standard list | 100 s, 3.3 km cells. Radiation every 3 steps. |
| **EAMxx-LR with ZM** (low resolution, "experimental, under development") | `F2010xx-ZM`, `F2010xx-ZM-CICE` | `zm`, then the standard list | 1800 s at ne30. `mac_aero_mic` sub-cycled 12 times (150 s). Radiation every 2 steps (3600 s). |

`tms` is on only for pg2 physics grids, which every production grid uses.

What each process is:

| Process | What it does | Upstream origin | Key default settings |
|---|---|---|---|
| `shoc` | Turbulence, cloud macrophysics and shallow convection in one scheme. Assumed double-Gaussian PDF, prognostic TKE. | SHOC, Bogenschutz and Krueger (2013), C++ translation in E3SM | `lambda_low=0.001, lambda_high=0.08, lambda_slope=2.65, lambda_thresh=0.02, thl2tune=qw2tune=qwthl2tune=w2tune=1, length_fac=0.5, c_diag_3rd_mom=7, coeff_kh=coeff_km=0.1, shoc_1p5tke=false`. One internal SHOC step per call. |
| `cld_fraction` | Ice cloud fraction from `qi`. Total = max(liquid, ice). | E3SM | `ice_cloud_threshold=1e-12`, `ice_cloud_for_analysis_threshold=1e-5` |
| `spa` | Simple Prescribed Aerosols. Time and vertical interpolation of a monthly climatology of CCN and aerosol optics. | E3SM | yearly-periodic interpolation |
| `p3` | Microphysics. Predicted Particle Properties, one ice category, prognostic `nc` with prescribed CCN. Lookup tables. | Morrison and Milbrandt P3, E3SM C++ | Ice table v4.1.1. `autoconversion_prefactor=1350, accretion_prefactor=67, spa_ccn_to_nc_factor=2000, max_total_ni=740e3`, full list in `reference/scripts/make_dp_case.py`. Sedimentation uses CFL sub-steps. |
| `rrtmgp` | Radiation. RTE+RRTMGP C++ (Kokkos). SW 112 and LW 128 g-points. Cloud optics from tables. McICA max-random overlap. SW two-stream. LW no-scattering, 1 Gaussian angle. | AER/CU RTE+RRTMGP, E3SM fork | `do_subcol_sampling=true`, `do_aerosol_rad=true`, gases h2o co2 o3 n2o co ch4 o2 n2, F2010 concentrations, orbit year 2010 |
| `tms` | Turbulent mountain stress (surface drag from sub-grid orography). Zero over ocean. | CAM | none |
| `iop_forcing` | DP/SCM only. Large-scale advective forcing, subsidence, optional nudging and Coriolis. | E3SM | per case |
| `zm` | Deep convection, LR only. Zhang–McFarlane with dilute CAPE, DCAPE trigger, ULL, MCSP. | CAM ZM, E3SM C++ | `trig_dcape=true, trig_ull=true, tau=3600, alfa=0.14, ke=2.5e-6, dmpdz=-0.7e-3, tpert_fac=2, tiedke_add=0.8, c0=0.002, num_cin=1, mcsp_enabled=true, mcsp_t_coeff=0.3` |

Also in the EAMxx tree but in no default process list: `gw` (gravity-wave drag), `cosp` (diagnostics), `nudging`, `spc`, `mam4xx` (prognostic aerosols, a separate compset), water tracers. None is part of the standard suite.

### Shallow and deep convection, stated plainly

- Shallow convection is inside SHOC. There is no separate shallow scheme.
- In the standard km-scale suite, deep convection is resolved. There is no deep scheme.
- ZM exists only in the experimental low-resolution configuration.

### Surface fluxes are not part of EAMxx

EAMxx computes no surface fluxes. It receives them from E3SM's coupler (ocean) or land model.
- Ocean fluxes come from `share/util/shr_flux_mod.F90`, `shr_flux_atmOcn`. E3SM's default is `ocn_surface_flux_scheme=0`. The scmlib DP scripts switch to scheme 2 (University of Arizona).
- Land fluxes come from ELM. Porting ELM is out of scope.

IOP files decide which cases need a flux scheme:

| Case | IOP surface data | Needs an interactive flux scheme? |
|---|---|---|
| BOMEX | shflx, lhflx, Tg | No |
| DYCOMS-II RF01 | shflx, lhflx, Tg | No |
| ARM SGP (ARM97) | shflx, lhflx, Tg (time series) | No |
| TWP-ICE (TWP06) | shflx, lhflx | No |
| RICO | Tg only | **Yes (ocean)** |
| GATE III | Tg only | **Yes (ocean)** |
| GABLS1 | Tg only (cooling surface) | **Yes** |
| RCE | fixed SST | **Yes (ocean)** |

TWP-ICE's IOP file has fluxes but **no surface temperature**. CIME takes the radiative surface temperature from the land model. The standalone reference therefore stops at step 1 with NaN `surf_lw_flux_up` (checked 18 Sep). GATE III (ocean, Tg given) is the deep-convection case to use once the ocean flux scheme is ported.

Momentum flux (`surf_mom_flux`) is never in an IOP file. In the standalone reference it stays at its initial value, zero. A CIME run would get it from the coupler.

## 2. Source code and pinned commits

| Repository | Commit | Date |
|---|---|---|
| E3SM-Project/E3SM (master) | `80fe0a71cfabd0091eaacecab7a1ce991563a6d7` | 2026-09-18 |
| E3SM-Project/rte-rrtmgp (submodule) | `2a887accea76f9e24369dda6abd2dd3d70b7403f` | |
| E3SM-Project/EKAT (submodule) | `3a1485aecfed7a036e4fa30ce845498fd37975e2` | |
| Kokkos (in EKAT) | `23745d8494203711a6090d0a201c3ae3bf57853b` | |
| E3SM-Project/scorpio | `8d794568f71d1f60b4afb5bc70ab6ff5939b1da8` | |
| ESMCI/cime | `bacb0a2c0601ca2bc0d08520d9eea12af55fe52f` | |
| E3SM-Project/scmlib (case scripts) | `2dc3f1073a5d03b5f32617e79d68fb20a63cfb43` | 2026-09-16 |

`reference/scripts/setup_reference.sh` recreates this tree and applies our patches.

## 3. Licenses

| Component | License | Translation and redistribution |
|---|---|---|
| EAMxx physics (SHOC, P3, cld_fraction, SPA, TMS, IOP forcing, ZM) and share code | E3SM 3-clause BSD | Allowed. Keep the notice. |
| ZM and TMS origins | CAM/CESM, public domain per E3SM's notice | Allowed |
| RTE+RRTMGP C++ | 3-clause BSD (file in the rrtmgp source) | Allowed. Keep the notice. |
| RRTMGP and P3 data tables | E3SM input data, distributed openly | We load them at run time. We do not commit them. |
| EKAT | Boost 1.0 | Allowed. Only small helpers, if any, will be translated. |
| `shr_flux_mod.F90` (ported: `surface_flux.py`) | E3SM/CESM share code, BSD | Allowed |

**Flag:** E3SM's top-level `LICENSE` table lists rrtmgp as "AER, non-commercial". The license file shipped in the rrtmgp source is plain 3-clause BSD. We follow the shipped file. The mismatch should be confirmed with E3SM before any public release.

No scheme in the suite blocks translation. CLUBB (unclear license) is not used by EAMxx.

This project is **BSD-3-Clause** (`LICENSE`). Translated notices are in `THIRD_PARTY_NOTICES.md`.

## 4. The reference

### How it is built

- Standalone EAMxx build. No CIME, no E3SM coupler. This is the build EAMxx developers use for tests (`components/eamxx/AGENTS.md`).
- Toolchain: an isolated conda-forge env `eamxx-ref` (clang 21, gfortran, OpenMPI, netCDF). Recipe in `reference/environment-ref.yml`. Nothing was installed system-wide.
- Release, double precision, pack size 16 (the CPU production default), Kokkos Serial backend, MPI for parallelism. HOMME dycore on. MAM off.
- Our driver target `reference/eamxx_ext/dp_case_l128` builds E3SM's own DP test driver (`homme_shoc_cld_spa_p3_rrtmgp_pg2_dp.cpp`) against a 128-level HOMME library. It links ZM too.
- Build time: about 5 minutes on 14 cores.

Patches to the E3SM clone (in `reference/patches/`, none touches physics code):

1. `0001` adds a CMake hook that builds our out-of-tree driver target.
2. `0002` adds the dump hook (`share/atm_process/ref_dump.hpp`). It is off unless `EAMXX_REF_DUMP_DIR` is set.
3. `0003` fixes a portability bug in EAMxx field bookkeeping (`eamxx_family_tracking.hpp`). In a `const` method, `weak_from_this()` returns `weak_ptr<const T>`. libc++ builds that through `lock()`, so an expired pointer becomes empty and a destructor check aborts. GCC's libstdc++ keeps it. EAMxx's own `field_header` unit test aborted here before the fix and passes after it. The fix changes no numbers.

Two macOS build settings, both outside physics:
- `-D_LIBCPP_DISABLE_AVAILABILITY`, because conda's libc++ hides `std::from_chars(double)` behind the macOS deployment target.
- EKAT refuses MPI on Apple, but only on re-configure. The configure script clears that cache entry.

### How to run it

```
bash reference/scripts/setup_reference.sh     # clone at pinned commits, apply patches
source reference/scripts/env.sh
bash reference/scripts/fetch_inputdata.sh     # 546 MB, E3SM public input data server
bash reference/scripts/configure_eamxx.sh
make -C reference/build/eamxx-release -j14 dp_eamxx_l128
bash reference/scripts/run_dp_case.sh bomex 1  # standard DP-EAMxx
python reference/scripts/make_dp_case.py bomex --lr   # EAMxx-LR + ZM variant
```

`make_dp_case.py` reproduces what CIME's EAMxx namelist generator writes for `FIOP-SCREAMv1-DP` on ne30pg2, plus the per-case settings from the scmlib scripts.

### Unit-level replay harnesses

Some inputs never occur in model dumps: values exactly at a threshold, NaN, infinity. For those, `reference/eamxx_ext/unit_replay/` builds small executables that call the EAMxx physics function itself on inputs read from a file. `reference/scripts/<scheme>_edge_cases.py` generates the inputs, runs the harness and saves a fixture. First use: `cld_fraction_replay` (calls `CldFractionFunctions::main`).

### Standalone physics runs (added 19 Sep)

DP cases never reach some regimes (no surface stress, no land, no cold surface). `reference/eamxx_ext/physics_standalone` builds EAMxx's generic atmosphere-driver program linked with every physics library we port. It runs any process list on a point grid from an IC file, with no dycore. `reference/scripts/make_standalone_case.py PRESET OUT` builds the IC from dumped DP columns plus overrides, and the input.yaml. The dump hook works as in DP runs, and `replay_check.py` replays these runs too.

- Preset `shoc_tms`: `tms`+`shoc` on 8 ARM97 columns with surface stress (including |τ| < 1e-12 and zero), land fraction 0/0.5/1 (`sgh30` = 300 m), a 175 K surface layer, and 400 W/m² heating.
- Preset `p3`: `p3` alone on 8 columns with supercooled liquid (homogeneous freezing), droplet number below `nsmall`, heavy subliming ice, rain evaporating into near-saturated air, and very high cloud water (the conservation limiters). `extra_p3_diags` is on, so the extra rate diagnostics are compared too.
- Point-grid facts: the grid must be named `physics_pg2` for TMS; `phis` and `sgh30` are YAML constants; vector fields are read per component (`surf_mom_flux_U`, `_V`); the IC file must be CDF5; the cell area is 4π/ncol, so SHOC's dx cap never binds there (the DP dumps cover it).
- SHOC resets `tke`, `sgs_buoy_flux`, `eddy_diff_mom`, `cldfrac_liq` and the wind perturbations on an initial run, so later steps are the interesting ones.

### Dumping scheme inputs and outputs

```
EAMXX_REF_DUMP_DIR=$PWD/dump EAMXX_REF_DUMP_STEPS=1,2,108,216 EAMXX_REF_DUMP_NCOLS=4 \
EAMXX_REF_DUMP_PROCS=shoc,p3   mpirun -n 1 .../dp_eamxx_l128
```

- Each call writes `<proc>/s<step>_c<subcycle>_r<rank>_{pre,post}.{bin,json}`.
- Saved: every required, computed and internal field, plus every member of the process's field groups (for example the tracers SHOC mixes). Also dt and time stamps.
- `reference/scripts/read_dump.py` reads them with numpy only.
- Scheme intermediates (inside `shoc_main` and so on) need extra, per-scheme instrumentation. It is added when each scheme is ported.
- `reference/scripts/make_dumps.sh CASE HOURS STEPS [NP] [NCOLS] [TAG]` runs a case on a 5×5-element, 50 km domain and dumps every process at the given steps. Domain size does not affect per-scheme replay.
- `reference/scripts/extract_fixtures.py` turns dumps into compact `.npz` fixtures for `tests/data/`.
- `reference/scripts/replay_check.py PROC DUMP_DIRS...` replays every dumped call through the port and prints the worst differences.

### Results so far

| Run | Result |
|---|---|
| DP-EAMxx BOMEX, 5×5 elements, 50 km, 100 s, 6 h, 1 core | Completed in 130 s. Fluxes 9.45 and 153.0 W/m² as prescribed. Cloud near 1.3 km, max layer cloud fraction 0.56. Boundary layer about 900 m. LWP grows to 117 g/m². Precipitation 0.24 mm/day in hour 6. |
| Same, 1 h, dump on vs off | Bitwise identical (0.0 difference in all 71 output variables) |
| Same, 1 h, 1 vs 4 MPI ranks | Max difference 8.7e-19. Comes from summation order in horizontal-average output. |
| DP-EAMxx-LR BOMEX with ZM: 5×5 elements of 333 km, 1800 s, 12 sub-cycles, 6 h, 4 cores | Completed. ZM active. Precipitation about 1 mm/day. |

These are "the reference runs" checks, not scientific validation. BOMEX LES gives much lower cloud fraction and LWP. We match the reference, not LES, during porting.

EAMxx itself repairs `qv` below 1e-13 at the model top before P3, every step. That repair is part of the suite and will be ported.

### Known differences between our reference setup and a CIME DP-EAMxx run

1. No coupler. Surface albedos are 0, `surf_mom_flux` is 0, and `landfrac` is 0 (it is a coupler import, even at ARM SGP). TMS therefore always returns zero drag in model runs; its non-zero path is validated with the `tms_replay` harness. Fluxes come from the IOP file when `iop_srf_prop` is on. BOMEX starts at local night and runs 6 h, so SW is zero anyway.
2. SPA reads the 177 MB ne4 aerosol file, nearest column. CIME would read the 4.4 GB ne30pg2 file. This changes aerosol input values, not code paths.
3. `sc_export` is omitted. It only sends fields to the coupler.
4. The scmlib BOMEX script sets `constant_zenith_deg=180` to turn off SW. That is a coupler setting. EAMxx's radiation still computes its own zenith angle, in CIME runs too.

None of these affects per-scheme replay. The port gets the same inputs as the reference.

### Validation reference builds (added 19 Sep)

Two build settings make exact comparison impossible, so the per-scheme reference is a second, stricter build of the same code:

| Build | FP contraction | Pack size | Role |
|---|---|---|---|
| `strict` | off (`-ffp-contract=off`, EAMxx's `SCREAM_FPMODEL=strict`) | 1 (EAMxx's GPU layout) | **Primary per-scheme reference. Target: bitwise.** |
| `release` | fast (compiler may fuse `a*b+c` into FMA) | 16 (CPU default) | Production CPU build. Compared at a stated roundoff tolerance. |
| `strict16` | off | 16 | Separates the two effects when needed. |

- With FMA allowed, clang fuses multiply-adds wherever it likes. PyTorch does not. Every arithmetic scheme then differs by about 1 ulp per fused operation.
- With packs of 16, EAMxx sums over levels in a pack-dependent order (e.g. `ColumnOps::column_scan`). With pack size 1 the sums are plain sequential. `torch.cumsum` is sequential on CPU (checked bitwise to 100,000 elements).
- `configure_eamxx.sh [release|strict|strict16]` builds each. `make_dumps.sh` uses `strict` by default (`REF_VARIANT` selects).

### PyTorch pitfalls found and handled (19–23 Sep)

1. **`python_scalar / tensor` is not IEEE division in PyTorch.** It is computed as `reciprocal(tensor) * scalar` and differs in the last bit for about 25% of inputs (measured on 10^6 values). All other scalar/tensor operators tested are exact. Port code uses `eamxx_torch.ekat.rdiv(a, t)`. `tests/conftest.py` makes the operator raise during tests.
2. **The compiler rewrites some `pow` calls (found 22 Sep).** clang turns `pow(x, 2.0)` into `x*x` and, on Darwin, `pow(10.0, x)` into `__exp10(x)` (LLVM SimplifyLibCalls). Each differs from libm's `pow` by 1 ulp for a small fraction of arguments (0.13% for the square). The port matches the build: it multiplies for the square and calls `__exp10` through `mathfn.exp10`. Related: numpy's `power` is *not* libm's `pow` for an exponent of 2 (numpy multiplies), so `mathfn`'s libm backend calls libm through ctypes for `pow`, as it already did for `tanh`, `erf` and `tgamma`. Every other numpy function used (`exp`, `log`, `log10`, `expm1`, `log1p`, `cbrt`, `sqrt`) matches libm exactly over the ranges tested.
3. **Transcendental functions differ from libm by about 1 ulp.** `torch.pow` differs from the system `pow` for 1.4% of values. numpy's `power`, `exp`, `log`, `log10`, `expm1`, `log1p`, `cbrt`, `sin`, `cos` match libm exactly; numpy's `tanh` does not. Port code calls `eamxx_torch.mathfn`. Its default backend is torch. A test-only `libm` backend calls libm (via numpy or ctypes). **Each scheme is replayed in both modes: libm must be bitwise, torch must agree within a stated ulp-based tolerance.**

4. **Thread oversubscription (found 22 Sep).** With the machine busy, torch's intra-op thread pool makes small calls wait on scheduling: an RRTMGP replay that needs 5 s of CPU took 79 minutes of wall time with 14 threads. `replay_check.py` runs single-threaded. This changes timing only, not results.
5. **`torch.pow` depends on the batch (found 23 Sep).** torch evaluates `pow` with a vectorized kernel on full SIMD blocks and a scalar loop on the remainder; the two differ by 1 ulp for some arguments (15 of 1000 random values). So in torch math mode a column's result can change at roundoff with the number of columns it runs with. `exp`, `log` and `log10` showed no such dependence. With the libm backend, columns are bitwise independent (tested for ZM).

### Check dumps (added 19 Sep)

EAMxx runs pre-condition checks (which may repair inputs) before `run_impl`, and post-condition checks (which may repair outputs) after it. The dump hook now also saves `s<step>_c-1_r<rank>_{prechk,postchk}` around the checks, so a scheme's port can be compared including its repairs.

### EAMxx-LR dumps (added 23 Sep)

`LR=1 bash reference/scripts/make_dumps.sh CASE HOURS STEPS NP NCOLS TAG` runs the ZM configuration (`make_dp_case.py --lr`: 1800 s, 12 `mac_aero_mic` sub-cycles, radiation every 2 steps, 5×5 elements of 333 km); `DUMP_PROCS` limits the dumped processes. ZM is dumped on one MPI rank with all 100 columns. Members of a sub-cycled group count their own sub-steps: model step S is member sub-steps 12(S−1)+1 … 12S.

### Ocean flux harness (added 23 Sep)

EAMxx has no surface flux scheme; E3SM's coupler computes ocean fluxes with `share/util/shr_flux_mod.F90`. `reference/scripts/build_flux_harness.sh` compiles that file unmodified, with E3SM's `shr_kind_mod`, `shr_const_mod` and `water_isotopes`, and stubs for logging, abort and the wave-coupling flag (`reference/eamxx_ext/flux_harness/`). The harness calls `shr_flux_atmOcn` as `seq_flux_mct` does for an EAMxx atmosphere: scheme 0, explicit stress, no gustiness from the atmosphere (SCREAM), at most 2 iterations, minimum wind 0.5 m/s. Two builds: `strict` (`-O -ffp-contract=off`, the bitwise target) and `release` (`-O`, E3SM's gnu flag).

### Single-column reference driver (added 23 Sep; decision 4)

`reference/eamxx_ext/scm_driver` builds `scm_eamxx`: EAMxx's atmosphere driver on a one-column point grid with no dycore, process list `[sc_import, scm_host, (scm_sfc_flux,) physics]`, where physics is the DP-EAMxx list (or the LR list with ZM). Our code is only the host side:
- `scm_host` does HOMME's job: layer thicknesses from the hybrid grid and the surface pressure (held at its initial value), pressures by `HommeDynamics::update_pressure`'s arithmetic (EAMxx's `ColumnOps`), omega from the IOP file.
- `scm_sfc_flux` (cases without IOP surface fluxes) gives open-ocean fluxes as E3SM's coupler would: the lowest-level quantities `SurfaceCouplingExporter` computes, E3SM's `shr_flux_atmOcn` (same build as the harness, linked through a C bridge), the coupler merge with ocean fraction 1 and the importer's sign conventions, at the IOP surface temperature `Tg`.
- The planar cell size SHOC needs is set as the grid's `dx_short`, as HOMME's DP grids manager does (3.333… km for the 50 km DP domains, 111 km for LR).

`reference/scripts/make_scm_case.py CASE [--lr]` writes the run directory: the physics settings of `make_dp_case.py`, and the port's single-column initial state as the IC file. EAMxx's own initialization from the IOP file (`set_fields_from_iop_data`) works only on the dycore's GLL grid, so the reference starts from the same state rather than building it; the comparison tests the time stepping. `reference/scripts/scm_compare.py RUN_DIR` runs the port alongside and compares every output field after every step.

### Validation against observations and LES (added 24 Sep)

`reference/scripts/validation_criteria.py` holds the pass criteria (written before the runs; sources in docs/validation.md). `reference/scripts/validate_obs.py run RUN_DIR` runs the port on a `make_scm_case.py` directory, records the diagnostics the criteria need and compares with EAMxx's output in the same directory; `score` applies the criteria; `dp` gives the same quantities from a DP-EAMxx run for context. `make_scm_case.py --every N` writes EAMxx's output every N steps for long runs. New case keys: `cass_scmlib` (E3SM's scmlib CASS settings) and `arm97_active` (the 8-day ARM97 window of Bogenschutz et al. 2023). `reference/scripts/plot_validation.py` draws `docs/figures/validation.png` (needs matplotlib, which the project environment does not include).

## 5. Vertical grid and time step

- 128 levels, `vertical_coordinates_L128_20220927.nc`. Hybrid sigma-pressure.
- Top interface 2.26 hPa (about 40 km). Lowest layer about 25 m thick.
- 24 layers below 1 km. 17 layers in 1–3 km. Spacing 216–289 m in 3–10 km.
- EAMxx indexes top to bottom, like our dycore contract. **No flip is needed.**
- Time step depends on the configuration choice (decision 1):
  - Standard/DP: 100 s, no sub-cycling, radiation every 3 steps.
  - LR with ZM: 1800 s, with SHOC–cld_fraction–SPA–P3 sub-cycled 12 times (150 s), radiation every 2 steps.

## 6. Port architecture

### Package layout (`src/eamxx_torch/`)

| Our file | Source (E3SM `components/eamxx/src/` unless noted) |
|---|---|
| `constants.py` | `share/physics/physics_constants.hpp`, `share/util/eamxx_universal_constants.hpp` |
| `physics_functions.py` | `share/physics/eamxx_common_physics_functions_impl.hpp` |
| `saturation.py` | `share/physics/physics_saturation_impl.hpp` |
| `column_ops.py` | `share/util/eamxx_column_ops.hpp` |
| `cld_fraction.py` | `physics/cld_fraction/cld_fraction_main_impl.hpp`, `eamxx_cld_fraction_process_interface.cpp` |
| `spa.py` | `physics/spa/eamxx_spa_process_interface.cpp`, `share/algorithm/eamxx_data_interpolation.cpp` (time and vertical interpolation) |
| `tms.py` | `physics/tms/impl/compute_tms_impl.hpp`, `eamxx_tms_process_interface.cpp` |
| `shoc/` (grid, length, tke, second and third moments, assumed PDF, pblintd, energy fixer, implicit update, main, interface) | `physics/shoc/impl/*.hpp` (57 files), `shoc_constants.hpp`, `eamxx_shoc_process_interface.cpp` |
| `p3/` (tables, main parts 1–3, sedimentation, process rates, conservation, interface) | `physics/p3/impl/*.hpp` (42 files), `p3_functions.hpp`, `eamxx_p3_process_interface.cpp`, `eamxx_p3_run.cpp` |
| `rrtmgp/` (gas optics, cloud optics, RTE SW and LW solvers, McICA, fluxes, interface, zenith angle) | `components/eam/src/physics/rrtmgp/external/cpp/{rrtmgp,rte,extensions/cloud_optics}`, `physics/rrtmgp/eamxx_rrtmgp_interface.hpp`, `eamxx_rrtmgp_process_interface.cpp`, `share/util/shr_orb_mod.F90` (`shr_orb_cosz`, `shr_orb_decl`) |
| `zm/` (constants and runtime options, entropy and its inversion, dilute CAPE, cloud properties and entrainment, downdraft, closure, tendencies, evaporation, MCSP, momentum transport, main, interface) | `physics/zm/impl/*.hpp` (21 files), `zm_functions.hpp`, `eamxx_zm_process_interface.cpp` |
| `surface_flux.py` | `share/util/shr_flux_mod.F90` (`shr_flux_atmOcn`, scheme 0) |
| `suite.py` | Process order, sub-cycling and radiation frequency: `share/atm_process/atmosphere_process_group.cpp` and the defaults above |
| `scm/iop.py`, `scm/driver.py` (separate from physics) | `physics/iop_forcing/*`, `share/data_managers/IOPDataManager.cpp`, `control/atmosphere_surface_coupling_{importer,exporter}.cpp` (IOP imports; the exporter quantities for ocean fluxes), `dynamics/homme/eamxx_homme_process_interface.cpp` (`update_pressure`) |

Sizes, excluding tests: SHOC 6.9k lines, P3 7.7k, ZM 6.0k, RRTMGP 3.3k interface plus 6.8k kernels, cld_fraction 1.0k, IOP forcing 0.6k, TMS 0.4k, SPA 0.2k plus interpolation.

### One public entry point

```python
out = eamxx_torch.physics_step(state, grid, dt, config)
# out.tendencies:  dT/dt, dq/dt for each tracer, du/dt, dv/dt   (ncol[, nmember], nlev)
# out.surface:     precipitation (liquid, ice), fluxes seen by the physics, radiative fluxes
# out.diagnostics: cloud fractions, TKE, PBL height, LWP/IWP, heating rates, ...
# out.state:       persistent physics state (TKE, P3 moments, SHOC moments, rad cache)
```

- Pressure interfaces and hybrid coefficients come from the host.
- Tensors are `(ncol, nlev)` or `(ncol, nmember, nlev)`. Index 0 is the model top.
- EAMxx updates the state process by process. Tendencies are (state after − state before) / dt over the whole physics step. That matches how HOMME receives physics in EAMxx.
- Persistent physics state (TKE, SHOC and P3 memory, radiation between calls) is carried explicitly. It is not hidden in globals.

### Translation rules for Kokkos code

These are how the code is written. They are not changes to the results.

- **Packs to tensors.** EAMxx works on SIMD packs of 16 levels. We work on whole `(ncol, nlev)` tensors. Level loops stay as Python loops where the source has vertical dependencies (tridiagonal solves, parcel ascent, sedimentation).
- **Data-dependent loops.** P3 sedimentation (`while dt_left > tol`) and ZM's Brent iteration (`zm_invert_entropy`) run a data-dependent number of times. We run a fixed number of iterations with a mask that freezes finished columns. The cap is set from a proven bound (for sedimentation, the largest allowed fall speed and the smallest layer depth). Every call also returns whether any column hit the cap. That is checked in tests, as a tensor, without `.item()`.
- **Branches become `torch.where`,** evaluated in the same order as the source.
- **Tables become gathers** with the same index arithmetic and interpolation as the source.
- **McICA random numbers.** EAMxx seeds each column from the fractional part of its lowest (SW) or second-lowest (LW) layer pressure, times 1e9, and starts one `conv::Random` per (column, layer, g-point) at seed + layer·ngpt + g-point. We port that generator bit for bit: its unsigned 64-bit arithmetic is held in int64 tensors (wrapping add/sub/xor/shift, logical right shift by masking), and the unsigned-to-double conversion rounds once, as the hardware does.
- **Summation order.** Kokkos Serial sums levels in order. `torch.sum` does not promise an order. Where a sum feeds a threshold, we use the source's order.
- **No in-place ops on tensors that need gradients.** New tensors or `torch.where` only.
- **Masked-out branches get safe inputs.** Where the source computes a quotient or root only under a mask, the port evaluates it everywhere under `torch.where`. The unselected entries are fed a harmless value (e.g. denominator 1), so they stay finite and cannot put NaN into gradients. Selected values are unchanged. Examples: TMS `z0oro`, SHOC `vvk` outside the PBL search range, the stress-direction factor, `qc2` in `inv_qc_relvar`.
- **Member dimension.** Scheme functions take `(ncol, …)`. Columns are independent, so the suite entry point flattens `(ncol, nmember, …)` to columns and back.

## 7. Porting order

Done (24 Sep): 1–11. Steps 1–10 match bitwise; so do the suite for one whole step in both configurations and multi-step single-column runs against EAMxx. The suite-level comparison with observations and LES is in docs/validation.md: criteria written first, 13 of 40 checks pass, and the failures are EAMxx's own. Differentiability report done (24 Sep, docs/differentiability.md): gradients finite everywhere tested, NaNs from masked branches removed with forward values unchanged, then options 1–2 and per-step checkpointing implemented, off by default (EAMxx behavior unchanged). Next: float32 and GPU.

1. **`cld_fraction`.** Small and self-contained. It proves the whole pipeline: dump, replay, compare, test. **Stop and report after this matches.**
2. **Shared helpers:** constants, common physics functions, saturation, column ops. Ported with their first user (SHOC), and tested through that scheme's dumps. (Order changed on 18 Sep: `cld_fraction` needs none of them.)
3. **SPA** and **TMS.** Small, and they feed P3, RRTMGP and SHOC.
4. **SHOC.** The core of the boundary layer and shallow convection.
5. **P3.**
6. **RRTMGP**, including McICA and the zenith angle.
7. **IOP forcing** and the SCM driver.
8. **ZM**, if the LR configuration is chosen.
9. **Surface fluxes** (`shr_flux_atmOcn`), if decision 2 says so. Its reference is a small Fortran harness built from E3SM's own `shr_flux_mod.F90`.
10. **Single-column EAMxx reference driver** (decision 4): physics and IOP forcing on one column, no dycore.
11. **Suite assembly** (order, sub-cycles, radiation frequency), then suite-level validation against that driver.

Each scheme follows the per-scheme workflow below. We do not move to the next scheme until the current one matches.

## 8. Per-scheme workflow and tolerances

1. Dump the scheme's inputs and outputs from the reference. Cases: BOMEX, DYCOMS-RF01, ARM97, TWP06, RICO, GATE III. Several steps each (first, early, mid, late), 4 columns each. Add intermediate dumps inside the scheme.
2. Translate line by line.
3. Replay the dumped inputs. Compare every output field. (Revised 19 Sep, after the strict build and `mathfn` existed.)
   - **libm math mode vs the `strict` build: bitwise.** Any difference is a translation error.
   - **torch math mode (the default):** torch's `pow`, `exp`, `log`, `erf`, `tanh` differ from libm by about 1 ulp. The tolerance is measured, explained and written in the scheme's test, per field (e.g. SHOC: normwise ≤ 1e-9; TMS: rtol 1e-11).
   - **vs `release`/`strict16` builds:** reported, not tested. They differ by FMA contraction and summation order.
   - **If a field exceeds the target,** find the first intermediate that diverges. A value that lands on the other side of a threshold is the usual cause. We never tune to close a gap.
   - **Branch coverage:** list which branches and limiters the replays exercise. Cover missing ones with a harness or a standalone run, or report them as untested.
4. Check water and energy closure. EAMxx's own column checks are off in these configurations, so the tests compute column water, total energy (SHOC energy-fixer definition, `eamxx_torch/budgets.py`) and tracer mass before and after, against the surface fluxes. The residual must be roundoff of the column sums.
5. Add a pytest that replays the saved dumps. Small fixtures (a few columns, a few steps) are committed to `tests/data/`. Full dumps stay in `reference/data/`, which is git-ignored.

## 9. Acceptance tests

### Per scheme (float64, CPU)

- Replay tests pass at the tolerance above, for every scheme, every dumped case and step.
- Budget checks pass.
- Float32 support is added only after the suite matches. It is checked against a single-precision reference build.

### Suite level

- Cases: BOMEX, RICO, DYCOMS-II RF01, GABLS1, ARM SGP (ARM97), one deep case (TWP-ICE via TWP06, or GATE III), and RCE.
- Compare over the full run: profiles of θ_l, q_t, u, v, cloud fraction, TKE; surface fluxes; LWP and IWP; precipitation.
- The reference for side-by-side runs is decision 4. The port must reproduce the reference's trajectories. Pass criteria are written down before the runs are made.
- Then compare with observations and LES where the case has them. Our earlier project already implements the published BOMEX targets (Siebesma et al. 2003). We would copy targets, with the source noted, not import code.
- Done 24 Sep: docs/validation.md (criteria, sources, results). The earlier project's BOMEX gates (0.5 K, 0.5 g/kg, 0.1 mm/day) are its own screening values, not published bands, so they were not used; the criteria come from the papers.
- Downloads over 1 GB need your approval. None has been needed so far.

### Differentiability (only after the suite matches)

- `torch.autograd.gradcheck` in float64 on each scheme at states taken from the dumps.
- A list of every zero or undefined gradient. Found so far (23 Sep): SHOC's assumed-PDF `sqrt(max(0, variance))` where the variance clips to 0. Masked-out branches that computed `log(0)` or divided by zero (SHOC's assumed PDF, P3's rain size distribution, autoconversion and rain immersion freezing) now get safe inputs, forward values unchanged. Known up front: SHOC PBL-height level picks, P3 thresholds and table cell picks, sedimentation sub-step counts, McICA masks, ZM trigger, launch level, cloud top and entropy iteration, RRTMGP table indexing, and the `qv` repair clips.
- Report first. Then propose smoothing, each behind a flag, off by default, re-validated against the reference.
- Done 24 Sep: docs/differentiability.md. Tools: `reference/scripts/grad_check.py`, `grad_horizon.py`; helpers `eamxx_torch/gradsafe.py`.

### GPU and coupling

- Batched runs on GPU. Report speed, memory, and CPU-versus-GPU agreement.
- **This Mac's GPU (Apple MPS) has no float64.** GPU checks here can be float32 only. Float64 GPU checks need a CUDA machine.
- Conservation across the dycore interface: column mass, water and energy before and after, with the host's pressure interfaces.
- Done 25 Sep: docs/float32_gpu.md. Float32 runs, against the `strict-sp` build where EAMxx has one (cld_fraction bitwise; SHOC and P3 within float precision, not bitwise). MPS runs in float32 and agrees with the CPU about as well as float32 agrees with float64. It is not faster at 1024 columns. Decided 25 Sep: float32 stays at tolerance level; P3's float overflow gets an off-by-default float64 option. Tools: `sp_compare.py`, `dtype_check.py`, `gpu_bench.py`, `device_diff.py`.

## 10. Deviations from the source

Every deviation is recorded here with its reason. Changes of form only (the translation rules in section 6) are not deviations, as long as results match.

| Scheme | Deviation | Reason | Effect on results |
|---|---|---|---|
| saturation | `check_temperature` returns a mask instead of aborting on T ≤ 0 or NaN | An abort needs `.item()` and a Python branch on tensor values | None for valid input |
| SHOC | `inv_exner = 1/exner` everywhere. The reference sets it only where `exner ≠ 0` and keeps stale buffer data elsewhere (debug builds abort). | Stale data cannot be reproduced | None for p > 0 |
| SHOC | Not ported: `do_3d_turbulence_shoc`, `check_flux_state_consistency`, column conservation checks | Off in the standard configuration | None |
| SHOC | `pblintd` starts `rino` at zeros. The release reference leaves unused entries uninitialized. | Entries never written are never read | None |
| P3 | The Courant substep loops (`while dt_left > tol`) run a fixed number of iterations (400) with finished columns masked off, and return `hit_substep_cap` | A data-dependent loop count per column needs `.item()` and per-column control flow | None. The most any reference call needs is 36 (rain), 11 (ice), 1 (cloud), and the flag is checked in the tests |
| P3 | The substep loop stops early once every column has finished (`EARLY_EXIT`, on by default) | Speed only; a masked-off column's state does not move. Batched or compiled callers can turn it off for a fixed loop shape | None (a test runs both ways and compares) |
| P3 | `find_top`/`find_bottom` follow the reference's `team_size()==1` branch | That is the branch a CPU build takes. It treats NaN as "hydrometeor present"; the multi-thread branch would not | None for valid input |
| P3 | `check_values` is not ported | It only prints, and aborts only with `force_abort`, which is off | None |
| RRTMGP | The vertical axis is taken as top-first (`top_at_1 = true`); the `top_at_1 = false` kernels are not ported | The reference derives the orientation from column 0's pressures. The host contract puts index 0 at the model top, which is what every EAMxx grid gives | None for top-first input |
| RRTMGP | The SW driver computes every column, with μ0 replaced by 1 on night columns, and then zeroes their fluxes, instead of compacting the daytime columns | Compaction needs data-dependent shapes; the reference's night columns are zero | None: each column is independent (a test compares day and night columns bitwise) |
| RRTMGP | Columns are not split into chunks (`column_chunk_size`) | Every quantity is per column | None |
| RRTMGP | Not ported: Pade cloud optics, overlap option 0, more than one LW quadrature angle, user `col_dry`/latitude in gas optics, `SCREAM_RRTMGP_DEBUG` range checks | EAMxx does not use them (the debug checks only print) | None |
| SCM driver | No dycore: the surface pressure stays at the IOP file's value at the start time; layer thicknesses come from the hybrid grid and it; pressures are recomputed each step as HOMME's `update_pressure` does | EAMxx has no single-column mode; this is the host's choice, and the single-column reference driver (`scm_eamxx`) makes the same one | Driver only; the physics is unchanged |
| SCM driver | SHOC's large-scale vertical velocity is the IOP omega | In DP runs it is the dycore's omega | Driver only |
| SCM driver | The initial T perturbation of DP runs (`perturbed_fields: [T_mid]`) is not applied | It only breaks the symmetry between columns of a DP domain | Driver only |
| IOP forcing | The nudging means run over a given set of columns (a DP domain), or each column alone in the single-column driver | A single-column run is a domain of one column (weight 1/1) | None |
| SHOC | Masked-out entries of σ in the assumed-PDF's `erf` and `exp` terms get σ = 1 | The translation rule for masked branches (section 6); it keeps gradients finite there | None (163 calls still bitwise) |
| P3 | Masked-out entries get 1 in `get_cdistr_logn0r`'s logs, autoconversion's powers and rain immersion freezing's logs | Same rule | None (212 calls still bitwise) |
| ZM | `invert_entropy`'s Brent loop runs at most LOOPMAX+1 = 101 iterations with converged elements frozen, stops early once all have converged, and returns a non-convergence flag instead of aborting | A data-dependent loop per element needs per-element control flow; an abort needs `.item()` | None. At most 9 iterations in the reference calls; the flag is checked in the replays |
| ZM | `ktm`, `kbm` (highest cloud top and base over active columns) are taken over the columns of the call, not of the MPI rank | Same idea at a different scope | None: they only bound loops that are zero outside each column's cloud (subset replays are bitwise) |
| ZM | Cloud properties, closure and tendencies are computed for every column and kept for active ones; the reference skips inactive columns | No per-column control flow | None (masked-out columns get finite stand-in inputs) |
| ZM | Not ported: `zm_transport_tracer` (EAMxx does not call it), `zm_microp`, `old_snow = false`, PERGRO | Off or unused in EAMxx | None |
| Surface fluxes | Only `shr_flux_atmOcn` scheme 0 on the path the coupler takes for EAMxx (explicit stress, no gustiness, no waves, no cold-air outbreak); water isotope fluxes not ported; ocean albedo not computed (albedos stay as the driver sets them, 0 as in DP runs) | The configuration EAMxx runs with; isotopes are unused | None for that configuration |
| SCM driver | Open-ocean fluxes (cases without IOP surface fluxes): EAMxx's exporter quantities, `shr_flux_atmOcn`, the coupler merge with ocean fraction 1, the importer's signs, at the IOP `Tg`, ocean at rest, computed at the start of each step | EAMxx has no flux scheme; this is what E3SM's coupler does over ocean | Driver only |
| SCM driver | The ACCUMULATED fields (surface precipitation masses, SHOC's wind perturbations) are reset each step, as `AtmosphereDriver::run` does; ZM inputs computed later in the step start at zero, as in EAMxx | Driver behavior | Driver only |
| SHOC, P3, ZM | Singular expressions in masked-out or floored lanes (√0, x/0, 0/0, x^0.55 at 0) are computed from safe inputs (`gradsafe.sqrt0`, `gradsafe.nonsingular`, `torch.where` on inputs); P3 computes `mu, dv, sc, dqsdt, dqsidt, ab, abi, kap` on every level instead of leaving 0 on inactive levels | Section 6 rule: NaN gradients (0*inf) otherwise | None: values identical (SHOC, P3, ZM bitwise on every dump) |
| RRTMGP, SHOC, SCM driver (options, off by default) | `RRTMGPRuntime.mcica_seed_from_pressure=False` (state-free McICA seeds); `SHOCRuntime.energy_fixer_tke_tol` (tolerance in the shoctop test); `SCM.run(checkpoint=True)` | Differentiability (docs/differentiability.md) | None when off (bitwise). On: see docs/differentiability.md; all 40 validation verdicts unchanged |
| Suite (float32 only) | With `dtype=float32`, `iop_forcing`, `zm`, `tms` and `rrtmgp` run in float64 (on the CPU when the device has no float64) and return float32 (`SuiteConfig.double_only_in_float64`, default True) | EAMxx builds these four only in double; in float32 RRTMGP's minor-gas scaling overflows (docs/float32_gpu.md) | None in float64. In float32: the configuration EAMxx's float build implies |
| SHOC, P3 (float32 only) | EKAT's float-to-double promotion (`Pack<float> op double` is `Pack<double>`) is not reproduced: float32 tensors stay float32 | Reproducing it needs an audit of every literal and constant in about 15k lines | Not bitwise against the `strict-sp` build; e32/e64 ≤ 0.94 for every field (docs/float32_gpu.md) |
| SCM driver (float32 only) | The initial state and IOP data are built in float64 and cast; ocean fluxes are computed in float64 on the CPU; McICA random numbers are generated in float64 on the CPU and cast | E3SM's coupler is double; MPS has no float64 | None in float64 |
| P3 (option, off by default) | `P3Runtime.immersion_freezing_in_float64`: in a float32 run, `cldliq_immersion_freezing` is evaluated in float64 and rounded to float32 | In float, `CONS6*cdist1*tgamma(7+mu_c)*expAimmDt` overflows below about 195 K, then inf·0 = NaN (as in EAMxx's float build); decided 25 Sep | None when off or in float64. On: 3 NaN float32 validation runs become finite |

## 11. Downloads so far

| What | Size | Source |
|---|---|---|
| E3SM, depth 1, plus submodules | 430 MB on disk | GitHub |
| Input data (IOP files, L128 grid, ne30 IC, ne4 SPA, RRTMGP and P3 tables, topography) | 546 MB | E3SM public input data server |
| conda envs `eamxx-ref`, `eamxx-torch` | 1.3 GB and 0.7 GB installed; many packages, each well under 1 GB | conda-forge |
| Reference build tree (local, not a download) | 1.2 GB | built here |

Not downloaded: the ne30pg2 SPA file (4.4 GB).

## Decisions for you

1. **Which configuration to port.** *Decided 18 Sep 2026: port the standard schemes first (shared by both configurations), then ZM.*
   - (a) Standard SCREAMv1: SHOC, P3, cloud fraction, SPA, TMS, RRTMGP. Tuned for km scale at 100 s. No deep scheme.
   - (b) EAMxx-LR: the same plus ZM, at 1800 s with 150 s sub-steps. Experimental upstream.
   - Recommendation: port (a)'s schemes first. They are shared by both. Add ZM second if your dycore is coarse, which it probably is. Grid is L128 either way.
2. **Surface fluxes.** *Decided 18 Sep 2026: port E3SM's coupler ocean flux routine `shr_flux_atmOcn`, E3SM default scheme 0.* EAMxx has none. Options: stay with prescribed-flux cases, or also port E3SM's coupler ocean flux (`shr_flux_atmOcn`) so that RICO, GATE, RCE and a slab ocean work. If ported, which scheme: E3SM default (0) or the UA scheme (2) that the scmlib DP scripts use? Recommendation: port it, scheme 0.
3. **SPA data.** *Decided 18 Sep 2026: keep the 177 MB ne4 file. No 4.4 GB download.* Keep the 177 MB ne4 aerosol file (recommended; code paths are identical), or approve the 4.4 GB ne30pg2 file for closer fidelity to CIME runs.
4. **Suite-level reference.** *Decided 18 Sep 2026: build the patched single-column EAMxx driver (physics and IOP forcing, no dycore) for side-by-side runs. Keep DP-EAMxx for context.* EAMxx has no pure single-column mode; its SCM is DP with HOMME. Options:
   - (a) DP-EAMxx as is. HOMME sits between physics steps, so the port cannot match it exactly.
   - (b) A small patched EAMxx driver that runs the same physics and IOP forcing on one column with no dycore. That is exactly what our SCM driver does.
   - Recommendation: build (b) for side-by-side runs and keep (a) for context.
