# Float32 and GPU

25 September 2026. Plan section 9: float32 is checked against a single-precision reference build; GPU runs report speed, memory and CPU-vs-GPU agreement. The float64 CPU default is unchanged and still bitwise (all tests pass).

## Verdict

- **cld_fraction in float32 matches EAMxx's single-precision build, bitwise.**
- **SHOC and P3 in float32 do not match it bitwise.** Most fields are much closer to EAMxx's float results than float precision itself moves them. A few fields (SHOC `T_mid`, `w3`, `qv`, `qc`; P3 `qm`, `bm`, `qr2qi_collect`, `qc2qi_hetero_freeze`) are only as close as a float64 run is. The likely cause is that EAMxx's float build is mixed precision at the expression level (below). That is not yet traced to specific lines.
- **RRTMGP, ZM, TMS and IOP forcing have no single-precision reference.** EAMxx builds them only in double. By default the port does the same: in a float32 run these four run in float64 (on the CPU when the device has no float64) and hand back float32.
- **In float32 the reference formulas overflow in two places.** The port reproduces both:
  - RRTMGP's minor-gas scaling overflows on every call, so the default above is needed.
  - P3's immersion freezing overflows for cloud liquid colder than about 195 K. It turned three of the twelve float32 validation runs into NaN (ARM97 at step 6507 of 6912; with ZM, ARM97 at 272 of 384 and GATE III at 383 of 960). EAMxx's float build computes this in float too. An option, off by default, evaluates it in float64; with it on, all three are finite (Decisions, below).
- **MPS (this Mac's GPU) runs the model in float32.** Against the CPU in float32 it differs by about as much as float32 differs from float64 on the CPU. It is not faster here: at 1024 columns a step takes 4.6 s on MPS and 3.9 s on the CPU. RRTMGP, which stays float64 on the CPU, takes 2.4 s of that.

## What EAMxx has in single precision

EAMxx's `SCREAM_DOUBLE_PRECISION=OFF` build (our `strict-sp` variant: `configure_eamxx.sh strict-sp`, no FMA, pack size 1) builds only P3, SHOC, cld_fraction and SPA. RRTMGP, TMS, ZM and IOP forcing are double-only, and HOMME is off. So the reference for float32 is scheme-level only: standalone runs of those four processes (`make_standalone_case.py` presets `shoc_sp`, `mac_sp`, and `p3`), dumped as in double.

Three things had to be fixed on the reference side first:
- **P3's single-precision tables (`*.dat4`) could not be obtained.** CMake downloads them, but the server was unreachable. EAMxx's own writer `p3_tables_setup` writes zeros: it writes a host mirror that is never filled. `eamxx_ext/unit_replay/p3_tables_write.cpp` computes the tables with EAMxx's `compute_tables<Real>` and writes them after copying device to host. In the double build its output matches the official `.dat8` files to 4e-15 relative. The float build's output is installed as the `.dat4` files.
- The extension CMake hook sat inside the double-only HOMME block, so it moved to `tests/multi-process/CMakeLists.txt` (patch 0001 regenerated). The extension targets now build only what exists in each precision.
- The single-column reference driver is double-only (it needs RRTMGP and IOP forcing).

**EKAT is mixed precision in the float build.** `ekat::Pack<float> op double` has type `Pack<double>` (common type), so any expression with a double literal or a double constant is evaluated in double and rounded to float when stored. The port does not model this: PyTorch keeps a float32 tensor times a Python float in float32. Matching EAMxx's float build bitwise would mean finding every such promotion in SHOC and P3 (about 15k lines) and reproducing it.

## Port vs EAMxx single precision

`reference/scripts/sp_compare.py` runs the port on each dumped single-precision call twice: in float32 (e32 = port f32 vs EAMxx float) and in float64 (e64 = port f64 vs EAMxx float, i.e. what float precision itself does). A port that follows EAMxx's float arithmetic has e32 ≪ e64. Libm math mode; 7 calls each for SHOC and P3 (`shoc_sp`, `p3`, `mac_sp`), 4 for cld_fraction.

| Scheme | Result |
|---|---|
| cld_fraction | **Bitwise** (e32 = 0) |
| SHOC | e32/e64 ≤ 0.1 for 18 of 37 fields (e.g. `nc` 0.0014, `qr` 0.003, `tke` 0.003; `ustar`, `tau_est` bitwise); ≥ 0.5 for 10, about 1 for `T_mid` (e32 2.2e-7), `w3` (8.9e-3), `qv`, `qc`, `shoc_cond`. Largest 1.0. Bitwise fraction per field 19–100% |
| P3 | e32/e64 ≤ 0.1 for 23 of 38 fields (e.g. `qc2qi_collect` 1e-4; `eff_radius_qr`, `diag_equiv_reflectivity`, `qc2qr_autoconv`, `qi2qv_sublim` bitwise); ≥ 0.5 for 7, 0.8–0.94 for `qm`, `bm`, `qr2qi_collect`, `qc2qi_hetero_freeze`. Largest 0.94. Bitwise fraction per field 23–100% |

## How the port runs in float32

`SuiteConfig(dtype=torch.float32, device=...)`:
- **P3 reads the `.dat4` tables.** The other data (RRTMGP coefficients, ZM's setup, SPA climatology, geometry) is read in float64 and cast (`eamxx_torch/precision.py`: precision first, then device, because MPS has no float64).
- **`double_only_in_float64=True` (the default)** runs `iop_forcing`, `zm`, `tms` and `rrtmgp` in float64 on the device, or on the CPU for MPS. Their outputs return in float32. `False` runs everything in float32, which has no reference; RRTMGP's longwave is then NaN (test `test_float32_scm_all_in_float32_gets_nan_longwave`).
- **Some work stays in float64 on the CPU:** the initial state and IOP data are built in float64 and cast; the ocean fluxes are computed in float64 on the CPU (E3SM's coupler is double); McICA's random numbers are generated in float64 on the CPU.
- **Epsilon-type constants follow the dtype:** `rte.py` `_tiny`/`_eps`; cloud optics `finfo(dt).eps`; SHOC's `std_s` floor `sqrt(finfo.tiny)*100`, as `numeric_limits<Real>` in the source.
- `reference/scripts/dtype_check.py` finds float64 leaking into a float32 run. None remain apart from the deliberate cases above.

## Findings about float32 in the reference code

| Where | What | Source |
|---|---|---|
| RRTMGP gas optics, minor absorbers | `scaling * mycol_gas_imnr * vmr_fact * dry_fact`, evaluated left to right: column amounts ~1e22 × ~3e21 molec/cm² > 3.4e38. Overflows on every call → inf, then NaN fluxes | `mo_gas_optics_kernels.h:458–460` (C++); the Fortran kernel has the same expression |
| P3 `cldliq_immersion_freezing` | `CONS6 * cdist1 * tgamma(7+mu_c) * expAimmDt * square(inv_lamc3)`: at T below ~195 K, `expAimmDt = exp(0.65·(273.15−T))` > 1.3e22 and the partial product exceeds 3.4e38 before `inv_lamc3²` shrinks it. `CONS6` is `Constants<Scalar>`, so float in the float build; no double promotion saves it. `cloud_water_conservation` then scales by ratio 0: inf·0 = NaN | `p3_cldliq_imm_freezing_impl.hpp:46–48`, `physics_constants.hpp:94`, `p3_conservation_impl.hpp` |
| RRTMGP SW two-stream | `1 − exp(−2kτ)` cancels for kτ ≲ 1e-7 in float (0 or one ulp). The `1e-12` floor on k² is a fixed constant; the `1 − k²μ²` guard uses the type's epsilon. The port follows both | `mo_rte_solver_kernels.h` `sw_two_stream` |
| P3 `p3_tables_setup` | Writes zeros (host mirror never filled) | EAMxx bug; our writer above |

## GPU (Apple MPS, float32)

MPS has no float64, so every GPU check here is float32. Float64 on a GPU needs CUDA; the code is device-generic but has not run on CUDA.

**Agreement** (`gpu_bench.py RUN --agree`): one column, 36 steps (1 hour), normwise relative difference. "MPS−CPU" is MPS float32 vs CPU float32; "f32−f64" is CPU float32 vs CPU float64 for scale.

| Case | Step 1, MPS−CPU (f32−f64) | Step 36, MPS−CPU (f32−f64) |
|---|---|---|
| BOMEX | T 8e-8 (1.5e-7), qv 7e-8 (9e-8) | T 1.3e-5 (1.8e-5), qc 0.31 (1.8), cloud 0.25 (2.1) |
| DYCOMS-RF01 | T 1.3e-7 (1.2e-7), qc 1e-4 (3e-5) | T 2.6e-5 (1.1e-5), qc 0.03 (0.014), TKE 0.11 (0.06) |
| ARM97 | T 1.3e-7 (1.4e-7) | T 1.3e-6 (1.6e-6), qc 3.5e-3 (4.5e-3) |

Step 1 agrees to float32 rounding, except trace ice in BOMEX. There P3's `qi − sub_melt·dt` is a residual of ±7e-15 whose sign differs between the devices (MPS behaves as if it fuses the multiply-add). Afterwards the difference grows just as the float32/float64 difference does: shallow cumulus amplifies rounding, stratocumulus and ARM97 less. Transcendental accuracy (float32, measured): MPS `exp`/`log`/`tanh` 1.4–2.6 ulp; CPU `torch.erf` up to 4.8e-7 absolute (SHOC uses `1 + erf`, so this is under 4 ulp of the result).

**Speed and memory** (`gpu_bench.py`, BOMEX, all columns from the same state, 6 timed steps after 2 warm-up; 8 CPU threads; 3 other runs were using 3 of the 16 cores):

| Device | dtype | 1 column, s/step | 64 | 1024 | Columns·steps/s at 1024 | Memory at 1024 |
|---|---|---|---|---|---|---|
| CPU | float64 | 0.035 | 0.44 | 3.86 | 266 | 10.0 GB peak RSS |
| CPU | float32 (RRTMGP f64) | 0.045 | 0.49 | 3.86 | 266 | (RSS peak shared with the row above) |
| MPS | float32 (RRTMGP f64 on CPU) | 0.42 | 1.06 | 4.58 | 224 | 1.3 GB GPU |

Per process at 1024 columns (s/step), CPU f32 / MPS: RRTMGP 2.41 / 2.63 (both float64 on the CPU), P3 0.30 / 0.21, SHOC 0.10 / 0.17, IOP forcing 0.03 / 0.14, TMS 0.01 / 0.09. RRTMGP decides the speed; on MPS the rest does not gain at this size. The many small kernels per step (P3's substep loops, SHOC's per-level solves) cost launch overhead that 1024 columns do not amortize.

## Float32 validation runs

All 12 validation runs repeated in float32 (torch math, CPU, defaults) and scored with the unchanged criteria file (SHA-256 `fc47c562…`). `validate_obs.py` now reports non-finite outputs. The first float32 attempt was all NaN (RRTMGP overflow) and its property-check count of 0 hid that.

- **With EAMxx's float arithmetic (option off), 9 of 12 runs are finite throughout.** Three go NaN through the P3 overflow above: ARM97 at step 6507 of 6912; with ZM, ARM97 at 272 of 384 and GATE III at 383 of 960 (each traced to P3, with immersion freezing non-finite in that step).
- **With `immersion_freezing_in_float64` on, all 12 are finite, and 2 of 40 verdicts change** (13 pass, as in float64): RICO-LR cloud cover 0.36 → 0.15 (now passes; it passed in 1 of 5 McICA draws in float64) and CASS-LR peak LWP 33.5 → 37.3 g m⁻² (band 23–35; now fails). Both were already marginal. Scores: `reference/runs/validation/results_f32.json`.
- The largest moves are in shallow cumulus with ZM: BOMEX-LR cloud cover 0.46 → 0.18, height of peak cloud 1670 → 470 m. BOMEX standard: θ_l tendency RMS 9.7 → 7.2 K/day.

## Tools

- `reference/scripts/sp_compare.py PROC RUN...`: port f32/f64 vs EAMxx float.
- `reference/scripts/dtype_check.py`: float64 leaks in a float32 run.
- `reference/scripts/gpu_bench.py RUN [--agree]`: speed, memory, agreement.
- `reference/scripts/device_diff.py SCHEME`: first operation that differs between CPU and GPU on the same inputs.
- `reference/scripts/validate_obs.py run RUN --dtype float32 [--device mps]`.
- Tests: `tests/test_precision.py`.

## Decisions (25 September 2026)

1. **P3's float32 overflow in immersion freezing: an option, off by default.** `P3Runtime.immersion_freezing_in_float64=True` evaluates `cldliq_immersion_freezing` in float64 in a float32 run (on the CPU if the device has no float64) and rounds its two tendencies to float32. It has no effect in float64 (test `test_p3_immersion_freezing_overflows_in_float32_unless_the_option_is_on`). With it off, float32 runs can go NaN, as EAMxx's float build would. With it on, the three float32 runs that went NaN (ARM97; ARM97 and GATE III with ZM; each traced to this P3 overflow) are finite to the end. `validate_obs.py run --p3-imm-f64` turns it on. It is not tied to the gradient options.
2. **Float32 stays at tolerance level.** EKAT's float-to-double promotion is not reproduced, so SHOC and P3 do not match the `strict-sp` build bitwise (table above). No audit of the 15k lines.
