# Differentiability report

24 September 2026. Plan section 9: report first, then propose smoothing, each behind a flag, off by default. Options 1 and 2 and checkpointing are implemented (section "Implemented options" at the end); the defaults are EAMxx's behavior, unchanged.

## Verdict

- **Gradients are finite everywhere tested.** Every scheme, on inputs dumped from EAMxx, and the whole single-column model over up to 36 steps (100 s) and 12 steps (ZM, 1800 s), in 8 cases.
- **Where the model is smooth, autograd is correct.** Per scheme, 91–100% of the input→output pairs that depend on each other agree with finite differences to 0.1%. Through one full model step they agree to 1e-3 or better in 7 of 8 cases (1% in GATE III with ZM).
- **The rest are discontinuities and kinks that are in EAMxx itself**, not errors in the gradient. Four sources account for nearly all of them: SHOC's energy-fixer level pick, RRTMGP's McICA random seed, P3's table indices, and ZM's trigger and level picks. Autograd returns the derivative with the pick held fixed.
- **Forward values are unchanged.** The fixes below only change gradients. SHOC, P3 and ZM replay bitwise against EAMxx on every dump, around the scheme and around its checks. The full test suite passes: 192 tests, no expected failures left.

## What was changed: NaN gradients from masked branches

All NaN gradients came from one pattern. A value is computed in lanes whose result is then discarded or floored, and the value there is singular (√0, x/0, 0/0, x^0.55 at 0). Autograd multiplies the discarded lane's gradient (0) by an infinite local slope and gets NaN. The true derivative there is 0 or finite.

Following the translation rule already in the plan (section 6: masked-out branches get safe inputs), these sites now compute from a safe input on the discarded lanes. `eamxx_torch/gradsafe.py` holds two helpers: `sqrt0` (√x, gradient 0 at x = 0) and `nonsingular(bad, fn, …)` (same values; gradient 0 on the named lanes). Values are identical for every input, including −0, NaN and negative arguments; `tests/test_gradients.py` checks this.

| Scheme | Sites |
|---|---|
| SHOC | surface stress magnitudes and u* (`ksrf`, `ustar` ×2, `uf`, Obukhov u*, the stress-perturbation scale); in-plume variance roots and the PDF standard deviation where a variance clips to 0; `sqrt(thlsec)`, `sqrt(qwsec)`, `sqrt(w_sec)` under their floors; `Skew_w`; the in-plume correlation divide (`testvar = 0`); the plume second moments at zero correlation (0/0, which EAMxx's `max_(0, NaN)` turns into 0); `−corr/w1` in masked lanes; `sqrt(sterm)` in masked lanes |
| P3 | the thermodynamic helpers `mu, dv, sc, dqsdt, dqsidt, ab, abi, kap` (EAMxx leaves them 0 on inactive levels; the port now computes them on every level, so later masked lanes divide by finite numbers); rain and cloud size-distribution roots and divides; the rain lookup at λr = 0; effective radii; accretion, self-collection, immersion-freezing and rime-density divides; the ice-table lookup at qi = 0; `tau_eff`; the CCN power law at zero CCN |
| ZM | rain evaporation's `sqrt(flxprec)` at zero flux (the flux limit wins there); the dilute parcel's mixing divide below the launch level |

My earlier note that SHOC's `std_s = sqrt(max(0, var))` was a true non-differentiable point was wrong. Where the variance clips to 0, `std_s` is constant nearby and is otherwise used squared, so the true derivative exists.

## How it was checked

`reference/scripts/grad_check.py` takes one dumped call (EAMxx's exact inputs), runs the port in float64 with PyTorch's math, and compares autograd with central differences (steps 1e-4, 1e-6, 1e-8, and one-sided) for every output field against every input field. Each pair is classified as agreeing, independent, kink (autograd equals a one-sided slope), jump (the difference grows like 1/step), missing or mismatch. For disagreements it bisects the perturbation to one element. It then reruns forward at ±step and names the first comparison or index computation that flips, with its source line.

`reference/scripts/grad_horizon.py` does the same through N steps of the whole single-column model, and in situ for each process of a running step (`--per-process`).

### Per scheme (26 dumped calls)

| Scheme | Calls | Dependent pairs | Agree | Jump | Kink | Other |
|---|---|---|---|---|---|---|
| cld_fraction | 2 | 4 | 100% | 0 | 0 | 0 |
| SPA | 1 | 5 | 100% | 0 | 0 | 0 |
| TMS | 2 | 14 | 100% | 0 | 0 | 0 |
| IOP forcing | 2 | 32 | 100% | 0 | 0 | 0 |
| SHOC | 5 | 1546 | 96% | 22 | 9 | 34 |
| P3 | 5 | 1320 | 95% | 0 | 19 | 47 |
| RRTMGP | 4 | 897 | 94% | 54 | 1 | 1 |
| ZM | 4 (8 columns each) | 584 | 91% | 22 | 1 | 27 |

In the running model, per process, every process agrees with finite differences except SHOC where input TKE sits exactly at its floor (the energy fixer, below).

### Through the whole model

Autograd vs central difference of a random functional of T and qv after N steps, with respect to the initial T and qv. The figures are the best relative agreement over steps 1e-5 and 1e-7; "jumps" means the two steps disagree with each other because a discontinuity was crossed.

| Case | N = 1 | Medium | Longest | Peak memory |
|---|---|---|---|---|
| BOMEX, 100 s | 1.5e-4 | N=10: jumps | N=36: 5e-4 | 1.3 GB |
| DYCOMS-RF01, 100 s | 5e-7 | N=10: 4e-2 | N=36: 6e-3 | 1.3 GB |
| CASS, 100 s | 9e-4 | N=10: 6e-2 | N=36: 8e-3 | 1.3 GB |
| RICO, 100 s | 2e-4 | N=10: jumps | N=36: jumps | 1.3 GB |
| BOMEX, ZM | 3e-4 | N=3: 3e-2 | N=12: jumps | 5.4 GB |
| RICO, ZM | 4e-4 | N=4: jumps | N=12: jumps | 5.0 GB |
| ARM97, ZM | 9e-4 | N=4: 9e-3 | N=12: 4e-3 | 4.9 GB |
| GATE III, ZM | 1e-2 | N=4: 1e-2 | N=12: 2e-2 | 5.2 GB |

Gradient norms stay of the same order over these horizons (no blow-up). Memory grows linearly: about 25 MB per 100 s step and 400 MB per ZM step.

## Where gradients are zero, one-sided or undefined

**Jumps (the output is discontinuous; autograd gives the derivative with the pick held fixed):**

1. **SHOC energy fixer** (`shoc/functions.py:887`). `shoctop` is the highest level where TKE ≠ its floor, found by exact equality. After the implicit diffusion solve, TKE above the boundary layer equals the floor only to rounding. So a one-ulp change anywhere can move `shoctop`, which changes how deep the energy residual is spread. This one line explains nearly all SHOC jumps and mismatches, and the SHOC disagreement in the running model.
2. **RRTMGP McICA seed** (`rrtmgp/mcica.py:83`). EAMxx seeds each column's random cloud subcolumns from the digits of the lowest (SW) or second-lowest (LW) layer pressure. Any change in that pressure, for example from qv anywhere in the column through the pressure update, redraws all subcolumns. All RRTMGP jumps trace here.
3. **P3 lookup-table indices** (`p3/tables.py:115`, truncation to a cell). Outputs are continuous within a table cell and jump or kink at cell edges. All located P3 disagreements trace here.
4. **ZM trigger and level picks** (launch level, LCL, cloud top, `ktm`/`kbm`) and the Brent iteration count.
5. Known by construction, not seen as flips in these calls: SHOC's PBL-height level search, P3's sedimentation substep counts, cld_fraction's ice fraction (a step function of qi at 1e-12), SHOC's `inv_qc_relvar` clips.

**Kinks (autograd returns one side's slope, a valid subgradient):** clips at a bound (TKE at its floor, cloud fraction at 1, P3's conservation limiters, `max(0, ·)` everywhere), and |τ| at exactly zero surface stress (subgradient 0).

**Near-zero sensitivities that look like mismatches:** ZM's outputs against surface geopotential `phis` (relative sensitivity ~1e-8). `phis` shifts every height equally and ZM uses height differences, so its true effect is ~0 and the finite differences see rounding.

**Non-finite:** none left.

## Proposed smoothing (1, 2 and checkpointing were then implemented; see below)

Each would be re-validated: with the flag off, bitwise against EAMxx as now; with it on, the single-column validation runs (docs/validation.md) re-scored to show the change is within noise.

1. **McICA seed from a caller-supplied integer** instead of pressure digits (for example the column index plus the step). The random subcolumns would then no longer redraw when pressure changes. With the flag on, values change: the random draws differ, with the same statistics. This is the largest jump source for any gradient involving qv or surface pressure. **Recommended first.**
2. **SHOC `shoctop` with a tolerance** (TKE > floor × (1 + 1e-12)) so that rounding cannot move it. With the flag on, forward output changes only where TKE is within rounding of the floor. **Recommended second.**
3. **Hold level picks and iteration counts from an unperturbed forward pass** (straight-through: ZM trigger and levels, SHOC PBL level, P3 substep counts). Autograd already does this for the gradient. A flag would matter only for users who need a forward model that is locally continuous for finite-difference or line-search methods.
4. **Soft versions of hard switches** (sigmoid ZM trigger, smooth P3 table interpolation across cells, soft clips). These change the physics more and would need their own validation. Not recommended unless a use needs them.

Separately (engineering, not smoothing): per-step activation checkpointing would make long adjoint horizons fit in memory. At 25–400 MB per step, a 1-day ZM run is about 20 GB without it.

## Files

- `src/eamxx_torch/gradsafe.py`: the two helpers.
- `reference/scripts/grad_check.py`: per-scheme gradient report on dumped calls (`--locate N`, `--nan-only`).
- `reference/scripts/grad_horizon.py`: gradients through N steps; `--per-process STEP` for each process in a running step.
- `tests/test_gradients.py`: helper values and gradients; finite gradients on every SHOC, P3 and ZM fixture. `tests/test_scm.py::test_scm_gradients_are_finite` (two model steps) is now a normal test.
- Results: `reference/runs/grad/` (JSON and logs per call; `horizon/` per case).

## Implemented options (24 September 2026)

All three are off by default; with them off the port is EAMxx, bitwise (all 195 tests pass, including every replay against the strict build).

| Option | How to turn on | What it does |
|---|---|---|
| State-free McICA seeds | `RRTMGPRuntime(mcica_seed_from_pressure=False)` | Seeds = step count × 2^40 + column × 2^20 (LW + 2^19); `state["mcica_seed"]` (int64, per column) replaces the column index. The random subcolumns no longer depend on pressure. |
| SHOC energy-fixer tolerance | `SHOCRuntime(energy_fixer_tke_tol=1e-14)` | `shoctop` treats \|tke − mintke\| ≤ tol·mintke as "at the floor". 1e-14 is about 50 ulp: in 100-step runs of five cases, 15,000 near-floor TKE values were within 1e-15 of the floor (rounding) and the rest spread evenly from 1e-15 to 1e-6, so there is no clean gap; 1e-14 moves only about 1% of the physical ones. |
| Checkpointing | `SCM.run(state, nsteps, checkpoint=True)` | Each step runs without a graph and is recomputed, with the host state it started from, during the backward pass. |

`reference/scripts/scm_case.py port_scm(..., smooth=True)` turns on both physics options; `grad_horizon.py --smooth [--only seeds|tke] [--checkpoint]`, `validate_obs.py run --smooth [--mcica-seed N]`.

### Checks

- **Checkpointing gives the same numbers.** Forward values and gradients are bitwise identical with and without it (72 steps at 100 s; 12 ZM steps), and the driver continues from the right step afterwards. Memory: 12 ZM steps 5.6 → 1.3 GB (1.5× the time); a full day with ZM (48 steps) needs 1.4 GB instead of about 20 GB. PyTorch's own checkpoint (non-reentrant) kept every graph node and used more memory than no checkpointing, so the step is a custom autograd function; it works with `torch.autograd.grad`.
- **The SHOC tolerance removes the jumps it targets.** Autograd vs finite differences through the whole model, one step: 3e-4 or better in all 8 cases, 1e-5 or better in 7 (before: 1e-3 in 7, 1e-2 in GATE III with ZM; GATE III with ZM is now 8e-8). Longer horizons improved in ARM97 and GATE III with ZM (4 and 12 steps: about 1e-2 → 1e-4 or better), BOMEX with ZM at 3 steps (3e-2 → 1e-5), DYCOMS at 36 steps (6e-3 → 3e-4). RICO in both configurations still crosses discontinuities (P3 tables, ZM trigger). Run separately, the tolerance alone gives these improvements; the seed option alone does not, because in the single column the seeds never change (below).
- **The seed option removes the pressure jump.** With EAMxx's seeds, raising the two lowest pressures by one part in 10^12 changes fluxes by more than 1e-3 W m⁻² in a partly cloudy column; with the option, by less than 1e-6 (`tests/test_gradients.py`).
- **Physics.** With both options on, every one of the 40 validation verdicts is unchanged (docs/validation.md, "With the gradient options on").

### Found along the way: in a single column, EAMxx's McICA sampling is frozen

The single-column host holds the surface pressure, and the layer thicknesses come from the hybrid grid, so the lowest layer pressures never change. EAMxx's seeds are the digits of those pressures, so **every radiation call of a single-column run uses the same seeds**: one frozen realization of the random cloud subcolumns for the whole run (checked: 20 radiation calls, one distinct seed, in BOMEX and ARM97). In DP-EAMxx and global runs the surface pressure varies and the seeds change every step. The state-free option draws new subcolumns each step. In the 100 s configuration this changes some cloud statistics beyond the spread of 4 different seeds (BOMEX LWP 73 g m⁻² with EAMxx's frozen draw, 66–68 with new draws each step; RICO cloud cover 0.35 vs 0.37–0.39), but no verdict.

### Where SHOC's energy fixer spreads energy in EAMxx

With EAMxx's exact test, `shoctop` is usually level 0–16 (near the model top), because some upper level holds TKE one ulp above the floor; the energy residual is then spread over almost the whole column. With the tolerance, `shoctop` is at the top of SHOC's active layer (levels 90–121 in BOMEX), which is what the code evidently intends. The effect on the solution is small (BOMEX LWP 73.0 vs 72.8 g m⁻²).
