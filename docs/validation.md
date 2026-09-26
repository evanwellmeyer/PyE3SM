# Suite validation against observations and LES

Criteria written 24 September 2026, 03:28 UTC, before any validation run. Results follow in a later section.

## What this tests

- The port already matches EAMxx bitwise in single-column runs (docs/status.md). So these runs test **EAMxx's physics run as a single column**, not the translation.
- A failure here is a property of EAMxx in this setting. It is reported, not tuned away.
- Every validation run is also run with EAMxx itself (`scm_eamxx`). In libm math mode the port must match it bitwise; then every number below is EAMxx's number too.
- Each case runs in both configurations: DP-EAMxx's (100 s, SHOC cell 3.33 km) and EAMxx-LR's (ZM, 1800 s, 12 sub-cycles, SHOC cell 111 km). The same bands apply to both.
- Case settings are those of E3SM's own scmlib scripts (`reference/scripts/make_dp_case.py`), with two additions below.
- The port runs in libm mode (bitwise with EAMxx) and again in torch mode. A verdict that differs between the two modes is reported as marginal.

## Rules for the bands

- Sources: the published LES intercomparison or observational paper for each case.
- LES cases (BOMEX, RICO): the band is the spread of the published LES, or the observation's stated range where the paper gives one.
- Cases the paper judges against observations (DYCOMS-RF01, CASS): the band is the observation with its uncertainty.
- Observation cases without a published acceptance band (ARM97, GATE III): the band is the largest error of E3SM's own doubly periodic SCREAM on the same case, window and observations (Bogenschutz et al. 2023). This asks: is single-column EAMxx no worse than E3SM's published doubly periodic runs of the same physics?
- Values read from figures by eye are marked so. Their bands are rounded outward.
- Quantities without a sourced band are reported with context and no verdict.
- The machine-readable criteria are in `reference/scripts/validation_criteria.py`. SHA-256 at the time of writing: `fc47c562e20696594bfc20a3e7cd39f839136eaf69fbe04c471192481f031001`.

## Criteria

Windows are hours after the start of the run unless marked LST.

| Id | Case | Quantity | Window | Pass band | Source |
|---|---|---|---|---|---|
| B1 | BOMEX | total cloud cover | h 3–6 | 0.08–0.17 | Siebesma et al. 2003, p. 1204: LES range 8–17%, ensemble 13% |
| B2 | BOMEX | maximum of the cloud fraction profile | h 3–6 | 0.05–0.07 | same, Fig. 6 and p. 1208: ≈0.06 (figure) |
| B3 | BOMEX | height of that maximum | h 3–6 | 500–700 m | same, Fig. 6; cloud base ≈500 m (figure) |
| B4 | BOMEX | liquid water path | h 3–6 | 2–15 g m⁻² | same, Fig. 2b (figure) |
| B5 | BOMEX | RMS tendency of θ_l and q_t below 2.5 km | h 3–6 | ≤ 1 K/day, ≤ 1 g/kg/day | same, p. 1204: "of the order 1 K day⁻¹ and 1 g kg⁻¹ day⁻¹" |
| R1 | RICO | total cloud cover | h 20–24 | 0.10–0.30 | van Zanten et al. 2011, p. 12: observed 0.17, "almost certainly between 0.1 and 0.3" |
| R2 | RICO | surface precipitation | h 20–24 | 0–31.2 W m⁻² (0–1.09 mm/day) | same, Table 3: LES range; observed ≈21 W m⁻² |
| D1 | DYCOMS-RF01 | total cloud cover | h 2–4 | ≥ 0.95 | Stevens et al. 2005, p. 1448: observed > 99% all night |
| D2 | DYCOMS-RF01 | cloud top (cloud fraction ≥ 0.5) | h 2–4 | 760–920 m | same, Fig. 3: observed ≈840 m (figure) |
| D3 | DYCOMS-RF01 | cloud base (cloud fraction ≥ 0.5) | h 2–4 | 520–680 m | same, Fig. 3: observed ≈600 m (figure) |
| C1 | CASS | largest hourly total cloud cover | 06–20 LST | 0.29–0.37 | Zhang et al. 2017, Fig. 7a: observed ≈33% ± 2% SE (figure) |
| C2 | CASS | cloud base (cloud fraction ≥ 0.01) | 13–14 LST | 1.7–2.1 km | same, Fig. 7c: observed ≈1.9 km (figure) |
| C3 | CASS | largest hourly LWP | 06–20 LST | 23–35 g m⁻² | same, Fig. 7e: observed ≈29 ± 3 SE (figure) |
| A1 | ARM97 | accumulated precipitation vs IOP observations | 23 Jun + 3 h to 1 Jul | within ±10% | Bogenschutz et al. 2023, Fig. 4a: DP-SCREAM −7% to 0 (figure) |
| A2 | ARM97 | largest time-mean T error, surface to 100 hPa | same | ≤ 3.5 K | same, Fig. 3a: ≈−3.2 to +2.5 K (figure) |
| A3 | ARM97 | largest time-mean q error, surface to 100 hPa | same | ≤ 0.8 g/kg | same, Fig. 3b: ≈−0.4 to +0.72 g/kg (figure) |
| G1 | GATE III | accumulated precipitation vs IOP observations | 20 days, first 3 h removed | within ±15% | same, Fig. 4a: DP-SCREAM +13% to +15% (figure) |
| G2 | GATE III | largest time-mean T error, surface to 100 hPa | same | ≤ 5.5 K | same, Fig. 3c: ≈−1 to +5.2 K (figure) |
| G3 | GATE III | largest time-mean q error, surface to 100 hPa | same | ≤ 1.1 g/kg | same, Fig. 3d: ≈−1.0 to +0.8 g/kg (figure) |

Reported without a verdict: BOMEX precipitation and cloud top; RICO LWP (LES ≈20 g m⁻²), cloud top (LES ≈2.3–2.5 km) and surface fluxes; DYCOMS-RF01 LWP (LES 5–60 g m⁻², observation-like upper quartile ≈45–60); CASS PBL depth (1.3 km at 11:30 and 1.8 km at 17:30 LST) and the hour of the cloud cover maximum (observed 13:30 LST); ARM97 and GATE III precipitable water.

Definitions (all EAMxx's own fields):
- Total cloud cover: EAMxx's `cldtot` (RRTMGP's cloud area over McICA subcolumns).
- Cloud fraction: `cldfrac_tot`. LWP: Σ qc·Δp/g. Precipitation: liquid plus ice surface mass per step.
- θ_l = θ − (θ/T)(L_v/c_p)·qc; q_t = qv + qc + qr.
- Heights are level midpoints above the surface. T and q errors are on the IOP file's pressure levels, with the model's hourly means interpolated in log p.

## Known differences from the published setups

These are E3SM's versions of the cases. They differ from the papers' setups:
- **Radiation is interactive** (RRTMGP) in every case. BOMEX and RICO LES prescribe radiative cooling. DYCOMS-RF01 LES use an idealized longwave formula and no sunlight. E3SM's DYCOMS case starts at 00 UTC, 16:00 local solar time, so the sun is up for the first 3.5 hours.
- **Surface fluxes:** BOMEX, DYCOMS-RF01, CASS and ARM97 use the IOP file's fluxes. RICO and GATE III use E3SM's coupler formula at the IOP sea surface temperature, not the LES bulk formula.
- **Aerosol and droplet number** come from SPA (prescribed aerosol), not the case's specified droplet number.
- **CASS** here uses E3SM's scmlib settings: start 06:00 LST, 14 h, wind nudging only (make_dp_case key `cass_scmlib`). The earlier `cass` runs used EAMxx's own test settings (start 10:58 LST, T/q nudging and Coriolis); they stay as they are for the bitwise tests.
- **ARM97** is judged on DP-SCREAM's 8-day window from 23 June 00 UTC (key `arm97_active`), run from that date. The DP-SCREAM paper used SCREAMv0, the Fortran predecessor of EAMxx with the same schemes, and a doubly periodic domain with a dycore.
- GATE III observed precipitation is the IOP file's `Prec` (derived from the Q2 budget).

## Not compared, and why

- **GABLS1.** Its surface is sea ice at 73° N with a prescribed cooling surface temperature and 0.1 m roughness. E3SM would take the fluxes from its sea-ice or land model. Neither is ported, and adding another scheme is not allowed.
- **RCE.** No observations or LES of this setup. E3SM's RCE uses a fixed solar constant of 551.58 W m⁻² with the model's own diurnal cycle, so RCEMIP's results do not apply directly. Its IOP file is not downloaded.
- **TWP-ICE.** EAMxx stops at step 1: the IOP file has no surface temperature.

## Seen before the criteria were written

To be clear about what could have influenced the bands:
- DP-EAMxx BOMEX (the dycore run, not the single column), 6 h: maximum layer cloud fraction 0.56, LWP 117 g m⁻², precipitation 0.24 mm/day in hour 6 (plan, section 4).
- Single-column runs made for the bitwise comparison: RICO surface fluxes 7 and 138 W m⁻² (sensible, latent); GATE III 48 h precipitation about 7 mm/day; ZM precipitation up to 19 mm/day (ARM97 LR) and 23 mm/day (GATE III LR). No cloud cover, LWP or profile from those runs was looked at against these targets.

## Results (24 September 2026)

### Verdict

- **The port matches EAMxx, bitwise, in all 12 validation runs** (libm math mode). Every output field was compared at every output step: 3,916 output steps over 27,436 model steps (every step, or hourly for the 100 s ARM97 and GATE III runs). So every number below is also EAMxx's.
- **EAMxx's physics as a single column does not match the observations and LES in most checks.** 13 of 40 checks pass.
- **Stratocumulus (DYCOMS-RF01): matches the observations** in both configurations.
- **Shallow cumulus (BOMEX, RICO, CASS): does not match.** Too much low cloud, and a cloud layer that drifts.
- **Deep convection (ARM97, GATE III): does not match** E3SM's own doubly periodic results. GATE III rain and temperature pass in both configurations; its humidity fails. ARM97 fails every check.
- **No verdict depends on the math mode.** libm and torch runs give the same verdict for every check.

The criteria file is unchanged since it was written (SHA-256 `fc47c562…`). After the results were seen, one thing was added to the runner, not to the criteria: EAMxx's low-cloud cover (`cldlow`, below 700 hPa) is now recorded for context.

### Checks

| Check | Band | 100 s | 1800 s + ZM |
|---|---|---|---|
| B1 BOMEX total cloud cover | 0.08–0.17 | 0.43 fail | 0.46 fail |
| B2 BOMEX maximum cloud fraction | 0.05–0.07 | 0.34 fail | 0.17 fail |
| B3 BOMEX height of that maximum | 500–700 m | 1250 m fail | 1670 m fail |
| B4 BOMEX LWP | 2–15 g m⁻² | 73 fail | 5.0 pass |
| B5 BOMEX RMS tendency θ_l, q_t | ≤ 1 K/day, ≤ 1 g/kg/day | 9.8, 9.4 fail | 3.8, 7.1 fail |
| R1 RICO total cloud cover | 0.10–0.30 | 0.35 fail | 0.36 fail |
| R2 RICO surface rain | 0–31.2 W m⁻² | 0.0 pass | 33.5 fail |
| D1 DYCOMS-RF01 total cloud cover | ≥ 0.95 | 1.00 pass | 1.00 pass |
| D2 DYCOMS-RF01 cloud top | 760–920 m | 796 pass | 796 pass |
| D3 DYCOMS-RF01 cloud base | 520–680 m | 545 pass | 545 pass |
| C1 CASS largest hourly cloud cover | 0.29–0.37 | 0.26 fail | 1.00 fail |
| C2 CASS cloud base, 13–14 LST | 1.7–2.1 km | 1.55 fail | 1.55 fail |
| C3 CASS largest hourly LWP | 23–35 g m⁻² | 65 fail | 33 pass |
| A1 ARM97 rain vs observed | ±10% | −30% fail | −16% fail |
| A2 ARM97 largest T error | ≤ 3.5 K | 9.8 fail | 6.5 fail |
| A3 ARM97 largest q error | ≤ 0.8 g/kg | 3.9 fail | 1.1 fail |
| G1 GATE III rain vs observed | ±15% | +11% pass | +10% pass |
| G2 GATE III largest T error | ≤ 5.5 K | 4.0 pass | 3.9 pass |
| G3 GATE III largest q error | ≤ 1.1 g/kg | 5.9 fail | 1.5 fail |

Values are from the libm runs. In the short cases the torch runs agree closely (cloud cover within 0.003). In the long ZM runs the two trajectories drift apart, as any convective run does: the largest T error moves by up to 2 K (GATE III 3.9 against 1.9 K, ARM97 6.5 against 4.9 K), the largest q error by up to 0.2 g/kg, rain by up to 2 percentage points. No verdict changes.

Reported without a verdict:

| Quantity | 100 s | 1800 s + ZM | Reference |
|---|---|---|---|
| BOMEX rain, h 3–6 | 0.0 mm/day | 1.4 mm/day (all ZM) | none in the case |
| BOMEX cloud top (cloud fraction ≥ 0.01) | 1660 m | 1670 m | ≈2000 m |
| RICO LWP, h 20–24 | 83 g m⁻² | 5.3 g m⁻² | ≈20 (LES) |
| RICO cloud top | 2720 m | 560 m | ≈2300–2500 m (LES) |
| RICO sensible, latent heat flux | 6.5, 128 W m⁻² | 19, 106 W m⁻² | coupler formula at the IOP SST |
| DYCOMS-RF01 LWP, h 2–4 | 65 g m⁻² | 56 g m⁻² | LES 5–60; observation-like upper quartile ≈45–60 |
| CASS hour of the cloud cover maximum | 16:30 LST | 16:30 LST | 13:30 observed |
| CASS largest hourly low-cloud cover | 0.25 | 0.20 | — |
| CASS PBL height, 11:30 and 17:30 LST | 1.95, 2.32 km | 1.93, 1.98 km | 1.3, 1.8 km |
| ARM97 rain, window total | 49 mm | 59 mm | 70 mm observed |
| ARM97 precipitable water, mean abs. error | 12.6 kg m⁻² | 5.8 kg m⁻² | DP-SCREAM up to ≈5 |
| ARM97 total cloud cover | 0.99 | 0.99 | 0.54 (GOES) |
| GATE III rain, total | 220 mm | 217 mm | 197 mm observed |
| GATE III precipitable water, mean abs. error | 26.6 kg m⁻² | 4.7 kg m⁻² | DP-SCREAM up to ≈5 |
| Daily rain, correlation with observed (ARM97, GATE III) | 0.96, 0.90 | 0.93, 0.88 | — |

Figure: `docs/figures/validation.png`.

### What the failures are

- **BOMEX, 100 s.** The cloud layer deepens and fills. LWP grows from 40 g m⁻² in the third hour to 81 in the sixth. The cloud fraction peaks at 0.34 near 1250 m; the LES peak is 0.06 near 600 m. Below 2.5 km, θ_l and q_t change by about 10 K and 9 g/kg per day; the LES change by about 1.
- **The same happens in EAMxx with its dycore.** EAMxx's own doubly periodic BOMEX run (the configuration the physics was tuned for) gives: maximum cloud fraction 0.37 at 1180 m, LWP 89 g m⁻², RMS tendencies 7.7 K/day and 10.4 g/kg/day (`validate_obs.py dp`). So the BOMEX failure belongs to EAMxx's physics at this resolution, not to the single-column setup. Bogenschutz et al. (2023) also found too much cloud and LWP for RICO at 3–5 km.
- **BOMEX, ZM.** LWP is within the LES band, but the cloud sits at 1670 m, and ZM rains 1.4 mm/day in a case without rain.
- **RICO.** Cloud cover is 0.35–0.36, all of it low cloud. At 100 s there is no surface rain (observed ≈21 W m⁻²), with LWP four times the LES. With ZM, ZM's rain is just above the LES maximum and the cloud stays below 600 m.
- **CASS.** At 100 s, cloud peaks three hours late (16:30 LST) and too low (0.26 against 0.33). Cloud base is too low and LWP twice the observed. SHOC's PBL is 0.5–0.65 km deeper than the observed mixed layer.
- **CASS, ZM: total cloud cover of 1.0 is an EAMxx diagnostic effect.** ZM leaves traces of ice at 10–12 km. EAMxx's `cld_fraction` sets the ice cloud fraction to 1 wherever qi > 1e-12 kg/kg (its default `ice_cloud_threshold`), and RRTMGP's cloud area counts any cloudy layer. Low cloud alone peaks at 0.20. The same effect puts ARM97's total cloud cover at 0.99 in both configurations (observed 0.54).
- **ARM97 and GATE III at 100 s.** This configuration has no deep-convection scheme. In DP-EAMxx the dycore resolves deep convection; a single column has only the prescribed large-scale ascent. So the lower troposphere keeps moistening: at 850 hPa in GATE III, from 13 to 18 g/kg in a week, and precipitable water is 27 kg m⁻² too high on average. ARM97's upper troposphere is on average up to 10 K too cold, and its lower troposphere up to 3.9 g/kg too moist. Running this configuration as a single column for deep convection is outside its design.
- **ARM97 and GATE III with ZM.** Errors are several times smaller: precipitable water off by 5.8 and 4.7 kg m⁻² on average, largest T error 6.5 and 3.9 K, largest q error 1.1 and 1.5 g/kg (ARM97, GATE III). They are still larger than DP-SCREAM's, except GATE III rain and T. ARM97 rain is 16% low.
- **Rain timing is good in both configurations.** Daily rain correlates with the observed at 0.88–0.96. In a column forced by observed advection, rain mostly follows the forcing.

### Context: EAMxx with its dycore (no verdict)

From the existing DP-EAMxx reference runs (50 km domain, 15-minute horizontal means; no `cldtot` in that output):

| Quantity | Single column, 100 s | DP-EAMxx, 100 s | DP-EAMxx-LR (ZM) | LES / observed |
|---|---|---|---|---|
| BOMEX maximum cloud fraction (height) | 0.34 (1250 m) | 0.37 (1180 m) | 0.14 (470 m) | 0.06 (≈600 m) |
| BOMEX LWP, h 3–6 | 73 g m⁻² | 89 g m⁻² | 4.7 g m⁻² | 2–15 |
| BOMEX RMS tendency θ_l, q_t | 9.8, 9.4 | 7.7, 10.4 | 3.3, 6.8 | ≈1, ≈1 |
| DYCOMS-RF01 cloud top, base | 796, 545 m | 796, 545 m | — | ≈840, ≈600 m |
| DYCOMS-RF01 LWP, h 2–4 | 65 g m⁻² | 64 g m⁻² | — | 45–60 (obs-like LES) |

### How to reproduce

```
python reference/scripts/make_scm_case.py CASE [--lr] --hours H [--every 36] --out reference/runs/val_NAME
# run scm_eamxx (strict build) inside reference/runs/val_NAME
python reference/scripts/validate_obs.py run reference/runs/val_NAME --math libm   # also compares with EAMxx
python reference/scripts/validate_obs.py run reference/runs/val_NAME --math torch
python reference/scripts/validate_obs.py score --json reference/runs/validation/results.json
python reference/scripts/validate_obs.py dp bomex reference/runs/dp_bomex_ne5/dp_bomex.horiz_avg.*.nc
python reference/scripts/plot_validation.py        # needs matplotlib
```

Cases and lengths: `bomex` 6 h, `rico` 24 h, `dycomsrf01` 6 h, `cass_scmlib` 14 h, `arm97_active` 192 h, `gateiii` 480 h (`--every 36` for the last two at 100 s). EAMxx needs 1–48 s per run; the port needs 8 s to 24 minutes.

## With the gradient options on (24 September 2026)

docs/differentiability.md adds two options that are not EAMxx behavior (state-free McICA seeds; SHOC energy-fixer tolerance). All 12 validation runs were repeated with both on (torch math), and the LES cases also with 4 more McICA seeds each, and scored with the same criteria file (unchanged, SHA-256 `fc47c562…`).

- **All 40 verdicts are the same as with EAMxx's behavior.**
- Values move, mostly because the random cloud sampling in radiation changes: in a single column EAMxx reuses one frozen draw for the whole run (its seeds come from pressures that never change there), while the option draws anew each step. In the 100 s configuration the EAMxx value is outside the spread of 5 draws for some quantities (BOMEX cloud cover 0.43 vs 0.46–0.49, LWP 73 vs 66–68 g m⁻², θ_l tendency 9.7 vs 7.5–8.0 K/day; RICO cloud cover 0.35 vs 0.37–0.39; CASS peak LWP 65 vs 54–58 g m⁻²), with the same verdicts.
- **In the ZM configuration some verdicts depend on the random draw:** BOMEX LWP passes in 4 of 5 draws, RICO cloud cover in 1 of 5, RICO rain in 1 of 5. These are marginal; the EAMxx-behavior verdicts reported above are single draws.
