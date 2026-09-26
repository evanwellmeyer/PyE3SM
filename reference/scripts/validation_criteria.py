"""Pass criteria for the suite-level comparison with observations and LES.

Written 23 September 2026, before any of the validation runs were made. docs/validation.md
states the rules, the sources and what had already been seen, and records this file's
SHA-256 so later edits are visible. Do not change a band after seeing results.

Each case lists
  run:     the make_scm_case.py case key (settings from E3SM's scmlib DP scripts)
  hours:   run length
  checks:  (id, quantity, window, lo, hi, source)   -> pass if lo <= value <= hi
  reported:(id, quantity, window, context)          -> reported, no verdict

Quantities are computed by validate_obs.py from the single-column run:
  cldtot       EAMxx's total cloud area (RRTMGP, McICA subcolumns), time mean
  cf_max       maximum over levels of the time-mean cloud fraction (cldfrac_tot)
  z_cf_max     height above the surface of that maximum [m]
  lwp          liquid water path sum(qc*dp/g), time mean [g m-2]
  precip       surface precipitation (liquid + ice), time mean [mm day-1]
  precip_wm2   the same in energy units, 28.5 W m-2 per mm day-1 (vanZanten et al. 2011, Table 1)
  rms_dthl     dp-weighted RMS over levels below 2500 m of the window-mean tendency of
               theta_l [K day-1]; rms_dqt likewise for q_t = qv + qc + qr [g kg-1 day-1]
  z_top_cf50, z_base_cf50   highest / lowest level where the time-mean cloud fraction >= 0.5 [m]
  z_top_cf01, z_base_cf01   the same with threshold 0.01 [m]
  cldtot_hmax, lwp_hmax     largest hourly mean over the run; cldtot_hmax_time its hour [LST]
  pbl_height   SHOC's PBL height at the given local times [m]
  shf_lhf      surface sensible and latent heat flux, time mean [W m-2]
  precip_rel   (model - observed) / observed, precipitation accumulated over the window
  t_bias_max, q_bias_max    largest |time-mean model - observed| on the IOP file's pressure
               levels between the surface and 100 hPa [K], [g kg-1]
  pw_mae       mean absolute error of hourly precipitable water against the IOP profiles [kg m-2]
Heights are midpoint heights of model levels above the surface.
Windows are hours after the start of the run, except "lst" windows (local standard time).
"""

CRITERIA = {
    # Siebesma et al. (2003), J. Atmos. Sci. 60, 1201-1219: 10 LES, 6 h, statistics over hours 3-6.
    "bomex": dict(
        run="bomex", hours=6,
        checks=[
            ("B1", "cldtot", (3, 6), 0.08, 0.17,
             "p. 1204: hours 3-6 mean total cloud cover ranges from 8% (UCLA) to 17% (NCAR); ensemble 13%"),
            ("B2", "cf_max", (3, 6), 0.05, 0.07,
             "p. 1208 and Fig. 6: cloud fraction maximum near cloud base about 0.06, 2-sigma band about "
             "+-0.005 (read from the figure), rounded outward"),
            ("B3", "z_cf_max", (3, 6), 500.0, 700.0,
             "Fig. 6: maximum near 600 m; p. 1207: cloud base about 500 m (read from the figure, rounded outward)"),
            ("B4", "lwp", (3, 6), 2.0, 15.0,
             "Fig. 2b: LES LWP 2-sigma band between hours 3 and 6 (read from the figure, rounded outward)"),
            ("B5a", "rms_dthl", (3, 6), 0.0, 1.0,
             "p. 1204: total tendencies over the last 3 h typically of the order 1 K/day and 1 g/kg/day"),
            ("B5b", "rms_dqt", (3, 6), 0.0, 1.0, "as B5a"),
        ],
        reported=[
            ("B6", "precip", (3, 6), "the LES case has no precipitation"),
            ("B7", "z_top_cf01", (3, 6), "LES cloud fraction reaches 0 near 2000 m (Fig. 6)"),
        ],
    ),
    # van Zanten et al. (2011), JAMES 3, M06001: 13 LES, 24 h, statistics over the last 4 h.
    "rico": dict(
        run="rico", hours=24,
        checks=[
            ("R1", "cldtot", (20, 24), 0.10, 0.30,
             "p. 12: cloud cover 'almost certainly between 0.1 and 0.3'; lidar 0.17, LES median 0.19"),
            ("R2", "precip_wm2", (20, 24), 0.0, 31.2,
             "Table 3: LES surface precipitation over the last 4 h ranges from 0.0 to 31.2 W m-2 "
             "(0 to 1.09 mm/day); observed about 21 W m-2 for the undisturbed period"),
        ],
        reported=[
            ("R3", "lwp", (20, 24), "LES quasi-steady near 20 g m-2 (Bogenschutz et al. 2023, Fig. 6)"),
            ("R4", "z_top_cf01", (20, 24), "LES cloud tops near 2300-2500 m (van Zanten et al. 2011, p. 9)"),
            ("R5", "shf_lhf", (20, 24), "surface fluxes; E3SM's coupler formula, not the LES bulk formula"),
        ],
    ),
    # Stevens et al. (2005), Mon. Wea. Rev. 133, 1443-1462: observations and 16 LES, 4 h;
    # analysis over hours 2-4. The paper judges the LES against the observations, so the
    # bands are the observations.
    "dycomsrf01": dict(
        run="dycomsrf01", hours=6,
        checks=[
            ("D1", "cldtot", (2, 4), 0.95, 1.0,
             "p. 1448: observed cloud fraction greater than 99% throughout the night "
             "(0.95 allows for McICA sampling noise in EAMxx's diagnostic)"),
            ("D2", "z_top_cf50", (2, 4), 760.0, 920.0,
             "Fig. 3: observed cloud top about 840 m; band +-80 m (about two model layers)"),
            ("D3", "z_base_cf50", (2, 4), 520.0, 680.0,
             "Fig. 3: observed cloud base about 600 m, descending 2 m/h; band +-80 m"),
        ],
        reported=[
            ("D4", "lwp", (2, 4), "LES full range about 5-60 g m-2, upper quartile (closest to the "
             "observations) about 45-60 (Fig. 2)"),
        ],
    ),
    # Zhang et al. (2017), J. Atmos. Sci. 74, 3229-3251 (CASS): observed composite of 76 days
    # and two LES. Local standard time = UTC - 6 h (the IOP fluxes peak at 18 UTC, noon LST).
    "cass": dict(
        run="cass_scmlib", hours=14,
        checks=[
            ("C1", "cldtot_hmax", "run", 0.29, 0.37,
             "Fig. 7a: observed total projected cloud fraction peaks at about 33% at 13:30 LST, "
             "1 standard error about 2%; band +-2 standard errors"),
            ("C2", "z_base_cf01", ("lst", 13, 14), 1700.0, 2100.0,
             "Fig. 7c: observed mean cloud base about 1.9 km at 13:30 LST; band +-0.2 km"),
            ("C3", "lwp_hmax", "run", 23.0, 35.0,
             "Fig. 7e: observed LWP peaks at about 29 g m-2, 1 standard error about 3; band +-2 standard errors"),
        ],
        reported=[
            ("C4", "pbl_height", ("lst", 11.5, 17.5), "mixed-layer depth 1.3 km at 11:30 LST and "
             "1.8 km at 17:30 LST, observed and LES (p. 15)"),
            ("C5", "cldtot_hmax_time", "run", "observed maximum at 13:30 LST; LES 11:30-13:30"),
        ],
    ),
    # Observations in E3SM's IOP file (ARM variational analysis). Bands: the largest error
    # E3SM's own doubly periodic SCREAM (same physics, with a dycore, 0.5-5 km) shows for
    # this case in Bogenschutz et al. (2023), JAMES 15, e2022MS003466, rounded outward. Same
    # 8-day window from 23 June 1997, first 3 h removed.
    "arm97": dict(
        run="arm97_active", hours=8 * 24,
        checks=[
            ("A1", "precip_rel", (3, 192), -0.10, 0.10,
             "Fig. 4a: DP-SCREAM accumulated precipitation 65-70 mm vs observed about 70 mm (-7% to 0)"),
            ("A2", "t_bias_max", (3, 192), 0.0, 3.5,
             "Fig. 3a: DP-SCREAM time-mean T bias within about -3.2 to +2.5 K"),
            ("A3", "q_bias_max", (3, 192), 0.0, 0.8,
             "Fig. 3b: DP-SCREAM time-mean q bias within about -0.4 to +0.72 g/kg"),
        ],
        reported=[
            ("A4", "pw_mae", (3, 192), "DP-SCREAM too moist most of the run, by up to about 5 kg m-2 (Fig. 1a)"),
            ("A5", "cldtot", (3, 192), "observed GOES total cloud (IOP file, totcld)"),
        ],
    ),
    "gateiii": dict(
        run="gateiii", hours=20 * 24,
        checks=[
            ("G1", "precip_rel", (3, 480), -0.15, 0.15,
             "Fig. 4a: DP-SCREAM accumulated precipitation 250-255 mm vs observed about 222 mm (+13% to +15%)"),
            ("G2", "t_bias_max", (3, 480), 0.0, 5.5,
             "Fig. 3c: DP-SCREAM time-mean T bias within about -1 to +5.2 K"),
            ("G3", "q_bias_max", (3, 480), 0.0, 1.1,
             "Fig. 3d: DP-SCREAM time-mean q bias within about -1.0 to +0.8 g/kg"),
        ],
        reported=[
            ("G4", "pw_mae", (3, 480), "DP-SCREAM wet bias up to about 5 kg m-2 in the last 5 days (Fig. 1b)"),
        ],
    ),
}

# Both configurations are judged with the same bands: DP-EAMxx's (100 s, SHOC cell 3.33 km)
# and EAMxx-LR's (ZM, 1800 s, 12 sub-cycles, SHOC cell 111 km).
CONFIGS = ("standard", "lr")
