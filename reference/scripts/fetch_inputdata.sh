#!/bin/bash
# Download the E3SM input data needed by the EAMxx reference build and the
# DP-EAMxx case runs. Source: E3SM public inputdata server (same files CIME
# would fetch). Total is about 0.55 GB. Files already present are skipped.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DEST="${SCREAM_INPUT_ROOT:-$HERE/../inputdata}"
SERVER=https://web.lcrc.anl.gov/public/e3sm/inputdata
FILES=(
  # Standalone EAMxx unit/process tests (non-MAM)
  atm/cam/scam/iop/CASS_iopfile_4scam.nc
  atm/cam/topo/USGS-gtopo30_ne4np4pg2_16x_converted.c20200527.nc
  atm/cam/topo/USGS-gtopo30_ne2np4pg2_x6t_20230331.nc
  atm/scream/init/rrtmgp-allsky.nc
  atm/scream/init/rrtmgp-cloud-optics-coeffs-lw.nc
  atm/scream/init/rrtmgp-cloud-optics-coeffs-sw.nc
  atm/scream/init/rrtmgp-data-lw-g128-210809.nc
  atm/scream/init/rrtmgp-data-sw-g112-210809.nc
  atm/scream/init/rrtmgp-data-lw-g256-2018-12-04.nc
  atm/scream/init/rrtmgp-data-sw-g224-2018-12-04.nc
  atm/scream/init/spa_file_unified_and_complete_ne2np4L72_20231222.nc
  atm/scream/init/spa_file_unified_and_complete_ne4_20220428.nc
  atm/scream/init/spa_init_ne2np4.nc
  atm/scream/maps/map_ne4np4_to_ne2np4_mono.nc
  atm/scream/maps/map_ne30pg2_to_20x40_20260112.nc
  atm/scream/tables/p3_lookup_table_1.dat-v4.1.1
  atm/scream/tables/mu_r_table_vals_v2.dat8
  atm/scream/tables/revap_table_vals_v2.dat8
  atm/scream/tables/vn_table_vals_v2.dat8
  atm/scream/tables/vm_table_vals_v2.dat8
  atm/scream/init/screami_unit_tests_ne2np4L72_20260813.nc
  atm/scream/init/screami_unit_tests_ne2np4L128_20260813.nc
  # DP-EAMxx case runs (L128 grid, ne30 IC used only for non-IOP fields)
  atm/scream/init/vertical_coordinates_L128_20220927.nc
  atm/scream/init/screami_ne30np4L128_20260813.nc
  atm/cam/topo/USGS-gtopo30_ne30np4pg2_x6t-SGH.c20210614.nc
  atm/cam/scam/iop/BOMEX_iopfile_4scam.nc
  atm/cam/scam/iop/RICO_iopfile_4scam.nc
  atm/cam/scam/iop/DYCOMSrf01_iopfile_4scam.nc
  atm/cam/scam/iop/GABLS_iopfile_4scam.nc
  atm/cam/scam/iop/ARM97_iopfile_4scam.nc
  atm/cam/scam/iop/GATEIII_iopfile_4scam.nc
  atm/cam/scam/iop/TWP06_iopfile_4scam.nc
)
for f in "${FILES[@]}"; do
  out="$DEST/$f"
  if [[ -s "$out" ]]; then continue; fi
  mkdir -p "$(dirname "$out")"
  echo "fetch $f"
  curl -sSfL --retry 3 -o "$out.part" "$SERVER/$f" && mv "$out.part" "$out"
done
du -sh "$DEST"
