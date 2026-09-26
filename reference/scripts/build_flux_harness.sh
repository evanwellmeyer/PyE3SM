#!/bin/bash
# Build the ocean surface flux reference harness from E3SM's own share/util sources.
#   bash reference/scripts/build_flux_harness.sh      -> reference/build/flux_harness/{strict,release}/flux_harness
# strict: -O -ffp-contract=off (no FMA contraction, the per-scheme bitwise reference);
# release: -O (E3SM's gnu.cmake release flag; the compiler may contract a*b+c into FMA).
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"
set -euo pipefail
SRC="$E3SM_ROOT/share/util"
EXT="$REF_ROOT/eamxx_ext/flux_harness"
for variant in strict release; do
  OUT="$REF_ROOT/build/flux_harness/$variant"
  rm -rf "$OUT"; mkdir -p "$OUT"; cd "$OUT"
  FLAGS="-O -ffree-line-length-none"
  [ "$variant" = strict ] && FLAGS="$FLAGS -ffp-contract=off"
  for f in "$SRC/shr_kind_mod.F90" "$SRC/shr_const_mod.F90" "$EXT/stubs.F90" \
           "$SRC/water_isotopes.F90" "$SRC/shr_flux_mod.F90" "$EXT/flux_harness.F90"; do
    gfortran $FLAGS -c "$f"
  done
  gfortran $FLAGS -o flux_harness *.o
  echo "$variant: $OUT/flux_harness"
done
