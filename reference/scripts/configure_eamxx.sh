#!/bin/bash
# Configure the standalone EAMxx reference build (Release, double precision,
# HOMME dycore for DP-EAMxx, MAM off). Run from anywhere:
#   bash reference/scripts/configure_eamxx.sh [release|strict|strict16]
#   release : production CPU build: -ffp-contract=fast (compiler may fuse a*b+c
#             into FMA), SIMD packs of 16 levels (summation order depends on packs)
#   strict  : validation reference: -ffp-contract=off (every operation rounded
#             as written) and pack size 1 (EAMxx's GPU layout: sequential sums)
#   strict16: -ffp-contract=off with packs of 16 (separates the two effects)
#   strict-sp: strict, single precision (Real = float). EAMxx turns HOMME off in single
#             precision, so only the dycore-free targets exist (scm_eamxx, standalone).
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"   # conda activation is not "set -u" safe
set -euo pipefail
VARIANT="${1:-release}"
DP=ON
DYN=HOMME
case "$VARIANT" in
  release)  BUILD="$REF_ROOT/build/eamxx-release";  FPMODEL=precise; PACK=16 ;;
  strict)   BUILD="$REF_ROOT/build/eamxx-strict";   FPMODEL=strict;  PACK=1  ;;
  strict16) BUILD="$REF_ROOT/build/eamxx-strict16"; FPMODEL=strict;  PACK=16 ;;
  strict-sp) BUILD="$REF_ROOT/build/eamxx-strict-sp"; FPMODEL=strict; PACK=1; DP=OFF; DYN=NONE ;;
  *) echo "unknown variant $VARIANT"; exit 1 ;;
esac
mkdir -p "$BUILD"
cd "$BUILD"
# EKAT refuses MPI on Apple, but only on re-configure (its check precedes the
# option). Clear the cached value so re-configures behave like the first one.
cmake -U EKAT_ENABLE_MPI -C "$REF_ROOT/machine/mac-arm64.cmake" \
  -DCMAKE_BUILD_TYPE=Release \
  -DSCREAM_DOUBLE_PRECISION=$DP \
  -DSCREAM_FPMODEL=$FPMODEL \
  -DSCREAM_PACK_SIZE=$PACK \
  -DSCREAM_DYNAMICS_DYCORE=$DYN \
  -DSCREAM_ENABLE_MAM=OFF \
  -DSCREAM_ENABLE_BASELINE_TESTS=OFF \
  -DSCREAM_TEST_MAX_TOTAL_THREADS=16 \
  -DSCREAM_TEST_MAX_RANKS=4 \
  -DEAMXX_ENABLE_PYTHON=OFF \
  -DEAMXX_REF_EXTRA_DIR="$REF_ROOT/eamxx_ext" \
  "$EAMXX_SRC" 2>&1 | tee "$BUILD/configure.log"
