#!/bin/bash
# Run a short DP-EAMxx case with per-process dumps at chosen steps.
#   [REF_VARIANT=strict|release] bash reference/scripts/make_dumps.sh CASE HOURS STEPS [NP] [NCOLS] [TAG]
# Uses a 5x5-element, 50 km domain for every case (cheap). Domain size does not
# affect per-scheme replay: the port is fed exactly the dumped inputs.
# Output: reference/runs/dumps_<variant>_<case>[_<tag>]/dump/<proc>/...  (never reuses another run's dir)
# REF_VARIANT (default strict) picks the reference build: strict = -ffp-contract=off.
# LR=1 runs the EAMxx-LR configuration instead (ZM on, 1800 s, make_dp_case.py --lr and
# its 5x5-element, 333 km domain); DUMP_PROCS limits the dumped processes.
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"
set -euo pipefail
CASE="$1"; HOURS="$2"; STEPS="$3"; NP="${4:-4}"; NCOLS="${5:-4}"; TAG="${6:-}"
VARIANT="${REF_VARIANT:-strict}"
EXE="$REF_ROOT/build/eamxx-$VARIANT/tests/multi-process/dynamics_physics/ref_extra/dp_case_l128/dp_eamxx_l128"
OUT="$REF_ROOT/runs/dumps_${VARIANT}_$CASE${TAG:+_$TAG}"
if [ "${LR:-0}" = 1 ]; then
  python "$HERE/make_dp_case.py" "$CASE" --lr --hours "$HOURS" --out "$OUT" >/dev/null
else
  python "$HERE/make_dp_case.py" "$CASE" --ne 5 --length 50000 --hours "$HOURS" --out "$OUT" >/dev/null
fi
cd "$OUT"
rm -rf dump
if [ -n "${DUMP_PROCS:-}" ]; then export EAMXX_REF_DUMP_PROCS="$DUMP_PROCS"; fi
EAMXX_REF_DUMP_DIR="$OUT/dump" EAMXX_REF_DUMP_STEPS="$STEPS" EAMXX_REF_DUMP_NCOLS="$NCOLS" \
  mpirun -n "$NP" "$EXE" > run.log 2>&1
grep -q "All tests passed" run.log && echo "$CASE OK: $(grep -c Iteration run.log) steps, dump $(du -sh dump | cut -f1)" \
  || { echo "$CASE FAILED, see $OUT/run.log"; exit 1; }
