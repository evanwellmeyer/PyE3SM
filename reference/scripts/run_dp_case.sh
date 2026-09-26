#!/bin/bash
# Run a DP-EAMxx IOP case with the standalone reference build.
#   bash reference/scripts/run_dp_case.sh bomex [nranks] [extra make_dp_case.py args...]
# Optional per-process dumps (see patches/0002): set EAMXX_REF_DUMP_DIR etc.
# REF_VARIANT=release (default) or strict selects the reference build.
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"
set -euo pipefail
CASE="$1"; NP="${2:-1}"; shift; shift || true
EXE="$REF_ROOT/build/eamxx-${REF_VARIANT:-release}/tests/multi-process/dynamics_physics/ref_extra/dp_case_l128/dp_eamxx_l128"
RUNDIR="$(python "$HERE/make_dp_case.py" "$CASE" "$@")"
cd "$RUNDIR"
echo "run dir: $RUNDIR"
mpirun -n "$NP" "$EXE" > run.log 2>&1
grep -q "All tests passed" run.log && echo "OK: $(grep -c Iteration run.log) steps" || { echo "FAILED, see $RUNDIR/run.log"; exit 1; }
