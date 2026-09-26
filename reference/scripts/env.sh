# Source this to get the reference (EAMxx) build environment.
#   source reference/scripts/env.sh
# Uses an isolated conda env; nothing is installed system-wide.
REF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-${(%):-%x}}")/.." && pwd)"
export REF_ROOT
export E3SM_ROOT="$REF_ROOT/E3SM"
export EAMXX_SRC="$E3SM_ROOT/components/eamxx"
export SCREAM_INPUT_ROOT="$REF_ROOT/inputdata"
export EAMXX_ENV_PREFIX="${EAMXX_ENV_PREFIX:-$HOME/miniconda3/envs/eamxx-ref}"
source "$HOME/miniconda3/bin/activate" "$EAMXX_ENV_PREFIX"
export OMPI_MCA_rmaps_base_oversubscribe=1
export NETCDF_C_PATH="$EAMXX_ENV_PREFIX"
export NETCDF_FORTRAN_PATH="$EAMXX_ENV_PREFIX"
