# EAMxx reference

The unmodified EAMxx physics, built standalone, used as the ground truth for the
port. Pinned commits, patches and known differences from a CIME run are in
[docs/plan.md](../docs/plan.md), section 4.

## Layout

| Path | What | In git? |
|---|---|---|
| `scripts/setup_reference.sh` | Clone E3SM and scmlib at the pinned commits, fetch submodules, apply patches | yes |
| `scripts/env.sh` | Activate the `eamxx-ref` toolchain env | yes |
| `scripts/fetch_inputdata.sh` | Download the 546 MB of E3SM input data used | yes |
| `scripts/configure_eamxx.sh` | CMake configure (Release, double, HOMME on, MAM off) | yes |
| `scripts/make_dp_case.py` | Write a DP-EAMxx case directory (`input.yaml`, `namelist.nl`, output yaml) | yes |
| `scripts/run_dp_case.sh` | Make a case and run it | yes |
| `scripts/read_dump.py` | Read per-process dumps (numpy only) | yes |
| `machine/mac-arm64.cmake` | EAMxx machine file for this Mac | yes |
| `eamxx_ext/dp_case_l128/` | Out-of-tree target: E3SM's DP driver at 128 levels | yes |
| `patches/` | Our three patches to the E3SM clone | yes |
| `environment-ref.yml` | conda-forge toolchain recipe | yes |
| `E3SM/`, `scmlib/` | Upstream clones | no |
| `inputdata/` | E3SM input data | no |
| `build/` | Build trees | no |
| `runs/` | Case run directories and dumps | no |
| `data/` | Saved reference data for replay (large files ignored) | small files only |

## Quick start

```bash
bash reference/scripts/setup_reference.sh
source reference/scripts/env.sh
bash reference/scripts/fetch_inputdata.sh
bash reference/scripts/configure_eamxx.sh
make -C reference/build/eamxx-release -j14 dp_eamxx_l128
bash reference/scripts/run_dp_case.sh bomex 1
```

## Dumps

```bash
cd reference/runs/dp_bomex_ne5
EAMXX_REF_DUMP_DIR=$PWD/dump EAMXX_REF_DUMP_STEPS=1,2 EAMXX_REF_DUMP_NCOLS=4 \
  mpirun -n 1 ../../build/eamxx-release/tests/multi-process/dynamics_physics/ref_extra/dp_case_l128/dp_eamxx_l128
python ../../scripts/read_dump.py dump/shoc 1
```
