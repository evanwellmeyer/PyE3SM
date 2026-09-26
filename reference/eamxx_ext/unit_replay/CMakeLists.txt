# Unit-level replay harnesses: call EAMxx physics functions directly on inputs
# read from a raw file, and write the outputs. Used for synthetic edge cases
# that never occur in model dumps (values at thresholds, NaN, ...).
include (ScreamUtils)

CreateUnitTestExec(cld_fraction_replay SOURCES cld_fraction_replay.cpp
  LIBS cld_fraction)

CreateUnitTestExec(interval_check_replay SOURCES interval_check_replay.cpp
  LIBS scream_share)

CreateUnitTestExec(tms_replay SOURCES tms_replay.cpp
  LIBS tms scream_share eamxx_physics_share)

# P3 non-ice lookup tables in the build's precision (single-precision reference build)
CreateUnitTestExec(p3_tables_write SOURCES p3_tables_write.cpp
  LIBS p3 scream_share)
