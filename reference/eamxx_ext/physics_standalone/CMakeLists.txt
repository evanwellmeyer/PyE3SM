# Standalone physics runs: EAMxx's generic atmosphere-driver test program
# (tests/eamxx_ad_test.cpp) linked against every physics library we port, so
# any process list can run on a point grid from an initial-condition file,
# without a dycore. Used to exercise branches that never occur in the DP dumps
# (e.g. nonzero surface stress, TMS drag over land) with synthetic inputs.
# Run from a directory holding input.yaml; the dump hook works as in DP runs.
include (ScreamUtils)

# EAMxx builds RRTMGP, TMS, ZM and IOP forcing in double precision only; a single-
# precision build links the processes that exist there (SHOC, P3, cld_fraction, SPA).
if (SCREAM_DOUBLE_PRECISION)
  set (STANDALONE_PHYSICS tms shoc cld_fraction spa p3 zm scream_rrtmgp)
else()
  set (STANDALONE_PHYSICS shoc cld_fraction spa p3)
endif()
CreateUnitTestExec(physics_standalone SOURCES ${EAMXX_AD_TEST_CPP}
  LIBS scream_control eamxx_io eamxx_diagnostics ${STANDALONE_PHYSICS})
