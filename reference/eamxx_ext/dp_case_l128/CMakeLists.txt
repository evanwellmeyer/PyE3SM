# Out-of-tree target for this project (not part of E3SM).
# Builds the stock DP-EAMxx standalone driver
#   tests/multi-process/dynamics_physics/homme_shoc_cld_spa_p3_rrtmgp_pg2_dp/*.cpp
# against a 128-level HOMME dynamics library, so that DP-EAMxx IOP cases can be
# run at L128 (the level count used by every EAMxx compset).
# The executable reads input.yaml and namelist.nl from its working directory.
include (ScreamUtils)

CreateDynamicsLib("theta-l_kokkos"  4   128   10)

set (DP_DRIVER_SRC
  ${SCREAM_BASE_DIR}/tests/multi-process/dynamics_physics/homme_shoc_cld_spa_p3_rrtmgp_pg2_dp/homme_shoc_cld_spa_p3_rrtmgp_pg2_dp.cpp)

CreateUnitTestExec(dp_eamxx_l128 SOURCES ${DP_DRIVER_SRC}
  LIBS cld_fraction tms shoc spa iop_forcing p3 zm scream_rrtmgp ${dynLibName} scream_control)
