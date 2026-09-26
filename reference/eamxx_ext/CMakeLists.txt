# Py-E3SM-EAMxx out-of-tree reference targets (not part of E3SM).
if ("${SCREAM_DYNAMICS_DYCORE}" STREQUAL "HOMME")   # needs HOMME (off in single precision)
  add_subdirectory(dp_case_l128)
endif()
add_subdirectory(unit_replay)
add_subdirectory(physics_standalone)
add_subdirectory(scm_driver)
