# Py-E3SM-EAMxx single-column reference driver (not part of E3SM): EAMxx physics and
# IOP forcing on a point grid, no dycore; optional open-ocean fluxes from E3SM's own
# share/util/shr_flux_mod.F90 (compiled here, unmodified, as in the flux harness).
include (ScreamUtils)
if (NOT SCREAM_DOUBLE_PRECISION)   # needs IOP forcing, RRTMGP, TMS, ZM: double only in EAMxx
  return()
endif()

set (E3SM_SHARE_UTIL ${SCREAM_BASE_DIR}/../../share/util)
set (SCM_FLUX_SRC
  ${CMAKE_CURRENT_SOURCE_DIR}/seq_flds_stub.F90
  ${E3SM_SHARE_UTIL}/water_isotopes.F90
  ${E3SM_SHARE_UTIL}/shr_flux_mod.F90
  ${CMAKE_CURRENT_SOURCE_DIR}/flux_bridge.F90)
add_library(scm_flux ${SCM_FLUX_SRC})
# Same flags as reference/scripts/build_flux_harness.sh (strict): the bitwise flux reference
set_source_files_properties(${SCM_FLUX_SRC} PROPERTIES
  COMPILE_OPTIONS "-O;-ffree-line-length-none;-ffp-contract=off")
set_target_properties(scm_flux PROPERTIES Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/scm_flux_mod)
target_include_directories(scm_flux PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/scm_flux_mod)
target_link_libraries(scm_flux PUBLIC csm_share)

CreateUnitTestExec(scm_eamxx SOURCES scm_driver.cpp scm_processes.cpp
  LIBS scream_control eamxx_io eamxx_diagnostics
       tms shoc cld_fraction spa p3 zm iop_forcing scream_rrtmgp scm_flux)
