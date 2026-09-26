# EAMxx standalone machine file for Apple Silicon (macOS), conda-forge toolchain.
# Written for this project; not part of E3SM. Modeled on
# components/eamxx/cmake/machine-files/linux-generic.cmake.
set(EKAT_MACH_FILES_PATH $ENV{E3SM_ROOT}/externals/ekat/cmake/machine-files)
set(SCREAM_MACHINE mac-arm64 CACHE STRING "")

# Kokkos: Serial backend (parallelism comes from MPI ranks).
include(${EKAT_MACH_FILES_PATH}/kokkos/generic.cmake)
include(${EKAT_MACH_FILES_PATH}/kokkos/serial.cmake)

set(EKAT_TEST_LAUNCHER_MANAGE_RESOURCES True CACHE BOOL "")
set(EKAT_MPIRUN_EXE "mpirun" CACHE STRING "")
set(EKAT_MPI_NP_FLAG "-n" CACHE STRING "")
set(EKAT_MPI_EXTRA_ARGS "--oversubscribe" CACHE STRING "")

set(CMAKE_C_COMPILER mpicc CACHE STRING "")
set(CMAKE_CXX_COMPILER mpicxx CACHE STRING "")
set(CMAKE_Fortran_COMPILER mpifort CACHE STRING "")
set(CMAKE_Fortran_FLAGS "-fallow-argument-mismatch -cpp" CACHE STRING "")

set(NetCDF_C_PATH $ENV{EAMXX_ENV_PREFIX} CACHE PATH "")
set(NetCDF_Fortran_PATH $ENV{EAMXX_ENV_PREFIX} CACHE PATH "")
set(BLAS_LIBRARIES $ENV{EAMXX_ENV_PREFIX}/lib/libblas.dylib CACHE STRING "")
set(LAPACK_LIBRARIES $ENV{EAMXX_ENV_PREFIX}/lib/liblapack.dylib CACHE STRING "")
set(SCREAM_INPUT_ROOT $ENV{SCREAM_INPUT_ROOT} CACHE PATH "")
set(Kokkos_ENABLE_DEPRECATED_CODE_4 OFF CACHE BOOL "")

# conda-forge libc++ hides newer C++17 features (std::from_chars for double)
# behind the macOS deployment target. We run on the build machine only, so
# disable the availability markup (conda-forge knowledge base recommendation).
set(CMAKE_CXX_FLAGS "-D_LIBCPP_DISABLE_AVAILABILITY" CACHE STRING "")
