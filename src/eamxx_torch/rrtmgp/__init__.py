"""RRTMGP radiation, as EAMxx runs it (RTE+RRTMGP C++/Kokkos, McICA, zenith angle).

Source: components/eamxx/src/physics/rrtmgp/ and the rte-rrtmgp submodule at
E3SM 80fe0a71. The coefficient and cloud-optics files are E3SM input data, read
at run time (load_rrtmgp).
"""
from .data import CloudOpticsData, GasOpticsData, RRTMGPData, load_rrtmgp
from .interface import RRTMGPRuntime, invariant_checks, radiation_do, rrtmgp_main, rrtmgp_run

__all__ = ["CloudOpticsData", "GasOpticsData", "RRTMGPData", "RRTMGPRuntime", "invariant_checks",
           "load_rrtmgp", "radiation_do", "rrtmgp_main", "rrtmgp_run"]
