"""Physical constants of EAMxx.

Translation of components/eamxx/src/share/physics/physics_constants.hpp at
E3SM 80fe0a71 (struct Constants<Real>, Real = double). Derived constants are
evaluated with the same operations in the same order, in IEEE double, so they
equal the C++ constexpr values bit for bit. Quantity<> arithmetic in EAMxx only
multiplies/divides the numeric values (units are bookkeeping), so plain floats
reproduce it.
"""
from __future__ import annotations

import sys

ZERO = 0.0
ONE = 1.0
THIRD = 1.0 / 3.0
SXTH = 1.0 / 6.0
Pi = 3.14159265358979323
PIOV3 = Pi * THIRD
PIOV6 = Pi * SXTH
macheps = sys.float_info.epsilon

Cpair = 1004.64
CP = Cpair
INV_CP = ONE / CP
Rair = 287.042
RD = Rair
RH2O = 461.505
RV = RH2O
RHO_H2O = 1000.0
INV_RHO_H2O = 1 / RHO_H2O
RHOW = RHO_H2O
INV_RHOW = 1 / RHOW
RhoIce = 917.0
MWH2O = 18.016
MWWV = MWH2O
MWdry = 28.966
ep_2 = MWH2O / MWdry
o2mmr = 0.23143
gravit = 9.80616
LatVap = 2501000.0
LatIce = 333700.0
CpLiq = 4188.0
Tmelt = 273.15
T_zerodegc = Tmelt
T_homogfrz = Tmelt - 40
T_rainfrz = Tmelt - 4

RHO_RIMEMIN = 50.0
RHO_RIMEMAX = 900.0
INV_RHO_RIMEMAX = 1.0 / RHO_RIMEMAX
BIMM = 2.0
CONS1 = PIOV6 * RHOW
CONS2 = 4. * PIOV3 * RHOW
CONS3 = 1.0 / (CONS2 * 1.562500000000000e-14)
CONS5 = PIOV6 * BIMM
CONS6 = PIOV6 * PIOV6 * RHOW * BIMM
CONS7 = 4. * PIOV3 * RHOW * 1.e-18
QSMALL = 1.e-14
QTENDSMALL = 1e-20
BSMALL = 1.e-15
NSMALL = 1.e-16
P0 = 100000.0
RHOSUR = P0 / (RD * Tmelt)
rhosui = 60000 / (RD * 253.15)
RHO_1000MB = P0 / (RD * Tmelt)
RHO_600MB = 60000 / (RD * 253.15)
dt_left_tol = 1.e-4
bcn = 2.
dropmass = 5.2e-7
NCCNST = 200.0e+6
incloud_limit = 5.1e-3
precip_limit = 1.0e-2
Karman = 0.4
Avogad = 6.02214e26
Boltz = 1.38065e-23
Rgas = Avogad * Boltz
RWV = Rgas / MWWV
ZVIR = (RWV / Rair) - 1.0
f1r = 0.78
f2r = 0.32
nmltratio = 1.0
basetemp = 300.0
r_earth = 6.376e6
stebol = 5.670374419e-8
omega = 7.292e-5
VTABLE_DIM0 = 300
VTABLE_DIM1 = 10
MU_R_TABLE_DIM = 150
orocnst = 1.0
z0fac = 0.075
IPARAM = 3
earth_ellipsoid1 = 111132.92
earth_ellipsoid2 = 559.82
earth_ellipsoid3 = 1.175
tau_pert_mag = 0.001

GAS_MOL_WEIGHT = {  # Constants::get_gas_mol_weight
    "h2o": MWH2O, "co2": 44.0095, "o3": 47.9982, "n2o": 44.0128, "co": 28.0101,
    "ch4": 16.04246, "o2": 31.998, "n2": 28.0134, "cfc11": 136., "cfc12": 120.,
}
