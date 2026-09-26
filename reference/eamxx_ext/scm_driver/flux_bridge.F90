! C bridge to E3SM's shr_flux_atmOcn for the single-column reference driver, with the
! arguments seq_flux_mct passes for an EAMxx atmosphere: ocn_surface_flux_scheme = 0,
! explicit stress (no wsresp/tau_est), no gustiness (ATM_SUPPLIES_GUSTINESS = FALSE for
! SCREAM), flux_convergence = 0, flux_max_iteration -1 -> 2, no cold-air outbreak.
! Same call as reference/eamxx_ext/flux_harness/flux_harness.F90. Not part of E3SM.
subroutine scm_shr_flux_atmocn(n, minwind, zbot, ubot, vbot, thbot, qbot, rbot, tbot, us, vs, &
                               ts, sen, lat, lwup, evap, taux, tauy) bind(C, name="scm_shr_flux_atmocn")
  use iso_c_binding, only: c_int, c_double
  use shr_kind_mod, only: R8 => SHR_KIND_R8, IN => SHR_KIND_IN
  use shr_flux_mod, only: shr_flux_atmOcn, shr_flux_adjust_constants
  implicit none
  integer(c_int), value :: n
  real(c_double), value :: minwind
  real(c_double), intent(in) :: zbot(n), ubot(n), vbot(n), thbot(n), qbot(n), rbot(n), tbot(n)
  real(c_double), intent(in) :: us(n), vs(n), ts(n)
  real(c_double), intent(out) :: sen(n), lat(n), lwup(n), evap(n), taux(n), tauy(n)
  integer(IN) :: mask(n)
  real(R8) :: s16O(n), sHDO(n), s18O(n), r16O(n), rHDO(n), r18O(n), e16O(n), eHDO(n), e18O(n)
  real(R8) :: tref(n), qref(n), duu10n(n), u10res(n), ustar(n), re(n), ssq(n)
  mask = 1
  s16O = qbot; sHDO = qbot; s18O = qbot; r16O = 1.0_R8; rHDO = 1.0_R8; r18O = 1.0_R8
  call shr_flux_adjust_constants(flux_convergence_tolerance=0.0_R8, &
       flux_convergence_max_iteration=2, coldair_outbreak_mod=.false.)
  call shr_flux_atmOcn(int(n, IN), zbot, ubot, vbot, thbot, qbot, s16O, sHDO, s18O, rbot, tbot, &
       us, vs, ts, mask, minwind, sen, lat, lwup, r16O, rHDO, r18O, evap, e16O, eHDO, e18O, &
       taux, tauy, tref, qref, 0, duu10n, u10res, ustar, re, ssq)
end subroutine scm_shr_flux_atmocn
