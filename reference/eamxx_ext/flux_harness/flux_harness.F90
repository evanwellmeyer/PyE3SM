! Reference harness: E3SM's shr_flux_atmOcn (share/util/shr_flux_mod.F90, unmodified),
! called as the MCT coupler calls it for an EAMxx atmosphere (driver-mct/main/
! seq_flux_mct.F90, seq_flux_atmocn_mct): ocn_surface_flux_scheme = 0,
! ATM_FLUX_INTEGRATION_METHOD = explicit (no wsresp/tau_est), ATM_SUPPLIES_GUSTINESS =
! FALSE for SCREAM (no ugust), no waves, flux_convergence = 0, flux_max_iteration = -1
! -> 2, coldair_outbreak_mod = false, seq_flux_atmocn_minwind from the input.
!
! Usage: flux_harness IN OUT
!   IN  (stream, native doubles): n, minwind, then n values each of
!       zbot ubot vbot thbot qbot rbot tbot us vs ts mask
!   OUT (stream, native doubles): n values each of
!       sen lat lwup evap taux tauy tref qref duu10n u10res ustar re ssq
! Py-E3SM-EAMxx reference harness; not part of E3SM.
program flux_harness
  use shr_kind_mod, only: R8 => SHR_KIND_R8, IN => SHR_KIND_IN
  use shr_flux_mod, only: shr_flux_atmOcn, shr_flux_adjust_constants
  implicit none
  character(len=1024) :: fin, fout
  real(R8) :: rn, minwind
  integer(IN) :: n
  real(R8), allocatable :: zbot(:), ubot(:), vbot(:), thbot(:), qbot(:), rbot(:), tbot(:)
  real(R8), allocatable :: us(:), vs(:), ts(:), rmask(:)
  integer(IN), allocatable :: mask(:)
  real(R8), allocatable :: s16O(:), sHDO(:), s18O(:), r16O(:), rHDO(:), r18O(:)
  real(R8), allocatable :: sen(:), lat(:), lwup(:), evap(:), e16O(:), eHDO(:), e18O(:)
  real(R8), allocatable :: taux(:), tauy(:), tref(:), qref(:), duu10n(:), u10res(:)
  real(R8), allocatable :: ustar(:), re(:), ssq(:)
  integer :: u

  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  open(newunit=u, file=trim(fin), access='stream', form='unformatted', status='old')
  read(u) rn, minwind
  n = nint(rn)
  allocate(zbot(n), ubot(n), vbot(n), thbot(n), qbot(n), rbot(n), tbot(n), us(n), vs(n), &
           ts(n), rmask(n), mask(n))
  read(u) zbot, ubot, vbot, thbot, qbot, rbot, tbot, us, vs, ts, rmask
  close(u)
  mask = nint(rmask)
  allocate(s16O(n), sHDO(n), s18O(n), r16O(n), rHDO(n), r18O(n))
  s16O = qbot; sHDO = qbot; s18O = qbot; r16O = 1.0_R8; rHDO = 1.0_R8; r18O = 1.0_R8
  allocate(sen(n), lat(n), lwup(n), evap(n), e16O(n), eHDO(n), e18O(n), taux(n), tauy(n), &
           tref(n), qref(n), duu10n(n), u10res(n), ustar(n), re(n), ssq(n))

  call shr_flux_adjust_constants(flux_convergence_tolerance=0.0_R8, &
       flux_convergence_max_iteration=2, coldair_outbreak_mod=.false.)
  call shr_flux_atmOcn(n, zbot, ubot, vbot, thbot, qbot, s16O, sHDO, s18O, rbot, tbot, us, vs, &
       ts, mask, minwind, sen, lat, lwup, r16O, rHDO, r18O, evap, e16O, eHDO, e18O, &
       taux, tauy, tref, qref, 0, duu10n, u10res, ustar, re, ssq)

  open(newunit=u, file=trim(fout), access='stream', form='unformatted', status='replace')
  write(u) sen, lat, lwup, evap, taux, tauy, tref, qref, duu10n, u10res, ustar, re, ssq
  close(u)
end program flux_harness
