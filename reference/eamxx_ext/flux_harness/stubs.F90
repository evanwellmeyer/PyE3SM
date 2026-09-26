! Stubs for the E3SM share/driver modules that shr_flux_mod.F90 uses but that do not
! enter the ocean flux calculation (logging, abort, the wave-coupling flag).
! Py-E3SM-EAMxx reference harness; not part of E3SM.
module shr_log_mod
  use shr_kind_mod, only: SHR_KIND_IN, SHR_KIND_CX
  use, intrinsic :: iso_fortran_env, only: output_unit
  implicit none
  integer(SHR_KIND_IN) :: shr_log_Level = 0
  integer(SHR_KIND_IN) :: shr_log_Unit  = output_unit
contains
  pure function shr_log_errMsg(file, line)
    character(len=SHR_KIND_CX) :: shr_log_errMsg
    character(len=*), intent(in) :: file
    integer, intent(in) :: line
    write(shr_log_errMsg, '(a,":",i0)') file, line
  end function shr_log_errMsg
end module shr_log_mod

module shr_sys_mod
  implicit none
contains
  subroutine shr_sys_abort(string)
    character(len=*), intent(in), optional :: string
    if (present(string)) write(*,*) 'shr_sys_abort: ', trim(string)
    error stop 1
  end subroutine shr_sys_abort
  subroutine shr_sys_flush(unit)
    integer, intent(in) :: unit
    flush(unit)
  end subroutine shr_sys_flush
end module shr_sys_mod

module seq_flds_mod
  implicit none
  ! driver-mct/shr/seq_flds_mod.F90 default (no wave-atmosphere coupling)
  character(len=80) :: wav_atm_coup = 'none'
end module seq_flds_mod
