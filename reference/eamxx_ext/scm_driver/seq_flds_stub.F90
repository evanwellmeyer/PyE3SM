! The one variable of driver-mct/shr/seq_flds_mod.F90 that shr_flux_mod.F90 uses, at its
! default (no wave-atmosphere coupling). Py-E3SM-EAMxx reference driver; not part of E3SM.
module seq_flds_mod
  implicit none
  character(len=80) :: wav_atm_coup = 'none'
end module seq_flds_mod
