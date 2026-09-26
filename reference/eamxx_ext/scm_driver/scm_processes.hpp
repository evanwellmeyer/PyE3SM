// Py-E3SM-EAMxx single-column reference driver: the host processes that stand in for
// the dycore and the coupler. Not part of E3SM.
//
//  scm_host     : what HOMME hands to physics in DP-EAMxx, without a dycore. The surface
//                 pressure stays at its initial value; layer thicknesses follow from the
//                 hybrid grid and it; wet and dry pressures are recomputed every step as
//                 HommeDynamics::update_pressure does; omega is the IOP omega.
//  scm_sfc_flux : open-ocean surface fluxes as E3SM's coupler would give them: EAMxx's
//                 exporter quantities (SurfaceCouplingExporter::compute_eamxx_exports),
//                 E3SM's shr_flux_atmOcn (scheme 0, explicit, as seq_flux_mct calls it),
//                 and the importer's conversions (sign -1 on the fluxes), for an ocean
//                 fraction of 1 at the IOP surface temperature Tg.
#ifndef SCM_PROCESSES_HPP
#define SCM_PROCESSES_HPP

#include "share/atm_process/atmosphere_process.hpp"

#include <ekat_parameter_list.hpp>

namespace scream {

class ScmHost : public AtmosphereProcess
{
public:
  ScmHost (const ekat::Comm& comm, const ekat::ParameterList& params);
  AtmosphereProcessType type () const override { return AtmosphereProcessType::Physics; }
  std::string name () const override { return "scm_host"; }
  void create_requests () override;

protected:
  void initialize_impl (const RunType run_type) override;
  void run_impl        (const double dt) override;
  void finalize_impl   () override {}
  void update_state (const util::TimeStamp& ts);

  std::shared_ptr<const AbstractGrid> m_grid;
  int m_num_cols;
  int m_num_levs;
};

class ScmSurfaceFlux : public AtmosphereProcess
{
public:
  ScmSurfaceFlux (const ekat::Comm& comm, const ekat::ParameterList& params);
  AtmosphereProcessType type () const override { return AtmosphereProcessType::Physics; }
  std::string name () const override { return "scm_sfc_flux"; }
  void create_requests () override;

protected:
  void initialize_impl (const RunType run_type) override;
  void run_impl        (const double dt) override;
  void finalize_impl   () override {}
  void compute_fluxes (const util::TimeStamp& ts);

  std::shared_ptr<const AbstractGrid> m_grid;
  int m_num_cols;
  int m_num_levs;
  Real m_minwind;
};

} // namespace scream

#endif
