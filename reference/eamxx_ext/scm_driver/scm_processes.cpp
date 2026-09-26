// Py-E3SM-EAMxx single-column reference driver: host processes (see scm_processes.hpp).
// The arithmetic is taken from EAMxx: HommeDynamics::update_pressure (pressures),
// SurfaceCouplingExporter::compute_eamxx_exports (lowest-level quantities for the
// coupler), atm_comp_mct/eamxx_cpl_indices.F90 (import signs). Not part of E3SM.
#include "scm_processes.hpp"

#include "share/physics/physics_constants.hpp"
#include "share/physics/eamxx_common_physics_functions.hpp"
#include "share/util/eamxx_column_ops.hpp"
#include "share/data_managers/IOPDataManager.hpp"

#include <ekat_team_policy_utils.hpp>
#include <ekat_pack_utils.hpp>
#include <ekat_assert.hpp>
#include <ekat_units.hpp>

#include <vector>

extern "C" {
// flux_bridge.F90: shr_flux_atmOcn as seq_flux_mct calls it for EAMxx (scheme 0, explicit)
void scm_shr_flux_atmocn(int n, double minwind,
                         const double* zbot, const double* ubot, const double* vbot,
                         const double* thbot, const double* qbot, const double* rbot,
                         const double* tbot, const double* us, const double* vs,
                         const double* ts,
                         double* sen, double* lat, double* lwup, double* evap,
                         double* taux, double* tauy);
}

namespace scream {

namespace {
using KT   = ekat::KokkosTypes<DefaultDevice>;
using Pack = ekat::Pack<Real,SCREAM_PACK_SIZE>;
using PC   = physics::Constants<Real>;
using PF   = PhysicsFunctions<DefaultDevice>;
using TPF  = ekat::TeamPolicyFactory<KT::ExeSpace>;
}

// =========================================================================================
ScmHost::ScmHost (const ekat::Comm& comm, const ekat::ParameterList& params)
  : AtmosphereProcess(comm, params) {}

void ScmHost::create_requests ()
{
  using namespace ekat::units;
  using namespace ShortFieldTagsNames;
  m_grid = m_grids_manager->get_grid("physics");
  const auto& gn = m_grid->name();
  m_num_cols = m_grid->get_num_local_dofs();
  m_num_levs = m_grid->get_num_vertical_levels();
  constexpr int ps = Pack::n;
  const auto s2d = m_grid->get_2d_scalar_layout();
  const auto mid = m_grid->get_3d_scalar_layout(LEV);
  const auto itf = m_grid->get_3d_scalar_layout(ILEV);
  add_field<Required>("ps", s2d, Pa, gn);
  add_tracer<Required>("qv", m_grid, kg/kg, ps);
  add_field<Computed>("pseudo_density",     mid, Pa, gn, ps);
  add_field<Computed>("pseudo_density_dry", mid, Pa, gn, ps);
  add_field<Computed>("p_int",     itf, Pa, gn, ps);
  add_field<Computed>("p_mid",     mid, Pa, gn, ps);
  add_field<Computed>("p_dry_int", itf, Pa, gn, ps);
  add_field<Computed>("p_dry_mid", mid, Pa, gn, ps);
  add_field<Computed>("omega",     mid, Pa/s, gn, ps);
}

void ScmHost::initialize_impl (const RunType)
{
  EKAT_REQUIRE_MSG(m_iop_data_manager, "Error! scm_host needs the IOP data manager (enable_iop).\n");
  update_state(start_of_step_ts());   // t0 at initialization
}

void ScmHost::run_impl (const double)
{
  update_state(start_of_step_ts());
}

void ScmHost::update_state (const util::TimeStamp& ts)
{
  using ColOps = ColumnOps<DefaultDevice,Real>;
  const int ncols = m_num_cols;
  const int nlevs = m_num_levs;
  const int npacks = ekat::npack<Pack>(nlevs);

  // Layer thickness from the hybrid grid and the (fixed) surface pressure, then
  // HommeDynamics::update_pressure: ps0 = hvcoord.ps0*hvcoord.hybrid_ai0
  const auto hyai = m_grid->get_geometry_data("hyai").get_view<const Real*>();
  const auto hybi = m_grid->get_geometry_data("hybi").get_view<const Real*>();
  const auto hyai_h = m_grid->get_geometry_data("hyai").get_view<const Real*,Host>();
  const Real P0 = PC::P0.value;
  const Real ps0 = P0 * hyai_h(0);

  const auto ps_v        = get_field_in("ps").get_view<const Real*>();
  const auto qv_view     = get_field_in("qv").get_view<const Pack**>();
  const auto dp_view     = get_field_out("pseudo_density").get_view<Pack**>();
  const auto p_int_view  = get_field_out("p_int").get_view<Pack**>();
  const auto p_mid_view  = get_field_out("p_mid").get_view<Pack**>();
  const auto dpd_view    = get_field_out("pseudo_density_dry").get_view<Pack**>();
  const auto pdi_view    = get_field_out("p_dry_int").get_view<Pack**>();
  const auto pdm_view    = get_field_out("p_dry_mid").get_view<Pack**>();

  const auto policy = TPF::get_thread_range_parallel_scan_team_policy(ncols, npacks);
  Kokkos::parallel_for(policy, KOKKOS_LAMBDA (const KT::MemberType& team) {
    const int icol = team.league_rank();
    auto dp     = ekat::subview(dp_view, icol);
    auto s_dp   = ekat::scalarize(dp);
    auto qv     = ekat::subview(qv_view, icol);
    auto p_int  = ekat::subview(p_int_view, icol);
    auto p_mid  = ekat::subview(p_mid_view, icol);
    auto dp_dry = ekat::subview(dpd_view, icol);
    auto p_dry_int = ekat::subview(pdi_view, icol);
    auto p_dry_mid = ekat::subview(pdm_view, icol);
    const Real ps = ps_v(icol);

    Kokkos::parallel_for(Kokkos::TeamVectorRange(team, nlevs), [&](const int k) {
      s_dp(k) = (hyai(k+1) - hyai(k))*P0 + (hybi(k+1) - hybi(k))*ps;
    });
    team.team_barrier();
    Kokkos::parallel_for(Kokkos::TeamVectorRange(team, npacks), [&](const int jpack) {
      dp_dry(jpack) = dp(jpack) * (1.0 - qv(jpack));
    });

    ColOps::column_scan<true>(team, nlevs, dp, p_int, ps0);
    team.team_barrier();
    ColOps::compute_midpoint_values(team, nlevs, p_int, p_mid);

    ColOps::column_scan<true>(team, nlevs, dp_dry, p_dry_int, ps0);
    team.team_barrier();
    ColOps::compute_midpoint_values(team, nlevs, p_dry_int, p_dry_mid);
  });

  // omega: the IOP large-scale vertical velocity (the dycore's in DP runs)
  auto omega = get_field_out("omega");
  if (m_iop_data_manager->has_iop_field("omega")) {
    m_iop_data_manager->read_iop_file_data(ts);
    const auto om_iop = m_iop_data_manager->get_iop_field("omega").get_view<const Real*>();
    const auto om = omega.get_view<Real**>();
    Kokkos::parallel_for(KT::RangePolicy(0, ncols*nlevs), KOKKOS_LAMBDA (const int idx) {
      const int i = idx / nlevs;
      const int k = idx % nlevs;
      om(i,k) = om_iop(k);
    });
  } else {
    omega.deep_copy(0);
  }
  Kokkos::fence();
}

// =========================================================================================
ScmSurfaceFlux::ScmSurfaceFlux (const ekat::Comm& comm, const ekat::ParameterList& params)
  : AtmosphereProcess(comm, params)
{
  m_minwind = m_params.get<double>("seq_flux_atmocn_minwind", 0.5);
}

void ScmSurfaceFlux::create_requests ()
{
  using namespace ekat::units;
  using namespace ShortFieldTagsNames;
  m_grid = m_grids_manager->get_grid("physics");
  const auto& gn = m_grid->name();
  m_num_cols = m_grid->get_num_local_dofs();
  m_num_levs = m_grid->get_num_vertical_levels();
  constexpr int ps = Pack::n;
  const auto s2d = m_grid->get_2d_scalar_layout();
  const auto v2d = m_grid->get_2d_vector_layout(2);
  const auto mid = m_grid->get_3d_scalar_layout(LEV);
  const auto itf = m_grid->get_3d_scalar_layout(ILEV);
  const auto v3d = m_grid->get_3d_vector_layout(LEV, 2);
  add_field<Required>("T_mid", mid, K, gn, ps);
  add_field<Required>("p_mid", mid, Pa, gn, ps);
  add_field<Required>("p_int", itf, Pa, gn, ps);
  add_field<Required>("pseudo_density", mid, Pa, gn, ps);
  add_tracer<Required>("qv", m_grid, kg/kg, ps);
  add_field<Required>("horiz_winds", v3d, m/s, gn, ps);
  add_field<Computed>("surf_sens_flux",   s2d, W/(m*m), gn);
  add_field<Computed>("surf_evap",        s2d, kg/(m*m)/s, gn);
  add_field<Computed>("surf_mom_flux",    v2d, N/(m*m), gn);
  add_field<Computed>("surf_lw_flux_up",  s2d, W/(m*m), gn);
  add_field<Computed>("surf_radiative_T", s2d, K, gn);
}

void ScmSurfaceFlux::initialize_impl (const RunType)
{
  EKAT_REQUIRE_MSG(m_iop_data_manager and m_iop_data_manager->has_iop_field("Tg"),
                   "Error! scm_sfc_flux needs the IOP surface temperature Tg.\n");
  compute_fluxes(start_of_step_ts());   // t0 at initialization
}

void ScmSurfaceFlux::run_impl (const double)
{
  compute_fluxes(start_of_step_ts());
}

void ScmSurfaceFlux::compute_fluxes (const util::TimeStamp& ts)
{
  const int ncols = m_num_cols;
  const int nlevs = m_num_levs;
  const int nlev_packs  = ekat::npack<Pack>(nlevs);
  const int nlevi_packs = ekat::npack<Pack>(nlevs+1);

  const auto p_int          = get_field_in("p_int").get_view<const Real**>();
  const auto pseudo_density = get_field_in("pseudo_density").get_view<const Pack**>();
  const auto qv             = get_field_in("qv").get_view<const Pack**>();
  const auto T_mid          = get_field_in("T_mid").get_view<const Pack**>();
  const auto horiz_winds    = get_field_in("horiz_winds").get_view<const Real***>();
  const auto p_mid          = get_field_in("p_mid").get_view<const Pack**>();

  KT::view_2d<Pack> dz("dz", ncols, nlev_packs), z_mid("z_mid", ncols, nlev_packs),
                    z_int("z_int", ncols, nlevi_packs);
  KT::view_1d<Real> Sa_z("Sa_z", ncols), Sa_u("Sa_u", ncols), Sa_v("Sa_v", ncols),
                    Sa_tbot("Sa_tbot", ncols), Sa_ptem("Sa_ptem", ncols),
                    Sa_shum("Sa_shum", ncols), Sa_dens("Sa_dens", ncols);

  // SurfaceCouplingExporter::compute_eamxx_exports, the quantities shr_flux_atmOcn uses
  const auto policy = TPF::get_thread_range_parallel_scan_team_policy(ncols, nlevs);
  Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const KT::MemberType& team) {
    const int i = team.league_rank();
    const auto qv_i             = ekat::subview(qv, i);
    const auto T_mid_i          = ekat::subview(T_mid, i);
    const auto p_mid_i          = ekat::subview(p_mid, i);
    const auto pseudo_density_i = ekat::subview(pseudo_density, i);
    const auto dz_i             = ekat::subview(dz, i);
    const auto s_p_mid_i = ekat::scalarize(p_mid_i);
    const auto s_T_mid_i = ekat::scalarize(T_mid_i);
    const auto z_int_i = ekat::subview(z_int, i);
    const auto z_mid_i = ekat::subview(z_mid, i);

    PF::calculate_dz(team, pseudo_density_i, p_mid_i, T_mid_i, qv_i, dz_i);
    team.team_barrier();
    const Real z_surf = 0.0;
    PF::calculate_z_int(team, nlevs, dz_i, z_surf, z_int_i);
    team.team_barrier();
    PF::calculate_z_mid(team, nlevs, z_int_i, z_mid_i);
    team.team_barrier();

    Kokkos::single(Kokkos::PerTeam(team), [&] {
      const auto s_z_mid_i = ekat::scalarize(z_mid_i);
      Sa_z(i) = s_z_mid_i(nlevs-1);
      Sa_u(i) = horiz_winds(i,0,nlevs-1);
      Sa_v(i) = horiz_winds(i,1,nlevs-1);
      Sa_tbot(i) = s_T_mid_i(nlevs-1);
      Sa_ptem(i) = s_T_mid_i(nlevs-1) / pow( s_p_mid_i(nlevs-1)/p_int(i,nlevs),
                                             PC::RD.value*PC::INV_CP.value);
      const auto s_qv_i = ekat::scalarize(qv_i);
      Sa_shum(i) = s_qv_i(nlevs-1);
      const auto s_dz_i = ekat::scalarize(dz_i);
      const auto s_pseudo_density_i = ekat::scalarize(pseudo_density_i);
      Sa_dens(i) = PF::calculate_density(s_pseudo_density_i(nlevs-1), s_dz_i(nlevs-1));
    });
  });
  Kokkos::fence();

  // Ocean state: at rest, at the IOP surface temperature
  m_iop_data_manager->read_iop_file_data(ts);
  auto tg = m_iop_data_manager->get_iop_field("Tg");
  tg.sync_to_host();
  const Real Tg = tg.get_view<const Real,Host>()();

  // The coupler works in double whatever the atmosphere's precision (the exporter converts);
  // in a double build these copies are exact.
  auto h = [](const KT::view_1d<Real>& v) {
    auto m = Kokkos::create_mirror_view(v); Kokkos::deep_copy(m, v);
    return std::vector<double>(m.data(), m.data() + m.size()); };
  auto zbot = h(Sa_z), ubot = h(Sa_u), vbot = h(Sa_v), tbot = h(Sa_tbot), thbot = h(Sa_ptem),
       qbot = h(Sa_shum), rbot = h(Sa_dens);
  std::vector<double> us(ncols, 0.0), vs(ncols, 0.0), tsv(ncols, static_cast<double>(Tg));
  std::vector<double> sen(ncols), lat(ncols), lwup(ncols), evap(ncols), taux(ncols), tauy(ncols);
  scm_shr_flux_atmocn(ncols, m_minwind, zbot.data(), ubot.data(), vbot.data(), thbot.data(),
                      qbot.data(), rbot.data(), tbot.data(), us.data(), vs.data(), tsv.data(),
                      sen.data(), lat.data(), lwup.data(), evap.data(), taux.data(), tauy.data());

  // Coupler merge with an ocean fraction of 1 (x2a = xao*ofrac), then the importer's
  // constant multiples (-1 for lwup, taux, tauy, sen, evap); Sx_t = So_t*ofrac.
  const Real ofrac = 1.0;
  auto sens_h = get_field_out("surf_sens_flux").get_view<Real*,Host>();
  auto evap_h = get_field_out("surf_evap").get_view<Real*,Host>();
  auto mom_h  = get_field_out("surf_mom_flux").get_view<Real**,Host>();
  auto lwup_h = get_field_out("surf_lw_flux_up").get_view<Real*,Host>();
  auto trad_h = get_field_out("surf_radiative_T").get_view<Real*,Host>();
  for (int i=0; i<ncols; ++i) {
    sens_h(i)   = -1.0*(sen[i]*ofrac);
    evap_h(i)   = -1.0*(evap[i]*ofrac);
    mom_h(i,0)  = -1.0*(taux[i]*ofrac);
    mom_h(i,1)  = -1.0*(tauy[i]*ofrac);
    lwup_h(i)   = -1.0*(lwup[i]*ofrac);
    trad_h(i)   = Tg*ofrac;
  }
  for (const auto& n : {"surf_sens_flux","surf_evap","surf_mom_flux","surf_lw_flux_up","surf_radiative_T"})
    get_field_out(n).sync_to_dev();
}

} // namespace scream
