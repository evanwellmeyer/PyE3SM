// Py-E3SM-EAMxx single-column reference driver (not part of E3SM).
//
// EAMxx physics and IOP forcing on a point grid with no dycore:
//   eamxx: [sc_import, scm_host, (scm_sfc_flux), physics]
// scm_host stands in for HOMME (pressures from the fixed surface pressure, IOP omega);
// scm_sfc_flux, when listed, gives open-ocean fluxes as E3SM's coupler would. The
// surface import is set up as in the DP-EAMxx test driver
// (homme_shoc_cld_spa_p3_rrtmgp_pg2_dp.cpp): the four IOP surface fields are imported
// as NaN and overwritten from the IOP file (iop_srf_prop), or imported as 0 and
// overwritten by scm_sfc_flux.
// SHOC's cell length is the "dx_short" geometry (km) of a planar DP grid, given as
// scm_driver: dx_short in input.yaml.
#include "catch2/catch.hpp"

#include "control/atmosphere_driver.hpp"
#include "physics/register_physics.hpp"
#include "share/diagnostics/register_diagnostics.hpp"
#include "control/register_surface_coupling.hpp"
#include "control/atmosphere_surface_coupling_importer.hpp"
#include "share/data_managers/mesh_free_grids_manager.hpp"
#include "share/atm_process/atmosphere_process.hpp"

#include "scm_processes.hpp"

#include <ekat_kokkos_types.hpp>
#include <ekat_yaml.hpp>

#include <algorithm>
#include <cstring>
#include <numeric>

TEST_CASE("scm_eamxx", "scm_eamxx") {
  using namespace scream;
  using namespace scream::control;

  ekat::Comm atm_comm (MPI_COMM_WORLD);

  std::string fname = "input.yaml";
  ekat::ParameterList ad_params("Atmosphere Driver");
  parse_yaml_file(fname,ad_params);
  ad_params.print();

  const auto& ts     = ad_params.sublist("time_stepping");
  const auto  dt     = ts.get<int>("time_step");
  const auto  nsteps = ts.get<int>("number_of_steps");
  const auto  t0_str = ts.get<std::string>("run_t0");
  const auto  t0     = util::str_to_time_stamp(t0_str);
  const auto  dx_short = ad_params.sublist("scm_driver").get<double>("dx_short");

  register_physics();
  register_diagnostics();
  register_surface_coupling();
  register_mesh_free_grids_manager();
  auto& proc_factory = AtmosphereProcessFactory::instance();
  proc_factory.register_product("scm_host",&create_atmosphere_process<ScmHost>);
  proc_factory.register_product("scm_sfc_flux",&create_atmosphere_process<ScmSurfaceFlux>);

  AtmosphereDriver ad;
  ad.set_comm(atm_comm);
  ad.set_params(ad_params);
  ad.init_scorpio ();
  ad.init_time_stamps (t0, t0);
  ad.create_output_managers ();
  ad.create_atm_processes ();
  ad.create_grids ();

  // Planar cell size, as HOMME's DP grids manager provides it (homme_grids_manager.cpp)
  {
    auto gm = ad.get_grids_manager();
    auto phys_grid = gm->get_grid_nonconst("physics");
    auto dx_short_f = phys_grid->create_geometry_data("dx_short",FieldLayout::scalar(),ekat::units::rad);
    dx_short_f.get_view<Real,Host>()() = dx_short;
    dx_short_f.sync_to_dev();
  }

  ad.create_fields ();

  const int ncols = ad.get_grids_manager()->get_grid("physics")->get_num_local_dofs();
  static constexpr int num_imports = 4;
  char import_names[num_imports][32];
  std::strcpy(import_names[0], "surf_radiative_T");
  std::strcpy(import_names[1], "surf_lw_flux_up");
  std::strcpy(import_names[2], "surf_sens_flux");
  std::strcpy(import_names[3], "surf_evap");
  // NaN as in the DP driver, overwritten from the IOP file; with open-ocean fluxes the
  // placeholder is 0 (the importer's NaN check runs before scm_sfc_flux overwrites them)
  const auto procs = ad_params.sublist("eamxx").get<std::vector<std::string>>("atm_procs_list");
  const bool ocean = std::find(procs.begin(), procs.end(), "scm_sfc_flux") != procs.end();
  KokkosTypes<HostDevice>::view_2d<Real> import_data("import_data",ncols, num_imports);
  Kokkos::deep_copy(import_data, ocean ? 0.0 : std::nan(""));
  KokkosTypes<HostDevice>::view_1d<int> import_cpl_indices("import_cpl_indices", num_imports);
  std::iota(import_cpl_indices.data(), import_cpl_indices.data()+num_imports, 0);
  KokkosTypes<HostDevice>::view_1d<int> import_vec_comps("import_vec_comps", num_imports);
  Kokkos::deep_copy(import_vec_comps, -1);
  KokkosTypes<HostDevice>::view_1d<Real> import_constant_multiple("import_constant_multiple", num_imports);
  Kokkos::deep_copy(import_constant_multiple, 1);
  KokkosTypes<HostDevice>::view_1d<bool> do_import_during_init("do_import_during_init", num_imports);
  Kokkos::deep_copy(do_import_during_init, false);
  do_import_during_init(2) = true; do_import_during_init(3) = true;

  ad.setup_surface_coupling_data_manager(SurfaceCouplingTransferType::Import,
                                         4, 4, ncols, import_data.data(), import_names[0], import_cpl_indices.data(),
                                         import_vec_comps.data(), import_constant_multiple.data(), do_import_during_init.data());
  ad.initialize_fields ();
  ad.initialize_atm_procs ();
  ad.reset_accumulated_fields();
  ad.initialize_output_managers ();

  for (int i=0; i<nsteps; ++i) {
    ad.run(dt);
    if (atm_comm.am_i_root()) {
      std::cout << "  - Iteration " << i+1 << " completed\n";
    }
  }
  ad.finalize();
  REQUIRE (true);
}
