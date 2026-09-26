// Replay harness for the TMS process (physics/tms).
// Reproduces TurbulentMountainStress::run_impl verbatim (its preprocessing calls
// PhysicsFunctions exner_function, calculate_dz, calculate_z_int, calculate_z_mid,
// then TMSFunctions::compute_tms), on columns read from a file. Used because
// the DP driver never sets landfrac (a coupler import), so model runs only
// exercise ksrf = ... * landfrac = 0.
// Input (REPLAY_IN, float64): ncols, nlevs, then per column:
//   u[nlev], v[nlev], T_mid[nlev], p_mid[nlev], pseudo_density[nlev], qv[nlev], sgh30, landfrac
// Output (REPLAY_OUT, float64): per column: ksrf, tau_x, tau_y, exner[nlev], z_mid[nlev]
#include "catch2/catch.hpp"
#include "physics/tms/tms_functions.hpp"
#include "share/physics/eamxx_common_physics_functions.hpp"
#include "share/core/eamxx_types.hpp"

#include <ekat_pack_kokkos.hpp>
#include <ekat_team_policy_utils.hpp>

#include <cstdlib>
#include <fstream>
#include <vector>

TEST_CASE("tms_replay") {
  using namespace scream;
  using TMSFunctions = tms::Functions<Real,DefaultDevice>;
  using PF   = PhysicsFunctions<DefaultDevice>;
  using Pack = ekat::Pack<Real,SCREAM_PACK_SIZE>;
  using KT   = TMSFunctions::KT;
  using TPF  = ekat::TeamPolicyFactory<KT::ExeSpace>;

  std::ifstream in(std::getenv("REPLAY_IN"),std::ios::binary);
  std::ofstream out(std::getenv("REPLAY_OUT"),std::ios::binary);
  auto rd = [&]() { double x; in.read(reinterpret_cast<char*>(&x),sizeof(x)); return x; };
  auto wr = [&](double x) { out.write(reinterpret_cast<const char*>(&x),sizeof(x)); };

  const int ncols = static_cast<int>(rd()), nlevs = static_cast<int>(rd());
  const int nlev_packs = ekat::npack<Pack>(nlevs), nlevi_packs = ekat::npack<Pack>(nlevs+1);

  KT::view_3d<Pack> horiz_winds("hw",ncols,2,nlev_packs);
  KT::view_2d<Pack> T_mid("T",ncols,nlev_packs), p_mid("p",ncols,nlev_packs),
                    pseudo_density("dp",ncols,nlev_packs), qv("qv",ncols,nlev_packs),
                    exner("ex",ncols,nlev_packs), dz("dz",ncols,nlev_packs),
                    z_mid("zm",ncols,nlev_packs), z_int("zi",ncols,nlevi_packs);
  KT::view_1d<Real> sgh30("sgh",ncols), landfrac("lf",ncols), ksrf("ksrf",ncols);
  KT::view_2d<Real> tau("tau",ncols,2);
  auto h_hw = Kokkos::create_mirror_view(horiz_winds);
  auto h_T = Kokkos::create_mirror_view(T_mid);
  auto h_p = Kokkos::create_mirror_view(p_mid);
  auto h_dp = Kokkos::create_mirror_view(pseudo_density);
  auto h_qv = Kokkos::create_mirror_view(qv);
  auto h_sgh = Kokkos::create_mirror_view(sgh30);
  auto h_lf = Kokkos::create_mirror_view(landfrac);
  for (int i=0; i<ncols; ++i) {
    for (int c=0; c<2; ++c) for (int k=0; k<nlevs; ++k) h_hw(i,c,k/Pack::n)[k%Pack::n] = rd();
    for (auto v : {&h_T,&h_p,&h_dp,&h_qv})
      for (int k=0; k<nlevs; ++k) (*v)(i,k/Pack::n)[k%Pack::n] = rd();
    h_sgh(i) = rd(); h_lf(i) = rd();
  }
  REQUIRE (in.good());
  Kokkos::deep_copy(horiz_winds,h_hw); Kokkos::deep_copy(T_mid,h_T); Kokkos::deep_copy(p_mid,h_p);
  Kokkos::deep_copy(pseudo_density,h_dp); Kokkos::deep_copy(qv,h_qv);
  Kokkos::deep_copy(sgh30,h_sgh); Kokkos::deep_copy(landfrac,h_lf);

  // ---- Verbatim from TurbulentMountainStress::run_impl ----
  const auto scan_policy = TPF::get_thread_range_parallel_scan_team_policy(ncols, nlev_packs);
  Kokkos::parallel_for(scan_policy, KOKKOS_LAMBDA (const KT::MemberType& team) {
    const int i = team.league_rank();
    const auto p_mid_i = ekat::subview(p_mid, i);
    const auto exner_i = ekat::subview(exner, i);
    const auto pseudo_density_i = ekat::subview(pseudo_density, i);
    const auto T_mid_i = ekat::subview(T_mid, i);
    const auto qv_i = ekat::subview(qv, i);
    const auto dz_i = ekat::subview(dz, i);
    const auto z_int_i = ekat::subview(z_int, i);
    const auto z_mid_i = ekat::subview(z_mid, i);
    PF::exner_function<Pack>(team, p_mid_i, exner_i);
    PF::calculate_dz(team, pseudo_density_i, p_mid_i, T_mid_i, qv_i, dz_i);
    const Real z_surf = 0.0;
    team.team_barrier();
    PF::calculate_z_int(team, nlevs, dz_i, z_surf, z_int_i);
    team.team_barrier();
    PF::calculate_z_mid(team, nlevs, z_int_i, z_mid_i);
  });
  TMSFunctions::compute_tms(ncols, nlevs,
                            ekat::scalarize(horiz_winds), ekat::scalarize(T_mid),
                            ekat::scalarize(p_mid), ekat::scalarize(exner),
                            ekat::scalarize(z_mid), sgh30, landfrac, ksrf, tau);
  // ---- end verbatim ----

  auto h_ksrf = Kokkos::create_mirror_view(ksrf); Kokkos::deep_copy(h_ksrf,ksrf);
  auto h_tau = Kokkos::create_mirror_view(tau);   Kokkos::deep_copy(h_tau,tau);
  auto h_ex = Kokkos::create_mirror_view(exner);  Kokkos::deep_copy(h_ex,exner);
  auto h_zm = Kokkos::create_mirror_view(z_mid);  Kokkos::deep_copy(h_zm,z_mid);
  for (int i=0; i<ncols; ++i) {
    wr(h_ksrf(i)); wr(h_tau(i,0)); wr(h_tau(i,1));
    for (int k=0; k<nlevs; ++k) wr(h_ex(i,k/Pack::n)[k%Pack::n]);
    for (int k=0; k<nlevs; ++k) wr(h_zm(i,k/Pack::n)[k%Pack::n]);
  }
  REQUIRE (out.good());
}
