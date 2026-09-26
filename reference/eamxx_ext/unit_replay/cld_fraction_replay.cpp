// Replay harness for CldFractionFunctions::main (physics/cld_fraction).
// Input  (REPLAY_IN,  float64, little endian):
//   nj, nk, ice_threshold, ice_4out_threshold, qi[nj*nk], liq[nj*nk]
// Output (REPLAY_OUT, float64):
//   ice[nj*nk], tot[nj*nk], ice_4out[nj*nk], tot_4out[nj*nk]
#include "catch2/catch.hpp"
#include "physics/cld_fraction/cld_fraction_functions.hpp"
#include "share/core/eamxx_types.hpp"

#include <ekat_pack_kokkos.hpp>

#include <cstdlib>
#include <fstream>
#include <vector>

TEST_CASE("cld_fraction_replay") {
  using namespace scream;
  using CF   = cld_fraction::CldFractionFunctions<Real,DefaultDevice>;
  using Pack = CF::Pack;
  using view2 = CF::view_2d<Pack>;

  const char* fin  = std::getenv("REPLAY_IN");
  const char* fout = std::getenv("REPLAY_OUT");
  REQUIRE (fin!=nullptr);
  REQUIRE (fout!=nullptr);

  std::ifstream in(fin,std::ios::binary);
  double hdr[4];
  in.read(reinterpret_cast<char*>(hdr),sizeof(hdr));
  const int nj = static_cast<int>(hdr[0]);
  const int nk = static_cast<int>(hdr[1]);
  const Real thr = hdr[2], thr4 = hdr[3];
  std::vector<double> qi(nj*nk), liq(nj*nk);
  in.read(reinterpret_cast<char*>(qi.data()),qi.size()*sizeof(double));
  in.read(reinterpret_cast<char*>(liq.data()),liq.size()*sizeof(double));
  REQUIRE (in.good());

  const int npack = ekat::npack<Pack>(nk);
  view2 v_qi("qi",nj,npack), v_liq("liq",nj,npack),
        v_ice("ice",nj,npack), v_tot("tot",nj,npack),
        v_ice4("ice4",nj,npack), v_tot4("tot4",nj,npack);
  auto h_qi  = Kokkos::create_mirror_view(v_qi);
  auto h_liq = Kokkos::create_mirror_view(v_liq);
  for (int j=0; j<nj; ++j) for (int k=0; k<nk; ++k) {
    h_qi (j,k/Pack::n)[k%Pack::n] = qi [j*nk+k];
    h_liq(j,k/Pack::n)[k%Pack::n] = liq[j*nk+k];
  }
  Kokkos::deep_copy(v_qi,h_qi);
  Kokkos::deep_copy(v_liq,h_liq);

  CF::main(nj,nk,thr,thr4,v_qi,v_liq,v_ice,v_tot,v_ice4,v_tot4);

  std::ofstream out(fout,std::ios::binary);
  for (auto v : {v_ice,v_tot,v_ice4,v_tot4}) {
    auto h = Kokkos::create_mirror_view(v);
    Kokkos::deep_copy(h,v);
    for (int j=0; j<nj; ++j) for (int k=0; k<nk; ++k) {
      const double x = h(j,k/Pack::n)[k%Pack::n];
      out.write(reinterpret_cast<const char*>(&x),sizeof(double));
    }
  }
  REQUIRE (out.good());
}
