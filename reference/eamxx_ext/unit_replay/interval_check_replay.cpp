// Replay harness for FieldWithinIntervalCheck (share/property_checks).
// Input  (REPLAY_IN, float64): ncase, then per case:
//   n0, n1, lb, ub, can_repair, lb_repairable, ub_repairable, values[n0*n1]
// Output (REPLAY_OUT, float64): per case: result (0 pass, 1 repairable->repaired,
//   2 fail), values[n0*n1] after repair (unchanged unless repaired)
// This follows AtmosphereProcess::run_property_check: repair() is called only
// when check() returns Repairable.
#include "catch2/catch.hpp"
#include "share/property_checks/field_within_interval_check.hpp"
#include "share/field/field.hpp"

#include <cstdlib>
#include <fstream>
#include <vector>

TEST_CASE("interval_check_replay") {
  using namespace scream;
  using namespace ShortFieldTagsNames;

  const char* fin  = std::getenv("REPLAY_IN");
  const char* fout = std::getenv("REPLAY_OUT");
  REQUIRE (fin!=nullptr);
  REQUIRE (fout!=nullptr);
  std::ifstream in(fin,std::ios::binary);
  std::ofstream out(fout,std::ios::binary);
  auto rd = [&]() { double x; in.read(reinterpret_cast<char*>(&x),sizeof(x)); return x; };
  auto wr = [&](double x) { out.write(reinterpret_cast<const char*>(&x),sizeof(x)); };

  const int ncase = static_cast<int>(rd());
  for (int c=0; c<ncase; ++c) {
    const int n0 = static_cast<int>(rd()), n1 = static_cast<int>(rd());
    const double lb = rd(), ub = rd();
    const bool can_repair = rd()!=0;
    const double lbr = rd(), ubr = rd();

    FieldIdentifier fid("f",FieldLayout({COL,LEV},{n0,n1}),ekat::units::Units::nondimensional(),"g");
    Field f(fid);
    f.allocate_view();
    auto v = f.get_view<Real**,Host>();
    for (int i=0; i<n0; ++i) for (int j=0; j<n1; ++j) v(i,j) = rd();
    f.sync_to_dev();

    FieldWithinIntervalCheck chk(f,nullptr,lb,ub,can_repair,lbr,ubr);
    const auto res = chk.check().result;
    double code = 2;
    if (res==CheckResult::Pass) {
      code = 0;
    } else if (res==CheckResult::Repairable) {
      chk.repair();
      code = 1;
    }
    f.sync_to_host();
    wr(code);
    for (int i=0; i<n0; ++i) for (int j=0; j<n1; ++j) wr(v(i,j));
  }
  REQUIRE (in.good());
  REQUIRE (out.good());
}
