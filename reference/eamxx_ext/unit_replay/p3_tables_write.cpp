// Writes P3's non-ice lookup tables (mu_r, vn, vm, revap) in the build's precision,
// computed by EAMxx's own compute_tables. EAMxx's p3_tables_setup writes host mirrors
// that are never filled from the computed views (with these view types Kokkos allocates
// a separate mirror), so its files are all zeros (found 24 Sep 2026); this tool copies
// device -> host before writing. Usage: p3_tables_write DIR  (writes *_v2.dat4 or .dat8)
#include "physics/p3/p3_functions.hpp"
#include "physics/p3/impl/p3_init_impl.hpp"
#include "share/core/eamxx_session.hpp"
#include <fstream>
#include <string>

int main(int argc, char** argv) {
  using P3F = scream::p3::Functions<scream::Real, ekat::DefaultDevice>;
  MPI_Init(&argc, &argv);
  scream::initialize_eamxx_session(argc, argv);
  {
    const std::string dir = argc > 1 ? argv[1] : ".";
    const std::string ext = sizeof(scream::Real) == 8 ? "8" : "4";
    typename P3F::P3LookupTables t;
    scream::p3::compute_tables<scream::Real, typename P3F::P3C>(
        true, t.mu_r_table_vals, t.vn_table_vals, t.vm_table_vals, t.revap_table_vals);
    auto write = [&](const std::string& name, const auto& v) {
      auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v);
      std::ofstream f(dir + "/" + name + "_v2.dat" + ext, std::ios::binary);
      f.write(reinterpret_cast<const char*>(h.data()), sizeof(scream::Real) * h.size());
    };
    write("mu_r_table_vals", t.mu_r_table_vals);
    write("revap_table_vals", t.revap_table_vals);
    write("vn_table_vals", t.vn_table_vals);
    write("vm_table_vals", t.vm_table_vals);
  }
  scream::finalize_eamxx_session();
  MPI_Finalize();
  return 0;
}
