#include <nanobind/nanobind.h>

#include "mpi.h"
#include <cassert>
#include <cxx-prettyprint/prettyprint.hpp>
#include <mxx/collective.hpp>
#include <mxx/comm.hpp>
#include <mxx/sort.hpp>
#include <mxx/timer.hpp>

#include <soque/utils.hpp>
#ifdef USE_PARALLEL_HDF5
#include <soque/mpi_hdf.hpp>
#endif

namespace nb = nanobind;

using namespace nb::literals;

template <typename T> struct Edge {
  T source;
  T target;
  Edge(T src, T tgt) : source(src), target(tgt) {}
  Edge() : source(0), target(0) {}

  void datatype(mxx::value_datatype_builder<Edge<T>> &builder) {
    builder.add_member(source);
    builder.add_member(target);
  }
};

template <typename T> using st_vec = std::vector<Edge<T>>;

template <typename T> std::vector<T> edge_sources(st_vec<T> &st_data) {
  std::vector<T> t_data(st_data.size());
  auto selector_fn = [](const Edge<T> &x) { return x.source; };
  std::transform(st_data.cbegin(), st_data.cend(), t_data.begin(), selector_fn);
  return t_data;
}

template <typename T> std::vector<T> edge_targets(st_vec<T> &st_data) {
  std::vector<T> t_data(st_data.size());
  auto selector_fn = [](const Edge<T> &x) { return x.target; };
  std::transform(st_data.cbegin(), st_data.cend(), t_data.begin(), selector_fn);
  return t_data;
}

st_vec<uint32_t> read_edges(const char *file_name) {
  hid_t file_id = H5File::open_rw(file_name, MPI_COMM_WORLD, MPI_INFO_NULL);
  mxx::comm wcomm;
  H5VecData<uint32_t, 1> tgt_nodes(file_id, "edges/left_local/target_node_id",
                                   wcomm.rank(), wcomm.size());
  H5VecData<uint32_t, 1> src_nodes(file_id, "edges/left_local/source_node_id",
                                   wcomm.rank(), wcomm.size());

  size_t local_size = tgt_nodes.counts()[0];
  assert(local_size == src_nodes.counts()[0]);

  // sort
  st_vec<uint32_t> st_data(local_size);
  for (size_t i = 0; i < local_size; i++) {
    st_data[i] = Edge<uint32_t>(tgt_nodes.data()[i], src_nodes.data()[i]);
  }

  H5Fclose(file_id);
  return st_data;
}

int save_edges(st_vec<uint32_t> &st_data, const char *out_file_name) {
  hsize_t ld_size = st_data.size();
  hsize_t n_data = mxx::allreduce(ld_size, std::plus<hsize_t>());
  hsize_t d_offset = mxx::exscan(ld_size, std::plus<hsize_t>());

  hid_t file_id = H5File::create(out_file_name, MPI_COMM_WORLD, MPI_INFO_NULL);
  hid_t group_id =
      H5Gcreate(file_id, "data", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  H5VecData<uint32_t, 1> h5vec_tgt(H5T_NATIVE_UINT, sizeof(uint32_t), {n_data},
                                   {ld_size}, {d_offset},
                                   edge_targets(st_data));

  H5VecData<uint32_t, 1> h5vec_src(H5T_NATIVE_UINT, sizeof(uint32_t), {n_data},
                                   {ld_size}, {d_offset},
                                   edge_sources(st_data));
  h5vec_tgt.write(group_id, "target_node_id");
  h5vec_src.write(group_id, "source_node_id");
  H5Gclose(group_id);
  return H5Fclose(file_id);
}

int sort_edges_by_target(const char *file_name, const char *out_file_name) {
  mxx::section_timer full_timer;
  mxx::section_timer sec_timer;
  st_vec<uint32_t> st_data(read_edges(file_name));
  sec_timer.end_section("READ EDGES");

  auto edge_cmp = [](const Edge<uint32_t> &x, const Edge<uint32_t> &y) {
    return x.target < y.target;
  };
  mxx::sort(st_data.begin(), st_data.end(), edge_cmp, MPI_COMM_WORLD);
  sec_timer.end_section("SORT EDGES");

  int ret = save_edges(st_data, out_file_name);
  sec_timer.end_section("SAVE EDGES");
  full_timer.end_section("COMPLETE");
  return ret;
}


NB_MODULE(soque_ext, m) {
  m.doc() = "This is a example module built with nanobind";
  m.def("sort_edges", &sort_edges_by_target, "file_name"_a, "out_file_name"_a,
        R"pbdoc(
    Sort data by target
    Args:
        file_name     : string input file path
        out_file_name : string output file path
    Returns:
        Batch Corrected Matrix )pbdoc");
}
