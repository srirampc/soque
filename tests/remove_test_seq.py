import typing as t
import logging
import os
import pandas as pd
import polars as pl
import h5py

from soque.comm_interface import default_comm
from soque.util import create_h5ds, NDIntArray
import time

def _log():
    return logging.getLogger(__name__)

HOME_DIR = os.environ['HOME']
EDGE_FILE = f"{HOME_DIR}/data/v1_test/network_v1/left_local_edges.edge_types_120.sorted.h5"
OUT_FILE = f"{HOME_DIR}/scratch/test_out_seq.h5"
POLARS_OUT_FILE = f"{HOME_DIR}/scratch/test_out_seq_polars_remove.h5"
PANDAS_OUT_FILE = f"{HOME_DIR}/scratch/test_out_seq_pandas_remove.h5"
NUMPY_OUT_FILE = f"{HOME_DIR}/scratch/test_out_seq_numpy_remove.h5"

@t.final
class SEArgs:
    def __init__(self, file_name=EDGE_FILE, out_file_name=OUT_FILE):
        self.file_name: str = file_name
        self.threshold: float = 0.5
        self.out_file_name: str = out_file_name
        if os.path.exists(out_file_name):
            os.remove(out_file_name)


def read_data(fname: str) -> dict[str, NDIntArray]:
    with h5py.File(fname) as hfx:
        return {"target": hfx["edges/left_local/target_node_id"][:], # pyright: ignore[reportIndexIssue, reportReturnType]
                "source": hfx["edges/left_local/source_node_id"][:]} # pyright: ignore[reportIndexIssue, reportReturnType]

def write_data(fname: str, tgt_col, src_col):
    with h5py.File(fname, 'w') as hfx:
        group_ptr = hfx.create_group("data")
        create_h5ds(group_ptr, "target_node_id", tgt_col)
        create_h5ds(group_ptr, "source_node_id", src_col)

def run_sort_save_polars():
    comm_ifx = default_comm()
    se_data = SEArgs(out_file_name=POLARS_OUT_FILE)
    comm_ifx.log_at_root(
        _log(), logging.DEBUG,
        f"MPI #PROC {comm_ifx.size}:: Input:: {se_data.file_name} :: Output :: {se_data.out_file_name}"
    )
    fstart_time = time.time()
    start_time = time.time()
    st_data = read_data(se_data.file_name)
    elapsed_time = time.time() - start_time
    comm_ifx.log_at_root(
        _log(), logging.DEBUG,
        f"DATA LOADED {st_data.keys()} in {elapsed_time} seconds"
    )
    start_time = time.time()
    st_df = pl.DataFrame(st_data).sort("target").sample(fraction=se_data.threshold)
    tgt_col = st_df.get_column("target").to_numpy()
    src_col = st_df.get_column("source").to_numpy()
    elapsed_time = time.time() - start_time
    #print(st_df)
    comm_ifx.log_at_root(
        _log(), logging.DEBUG,
        f" DATA SAMPLED {st_df.columns}:{st_df.shape} in {elapsed_time} seconds"
    )
    start_time = time.time()
    write_data(se_data.out_file_name, tgt_col, src_col)
    elapsed_time = time.time() - start_time
    felapsed_time = time.time() - fstart_time
    comm_ifx.log_at_root(_log(), logging.DEBUG, f" DATA WRITTEN in {elapsed_time} seconds")
    comm_ifx.log_at_root(_log(), logging.DEBUG, f" TOTAL in {felapsed_time} seconds")


def run_sort_save_pandas():
    comm_ifx = default_comm()
    se_data = SEArgs(out_file_name=PANDAS_OUT_FILE)
    comm_ifx.log_at_root(
        _log(), logging.DEBUG,
        f"MPI #PROC {comm_ifx.size}:: Input:: {se_data.file_name} :: Output :: {se_data.out_file_name}"
    )
    fstart_time = time.time()
    start_time = time.time()
    st_data = read_data(se_data.file_name)
    elapsed_time = time.time() - start_time
    comm_ifx.log_at_root(
        _log(), logging.DEBUG,
        f" DATA LOADED {st_data.keys()} in {elapsed_time} seconds"
    )
    start_time = time.time()
    st_df = pd.DataFrame(st_data)
    st_df.sort_values(by='target')
    st_df = st_df.sample(frac=se_data.threshold)
    tgt_col = st_df.loc[:, "target"].to_numpy()
    src_col = st_df.loc[:, "source"].to_numpy()
    elapsed_time = time.time() - start_time
    comm_ifx.log_at_root(
        _log(), logging.DEBUG,
        f" DATA SAMPLED {st_df.columns}:{st_df.shape} in {elapsed_time} seconds"
    )
    start_time = time.time()
    write_data(se_data.out_file_name, tgt_col, src_col)
    elapsed_time = time.time() - start_time
    felapsed_time = time.time() - fstart_time
    comm_ifx.log_at_root(_log(), logging.DEBUG, f" DATA WRITTEN in {elapsed_time} seconds")
    comm_ifx.log_at_root(_log(), logging.DEBUG, f" TOTAL in {felapsed_time} seconds")


def test_sample_save(method: t.Literal["polars", "pandas", "numpy"]):
    comm_ifx = default_comm()
    if comm_ifx.rank == 0:
        #run_sort_save_polars()
        match method:
            case "numpy":
                print("NOT IMPLEMENTED")
            case "polars":
                run_sort_save_polars()
            case "pandas":
                run_sort_save_pandas()
    comm_ifx.barrier()


def main(method: t.Literal["polars", "pandas", "numpy"]):
    logging.basicConfig(level=logging.DEBUG)
    test_sample_save(method)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
       prog="Test Seq with Numpy/Polars/Pandas",
       description="Generate GRNs w. PIDC for Single Cell Data"
    )
    parser.add_argument(
        "method",
        default="polars",
        choices=["polars", "pandas", "numpy"],
        help=f"Yaml Input file with a given configuration."
    )
    run_args = parser.parse_args()
    main(run_args.method)
