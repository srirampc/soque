import logging
import typing as t
import os

from soque import remove_edges
from soque.comm_interface import default_comm


def _log():
    return logging.getLogger(__name__)

HOME_DIR = os.environ['HOME']
EDGE_FILE = f"{HOME_DIR}/data/v1_test/network_v1/left_local_edges.edge_types_120.sorted.h5"
OUT_FILE = f"{HOME_DIR}/scratch/test_out_remove.h5"

@t.final
class SEArgs:
    def __init__(self, file_name=EDGE_FILE, out_file_name=OUT_FILE):
        self.in_file: str = file_name
        self.threshold: float = 0.5
        self.out_file: str = out_file_name

    def del_output(self):
        if os.path.exists(self.out_file):
            os.remove(self.out_file)

def test_sort_save():
    comm_ifx = default_comm()
    se_data = SEArgs()
    if comm_ifx.rank == 0:
        se_data.del_output()
    comm_ifx.log_at_root(
        _log(), logging.DEBUG,
        f"MPI #P {comm_ifx.size}; IN: {se_data.in_file}; OUT: {se_data.out_file}"
    )
    remove_edges(se_data.in_file, se_data.threshold, se_data.out_file)
    comm_ifx.log_at_root(
        _log(), logging.DEBUG,
        f"SORTED and WRITTEN TO {se_data.out_file}"
    )

def main():
    logging.basicConfig(level=logging.DEBUG)
    test_sort_save()

if __name__ == "__main__":
    main()
