## A prototype for fast processing SONATA edges

Contains template code to test sort a SONATA file.

### Installation

1. MPI and parallel HDF5 should be already installed in the system.
   MPI and parallel HDF5 installation can be verified by commands
   `mpicc -show` and `h5pcc -show` respectively.

2. Install python and other dependencies via conda as below.

   ```sh
   conda env create -n soque -f env.yml
   conda activate soque
   ```

3. Install mpi4py and h5py via pip from souces as below.
   ```sh
   pip install --no-binary=mpi4py mpi4py
   CC="mpicc" HDF5_MPI="ON" HDF5_DIR=/path/to/opt/spack/linux-cascadelake/hdf5-1.14.6-z5pldojbgep4zqq6bh3uevc6ggwjncn5/  pip install --no-binary=h5py h5py
   ```

### Test scripts

Test scripts are given in tests directory and they can be run as follows. Parallel
implementation can be run as below:

```sh
mpirun -np 4 python tests/sort_test_par.py
```

Sequential implementation with numpy, pandas and polars can be run as below:

```sh
mpirun -np 1 python tests/sort_test_seq.py numpy
mpirun -np 1 python tests/sort_test_seq.py pandas
mpirun -np 1 python tests/sort_test_seq.py polars
```

### Code name

Repo is name soque after Soque river in Georgia

![Soque river](https://upload.wikimedia.org/wikipedia/commons/c/c6/Soque_river_georgia.jpg)
