# Running and benchmarking 3D-rot-strat

## Setup
```bash
git clone https://github.com/Rajarshi-prime/3D-rot-strat
cd 3D-rot-strat
pip install numpy scipy sympy tqdm h5py mpi4py jax jaxlib
```
- `mpi4py` **is only needed for running CPU codes. Not needed for GPU codes.** `mpi4py` needs a system MPI (e.g. `apt install openmpi-bin libopenmpi-dev`) before `pip install` will work.  
- `jax`/`jaxlib` need a CUDA build to actually use a GPU; CPU-only `jax` will run but is not what the code is written for (README: "GPU code ... written for single GPU").
- `h5py` is imported in the MPI scripts but only used in commented-out lines — still required so the import doesn't fail.

## Categories of files
### CPU codes : 
`3d_bsnq_MPI.py`, `3d_bsnq_MPI_ps.py`.
### GPU codes : 
`3d_bsnq_JAX.py`, `3d_bsnq_JAX_ps.py`, `3d_bsnq_JAX_ps_const_tide_forcing.py`.

## Running each file
**3d_bsnq_JAX.py** — no arguments. `idx` (picks `N_b` from `[5,10,15,20]`) hardcoded to `3`.
```bash
python 3d_bsnq_JAX.py
```
**3d_bsnq_JAX_ps.py** — same as above, phase-shifted dealiasing variant, no arguments.
```bash
python 3d_bsnq_JAX_ps.py
```
**3d_bsnq_JAX_ps_const_tide_forcing.py** — one required argument, index 0/1/2 selecting hyperviscosity strength `m` from `[10,15,20]`. (`N_b` index is separately hardcoded to `1` in-file, list `[15,20]`.)
```bash
python 3d_bsnq_JAX_ps_const_tide_forcing.py 0
```
**3d_bsnq_MPI.py** — CPU/MPI code, one required argument, index 0/1 selecting `N_b` from `[15,20]`.
```bash
mpirun -n 4 python 3d_bsnq_MPI.py 0
```
**3d_bsnq_MPI_ps.py** — same calling convention, phase-shifted dealiasing variant.
```bash
mpirun -n 4 python 3d_bsnq_MPI_ps.py 0
```
Default is `T = 10000` with `dt` on the order of `0.01–0.03`, i.e. hundreds of thousands of steps. Lower `T` in the file before running anything as a quick benchmark.
## Benchmarking with and without saving
Change the following variables to check scaling.
-  `T` : Controls the final time till which the simulation is run. 
    * Change it to `10*dt` to run the code for 10 time-step for example.
- `dt_save` : Controls the simulation after which the fields are saved. By default it is set to 1.0. 
    * Change it to `dt` to save after every time step. 
    * Change it to `np.inf` to never save during the run.


The output of average time-step per simulation is given in the following format at the end of the simulation:

```
Average time taken to run 10 steps while saving after every after 1 step is: 10s
```

In addition, the CPU and GPU live updates the time taken for each step. 

CPU codes prints the time taken at each step in the error file in the following format:
```
...
step 19 in time 1.833325
step 20 in time 1.833224
...
```

GPU Code live updates the it/s or s/it in the following format: 
```
 25%|██▌       | 25/100 [00:09<00:28,  2.68it/s]
```
or 
```
 25%|██▌       | 25/100 [00:09<00:28,  2.68s/it]
```