# Running and benchmarking 3D-rot-strat
Verified by pulling the raw files from the repo (`main` branch). No requirements file exists; dependencies below are inferred from the imports.
## Setup (my inference, not stated in repo)
```bash
git clone https://github.com/Rajarshi-prime/3D-rot-strat
cd 3D-rot-strat
pip install numpy scipy sympy tqdm h5py mpi4py jax jaxlib
```
- `mpi4py` **is only needed for running CPU codes. Not needed for GPU codes.** `mpi4py` needs a system MPI (e.g. `apt install openmpi-bin libopenmpi-dev`) before `pip install` will work.  
- `jax`/`jaxlib` need a CUDA build to actually use a GPU; CPU-only `jax` will run but is not what the code is written for (README: "GPU code ... written for single GPU").
- `h5py` is imported in the MPI scripts but only used in commented-out lines — still required so the import doesn't fail.

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
Shipped default is `T = 10000` with `dt` on the order of `0.01–0.03`, i.e. hundreds of thousands of steps. Lower `T` in the file before running anything as a quick benchmark.
## Benchmarking with vs. without saving
The two families of scripts already handle this differently.
**MPI.py / MPI_ps.py — no edit needed.** `evolve_and_save` keeps two separate timers:
- `calc_time`, accumulated only over the RK4/RHS math (save time is excluded), printed at the end as `average calculation time per step`.
- `t2`, the wall time of the whole run including saving, printed and appended to `data/.../calcTime.txt`.
One run gives you both: "without saving" = `calc_time/(nsteps-1)`, "with saving" = `t2/(nsteps-1)`.
**JAX.py / JAX_ps.py / JAX_ps_const_tide_forcing.py — need two runs.** These only compute `t2 = time() - t1` for the whole loop and never print or log it (add `print(t2)` yourself). To isolate save cost, comment out the save call inside `evolve_and_save`:
```python
if i % st ==0 :save(ti,uk,bk)   # comment out for the "no saving" run
```
Run once with this line active and once commented out, same `N`/`T`/`idx`, and compare `t2`.
