import numpy as np 
# from pyfftw.interfaces.scipy_fft import fft ,  ifft ,  irfft2 ,  rfft2 , irfftn ,  rfftn, fftfreq, dst, dct, idst, idct, rfft,  irfft
from scipy.fft import fft ,  ifft ,  irfft2 ,  rfft2 , irfftn ,  rfftn, fftfreq, dst, dct, idst, idct, rfft,  irfft #type: ignore
from mpi4py import MPI
from time import time
import pathlib
import os,sys,json

curr_path = pathlib.Path(__file__).parent

## ---------------MPI things--------------
comm = MPI.COMM_WORLD
num_process =  comm.Get_size()
rank = comm.Get_rank()
## ---------------------------------------

## --------- Loading from the parameters file ------------
with open(curr_path/"parameters.json") as f:
    param = json.load(f)

T = param["Final_time"]
dt = param["time_step"]
dt_save = param["save_step"]
dt_save_r = param["save_step_r"]
N = param["N"]
ro = param["Rossby"]
nu = param["nu"]
lp = param["hyperviscous"]
alph = param["Alpha"]
aa = N**3*param["Forcing amplitude"]
aa_s = N**3*param["Balanced Forcing amplitude"]
aa_f = N**3*param["Wave Forcing amplitude"]
einit = N**3*param["Initial balanced amplitude"]
kinit = param["k limit"]
omega = param["Forcing frequency"]
forcestart = param["forcestart"]
low_wave = param["Low wave"]
## -------------------------------------------------------

## -------------- paths ------------------
lw = "_LW" if low_wave else ""
omega = 1.7277
loadPath = pathlib.Path(f"/mnt/pfs/rajarshi.chattopadhyay/boussinesq/spectrum-development/nu_{nu}_N_{N}/Ro_{ro}/forcedTide_ring_{omega:.2f}{lw}")

savePath = loadPath
savePlot = curr_path/f"Plots/nu_{nu}_N_{N}/Ro_{ro:.1f}/forcedTide_ring_{omega:.2f}{lw}/"


with open(loadPath/"parameters.json") as f:
    param = json.load(f)
    num_slabs = param["num_processes"]

try: savePlot.mkdir(parents = True,exist_ok = True)
except FileExistsError: pass
try: savePath.mkdir(parents = True,exist_ok = True)
except FileExistsError: pass
if rank == 0: print("saved in " ,str(savePath))
    
## ---------------------------------------

## -------------Defining the grid ---------------
PI = np.pi
TWO_PI = 2*PI
Nf = N//2 + 1
Np = N//num_process

Ns = num_slabs//num_process
Nslab = N// num_slabs
if rank ==0: print(f"# slabs and Ns of the sim : {num_slabs},{Ns} ")


sx = slice(rank*Np ,  (rank+1)*Np)
L = TWO_PI
Lz = PI
X = Y = np.linspace(0, L, N, endpoint= False)
dx,dy = X[1]-X[0], Y[1]-Y[0]
Z = np.linspace(0+ Lz/(2*N), Lz + Lz/(2*N), N, endpoint= False)
dz = Z[1]- Z[0]
x, y, z = np.meshgrid(X[sx], Y, Z, indexing='ij')

Kx = Ky = fftfreq(N,  1./N)*TWO_PI/L
Kzc = np.arange(N)* PI/Lz
dkx, dky,dkz = Kx[1]- Kx[0], Ky[1]  - Ky[0], Kzc[1]- Kzc[0]
Kzs = Kzc + 1
kx,  ky,  kzc = np.meshgrid(Kx,  Ky[sx],  Kzc,  indexing = 'ij')
kzs = kzc + 1
kc = (kx**2 + ky**2 + kzc**2 )**0.5
ks = (kx**2 + ky**2 + kzs**2 )**0.5
kh = np.mean((kx**2 + ky**2 )**0.5,axis = 2)
shells = np.arange(-0.5,N//2 + 1)
shells[0] = 0.
kv = Kzc
## -----------------------------------------------



## --------- kx and ky for differentiation --------------------------
# kx.shape = (Nf,  Np,  N )
# ky.shape = (Np,  Nf,  N)
kx_diff = kx[:Nf,  :, :].copy()
kx_diff[-1, :, :] = -kx_diff[-1, :, :]
ky_diff = np.swapaxes(kx_diff, 0, 1).copy()

## ------------------------------------------------------------------

## ------------Useful Operators-------------------
dealias_cos = (abs(kx)<N//3)*(abs(ky)<N//3)*(kzc<(2*N)//3)
dealias_sin = (abs(kx)<N//3)*(abs(ky)<N//3)*(kzs<(2*N)//3)

lapc = -(kx**2 + ky**2 + kzc**2 )
laps = -(kx**2 + ky**2 + kzs**2 )

kc = (-lapc )**0.5
ks = (-laps )**0.5
lappress = -(kx**2 + ky**2 + alph**2 * kzc**2)
invlapc = 1.0/np.where(lapc == 0, np.inf,  lapc)
invlaps = 1.0/laps
invpress = 1.0/np.where(lappress == 0,  np.inf ,  lappress)* dealias_cos

# Hyperviscous operators
vis_cos = nu*(kc)**(2*lp) ## This is in Fourier Space
vis_sin = nu*(ks)**(2*lp) ## This is in Fourier Space
## -------------------------------------------------

## -------------Empty arrays ----------------------------------------

u  = np.empty((3, Np, N, N), dtype= np.float64)
u1 = np.empty((3, Np, N, N), dtype= np.float64)

p = u[0].copy()
b = u[0].copy()
b1 = np.empty_like(b)


uk = np.empty((3, N, Np, N), dtype= np.complex128)


pk = uk[0].copy()
bk = pk.copy()





rhsu = np.empty_like(p)
rhsu1 = np.empty_like(p)
rhsu2 = np.empty_like(p)
rhsu3 = np.empty_like(p)

rhsv = rhsu.copy()
rhsv2 = rhsu.copy()
rhsv1 = rhsu.copy()
rhsv3 = rhsu.copy()

rhsw = rhsu.copy()
rhsw1 = rhsu.copy()
rhsw2 = rhsu.copy()
rhsw3 = rhsu.copy()

b_t1 = rhsu.copy()
b_t2 = rhsu.copy()
b_t3 = rhsu.copy()
p1 = rhsu.copy()
p11 = rhsu.copy()
p12 = rhsu.copy()
p13 = rhsu.copy()
u_temp = p1.copy()




arr_temp_r = np.empty((Np, N, N),dtype = np.float64)
arr_temp_k = np.empty((N, Np, N),dtype= np.float64)
arr_temp_fr = np.empty((Np, N, N), dtype= np.complex128)      
arr_temp_ifr = np.empty((N, Np, N), dtype= np.complex128)
arr_mpi = np.empty((num_process,  Np,  Np, N), dtype= np.complex128)
arr_mpi_r = np.empty((num_process,  Np,  Np, N), dtype= np.float64)
arr_mpi_Z = np.empty((num_process,  Np, N,  Np), dtype= np.complex128)



## ------------------------------------------------------------------

## ------------------------------------------------------------------
##                      Functions to use
## ------------------------------------------------------------------

def create_dir(path):
    path = pathlib.Path(path)
    try:
        path.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        pass
    
## --------------------- FFT + diff fns ---------------------------
def fft_cos(u, fu):
    arr_temp_r[:] = dct(u,type=2, axis= 2)
    arr_temp_fr[:] = fft(arr_temp_r, axis = 1)
    arr_mpi[:] = np.swapaxes(np.reshape(arr_temp_fr, (Np,  num_process,  Np, N)), 0, 1)
    comm.Alltoall([arr_mpi,  MPI.DOUBLE_COMPLEX], [fu,  MPI.DOUBLE_COMPLEX])
    fu[:] = fft(fu, axis = 0)
    return fu

def fft_sin(u, fu):
    arr_temp_r[:] = dst(u,type=2, axis= 2)
    arr_temp_fr[:] = fft(arr_temp_r, axis = 1)
    arr_mpi[:] = np.swapaxes(np.reshape(arr_temp_fr, (Np,  num_process,  Np, N)), 0, 1)
    comm.Alltoall([arr_mpi,  MPI.DOUBLE_COMPLEX], [fu,  MPI.DOUBLE_COMPLEX])
    fu[:] = fft(fu, axis = 0)
    return fu

def ifft_cos(fu, u):
    arr_temp_ifr[:] = ifft(fu,  axis = 0)
    comm.Alltoall([arr_temp_ifr,  MPI.DOUBLE_COMPLEX], [arr_mpi, MPI.DOUBLE_COMPLEX])
    arr_temp_fr[:] = np.reshape(np.swapaxes(arr_mpi,  0, 1), (Np,  N,  N))
    arr_temp_r[:] = ifft(arr_temp_fr, axis =1).real
    u[:] = idct(arr_temp_r,type = 2, axis=2)
    return u

def ifft_sin(fu, u):
    arr_temp_ifr[:] = ifft(fu,  axis = 0)
    comm.Alltoall([arr_temp_ifr,  MPI.DOUBLE_COMPLEX], [arr_mpi, MPI.DOUBLE_COMPLEX])
    arr_temp_fr[:] = np.reshape(np.swapaxes(arr_mpi,  0, 1), (Np,  N,  N))
    arr_temp_r[:] = ifft(arr_temp_fr, axis =1).real
    u[:] = idst(arr_temp_r,type = 2, axis=2)
    return u

def diff_x(u,  u_x):
    arr_mpi_r[:] = np.swapaxes(np.reshape(u, (Np,  num_process,  Np,  N)), 0, 1)
    comm.Alltoall([arr_mpi_r,  MPI.DOUBLE], [arr_temp_k,  MPI.DOUBLE])
    arr_temp_k[:] = irfft(1j * kx_diff*rfft(arr_temp_k,  axis = 0), N,  axis=0)
    comm.Alltoall([arr_temp_k,  MPI.DOUBLE], [arr_mpi_r,  MPI.DOUBLE])
    u_x[:] = np.reshape(np.swapaxes(arr_mpi_r,  0, 1), (Np,  N, N))
    return u_x

def diff_y(u, u_y):
    u_y[:] = irfft(1j*ky_diff*rfft(u, axis= 1), N, axis= 1)
    return u_y
    
def diff_z_sin(u, u_z):
    u_temp[:] = np.roll(Kzs*dst(u,type=2,axis=2), 1, axis = 2) # type: ignore
    u_temp[:, :, 0] = 0
    u_z[:] = idct(u_temp,type = 2, axis = 2) 
    return u_z
    
    
def diff_z_cos(u, u_z):
    u_temp[:] = - np.roll(Kzc*dct(u,type =2,  axis = 2), -1, axis = 2) # type: ignore
    u_temp[:, :, -1] = 0
    u_z[:] = idst(u_temp,type = 2, axis = 2) 
    return u_z    
    
## -----------------------------------------------------------------

## --------------------- Other functions ---------------------------
def load_data(Ns,Nslab,nu,N,u,b):
    for j in range(Ns):
        field = np.load(loadPath/f"time_{t[i]:.1f}/Fields_{rank*Ns+j}.npz")
        u[0,j*Nslab:(j+1)*(Nslab),...] = field["u"]
        u[1,j*Nslab:(j+1)*(Nslab),...] = field["v"]
        u[2,j*Nslab:(j+1)*(Nslab),...] = field["w"]
        b[j*Nslab:(j+1)*(Nslab),...] = field["b"]
        # u[0,j*Nslab:(j+1)*(Nslab),...] = np.load(loadPath/f"T{np.round(t[i],2)}/U_files0/U{rank}.npz")["u"]
        # u[1,j*Nslab:(j+1)*(Nslab),...] = np.load(loadPath/f"T{np.round(t[i],2)}/V_files0/V{rank}.npz")["v"]
        # u[2,j*Nslab:(j+1)*(Nslab),...] = np.load(loadPath/f"T{np.round(t[i],2)}/W_files0/W{rank}.npz")["w"]
        # b[j*Nslab:(j+1)*(Nslab),...] = np.load(loadPath/f"T{np.round(t[i],2)}/B_files0/B{rank}.npz")["b"]
    return u,b    
## -----------------------------------------------------------------


## -------------------- Giving the RHS of the balanced flow --------    

## ------------------ RHS for Boussinesq -----------------
def RHS(u, b, u_t, b_t):
    ## The RHS terms of u, v and w excluding the pressure and the hypervisocsity term 
    rhsu[:] = -ro*(u[0]*diff_x(u[0], rhsu1) + u[1]*diff_y(u[0], rhsu2) + u[2]*diff_z_cos(u[0], rhsu3)) + u[1]
    
    rhsv[:] = -ro*(u[0]*diff_x(u[1], rhsv1) + u[1]*diff_y(u[1], rhsv2) + u[2]*diff_z_cos(u[1], rhsv3)) - u[0]
    
    rhsw[:] = -ro*(u[0]*diff_x(u[2], rhsw1) + u[1]*diff_y(u[2], rhsw2) + u[2]*diff_z_sin(u[2], rhsw3)) + alph**2*b
    
    b_t[:] = -ro*(u[0]*diff_x(b, b_t1) + u[1]*diff_y(b, b_t2) + u[2]*diff_z_sin(b, b_t3)) - u[2]
    
    ## The pressure term
    p1[:] =  diff_x(rhsu, p11) + diff_y(rhsv, p12) + diff_z_sin(rhsw, p13)
    pk[:] = invpress  * fft_cos(p1, pk)
    p[:] = ifft_cos(pk, p)
    
    
    u_t[0] = rhsu - diff_x(p, p1)
    u_t[1] = rhsv - diff_y(p,  p1)
    u_t[2] = rhsw - alph**2 * diff_z_cos(p, p1)
    
        
    return u_t, b_t
## -------------------------------------------------------



## -------------------------------------------------------------------
## 
## -------------------------------------------------------------------

## ----------------- Loading the data ------------------------------
interval = 10.
Ti = 780 - 10*PI
Tf = Ti + 10*PI
t = np.arange(Ti,Tf,interval)

e_arr = np.empty_like(t)
e_arr_f = np.empty_like(t)
e_arr_s = np.empty_like(t)

T_f = np.empty_like(t, dtype = np.float64)
T_s = np.empty_like(t, dtype = np.float64)

if rank ==0:
    fluxes = np.zeros(len(t))
    fluxes_f = np.zeros(len(t))
    fluxes_s = np.zeros(len(t))
    energies = np.zeros(len(t))
    energies_f = np.zeros(len(t))
    energies_s = np.zeros(len(t))
## -------------------------------------------------------------------
for i in range(len(t)):
    if rank ==0: print(f"Time : {t[i]:.1f}")
    u[:],b[:] = load_data(Ns,Nslab,nu,N,u,b)
    
    u1[:], b1[:] = RHS(u,b,u1,b1)
    
    savePath = loadPath/f"time_{t[i]:.1f}"
    create_dir(savePath)
    
    np.savez_compressed(savePath/f"RHS_{rank}.npz",u1 = u1, b1 = b1)
    
