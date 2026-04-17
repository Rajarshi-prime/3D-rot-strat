""" 
The code forces the dimensional boussinesq equations with constant power input to the waves with wavenunbers between 1 and 4 separately.
"""
#%%
import numpy as np 
from scipy.fft import fft ,  ifft ,  irfft2 ,  rfft2 , irfftn ,  rfftn,   rfft,  irfft,fftfreq
from mpi4py import MPI
from time import time
import pathlib,sys
curr_path = pathlib.Path(__file__).parent
forcestart = False
idx = int(float(sys.argv[-1]))
# forcestart = bool(float(sys.argv[-1]))
#%%

## ---------------MPI things--------------
comm = MPI.COMM_WORLD
num_process =  comm.Get_size()
rank = comm.Get_rank()
#%%

isforcing = True
viscosity_integrator = "implicit" 
# viscosity_integrator = "explicit" #! Do not use this for hyperviscous simulations or cases with high resolution simulations.
# viscosity_integrator = "exponential"
if viscosity_integrator == "explicit": isexplicit = 1.
else : isexplicit = 0.
## ---------------------------------------
#%%

## ------------- Time steps --------------
N = 192
dt = 0.256/N   #! Such that increasing resolution will decrease the dt
f_corr = 5
N_bs = [15,20]
N_b = N_bs[idx]
T = 100 if forcestart else 31.4/f_corr
dt_save = 1.0 if forcestart else 0.1/f_corr
st = round(dt_save/dt)

## ---------------------------------------
#%%

## -------------Defining the grid ---------------
PI = np.pi
TWO_PI = 2*PI
Nf = N//2 + 1
Np = N//num_process
sx = slice(rank*Np ,  (rank+1)*Np)
L = TWO_PI
X = Y = Z = np.linspace(0, L, N, endpoint= False)
dx,dy,dz = X[1]-X[0], Y[1]-Y[0], Z[1]-Z[0]
x, y, z = np.meshgrid(X[sx], Y, Z, indexing='ij')

Kx = Ky = fftfreq(N,  1./N)*TWO_PI/L
Kz = np.abs(Ky[:Nf])

kx,  ky,  kz = np.meshgrid(Kx,  Ky[sx],  Kz,  indexing = 'ij')
## -----------------------------------------------

#%%


## --------- kx and ky for differentiation ---------    
kx_diff = np.moveaxis(kz,[0,1,2],[2,1,0]).copy()
ky_diff = np.swapaxes(kx_diff, 0, 1).copy()
kz_diff = np.moveaxis(kz, [0,1], [1,0]).copy()

if rank ==0 : print(kx_diff.shape, ky_diff.shape, kz_diff.shape)

## -------------------------------------------------
#%%

## ----------- Parameters ----------
lp = 8 # Hyperviscosity power
nu0 = 0.5 #! Viscosity for N = 1
m = 1 #! Desired kmax*eta
nu = nu0*(3*m/N)**(2*(lp - 1/3))  #? scaling with resolution. For 512, nu = 0.002 #! Need to add scaling for hyperviscosity

fbyN = f_corr/N_b if N_b != 0 else 0.0
einit = 1*TWO_PI**3 # Initial energy
nshells = 5 # Number of consecutive shells to be forced
shell_no = np.arange(4,4+nshells) # the shells to be forced 
#%%

#----  Kolmogorov length scale - \eta \epsilon etc...---------

f0 = 0.5*TWO_PI**3/ nshells #! Total power input at each shells
# f0 = 0.02 /(N_b**2)*nshells#! Total power input at each shells
re = np.inf if nu==0 else 1/nu
if rank ==0 : print(f" Power input  : {nshells*f0} \n Viscosity : {nu}, Re : {re},dt : {dt}")
#%%
param = dict()
param["nu"] = nu
param["hyperviscous"] = lp
param["Initial energy"] = einit
param["Gridsize"] = N
param["Processes"] = num_process
param["Final_time"] = T
param["time_step"] = dt
param["interval of saving indices"] = st

## ---------------------------------
#%%
nu,f0*nshells
#%%


if nu!= 0: savePath = pathlib.Path(f"./data/bsnq/f_{f_corr:.1f}_Nb_{N_b:.1f}/forced_{isforcing}/N_{N}_Re_{re:.1f}")
else: savePath = pathlib.Path(f"./data/bsnq/forced_{isforcing}/N_{N}_Re_inf")

if rank == 0:
    print(savePath)
    try: savePath.mkdir(parents=True,  exist_ok=True)
    except FileExistsError: pass

## ------------Useful Operators-------------------
#%%

lap = -1.0*(kx**2 + ky**2 + kz**2 )
k = (-lap)**0.5
kint = np.clip(np.round(k,0).astype(int),None,N//2)
kh = (kx**2 + ky**2)**0.5
# dealias = kint<=N/3 #! Spherical dealiasing
# dealias = (abs(kx)<N//3)*(abs(ky)<N//3)*(abs(kz)<N//3) #! Cubic 2/3 dealiasing
# dealias = np.exp(-(N//10)*((1.0*kx/N)**(N//10) +(1.0*kz/N)**(N//10) + (1.0*kz/N)**(N//10))) #! Exponential dealiasing a la. Sanjay for 360^3.
dealias = kint < 2**0.5*N/3 #! phase shifted dealiasing
phase_k = np.exp(1j*(kx*dx/2 + ky*dy/2 + kz*dz/2))*dealias
conjphase_k = np.conjugate(phase_k)*dealias

invlap = dealias/np.where(lap == 0, np.inf,  lap)
lapwv = -1.0*(kx**2 + ky**2 + (fbyN)**2*kz**2 )
invlapwv = dealias/np.where(lapwv == 0, np.inf,lapwv)

# Hyperviscous operator
vis = nu*(k)**(2*lp) ## This is in Fourier Space

normalize = np.where((kz== 0) + (kz == N//2) , 1/(N**6/TWO_PI**3),2/(N**6/TWO_PI**3))
shells = np.arange(-0.5,Nf, 1.)
shells[0] = 0.

cond_ky = np.abs(np.round(Ky))<=N//3
cond_kz = np.abs(np.round(Kz))<=N//3
## -------------------------------------------------


## -------------zeros arrays -----------------------
u  = np.zeros((3, Np, N, N), dtype= np.float64)
b = np.zeros_like(u[0])
b1 = b.copy()
b2 = b.copy()
b3 = b.copy()
omg= np.zeros((3, Np, N, N), dtype= np.float64)


uk = np.zeros((3, N, Np, Nf), dtype= np.complex128)
uk_w = uk.copy()
uk_v = uk.copy()
pk = uk[0].copy()
pv = uk[0].copy()
bk = pk.copy()
theta = pk.copy()
sig = pk.copy()
bk_w = bk.copy()
bk_v = bk.copy()

ek = np.zeros_like(pk, dtype = np.float64)
Pik = np.zeros_like(pk, dtype = np.float64)
ek_arr = np.zeros(Nf)
Pik_arr = np.zeros(Nf)
factor = np.zeros(Nf)
factor3d = np.zeros_like(pk,dtype= np.float64)
uknew = np.zeros_like(uk)
bknew = np.zeros_like(bk)


fk = np.zeros_like(uk)
f1k = np.zeros_like(uk)
f2k = np.zeros_like(uk)
fkb = np.zeros_like(bk)
f1bk = np.zeros_like(bk)
f2bk = np.zeros_like(bk)

rhsuk = np.zeros_like(pk)
rhsvk = rhsuk.copy()
rhswk = rhsuk.copy()
rhsbk = rhsuk.copy()

rhsu = np.zeros_like(u[0])
rhsv = rhsu.copy()
rhsw = rhsu.copy()
rhsb = rhsu.copy()


k1u = np.zeros((3, N, Np, Nf), dtype = np.complex128)
k2u = np.zeros((3, N, Np, Nf), dtype = np.complex128)
k3u = np.zeros((3, N, Np, Nf), dtype = np.complex128)
k4u = np.zeros((3, N, Np, Nf), dtype = np.complex128)

k1b = np.zeros((N, Np, Nf), dtype = np.complex128)
k2b = np.zeros((N, Np, Nf), dtype = np.complex128)
k3b = np.zeros((N, Np, Nf), dtype = np.complex128)
k4b = np.zeros((N, Np, Nf), dtype = np.complex128)

arr_temp_k = np.zeros((N, Np, N),dtype= np.float64)
arr_temp_fr = np.zeros((Np, N, Nf), dtype= np.complex128)      
arr_temp_ifr = np.zeros((N, Np, Nf), dtype= np.complex128)      
arr_mpi = np.zeros((num_process,  Np,  Np, Nf), dtype= np.complex128)
arr_mpi_r = np.zeros((num_process,  Np,  Np, N), dtype= np.float64)


## -----------------------------------------------------


## ------FFT + iFFT + derivative functions------- 
def rfft_mpi(u, fu):
    arr_temp_fr[:] = rfft2(u,  axes=(1, 2))
    arr_mpi[:] = np.swapaxes(np.reshape(arr_temp_fr, (Np,  num_process,  Np, Nf)), 0, 1)
    comm.Alltoall([arr_mpi,  MPI.DOUBLE_COMPLEX], [fu,  MPI.DOUBLE_COMPLEX])
    fu[:] = fft(fu, axis = 0)
    return fu

def irfft_mpi(fu, u):
    arr_temp_ifr[:] = ifft(fu,  axis = 0)
    comm.Alltoall([arr_temp_ifr,  MPI.DOUBLE_COMPLEX], [arr_mpi, MPI.DOUBLE_COMPLEX])
    arr_temp_fr[:] = np.reshape(np.swapaxes(arr_mpi,  0, 1), (Np,  N,  Nf))
    u[:] = irfft2(arr_temp_fr, (N, N), axes = (1, 2))
    return u    


def diff_x(u,  u_x):
    arr_mpi_r[:] = np.moveaxis(np.reshape(u, (Np,  num_process,  Np,  N)),[0,1], [1,0])
    comm.Alltoall([arr_mpi_r,  MPI.DOUBLE], [arr_temp_k,  MPI.DOUBLE])
    arr_temp_k[:] = irfft(1j * kx_diff*rfft(arr_temp_k,  axis = 0), N,  axis=0)
    comm.Alltoall([arr_temp_k,  MPI.DOUBLE], [arr_mpi_r,  MPI.DOUBLE])
    u_x[:] = np.reshape(np.moveaxis(arr_mpi_r,  [0,1], [1,0]), (Np,  N, N))
    return u_x

def diff_y(u, u_y):
    u_y[:] = irfft(1j*ky_diff*rfft(u, axis= 1), N, axis= 1)
    return u_y
    
def diff_z(u, u_z):
    u_z[:] = irfft(1j*kz_diff*rfft(u, axis= 2), N, axis= 2)
    return u_z

def e3d_to_e1d(x): #1 Based on whether k is 2D or 3D, it will bin the data accordingly. 
    return np.histogram(k.ravel(),bins = shells,weights=x.ravel())[0] 

 
def vortex(uk,bk): 
    global uk_v,bk_v
    """
    Projects the velocity and buoyancy in the vortical modes. 
    """
    pv[:] = 1j*(kx*uk[1] - ky*uk[0] + kz*bk*(fbyN))
    uk_v[0] = -1j*ky*pv*invlapwv
    uk_v[1] = 1j*kx*pv*invlapwv
    uk_v[2] = 0. + 0.j
    bk_v = 1j*kz*pv*invlapwv*(fbyN)
    
    return uk_v, bk_v
    


def forcing_old(uk,bk):
    """
    Calculates the net dissipation of the flow and injects that amount into larges scales of the horizontal flow
    """
    global fk, fkb, factor3d, factor, ek_arr,kint
    uk_v[:],bk_v[:] = vortex(uk,bk)
    # uk_w[:],bk_w[:] = uk - uk_v, bk - bk_v
    uk_w[:],bk_w[:] = uk , bk 
    
    
    ek[:] = 0.5*(np.abs(uk_w[0])**2 + np.abs(uk_w[1])**2 + np.abs(uk_w[2])**2 + np.abs(bk_w)**2)*dealias*normalize*(kh>0.5)*(kz > 0.5) #! This is the 3D ek array of waves
    # ek[:] = 0.5*(np.abs(uk[0])**2 + np.abs(uk[1])**2 + np.abs(uk[2])**2 + np.abs(bk)**2)*dealias*normalize*(kh>0.5)*(kz > 0.5) #! This is the 3D ek array of waves
    
    ek_arr[:] = comm.allreduce(e3d_to_e1d(ek),op = MPI.SUM) #! This is the shell-summed ek array.
    #? Only if you are forcing 1 or two shells 
    # ek_arr[:] = 0.0
    # for shell in shell_no:
    #     ek_arr[shell] = comm.allreduce(np.sum(ek*(kint>= shell-0.5)*(kint< shell +0.5)),op = MPI.SUM)
    
    ek_arr[:] = np.where(np.abs(ek_arr)< 1e-10,np.inf, ek_arr)
    """Change forcing starts here"""
    # Const Power Input
    factor[:] = 0.0
    factor[shell_no] = f0/(2*ek_arr[shell_no])
    factor3d[:] = factor[kint]*dealias*(kh>0.5)*(kz > 0.5)
    
    
    # # Constant shell energy
    # factor[:] = np.tanh(np.where(np.abs(ek_arr0) < 1e-10, 0, (ek_arr0/ek_arr)**0.5 - 1)) #! The factors for each shell is calculated
    # factor3d[:] = factor[kint]

    
    fk[0] = factor3d*uk_w[0]
    fk[1] = factor3d*uk_w[1]
    fk[2] = factor3d*uk_w[2]
    fkb[:] = factor3d*bk_w
    # fk[0] = factor3d*uk[0]
    # fk[1] = factor3d*uk[1]
    # fk[2] = factor3d*uk[2]
    # fkb[:] = factor3d*bk

    """Change forcing ends here here"""
    
    pk[:] = invlap  * (kx*fk[0] + ky*fk[1] + kz*fk[2])*dealias
    
    fk[0] = fk[0] + kx*pk
    fk[1] = fk[1] + ky*pk
    fk[2] = fk[2] + kz*pk
    
    
    return fk*isforcing*dealias, fkb*isforcing*dealias
    
# -----------------------------------------



def forcing(t, cond,f0,uk,bk ,h = dt,theta= theta,pk = pk,f1uk = f1k,f1bk = f1bk,f2uk = f2k, f2bk = f2bk,fk = fk, fkb = fkb):


    
    
    # ------------------- negative frequency ------------------- #
    theta[:] = np.random.uniform(0,TWO_PI,(N,Np,Nf))
    pk[:] = np.exp(1j*theta)*cond
    
    sig[:] = -(-invlap*(kh**2*N_b**2 +f_corr**2 *kz**2 ))**0.5
    
    f1uk[0,:] = (pk* (1j*ky + kx*sig)/(sig**2 - f_corr**2+ 1e-16))*np.exp(1j*sig*t)
    f1uk[1,:] = (pk* (-1j*kx + ky*sig)/(sig**2 - f_corr**2+ 1e-16))*np.exp(1j*sig*t)
    f1bk[:] = (1j*kz*N_b/(N_b**2 - sig**2 + 1e-16)*pk)*np.exp(1j*sig*t)
    f1uk[2,:] = (1j*sig*bk/N_b)*np.exp(1j*sig*t)
    
    neg_corr = comm.allreduce(np.sum(normalize*(np.einsum('i...,i...->...',np.conjugate(uk),f1uk) + np.conjugate(bk)*f1bk).real), op = MPI.SUM)
    # ---------------------------------------------------------- #
    # ------------------- positive frequency ------------------- #
    theta[:] = np.random.uniform(0,TWO_PI,(N,Np,Nf))
    pk[:] = np.exp(1j*theta)*cond
    
    sig[:] = (-invlap*(kh**2*N_b**2 +f_corr**2 *kz**2 ))**0.5
    
    f2uk[0,:] = (pk* (1j*ky + kx*sig)/(sig**2 - f_corr**2+ 1e-16))*np.exp(1j*sig*t)
    f2uk[1,:] = (pk* (-1j*kx + ky*sig)/(sig**2 - f_corr**2+ 1e-16))*np.exp(1j*sig*t)
    f2bk[:] = (1j*kz*N_b/(N_b**2 - sig**2 + 1e-16)*pk)*np.exp(1j*sig*t)
    f2uk[2,:] = (1j*sig*bk/N_b)*np.exp(1j*sig*t)
    
    pos_corr = comm.allreduce(np.sum(normalize*(np.einsum('i...,i...->...',np.conjugate(uk),f2uk) + np.conjugate(bk)*f2bk).real), op = MPI.SUM)
    
    # ---------------------------------------------------------- #
    norm = comm.allreduce(np.sum(normalize*(np.einsum('i...,i...->...',np.conjugate(f1uk),f1uk) + np.conjugate(f1bk)*f1bk).real))**0.5
    
    if np.abs(neg_corr) > 1e-6*(2*f0*h)**0.5*norm: 
        beta = -pos_corr/neg_corr
        fk[:] = beta*f1uk  + f2uk
        fkb[:] = beta*f1bk  + f2bk

    else: 
        beta = 1.0
        fk[:] = beta*f1uk  
        fkb[:] = beta*f1bk 
        
    alpha = (2.0*f0/comm.allreduce(np.sum(normalize*(np.einsum('i...,i...->...',np.conjugate(fk),fk) + np.conjugate(fkb)*fkb).real))/h)**0.5
    return alpha*fk, alpha*fkb


uk = (np.random.random((3,N,Np,Nf)) + 1j*np.random.random((3,N,Np,Nf)))*(kz> 0) 
bk = (np.random.random((N,Np,Nf)) + 1j*np.random.random((N,Np,Nf)))*(kz> 0)

cond  = (kh>=shell_no[0])*(kh<=shell_no[-1])*(kz > 1)*(kz<6)

fk[:],fkb[:] = forcing(0,cond,f0*nshells,uk,bk)

correlation = comm.allreduce(np.sum(normalize*(np.einsum('i...,i...->...',np.conjugate(uk),fk) + np.conjugate(bk)*fkb).real), op = MPI.SUM)
pinput = comm.allreduce(np.sum(normalize*(np.einsum('i...,i...->...',np.conjugate(fk),fk) + np.conjugate(fkb)*fkb).real))*0.5*dt

fk[:],fkb[:] = forcing_old(uk,bk)
pinput_old = comm.allreduce(np.sum(normalize*(np.einsum('i...,i...->...',np.conjugate(uk),fk) + np.conjugate(bk)*fkb).real),op = MPI.SUM)
if rank ==0 : print(f" Coorelation : {correlation}, power input : {pinput}, prescribed : {f0*nshells}, old power input : {pinput_old}")
# %%
