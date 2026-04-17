""" The idea
1. Select a time window with a particular dt such that the sepearation between around omega = 1 is clear. 
2. Load the data serially in that window. 
3. Split the data across the frequecy. 
4. Ifft the data and save. 

#! To save storage space, we will only save the fast data. When required, the fast data can be computed on the fly using the fast data and the original data.

6. Write a separate code that loads the split data and computes the transfer and the flux. 
"""

import numpy as np
# from pyfftw.interfaces.scipy_fft import fftfreq,fftn, ifftn,dct,dst,rfft,irfft
from scipy.fft import fftfreq,fftn, ifftn,dct,dst,rfft,irfft #type: ignore
import pathlib
import os,sys,json,time
from mpi4py import MPI
# from pyevtk.hl import imageToVTK

curr_path = pathlib.Path("/mnt/pfs/rajarshi.chattopadhyay/boussinesq/spectrum-development/")


# ## ---------------MPI things--------------
# comm = MPI.COMM_WORLD
# num_process =  comm.Get_size()
# rank = comm.Get_rank()
# ## ---------------------------------------


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
aa_v = N**3*param["Balanced Forcing amplitude"]
aa_w = N**3*param["Wave Forcing amplitude"]
einit = N**3*param["Initial balanced amplitude"]
kinit = param["k limit"]
omega = param["Forcing frequency"]
forcestart = param["forcestart"]
low_wave = param["Low wave"]
## -------------------------------------------------------

## --------------- Params ----------------
TWO_PI = 2*np.pi
PI = np.pi
Ti = 780 - 10*PI
Tf = Ti + 10*PI
nu = 1e-32
N = 384
nPts = 30
times_o = np.arange(Ti,Tf,0.1)
Ntimes = len(times_o)
freqs = 2*np.pi/(times_o[-1] - times_o[0])*fftfreq(Ntimes,1./Ntimes)[:Ntimes//2 + 1]
freqs[-1] = abs(freqs[-1])

cond = freqs>0.8

print(f"final time = {(Tf)/PI :.0f} pi")
## ---------------------------------------

## -------------- paths ------------------
lw = "_LW" if low_wave else ""
omega = 1.7277
loadPath = pathlib.Path(f"/mnt/pfs/rajarshi.chattopadhyay/boussinesq/spectrum-development/nu_{nu}_N_{N}/Ro_{ro}/forcedTide_ring_{omega:.2f}{lw}")


with open(loadPath/"parameters.json") as f:
    param = json.load(f)
    num_slabs = param["num_processes"]
    
## ---------------------------------------

# ------------------ Grid ------------------------
Nf = N//2 + 1
Np = N//num_slabs
PI = np.pi
TWO_PI = 2*PI
L = TWO_PI
Lz = PI
X = Y = np.linspace(0, L, N, endpoint= False)
Z = np.linspace(0+ Lz/(2*N), Lz + Lz/(2*N), N, endpoint= False)
dx = X[1] - X[0]
dy = Y[1] - Y[0]
dz = Z[1] - Z[0]
Kx = Ky = fftfreq(N,  1./N)*TWO_PI/L
Kzc = np.arange(N)* PI/Lz

Kz = Kx
# Kz = Kzc.copy()
# Kz[N//2:] = Kz[N//2: ] - N
dkx = Kx[1] -Kx[0]
dky = Ky[1] -Ky[0]
dkz = Kzc[1] -Kzc[0]
# ------------------------------------------------

## ---------- Empty arrays ---------------
ut = np.empty((3,Ntimes,Np,N,N))
bt = np.empty((Ntimes,Np,N,N))

u1t = np.empty((3,Ntimes,Np,N,N))
b1t = np.empty((Ntimes,Np,N,N))


uom = np.empty((3,Ntimes//2 + 1,Np,N,N),dtype= np.complex128)
bom = np.empty((Ntimes//2+1,Np,N,N),dtype= np.complex128)

u_fast = np.empty((3,Ntimes,Np,N,N))
b_fast = np.empty((Ntimes,Np,N,N))

u1_fast = np.empty((3,Ntimes,Np,N,N))
b1_fast = np.empty((Ntimes,Np,N,N))

T_fast = np.zeros(Ntimes)

## ---------------------------------------
print(f"stating to load the data")

for kk in range(N//num_slabs):
    for jj,time in enumerate(times_o):

        file = np.load(loadPath/f"time_{time:.1f}/Fields_{kk}.npz")
        ut[0,jj,...] = file["u"].ravel()
        ut[1,jj,...] = file["v"].ravel()
        ut[2,jj,...] = file["w"].ravel()
        bt[jj,...] = file["b"].ravel()
        
        file = np.load(loadPath/f"time_{time:.1f}/RHS_{kk}.npz")
        u1t[0,jj,...] = file["u1"].ravel()
        u1t[1,jj,...] = file["v1"].ravel()
        u1t[2,jj,...] = file["w1"].ravel()
        b1t[jj,...] = file["b1"].ravel()

        
        
        print(f"Loading for time {time:.2f}: Done!")

    
    uom[:] = rfft(ut,axis = 1)/Ntimes
    bom[:] = rfft(bt,axis = 0)/Ntimes

    u_fast[:] = irfft(uom*cond[None,:,None,None,None],axis = 1)*Ntimes
    b_fast[:] = irfft(bom*cond[:,None,None,None],axis = 0)*Ntimes
    
    uom[:] = rfft(u1t,axis = 1)/Ntimes
    bom[:] = rfft(b1t,axis = 0)/Ntimes
    
    u1_fast[:] = irfft(uom*cond[None,:,None,None,None],axis = 1)*Ntimes
    b1_fast[:] = irfft(bom*cond[:,None,None,None],axis = 0)*Ntimes
    
    T_fast += np.sum(u_fast[0]*u1_fast[0] + u_fast[1]*u1_fast[1] + u_fast[2]*u1_fast[2]/alph**2 + b_fast*b1_fast,axis = (1,2,3))
    print(f"Saving the data for slab {kk}")

    ## ----------------- Saving the data ----------------
    for jj,time in enumerate(times_o):
        savePath = loadPath/f"time_{time:.1f}/"
        try: savePath.mkdir(parents=True,  exist_ok=True)
        except FileExistsError: pass
        np.savez(savePath/f"Fields_fast_{kk}.npz",u = u_fast[:,jj],b = b_fast[jj])
        np.savez(savePath/f"RHS_fast_{kk}.npz",u1 = u1_fast[:,jj],b1 = b1_fast[jj])
        # os.remove(loadPath/f"time_{time:.1f}/RHS_{kk}.npz")
    print(f"Data saved for slab {kk}")
np.savez(savePath/f"Transfer_fast.npz",T = T_fast[jj])
## --------------------------------------------------
