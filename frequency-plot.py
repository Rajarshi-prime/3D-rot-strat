
#? Take u,v ,w , b from 30 random points in the grid. Calculate the energy spectrum in time for them. 
#? Average that and plot it as a function of frequency.


import numpy as np
import matplotlib.pyplot as plt
# from pyfftw.interfaces.scipy_fft import fft ,  ifft ,  irfft2 ,  rfft2 , irfftn ,  rfftn, fftfreq, dst, dct, idst, idct, rfft,  irfft
from scipy.fft import fft ,  ifft ,  irfft2 ,  rfft2 , irfftn ,  rfftn, fftfreq, dst, dct, idst, idct, rfft,  irfft
import pathlib
import matplotlib as mpl 
mpl.rc('text', usetex = True)
import os
import sys
from mpi4py import MPI
# from pyevtk.hl import imageToVTK




## ---------------MPI things--------------
comm = MPI.COMM_WORLD
num_process =  comm.Get_size()
rank = comm.Get_rank()
## ---------------------------------------


## --------------- Params ----------------
TWO_PI = 2*np.pi
PI = np.pi
# Tf_glob = [2*PI ,4*PI] + [PI*i for i in range(5,101,5)]
# Tf = np.round(Tf_glob[int(sys.argv[-1])],1)
Ti = 780 - 100*PI
Tf = Ti + 100*PI
nu = 1e-31
N = 384
ro = 0.1
num_slabs = 192
nPts = int(N*N * (N//num_slabs))
times_o = np.arange(Ti,Tf,0.1)
Ntimes = len(times_o)
freqs = 2*np.pi/(times_o[-1] - times_o[0])*fftfreq(Ntimes,1./Ntimes)[:Ntimes//2 + 1]
print(f"domega = {freqs[1] - freqs[0]:.2e}")
freqs[-1] = abs(freqs[-1])
print(Ti,Tf,f"final time = {(Tf)/PI :.1f} pi")
omega = 1.7277
## ---------------------------------------

## ------------ Paths --------------------
curr_path = pathlib.Path("/mnt/pfs/rajarshi.chattopadhyay/boussinesq/spectrum-development/")
loadPath = curr_path/f"nu_{nu}_N_{N}/Ro_{ro}/forcedTide_ring_{omega:.2f}"
savePath = pathlib.Path(f"/mnt/pfs/rajarshi.chattopadhyay/codes/boussinesq/Plots/nu_{nu}_N_{N}/Ro_{ro}/forcedTide_ring_{omega:.2f}")
savePath.mkdir(parents=True,  exist_ok=True)
## ---------------------------------------

## ---------- rest of the parameters --------------
paramfile = (loadPath/f"params.txt")
if paramfile.exists():
    ## ---------- Beginning from existing data -----------
    """Load the parameters in the param file"""
    with open((loadPath/f"params.txt"),"r") as param_file:
        param = eval(param_file.read()) 
    # times = sorted([float(str(x).split("time_")[-1]) for x in (loadPath/f"E_k").iterdir() if "time_" in str(x)])
    # num_slabs = len([x for x in (loadPath/f"E_k/time_100.0").iterdir() if "e_" in str(x)])
    # print(num_slabs)
    ro = param["Rossby"]
    lp = param["hyperviscous"] 
    alph = param["Alpha"] 
    T = param["Final_time"]
    dt = param["time_step"]
    st = param["interval of saving indices"] 
    
    del paramfile,param_file,param
    # print(times)
    
else: 
    raise ValueError(str(loadPath),"not found")
if rank == 0:
    print(num_slabs)
    # print(len(times))



# times = np.arange(0,1001,5)
Ns = N//num_slabs

# times_o = np.copy(times)
time_range = Ntimes//num_process
# print(f"times_range Rank: {time_range}")
# if rank ==  num_process -1 : 
#     times = times[rank*(time_range):]
# else :
#     times = times[rank*(time_range):(rank+1)*time_range]
# times = [rank*2]
# print(f"rank {rank} : {times}")
## ------------------------------------------------

# ------------------ Grid ------------------------
Nf = N//2 + 1
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
kx,  ky,  kz = np.meshgrid(Kx,  Ky,  Kz ,  indexing = 'ij')
K = np.arange(Nf)
k = (kx**2 + ky**2 + kz**2 )**0.5
kh = np.mean((kx**2 + ky**2)**0.5,axis=2)
dkx = Kx[1] -Kx[0]
dky = Ky[1] -Ky[0]
dkz = Kzc[1] -Kzc[0]
# ------------------------------------------------

# -------------- creating functions --------------


# ---------- Empty arrays -------------
ut = np.empty((3,nPts, Ntimes))
bt = np.empty((nPts, Ntimes))

uom = np.empty((3,nPts, Ntimes//2+1),dtype= np.complex128)
bom = np.empty((nPts, Ntimes//2+1),dtype= np.complex128)
eom = np.empty((nPts, Ntimes//2+1),dtype= np.float64)

u = np.empty((3,N,N,N))
b = np.empty((N,N,N))

uk = np.empty((3,N,N,N),dtype = np.complex128)
bk = np.empty((N,N,N),dtype=np.complex128)

# ek = np.empty((N,N,N))
# ekTemp = np.empty((N,N,N))
# ek_w = np.empty((N,N,N))
# ek_v = np.empty((N,N,N))
# e1dk = np.empty(Nf)
# e1dk_w = np.empty(Nf)
# e1dk_v = np.empty(Nf)
# e = np.empty((N,N,N))
# eTemp = np.empty((N,N,N))
# e_w = np.empty((N,N,N))
# e_v = np.empty((N,N,N))
# Pi = np.ones((N,N,N))
# Pi_w = np.ones((N,N,N))
# Pi_v = np.ones((N,N,N))

# PIk = np.empty((N,N,N))
# PIk_w = np.empty((N,N,N))
# PIk_v = np.empty((N,N,N))
# PI1dk = np.empty(Nf)
# PI1dk_w = np.empty(Nf)
# PI1dk_v = np.empty(Nf)

# ekh = np.empty((N,N))
# ekh_w = np.empty((N,N))
# ekh_v = np.empty((N,N))
# e1dkh = np.empty(Nf)
# e1dkh_w = np.empty(Nf)
# e1dkh_v = np.empty(Nf)
    

# ekv = np.empty(N)
# ekv_w = np.empty(N)
# ekv_v = np.empty(N)

# pv = np.empty((N,N,N))
# if rank == 0:
#     E = np.empty(len(times_o))
#     ek_t = np.empty((Ntimes,N))
#     ekh_t = np.empty((Ntimes,N))
#     ekv_t = np.empty((Ntimes,N))

## -------------------------------------

## ------ Creating directories ----------

# Net PV
# new_dir_name = loadPath/f"3D_PV"
# new_dir = pathlib.Path(curr_path,  new_dir_name)
# new_dir.mkdir(parents=True,  exist_ok=True)

# Plots 
# new_dir_name = savePath/f"PV"
# new_dir = pathlib.Path(curr_path,  new_dir_name)
# new_dir.mkdir(parents=True,  exist_ok=True)

# new_dir_name = savePath/f"E_k-k"
# new_dir = pathlib.Path(curr_path,  new_dir_name)
# new_dir.mkdir(parents=True,  exist_ok=True)
# new_dir_name = savePath/f"PI_k-k"
# new_dir = pathlib.Path(curr_path,  new_dir_name)
# new_dir.mkdir(parents=True,  exist_ok=True)
# new_dir_name = savePath/f"E_kv-kv"
# new_dir = pathlib.Path(curr_path,  new_dir_name)
# new_dir.mkdir(parents=True,  exist_ok=True)
# new_dir_name = savePath/f"E_kh-kh"
# new_dir = pathlib.Path(curr_path,  new_dir_name)
# new_dir.mkdir(parents=True,  exist_ok=True)
## --------------------------------------
# times_o = np.arange(100,1600,100)

# times = times_o
# ektot = np.empty(len(times))
# etot = np.empty(len(times))
# eratio = np.empty(len(times))
# ekwtot = np.empty(len(times))
# ekvtot = np.empty(len(times))
# ektide = np.empty(len(times))
np.random.seed(23)
indices = np.random.randint(0,1,(nPts,3))
indices[:,0] = indices[:,0]*Ns
indices[:,1] = indices[:,1]*N
indices[:,2] = indices[:,2]*N
for jj,time in enumerate(times_o):
    print(f"Time {time:.2f}")
    # ekv[:] = 0.
    
    # for j in range(num_slabs):
    #     u[0,j*Ns:(j+1)*Ns,:] = np.load(loadPath/f"time_{time:.2f}/u_{j}.npy")
    #     u[1,j*Ns:(j+1)*Ns,:] = np.load(loadPath/f"time_{time:.2f}/v_{j}.npy")
    #     u[2,j*Ns:(j+1)*Ns,:] = np.load(loadPath/f"time_{time:.2f}/w_{j}.npy")
    #     b[j*Ns:(j+1)*Ns,:] = np.load(loadPath/f"time_{time:.2f}/b_{j}.npy")
    #     # ----------------------- Loading total energy -------------------------------------
    # for ii in range(nPts):
    #     #? The points along x are split into slabs.
    #     # slab = indices[ii,0]//Ns
    #     # slabidx = indices[ii,0]%Ns
    #     # # print(slab,ii,jj,slabidx,indices[ii,0],indices[ii,1])
    #     # ut[0,ii,jj] = np.load(loadPath/f"time_{time:.2f}/u_{slab}.npy")[slabidx,indices[ii,1],indices[ii,2]]
    #     # ut[1,ii,jj] = np.load(loadPath/f"time_{time:.2f}/v_{slab}.npy")[slabidx,indices[ii,1],indices[ii,2]]
    #     # ut[2,ii,jj] = np.load(loadPath/f"time_{time:.2f}/w_{slab}.npy")[slabidx,indices[ii,1],indices[ii,2]]
    #     # bt[ii,jj] = np.load(loadPath/f"time_{time:.2f}/b_{slab}.npy")[slabidx,indices[ii,1],indices[ii,2]]
    #     # since we have only one slab,we do not need splitting. 
        
    #     slab = 0
    #     slabidx = indices[ii,0]
    #     # print(slab,ii,jj,slabidx,indices[ii,0],indices[ii,1])
    #     file = np.load(loadPath/f"time_{time:.1f}/Fields_{slab}.npz")
        
    #     ut[1,ii,jj] = file["u"][slabidx,indices[ii,1],indices[ii,2]]
    #     ut[0,ii,jj] = file["v"][slabidx,indices[ii,1],indices[ii,2]]
    #     ut[2,ii,jj] = file["w"][slabidx,indices[ii,1],indices[ii,2]]
    #     bt[ii,jj] = file["b"][slabidx,indices[ii,1],indices[ii,2]]
    #     # ut[0,ii,jj] = np.sin(10*time) + 1
    #     # ut[1,ii,jj] = indices[ii,1]*np.cos(10*time)
    #     # ut[2,ii,jj] = indices[ii,2]*np.sin(30*time)
    #     # bt[ii,jj] = indices[ii,0]*np.cos(40*time)
    #     # ut[1,ii,jj] = 0.
    #     # ut[2,ii,jj] = 0.
    #     # bt[ii,jj] = 0.
    
    
    file = np.load(loadPath/f"time_{time:.1f}/Fields_{0}.npz")
    ut[0,...,jj] = file["u"].ravel()
    ut[1,...,jj] = file["v"].ravel()
    ut[2,...,jj] = file["w"].ravel()
    bt[...,jj] = file["b"].ravel()
    print(f"Loading for time {time:.2f}: Done!")
        
        
uom[:] = rfft(ut,axis = 2)/Ntimes
bom[:] = rfft(bt,axis = 1)/Ntimes
eom[:] = 0.5*(np.abs(uom[0])**2 + np.abs(uom[1])**2 + np.abs(uom[2])**2/alph**2 + np.abs(bom)**2)

eplot = np.mean(eom,axis = 0)
np.save(loadPath/f"e_omega_mean.npy",eplot)
eplot = np.load(loadPath/f"e_omega_mean.npy")
plt.figure(figsize=(8, 6))
plt.ylim(1e-8,5e-2)
plt.plot(freqs,eplot,'.-')#,markersize = ,color = #,label = )
plt.plot(np.arange(8,32),2e-4*np.arange(8,32)**(-2.),color = "#000000", linestyle='--')
plt.text(16,1e-7, r"$\omega^{-2}$", fontsize=20)
plt.plot(np.arange(4,22),2e-4*np.arange(4,22)**(-1.),color = "#000000", linestyle='--')
plt.text(8,5e-5, r"$\omega^{-1}$", fontsize=20)
lines = [[1e-2,1e-5],[3e-4,6e-6],[5e-4,5e-6],[5e-5,7e-6]]#,[5e-6,1e-8]]
textpos = np.array([[2,5e-3],[1.7,2e-6],[4,3e-4],[5.5,5e-5],[8,1e-6]])

for i in range(0,3):
    plt.plot([omega*(1+0.5*i)]*2,lines[i], color="#000000", linestyle='--') # Remove the label argument
    plt.text(textpos[i,0],textpos[i,1], fr"${i*0.5+1}\omega_t$", fontsize=20) # Add text beside the lines
plt.plot([omega*.5]*2,[1e-3,1e-5], color="#000000", linestyle='--') # Remove the label argument
plt.text(0.4,7e-4, fr"$0.5\omega_t$", fontsize=20) # Add text beside the lines
plt.yscale("log")
plt.xscale("log",base = 2)
# plt.xscale("log")
plt.xticks([2**i for i in [-4,-3,-2,-1,0,1,2,3,4,5]],fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel(r"$\omega$", fontsize=30)
plt.ylabel(r"$E_{\omega}$", fontsize=30,rotation = 0,labelpad = 20)
plt.grid( linestyle='--') # Add the linestyle parameter with value '--'

# plt.legend()
# plt.title(r"$E_{\omega}$ Frequecy spectrum")
plt.tight_layout()
plt.savefig(savePath/f"Frequency-Plot-fTime-{Ti:.1f}-{Tf:.1f}.png")
print(f"plotting done")
## -------- Testing the code --------------------
# fx = np.sin(50*times_o) + np.cos(70*times_o)
# fk = rfft(fx)



# plt.figure(figsize=(16, 12))
# plt.loglog(freqs,np.abs(fk),'.-')#,markersize = ,color = #,label = )
# plt.loglog([50., 50.],plt.ylim(),color = "#000000")
# plt.loglog([70., 70.],plt.ylim(),color = "#000000")
# plt.xticks(fontsize = 20)
# plt.yticks(fontsize = 20)
# plt.xlabel(r"$\omega$",fontsize = 30)
# plt.ylabel(r"$E_{\omega}$",fontsize = 30)
# # plt.legend()
# plt.title("Frequecy spectrum of sin(t)")
# plt.savefig(savePath/"Frequency-test-Plot.png")

## ----------------------------------------------
# plt.show()
# plt.show()




# """
#     time nohup mpirun -n 100 python -u plotting.py > errors-outputs/plotting.out &
# """