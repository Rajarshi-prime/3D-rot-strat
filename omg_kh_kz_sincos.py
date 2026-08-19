
#? Take u,v ,w , b from 30 random points in the grid. Calculate the energy spectrum in time for them. 
#? Average that and plot it as a function of frequency.


import numpy as np
import matplotlib.pyplot as plt
# from pyfftw.interfaces.scipy_fft import fft ,  ifft ,  irfft2 ,  rfft2 , irfftn ,  rfftn, fftfreq, dst, dct, idst, idct, rfft,  irfft
from scipy.fft import fft ,  ifft ,  irfft2 ,  rfft2 , irfftn ,  rfftn, fftfreq, dst, dct, idst, idct, rfft,  irfft, rfftn
import pathlib
import matplotlib as mpl 
mpl.rc('text', usetex = True)
import os,sys,json,tracemalloc
from mpi4py import MPI
# from pyevtk.hl import imageToVTK
curr_path = pathlib.Path("/mnt/pfs/rajarshi.chattopadhyay/codes/boussinesq/")


## ---------------MPI things--------------
comm = MPI.COMM_WORLD
num_process =  comm.Get_size()
rank = comm.Get_rank()
## ---------------------------------------

## --------- Loading from the parameters file ------------
if float(sys.argv[-1]) == 1.0:
    with open(curr_path/"parameters_OT.json") as f: param = json.load(f) #! Does not initialize balanced flow.
else:
    with open(curr_path/"parameters.json") as f: param = json.load(f)
## --------- Loading from the parameters file ------------
    

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
aa_w = 0.0 if float(sys.argv[-1]) == 2.0 else N**3*param["Wave Forcing amplitude"]
einit = N**3*param["Initial balanced amplitude"]
kinit = param["k limit"]
omega = param["Forcing frequency"]
forcestart = param["forcestart"]
low_wave = param["Low wave"]
param["num_process"] = num_process

## -------------------------------------------------------

## ---------------------------------------

## --------------- Params ----------------
TWO_PI = 2*np.pi
PI = np.pi
Np = N//num_process
Nf = N//2 + 1
# Tf_glob = [2*PI ,4*PI] + [PI*i for i in range(5,101,5)]
# Tf = np.round(Tf_glob[int(sys.argv[-1])],1)
Ti = 130 #- 50*PI
Tf = Ti + 7.5*PI
nPts = N*N
omega = 1.7277
times_o = np.arange(Ti,Tf,0.1)
Ntimes = len(times_o)
freqs = 2*np.pi/(times_o[-1] - times_o[0])*fftfreq(Ntimes,1./Ntimes)[:Ntimes//2 + 1]
domega = freqs[1] - freqs[0]
if rank ==0 :print(f"domega = {freqs[1] - freqs[0]:.2e}")
freqs[-1] = abs(freqs[-1])
if rank ==0 :print(Ti,Tf,f"final time = {(Tf)/PI :.1f} pi")
shells = np.arange(-0.5,Nf)
shells[0] = 0.
## ---------------------------------------

## ------------ Paths --------------------
curr_path = pathlib.Path("/mnt/pfs/rajarshi.chattopadhyay/codes/boussinesq/")
# curr_path = pathlib.Path("/mnt/pfs/rajarshi.chattopadhyay/boussinesq/spectrum-development/")
# loadPath = curr_path/f"nu_{nu}_N_{N}/Ro_{ro}/forcedTide_ring_OB_{omega:.2f}"
loadPath = pathlib.Path(f"/mnt/pfs/rajarshi.chattopadhyay/boussinesq/data_final/nu_{nu}_N_{N}/Ro_{ro}/forcedTide_ring_OB")
# savePath = pathlib.Path(f"/mnt/pfs/rajarshi.chattopadhyay/codes/boussinesq/Plots/nu_{nu}_N_{N}/Ro_{ro}/forcedTide_ring_OB_{omega:.2f}")
savePath = pathlib.Path(f"/mnt/pfs/rajarshi.chattopadhyay/codes/boussinesq/Plots/nu_{nu}_N_{N}/Ro_{ro}/forcedTide_ring_OB")
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
    
elif (loadPath/f"parameters.json").exists():
    with open(loadPath/f"parameters.json") as f: param = json.load(f) #! Does not initialize balanced flow.
    ro = param["Rossby"]
    lp = param["hyperviscous"] 
    alph = param["Alpha"] 
    T = param["Final_time"]
    dt = param["time_step"]
    st = param["save_step_r"] 
    
else: 
    raise ValueError(f"Could not load parameters in {loadPath}")


# times = np.arange(0,1001,5)
num_slabs = 192
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
PI = np.pi
TWO_PI = 2 * PI
Nf = N // 2 + 1
Np = N // num_process
sx = slice(rank * Np, (rank + 1) * Np)
L = TWO_PI
Lz = PI
X = Y = np.linspace(0, L, N, endpoint= False)
dx,dy = X[1]-X[0], Y[1]-Y[0]
Z = np.linspace(0+ Lz/(2*N), Lz + Lz/(2*N), N, endpoint= False)
dz = Z[1]- Z[0]
x, y, z = np.meshgrid(X[sx], Y, Z, indexing='ij')

Kx = Ky = fftfreq(N,  1./N)*TWO_PI/L
Kzc = np.arange(N)* PI/Lz
Kzs = Kzc + 1
kx,  ky,  kzc = np.meshgrid(Kx,  Ky[sx],  Kzc,  indexing = 'ij')
kzs = kzc + 1
kzc_int = (np.round(kzc)).astype(int)
kzs_int = (np.round(kzs)).astype(int)
normalize_c = dx*dy*dz*np.where(kzc_int == 0, 0.25,0.5 )/(N**3) 
normalize_s = dx*dy*dz*np.where(kzs_int == N, 0.25,0.5)/(N**3)
del kzc_int,kzs_int
K = np.arange(Nf)
kc = (kx**2 + ky**2 + kzc**2 )**0.5
ks = (kx**2 + ky**2 + kzs**2 )**0.5
kxfull,kyfukk = np.meshgrid(Kx,  Ky,indexing = 'ij')
khfull = (kxfull**2 + kyfukk**2)**0.5
dkx = Kx[1] -Kx[0]
dky = Ky[1] -Ky[0]
dkz = Kzc[1] -Kzc[0]

# ------------------------------------------------

# -------------- creating functions --------------
       
def y_to_z(aa,bb):
    """reshapes any scalar array slabbed in x direction to an array slabbed in z direction

    Args:
        aa (nd array): array slabbed in y direction
        arr_theta (nd array): array required in to pass in the Alltoall function
        bb (nd array): array slabbed in z direction
    Returns:
        (nd array): array slabbed in z direction
    """
    arr_theta[:] = np.moveaxis(aa.reshape(N,Np,num_process,Np),[0,1,2,3],[2,1,0,3]) 
    comm.Alltoall([arr_theta,  MPI.DOUBLE], [arr_theta_1,  MPI.DOUBLE])
    bb[:] = np.moveaxis(arr_theta_1,[0,1,2],[1,0,2])
    
    return bb

def sin_to_cos(x):
    """reshapes any dst transformed array to one in dct form (in axis =2 )

    Args:
        x (nd array): dst appropriate array

    Returns:
        (nd array) : dct appropriate array
    """
    
    
    x[:] = np.roll(x, 1, axis = -1)
    x[:, :, 0] = 0.
    return x


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
def e2d_to_e1d(x): #1 Based on whether k is 2D or 3D, it will bin the data accordingly. 
    return np.histogram(khfull.ravel(),bins = shells,weights = x.ravel())[0]
# ------------------------------------------------
# ---------- Empty arrays -------------


u = np.empty((3,Np,N,N))
b = np.empty((Np,N,N))

uk = np.empty((3,N,Np,N),dtype = np.complex128)
bk = np.empty((N,Np,N),dtype=np.complex128)

ukz = np.empty((3,N,N,Np),dtype = np.complex128)
bkz = np.empty((N,N,Np),dtype=np.complex128)

arr_theta = np.zeros((num_process,Np,N,Np),dtype = np.complex128)
arr_theta_1 = np.zeros((N,N,Np),dtype = np.complex128)

arr_temp_r = np.zeros((Np, N, N),dtype = np.float64)
arr_temp_k = np.zeros((N, Np, N),dtype= np.float64)
arr_temp_fr = np.zeros((Np, N, N), dtype= np.complex128)      
arr_temp_ifr = np.zeros((N, Np, N), dtype= np.complex128)
arr_mpi = np.zeros((num_process,  Np,  Np, N), dtype= np.complex128)
arr_mpi_r = np.zeros((num_process,  Np,  Np, N), dtype= np.float64)

ukt = np.zeros((Ntimes,N,N,Np),dtype = np.complex128)
vkt = np.zeros((Ntimes,N,N,Np),dtype = np.complex128)
wkt = np.zeros((Ntimes,N,N,Np),dtype = np.complex128)
bkt = np.zeros((Ntimes,N,N,Np),dtype = np.complex128)


for jj,time in enumerate(times_o):
    if rank == 0: print(f"Time {time:.2f}")
    
    
    load_num_slabs = num_slabs

    data_per_rank = N//load_num_slabs
    rank_data = range(rank*Np,(rank + 1)*Np) # The rank contains these slices 
    slab_old = np.inf
    for lidx,j in enumerate(rank_data):
        slab = j//data_per_rank
        idx = j%data_per_rank
        
        # print(f"Rank {rank} is loading slab {slab} and idx {idx}")
        
        """Loading the truncated data"""
        if slab_old != slab:  
            file = np.load(loadPath/f"time_{time:.1f}/Fields_{rank}.npz")

        slab_old = slab
        u[0,lidx] = file['u'][idx]
        u[1,lidx] = file['v'][idx]
        u[2,lidx] = file['w'][idx]
        b[lidx] = file['b'][idx]

    if rank ==0 : print(f"Loading for time {time:.2f}: Done!")
    
    uk[0] = fft_cos(u[0],uk[0])
    uk[1] = fft_cos(u[1],uk[1])
    uk[2] = fft_sin(u[2],uk[2])
    bk[:] = fft_sin(b,bk)
    
    #? making w and b in cos form because we are interested in the energies
    uk[2] = sin_to_cos(uk[2])
    bk[:] = sin_to_cos(bk)
    
        
    ukz[0] = y_to_z(uk[0]*(normalize_c**0.5),ukz[0])
    ukz[1] = y_to_z(uk[1]*(normalize_c**0.5),ukz[1])
    ukz[2] = y_to_z(uk[2]*(normalize_c**0.5),ukz[2])
    bkz[:] = y_to_z(bk*(normalize_c**0.5),bkz)
    
    ukt[jj] = ukz[0]
    vkt[jj] = ukz[1]
    wkt[jj] = ukz[2]
    bkt[jj] = bkz
    if rank ==0 : print(f"Storing for time {time:.2f}: Done!")

# ent = dx*dy*dz*comm.allreduce(np.sum(bkt**2),op = MPI.SUM)
del ukz,bkz
if rank ==0: print(f"ukt shape = {ukt.shape}")
e_uk_omg = np.abs(fft(ukt,axis = 0))**2
del ukt
e_vk_omg = np.abs(fft(vkt,axis = 0))**2
del vkt
e_wk_omg = np.abs(fft(wkt,axis = 0))**2
del wkt
e_bk_omg = np.abs(fft(bkt,axis = 0))**2
del bkt
if rank ==0: print("FFT done!")
# enw = dkx*dky*dkz*comm.allreduce(np.sum(np.abs(bk_omg)**2),op = MPI.SUM)
# if rank ==0: print(f"Energy = {ent:.2e}, Energy in w = {enw:.2e}, ratio = {enw/ent:.2e}")
# snapshot = tracemalloc.take_snapshot()
# top_stats = snapshot.statistics('lineno')

# if rank ==0: 
#     print("[Top 10]")   
#     for stat in top_stats[:10]:
#         print(stat)
e = 0.5*domega*comm.allreduce(np.sum(e_uk_omg + e_vk_omg + e_wk_omg/alph**2 + e_bk_omg),op = MPI.SUM)
if (Kzc[sx]<Nf).any(): 
    e_u_omg_kh_kz = np.zeros((3,Ntimes,Nf,Np),dtype = np.float64)
    e_b_omg_kh_kz = np.zeros((Ntimes,Nf,Np),dtype = np.float64)
    for jj,time in enumerate(times_o):
        for kk in range(Np):
                e_u_omg_kh_kz[0,jj,:,kk]  = e2d_to_e1d(e_uk_omg[jj,...,kk])
                e_u_omg_kh_kz[1,jj,:,kk]  = e2d_to_e1d(e_vk_omg[jj,...,kk])
                e_u_omg_kh_kz[2,jj,:,kk]  = e2d_to_e1d(e_wk_omg[jj,...,kk])
                e_b_omg_kh_kz[jj,:,kk] = e2d_to_e1d(e_bk_omg[jj,...,kk])
    if rank ==0: print("Energy_shape: ",e_u_omg_kh_kz.shape)
    np.savez_compressed(savePath/f"omg_kh_kzslab_{rank}.npz",e_u_omg_kh_kz = e_u_omg_kh_kz,e_b_omg_kh_kz = e_b_omg_kh_kz,energy = e )