
#? Take u,v ,w , b from 30 random points in the grid. Calculate the energy spectrum in time for them. 
#? Average that and plot it as a function of frequency.


import numpy as np
# from pyfftw.interfaces.scipy_fft import fft ,  ifft ,  irfft2 ,  rfft2 , irfftn ,  rfftn, fftfreq, dst, dct, idst, idct, rfft,  irfft
from scipy.fft import fft ,  ifft ,  irfft2 ,  rfft2 , irfftn ,  rfftn, fftfreq, dst, dct, idst, idct, rfft,  irfft, rfftn
import pathlib


import os,sys,json,h5py
from mpi4py import MPI
# from pyevtk.hl import imageToVTK
curr_path = pathlib.Path(__file__).parent
forcestart = False
idx = int(float(sys.argv[-1]))

## ---------------MPI things--------------
comm = MPI.COMM_WORLD
num_process =  comm.Get_size()
rank = comm.Get_rank()
## ---------------------------------------


N = 192
dt = 0.256/N   #! Such that increasing resolution will decrease the dt
f_corr = 1
N_bs = [15,20]
N_b = N_bs[idx]
T = 31.4/f_corr
dt_save = 0.1/f_corr
st = round(dt_save/dt)

## --------- Loading from the parameters file -----------
# param["num_process"] = num_process
## -------------------------------------------------------

## ---------------------------------------

## --------------- Params ----------------
PI = np.pi
TWO_PI = 2*PI
Nf = N//2 + 1
Np = N//num_process
sx = slice(rank*Np ,  (rank+1)*Np)
L = TWO_PI


Kx = Ky = fftfreq(N,  1./N)*TWO_PI/L
Kz = np.abs(Ky[:Nf])

kx,  ky,  kz = np.meshgrid(Kx,  Ky[sx],  Kz,  indexing = 'ij')
kh = (kx**2 + ky**2)**0.5
khint = np.round(kh).astype(np.int32)

Ti = 0 
Tf = T
times_o = np.arange(Ti,Tf,dt_save)
Ntimes = len(times_o)
freqs = 2*np.pi/(times_o[-1] - times_o[0])*fftfreq(Ntimes,1./Ntimes)[:Ntimes//2 + 1]
domega = freqs[1] - freqs[0]
freqs[-1] = abs(freqs[-1])
if rank ==0 :print(f"domega = {freqs[1] - freqs[0]:.2e}")
if rank ==0 :print(Ti,Tf,f"final time = {(Tf)/PI :.1f} pi")
shells = np.arange(-0.5,Nf)
shells[0] = 0.
## ---------------------------------------

## ------------ Paths --------------------
lp = 8 # Hyperviscosity power
nu0 = 0.5 #! Viscosity for N = 1
m = 1 #! Desired kmax*eta
nu = nu0*(3*m/N)**(2*(lp - 1/3))  #? scaling with resolution. For 512, nu = 0.002 #! Need to add scaling for hyperviscosity
re = np.inf if nu==0 else 1/nu

fbyN = f_corr/N_b if N_b != 0 else 0.0
einit = 1*TWO_PI**3 # Initial energy
nshells = 2 # Number of consecutive shells to be forced
shell_no = np.arange(4,4+nshells) # the shells to be forced
isforcing = True
if nu!= 0: loadPath = pathlib.Path(f"./data/bsnq/f_{f_corr:.1f}_Nb_{N_b:.1f}/forced_{isforcing}/N_{N}_Re_{re:.1f}")
else: loadPath = pathlib.Path(f"./data/bsnq/forced_{isforcing}/N_{N}_Re_inf")

arr_theta = np.zeros((num_process,Np,N,Np),dtype = np.complex128)
arr_theta_1 = np.zeros((N,N,Np),dtype = np.complex128)


# ------------------ Grid ------------------------
# ------------------------------------------------

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


def load_npz(paths,uk,bk):
    load_num_slabs = len([x for x in (paths).iterdir() if "Fields" in str(x) and ".npz" in str(x)])
    data_per_rank = N//load_num_slabs
    rank_data = range(rank*Np,(rank + 1)*Np) # The rank contains these slices 
    slab_old = np.inf
    for lidx,j in enumerate(rank_data):
        slab = j//data_per_rank
        idx = j%data_per_rank
        
        # print(f"Rank {rank} is loading slab {slab} and idx {idx}")
        
        """Loading the truncated data"""
        if slab_old != slab:  
            Field = np.load(paths/f"Fields_k_{slab}.npz")
        slab_old = slab
        uk[0,:,lidx] = Field['uk'][:,idx]
        uk[1,:,lidx] = Field['vk'][:,idx]
        uk[2,:,lidx] = Field['wk'][:,idx]
        bk[:,lidx] = Field['bk'][:,idx]
        
    
        # u[0,lidx] = load_trunc(Field['u'][idx])
        # u[1,lidx] = load_trunc(Field['v'][idx])
        # u[2,lidx] = load_trunc(Field['w'][idx])
        # b[lidx] = load_trunc(Field['b'][idx])

        
        
        """Loading the OG data"""
        # if slab_old != slab:  Field = np.load(paths/f"Fields_{slab}.npz")
        # slab_old = slab
        # u[0,lidx] = Field['u'][idx]
        # u[1,lidx] = Field['v'][idx]
        # u[2,lidx] = Field['w'][idx]
      

    return uk,bk

# -------------- creating functions --------------
def e2d_to_e1d(x): #1 Based on whether k is 2D or 3D, it will bin the data accordingly. 
    return np.histogram(khint.ravel(),bins = shells,weights = x.ravel())[0]
def e3d_to_e2d(x): #converst kx,ky,kz to kh,kz
    return np.histogram2d(khint.ravel(),kz.ravel(),bins = [shells,shells],weights = x.ravel())[0]
# ------------------------------------------------
# ---------- Empty arrays -------------



ukt = np.zeros((Ntimes,3,N,Np,Nf),dtype = np.complex128)
bkt = np.zeros((Ntimes,N,Np,Nf),dtype = np.complex128)


for jj,time in enumerate(times_o):
    # uk[:] = file["uk"]
    # bk[:] = file["bk"]
    # if rank ==0 : print(f"Loading for time {time:.2f}: Done!")
    
    
    ukt[jj],bkt[jj] = load_npz(loadPath/f"time_{time:.1f}",ukt[jj],bkt[jj])
    if rank ==0 : print(f"Loading for time {time:.2f}: Done!")

# ent = dx*dy*dz*comm.allreduce(np.sum(bkt**2),op = MPI.SUM)
if rank ==0: print(f"ukt shape = {ukt.shape}")
e_uk_omg = np.abs(fft(ukt,axis = 0))**2
e_bk_omg = np.abs(fft(bkt,axis = 0))**2
del bkt
if rank ==0:
    print("FFT done!")
    with h5py.File(loadPath/f"e_omg_kh_kz_ti_{Ti:.3f}_tf_{Tf:.3f}.hdf5","w") as f:
        f.create_dataset('e_u_omg_kh_kz',(Ntimes,3,Nf,Nf),dtype = np.float64,compression = 'gzip', chunks = (1,3,Nf,Nf))
        f.create_dataset('e_b_omg_kh_kz',(Ntimes,Nf,Nf),dtype = np.float64,compression = 'gzip', chunks = (1,Nf,Nf))
for jj,time in enumerate(times_o):
    e_u_omg_kh_kz = e3d_to_e2d(e_uk_omg[jj,0])
    e_v_omg_kh_kz = e3d_to_e2d(e_uk_omg[jj,1])
    e_w_omg_kh_kz = e3d_to_e2d(e_uk_omg[jj,2])
    e_b_omg_kh_kz = e3d_to_e2d(e_bk_omg[jj])
    if rank ==0: 
        with h5py.File(loadPath/f"e_omg_kh_kz_ti_{Ti:.3f}_tf_{Tf:.3f}.hdf5","r+") as f:  
            f['e_u_omg_kh_kz'][jj,0] = e_u_omg_kh_kz
            f['e_u_omg_kh_kz'][jj,1] = e_v_omg_kh_kz
            f['e_u_omg_kh_kz'][jj,2] = e_w_omg_kh_kz
            f['e_b_omg_kh_kz'][jj] = e_b_omg_kh_kz
            

# e = 0.5*domega*comm.allreduce(np.sum(e_uk_omg + e_bk_omg),op = MPI.SUM)
# extra = (len(times_o)%num_process)//rank 
# times_in_this_proc = times_o//num_process + extra
# times_loc = times_o//num_process
# if (Kz[sx]<Nf).any(): 
#     e_u_omg_kh_kz = np.zeros((3,Ntimes,Nf,Np),dtype = np.float64)
#     e_b_omg_kh_kz = np.zeros((Ntimes,Nf,Np),dtype = np.float64)
#     for jj,time in enumerate(times_o):
#         for kk in range(Np):
#                 e_u_omg_kh_kz[0,jj,:,kk]  = e2d_to_e1d(e_uk_omg[jj,...,kk])
#                 e_u_omg_kh_kz[1,jj,:,kk]  = e2d_to_e1d(e_vk_omg[jj,...,kk])
#                 e_u_omg_kh_kz[2,jj,:,kk]  = e2d_to_e1d(e_wk_omg[jj,...,kk])
#                 e_b_omg_kh_kz[jj,:,kk] = e2d_to_e1d(e_bk_omg[jj,...,kk])
#     if rank ==0: print("Energy_shape: ",e_u_omg_kh_kz.shape)
#     np.savez_compressed(savePath/f"omg_kh_kzslab_{rank}.npz",e_u_omg_kh_kz = e_u_omg_kh_kz,e_b_omg_kh_kz = e_b_omg_kh_kz,energy = e )