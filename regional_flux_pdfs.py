import numpy as np
from scipy.fft import fftn,ifftn, fft ,  ifft ,  irfft2 ,  rfft2 , irfftn ,  rfftn, fftfreq, dst, dct, idst, idct, rfft,  irfft
import matplotlib.pyplot as plt
from mpi4py import MPI
import pathlib,os,sys
import matplotlib as mpl 
import matplotlib.colors as colors
mpl.rc('text', usetex = True)
mpl.rcParams['font.size'] = 25

## ---------------MPI things--------------
comm = MPI.COMM_WORLD
num_process =  comm.Get_size()
rank = comm.Get_rank()
low_wave = True
## ---------------------------------------

N = 384
Np = N// num_process
ro = 0.1
omega = 1.7277
nu = 1e-31
alph = 20
num_slabs = 192
Ns = num_slabs// num_process
n_slab = N//num_slabs

if low_wave:
    factor = 1/0.5
    factor1 = 10
    loadPathdata = pathlib.Path(f"/mnt/pfs/rajarshi.chattopadhyay/boussinesq/spectrum-development/nu_{nu}_N_{N}/Ro_{ro}/forcedTide_ring_{omega:.2f}_LW")
    savePlot = pathlib.Path(f"/mnt/pfs/rajarshi.chattopadhyay/codes/boussinesq/Plots/nu_{nu}_N_{N}/Ro_{ro}/forcedTide_ring_{omega:.2f}_LW/filtered_flux")
    
else:
    factor = 1.0
    factor1 = factor
    loadPathdata = pathlib.Path(f"/mnt/pfs/rajarshi.chattopadhyay/boussinesq/spectrum-development/nu_{nu}_N_{N}/Ro_{ro}/forcedTide_ring_{omega:.2f}")
    savePlot = pathlib.Path(f"/mnt/pfs/rajarshi.chattopadhyay/codes/boussinesq/Plots/nu_{nu}_N_{N}/Ro_{ro}/forcedTide_ring_{omega:.2f}/filtered_flux")
    
savePath = loadPathdata/f"filtered_flux/"
try: savePlot.mkdir(parents = True,exist_ok = True)
except FileExistsError: pass
try: savePath.mkdir(parents = True,exist_ok = True)
except FileExistsError: pass
if rank == 0: print("saved in " ,str(savePath))
# loadPathdata.exists(),str(loadPath)
# len(os.listdir(loadPath))

u = np.zeros((3,Np,N,N))
b = np.zeros((Np,N,N))
q = np.empty((Np,N,N))
omg = np.empty((Np,N,N))
div_h = np.empty((Np,N,N))
omg_g = np.empty((Np,N,N))

# omg_factor = 0.25
# q_factor = 0.25
# div_h_factor = 0.25
# omg_g_factor = 0.25
factors = np.logspace(np.log10(0.1),np.log10(0.9),10)
q_fraction = np.zeros((len(factors),len(factors)))
omg_fraction = np.zeros((len(factors),len(factors)))
div_h_fraction = np.zeros((len(factors),len(factors)))
omg_g_fraction = np.zeros((len(factors),len(factors)))
                          
if rank ==0: print(factors)
# print(i,end = "\r")

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
kc = (kx**2 + ky**2 + kzc**2 )**0.5
kcint = np.clip(np.round(kc,0),0,N//2)
ks = (kx**2 + ky**2 + kzs**2 )**0.5
ksint = np.clip(np.round(ks,0),1,N//2+1)
kh = (kx**2 + ky**2 )**0.5
invlapc = -1/np.where(kc == 0, np.inf, kc**2)


kx_diff = kx[:Nf,  :, :].copy()
kx_diff[-1, :, :] = -kx_diff[-1, :, :]
ky_diff = np.swapaxes(kx_diff, 0, 1).copy()



## –--------------- empty arrays for mpi communication ----------------- ##
u_temp = np.zeros((Np,N,N))
v_temp = np.zeros((Np,N,N))
b_temp = np.zeros((Np,N,N))
pk = np.zeros((N,Np,N),dtype = np.complex128)

arr_theta = np.zeros((num_process, Np, N, Np),dtype = np.float64)
arr_temp_theta1 = np.zeros((N, N, Np),dtype = np.float64)

arr_temp_r = np.zeros((Np, N, N),dtype = np.float64)
arr_temp_k = np.zeros((N, Np, N),dtype= np.float64)
arr_temp_fr = np.zeros((Np, N, N), dtype= np.complex128)      
arr_temp_ifr = np.zeros((N, Np, N), dtype= np.complex128)
arr_mpi = np.zeros((num_process,  Np,  Np, N), dtype= np.complex128)
arr_mpi_r = np.zeros((num_process,  Np,  Np, N), dtype= np.float64)

u_filt = np.zeros_like(u)
b_filt = np.zeros_like(b)

tau = np.zeros((3,3,Np,N,N))
A = np.ones((3,3,Np,N,N))
B = np.zeros((3,Np,N,N))
grdB = np.zeros((3,Np,N,N))


pi_h = np.zeros((Np,N,N))
pi_v = np.zeros((Np,N,N))
pi_b = np.zeros((Np,N,N))
pih = np.zeros((N,N,Np))
piv = np.zeros((N,N,Np))
pib = np.zeros((N,N,Np))
pi = np.zeros((N,N,Np))
omgz = np.zeros((N,N,Np))
div_hz = np.zeros((N,N,Np))
omg_gz = np.zeros((N,N,Np))
sig_s = np.zeros((Np,N,N))
sig_n = np.zeros((Np,N,N))
sig = np.zeros((Np,N,N))
sigz = np.zeros((N,N,Np))
qz = np.zeros((N,N,Np))
q_cond = np.zeros((N,N,Np),dtype = bool)
omg_cond = np.zeros((N,N,Np),dtype = bool)
div_h_cond = np.zeros((N,N,Np),dtype = bool)
omg_g_cond = np.zeros((N,N,Np),dtype = bool)
pi_cond = np.zeros((N,N,Np),dtype = bool)

## --------------------------------------------------------------------- ##

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
    u_temp[:] = np.roll(Kzs*dst(u,type = 2,  axis = 2), 1, axis = 2)
    u_temp[:, :, 0] = 0
    u_z[:] = idct(u_temp,type = 2, axis = 2) 
    return u_z
    
    
def diff_z_cos(u, u_z):
    u_temp[:] = - np.roll(Kzc*dct(u,type =2,  axis = 2), -1, axis = 2)
    u_temp[:, :, -1] = 0
    u_z[:] = idst(u_temp,type = 2, axis = 2) 
    return u_z   

def del_cos(i,u,u_i):
    if i ==0 : 
        return diff_x(u,u_i)
    elif i ==1 : 
        return diff_y(u,u_i)
    elif i == 2: 
        return diff_z_cos(u,u_i)
    
def del_sin(i,u,u_i):
    if i ==0 : 
        return diff_x(u,u_i)
    elif i ==1 : 
        return diff_y(u,u_i)
    elif i == 2: 
        return diff_z_sin(u,u_i)
        
        
def x_to_z(a,b):
    """reshapes any scalar array slabbed in x direction to an array slabbed in z direction

    Args:
        a (nd array): array slabbed in z direction
        c (nd array): array required in to pass in the Alltoall function
        b (nd array): array slabbed in y direction but z axis is in front
    Returns:
        (nd array): array slabbed in y direction
    """
    arr_theta[:] = np.moveaxis(a.reshape(Np,N,num_process,Np),[0,1,2,3],[1,2,0,3]) 
    comm.Alltoall([arr_theta,  MPI.DOUBLE], [b,  MPI.DOUBLE])
    # print(arr_temp_theta1.shape)
    # b[:] = np.moveaxis(arr_temp_theta1,[0,1,2],[2,0,1])
    return b
    
def all_mean(x):
    return comm.allreduce(np.sum(x), op = MPI.SUM)/N**3
def all_std(x,x_mean):
    return np.sqrt(comm.allreduce(np.sum((x - x_mean)**2) , op = MPI.SUM))/N**3

pi_abs_max = 0.0028263017260083466 if low_wave else 1.0506167545921938

    
pih_max = 0.05/factor1
pih_min = -0.05/factor1
piv_max = 0.05/factor1
piv_min = -0.05/factor1
pib_max = 0.05/factor1
pib_min = -0.05/factor1
pi_max = pi_abs_max
pi_min = -pi_abs_max

k_filts = np.array([20,40,60])


omg_max = 30/factor
omg_min = -30/factor

q_max = 15/factor
q_min = -15/factor

div_h_max = 15/factor
div_h_min = -15/factor

omg_g_max = 15/factor
omg_g_min = -15/factor
    

pih_bins = np.linspace(pih_min,pih_max,601)
piv_bins = np.linspace(piv_min,piv_max,601)
pib_bins = np.linspace(pib_min,pib_max,601)
pi_bins = np.linspace(pi_min,pi_max,601)
omg_bins = np.linspace(omg_min,omg_max,301)
q_bins = np.linspace(q_min,q_max,301)
div_h_bins = np.linspace(div_h_min,div_h_max,301)
omg_g_bins = np.linspace(omg_g_min,omg_g_max,301)

sig_bins = np.linspace(0,30,301)
omg_sig_2Dhist = np.zeros((len(omg_bins[:-1]),len(sig_bins[:-1])))
pi_omg_sig_2Dhist = np.zeros((len(k_filts),len(omg_bins[:-1]),len(sig_bins[:-1])))

pih_hist = np.zeros((len(k_filts),len(pih_bins[:-1])))
piv_hist = np.zeros((len(k_filts),len(piv_bins[:-1])))
pib_hist = np.zeros((len(k_filts),len(pib_bins[:-1])))
pi_hist = np.zeros((len(k_filts),len(pi_bins[:-1])))
# pih_omg_hist = np.zeros((len(k_filts),len(omg_bins[:-1])))
# piv_omg_hist = np.zeros((len(k_filts),len(omg_bins[:-1])))
# pib_omg_hist = np.zeros((len(k_filts),len(omg_bins[:-1])))
pi_omg_hist = np.zeros((len(k_filts),len(omg_bins[:-1])))
pi_omg_hist_high = np.zeros((len(factors),len(k_filts),len(pi_bins[:-1])))
pi_omg_hist_low = np.zeros((len(factors),len(k_filts),len(pi_bins[:-1])))
# pih_q_hist = np.zeros((len(k_filts),len(q_bins[:-1])))
# piv_q_hist = np.zeros((len(k_filts),len(q_bins[:-1])))
# pib_q_hist = np.zeros((len(k_filts),len(q_bins[:-1])))
pi_q_hist = np.zeros((len(k_filts),len(q_bins[:-1])))
pi_q_hist_high = np.zeros((len(factors),len(k_filts),len(pi_bins[:-1])))
pi_q_hist_low = np.zeros((len(factors),len(k_filts),len(pi_bins[:-1])))
# pih_div_h_hist = np.zeros((len(k_filts),len(div_h_bins[:-1])))
# piv_div_h_hist = np.zeros((len(k_filts),len(div_h_bins[:-1]))) 
# pib_div_h_hist = np.zeros((len(k_filts),len(div_h_bins[:-1])))
pi_div_h_hist = np.zeros((len(k_filts),len(div_h_bins[:-1])))
pi_div_h_hist_high = np.zeros((len(factors),len(k_filts),len(pi_bins[:-1])))
pi_div_h_hist_low = np.zeros((len(factors),len(k_filts),len(pi_bins[:-1])))
# pih_omg_g_hist = np.zeros((len(k_filts),len(omg_g_bins[:-1])))
# piv_omg_g_hist = np.zeros((len(k_filts),len(omg_g_bins[:-1])))
# pib_omg_g_hist = np.zeros((len(k_filts),len(omg_g_bins[:-1])))
pi_omg_g_hist = np.zeros((len(k_filts),len(omg_g_bins[:-1])))
pi_omg_g_hist_high = np.zeros((len(factors),len(k_filts),len(pi_bins[:-1])))
pi_omg_g_hist_low = np.zeros((len(factors),len(k_filts),len(pi_bins[:-1])))

# pih_omg_2Dhist = np.zeros((len(k_filts),len(pih_bins[:-1]),len(omg_bins[:-1])))
# piv_omg_2Dhist = np.zeros((len(k_filts),len(pih_bins[:-1]),len(omg_bins[:-1])))
# pib_omg_2Dhist = np.zeros((len(k_filts),len(pih_bins[:-1]),len(omg_bins[:-1])))
# pi_omg_2Dhist = np.zeros((len(k_filts),len(pih_bins[:-1]),len(omg_bins[:-1])))

# pih_q_2Dhist = np.zeros((len(k_filts),len(pih_bins[:-1]),len(q_bins[:-1])))
# piv_q_2Dhist = np.zeros((len(k_filts),len(pih_bins[:-1]),len(q_bins[:-1])))
# pib_q_2Dhist = np.zeros((len(k_filts),len(pih_bins[:-1]),len(q_bins[:-1])))
# pi_q_2Dhist = np.zeros((len(k_filts),len(pih_bins[:-1]),len(q_bins[:-1])))

# pih_div_h_2Dhist = np.zeros((len(k_filts),len(pih_bins[:-1]),len(div_h_bins[:-1])))
# piv_div_h_2Dhist = np.zeros((len(k_filts),len(pih_bins[:-1]),len(div_h_bins[:-1])))
# pih_div_h_2Dhist = np.zeros((len(k_filts),len(pih_bins[:-1]),len(div_h_bins[:-1])))
# pi_div_h_2Dhist = np.zeros((len(k_filts),len(pih_bins[:-1]),len(div_h_bins[:-1])))

# pih_omg_g_2Dhist = np.zeros((len(k_filts),len(pih_bins[:-1]),len(omg_g_bins[:-1])))
# piv_omg_g_2Dhist = np.zeros((len(k_filts),len(pih_bins[:-1]),len(omg_g_bins[:-1])))
# pib_omg_g_2Dhist = np.zeros((len(k_filts),len(pih_bins[:-1]),len(omg_g_bins[:-1])))
# pi_omg_g_2Dhist = np.zeros((len(k_filts),len(pih_bins[:-1]),len(omg_g_bins[:-1])))


tflux = np.zeros(len(k_filts))
# t = 600.0
times = np.arange(900,1000,1.)
flux_mean = np.zeros((len(k_filts),len(times)))
flux_std = np.zeros((len(k_filts),len(times)))
div_omg_g_hist = 0.

omg_abs_max = 0. 
q_abs_max = 0.
div_h_abs_max = 0.
omg_g_abs_max = 0.

for jj,t in enumerate(times):
    if rank ==0: print(t, num_process,Ns)
    loadPath = loadPathdata/f"time_{t:.1f}"
    for qq in range(Ns):
        Fields = np.load(loadPath/f"Fields_{(rank*Ns + qq):.0f}.npz")
        u[0,qq*n_slab:(qq + 1)*n_slab]  = Fields["u"]
        u[1,qq*n_slab:(qq + 1)*n_slab]  = Fields["v"]
        u[2,qq*n_slab:(qq + 1)*n_slab]  = Fields["w"]
        b[qq*n_slab:(qq + 1)*n_slab]  = Fields["b"]
        # q[qq*n_slab:(qq + 1)*n_slab] = np.load(loadPathdata/f"PV/time_{t:.1f}/PV_{(rank*Ns + qq):.0f}.npy")
        # omg[qq*n_slab:(qq + 1)*n_slab] = np.load(loadPathdata/f"zeta/time_{t:.1f}/zeta_{(rank*Ns + qq):.0f}.npy")
    
    comm.Barrier()
    q[:] = diff_x(u[1],v_temp) - diff_y(u[0],u_temp) + diff_z_sin(b,b_temp)
    omg[:] = diff_x(u[1],v_temp) - diff_y(u[0],u_temp)
    omg_g[:] = -ifft_cos(kh**2*invlapc*fft_cos(q,pk),omg_g)
    div_h[:] = diff_x(u[0],u_temp) + diff_y(u[1],v_temp)
    # sig_s[:] = diff_x(u[1],u_temp) + diff_y(u[0],v_temp)
    # sig_n[:] = diff_x(u[0],u_temp) - diff_y(u[1],v_temp)
    # sig[:] = (sig_s**2 + sig_n**2)**0.5
    
    # omg_min = min(omg_min, comm.allreduce(np.min(omg), op = MPI.MIN))
    # omg_max = max(omg_max, comm.allreduce(np.max(omg), op = MPI.MAX))
    
    # q_min = min(q_min, comm.allreduce(np.min(q), op = MPI.MIN))
    # q_max = max(q_max, comm.allreduce(np.max(q), op = MPI.MAX))
    
    # div_h_min = min(div_h_min, comm.allreduce(np.min(div_h), op = MPI.MIN))
    # div_h_max = max(div_h_max, comm.allreduce(np.max(div_h), op = MPI.MAX))
    
    # omg_g_min = min(omg_g_min, comm.allreduce(np.min(omg_g), op = MPI.MIN))
    # omg_g_max = max(omg_g_max, comm.allreduce(np.max(omg_g), op = MPI.MAX))

    omg_abs_max = max(omg_abs_max, comm.allreduce(np.max(np.abs(omg)), op = MPI.MAX))
    q_abs_max = max(q_abs_max, comm.allreduce(np.max(np.abs(q)), op = MPI.MAX))
    div_h_abs_max = max(div_h_abs_max, comm.allreduce(np.max(np.abs(div_h)), op = MPI.MAX))
    omg_g_abs_max = max(omg_g_abs_max, comm.allreduce(np.max(np.abs(omg_g)), op = MPI.MAX))




for jj,t in enumerate(times):
    if rank ==0: print(t, num_process,Ns)
    loadPath = loadPathdata/f"time_{t:.1f}"
    for qq in range(Ns):
        Fields = np.load(loadPath/f"Fields_{(rank*Ns + qq):.0f}.npz")
        u[0,qq*n_slab:(qq + 1)*n_slab]  = Fields["u"]
        u[1,qq*n_slab:(qq + 1)*n_slab]  = Fields["v"]
        u[2,qq*n_slab:(qq + 1)*n_slab]  = Fields["w"]
        b[qq*n_slab:(qq + 1)*n_slab]  = Fields["b"]

    comm.Barrier()
    q[:] = diff_x(u[1],v_temp) - diff_y(u[0],u_temp) + diff_z_sin(b,b_temp)
    omg[:] = diff_x(u[1],v_temp) - diff_y(u[0],u_temp)
    sig_s[:] = diff_x(u[1],u_temp) + diff_y(u[0],v_temp)
    div_h[:] = diff_x(u[0],u_temp) + diff_y(u[1],v_temp)
    sig_n[:] = diff_x(u[0],u_temp) - diff_y(u[1],v_temp)
    sig[:] = (sig_s**2 + sig_n**2)**0.5
    omg_g[:] = -ifft_cos(kh**2*invlapc*fft_cos(q,pk),omg_g)
    

    omgz[:] = x_to_z(omg,omgz)
    qz[:] = x_to_z(q,qz)
    div_hz[:] = x_to_z(div_h,div_hz)
    omg_gz[:] = x_to_z(omg_g,omg_gz)
    sigz[:] = x_to_z(sig, sigz)
    
    
    
    if rank ==0: print("Calculated")
    
    # q_cond[:] = np.abs(qz)/q_abs_max > q_factor
    # omg_cond[:] = np.abs(omgz)/omg_abs_max > omg_factor
    # div_h_cond[:] = np.abs(div_hz)/div_h_abs_max > div_h_factor
    # omg_g_cond[:] = np.abs(omg_gz)/omg_g_abs_max > omg_g_factor
    
    


    k_filts = 1.0*np.array([20,40,60])
    for ii,k_filt in enumerate(k_filts):
        l_filt = TWO_PI/k_filt
        cond_c = (kcint< k_filt)
        cond_s = (ksint< k_filt)
        
        def filtered_cos(u,uf):
            uf[:] = ifft_cos(fft_cos(u,pk)*cond_c,uf)
            return uf

        def filtered_sin(u,uf):
            uf[:] = ifft_sin(fft_sin(u,pk)*cond_s,uf)
            return uf

        u_filt[0] = filtered_cos(u[0],u_filt[0])
        u_filt[1] = filtered_cos(u[1],u_filt[1])
        u_filt[2] = filtered_sin(u[2],u_filt[2])
        b_filt[:] = filtered_sin(b,b_filt)



        # # Calculating the filtered stress and strain tensor


        count = 0.
        for i in range(3):
            for j in range(i+1):
                if i < 2:
                    tau[i,j] = filtered_cos(u[i]*u[j],u_temp) - u_filt[i]*u_filt[j]
                    A[i,j] = (del_cos(j,u_filt[i],u_temp))
                    if i!= j:
                        A[j,i] = (del_cos(i,u_filt[j],v_temp))
                    count = count + 1
                elif i == 2 and j <2:
                    tau[i,j] = filtered_sin(u[i]*u[j],u_temp) - u_filt[i]*u_filt[j]
                    A[i,j] = (del_sin(j,u_filt[i],u_temp))
                    A[j,i] = (del_cos(i,u_filt[j],v_temp))
                    count = count + 1
                    
                else : 
                    tau[i,j] = filtered_cos(u[i]*u[j],u_temp) - u_filt[i]*u_filt[j]
                    A[i,j] = (del_sin(j,u_filt[i],u_temp))
                    count = count + 1
                
                tau[j,i] = tau[i,j]
            
            if i<2: 
                B[i] = filtered_sin(b*u[i],u_temp) - u_filt[i]* b_filt
            else: 
                B[i] = filtered_cos(b*u[i],u_temp) - u_filt[i]* b_filt
                
            grdB[i] = del_sin(i,b_filt,u_temp)


        mxtrc = comm.allreduce(np.max(np.abs(A[1,1] + A[2,2]+ A[0,0])),op = MPI.MAX)
        if rank ==0 : print(count, mxtrc)


        # # Calculating different flux components

        pi_h[:] = 0.
        for i in range(2):
            for j in range(2):
                    pi_h[:] = pi_h + tau[i,j]*A[j,i]

        pi_v[:] = 0.
        for i in range(2):
            pi_v[:] = pi_v + tau[2,i]*A[i,2] + 1/alph**2 *tau[i,2]*A[2,i]

        pi_v[:] = pi_v + 1/alph**2 *tau[2,2]*A[2,2]
            

        pi_b[:] = 0.
        for i in range(3):
            pi_b[:] = pi_b + B[i]*grdB[i]


        ## ------ corresponding with the notation --------------- ##
        pi_h[:] = - pi_h*ro
        pi_v[:] = - pi_v*ro
        pi_b[:] = - pi_b*ro
        
        pih[:] = x_to_z(pi_h,pih)
        piv[:] = x_to_z(pi_v,piv)
        pib[:] = x_to_z(pi_b,pib)
        
        
        pi[:] = pih + piv + pib
        
        pi_abs_max = max(pi_abs_max,comm.allreduce(np.max(np.abs(pi)), op = MPI.MAX))
        pi_hist[ii] += np.histogram(pi.ravel(), bins = pi_bins)[0]/len(times)
        
        for kkk,factor2 in enumerate(factors):
            q_cond[:] = np.abs(qz)/q_abs_max > factor2
            omg_cond[:] = np.abs(omgz)/omg_abs_max > factor2
            div_h_cond[:] = np.abs(div_hz)/div_h_abs_max > factor2
            omg_g_cond[:] = np.abs(omg_gz)/omg_g_abs_max > factor2
            
            # vol_fraq_q = comm.allreduce(q_cond.sum()/N**3, op= MPI.SUM)
            # vol_fraq_omg = comm.allreduce(omg_cond.sum()/N**3, op= MPI.SUM)
            # vol_fraq_div_h = comm.allreduce(div_h_cond.sum()/N**3, op= MPI.SUM)
            # vol_fraq_omg_g = comm.allreduce(omg_g_cond.sum()/N**3, op= MPI.SUM)
            
            # if rank ==0: print(f"vol fracs with factor {factor2:.1f} for q, omg, div_h and omg_g:{vol_fraq_q, vol_fraq_omg, vol_fraq_div_h, vol_fraq_div_h}")
            for jjj, factorpi in enumerate(factors):
                pi_cond[:] = np.abs(pi)/pi_abs_max > factorpi
                q_fraction[kkk,jjj] += np.sum(~pi_cond)*np.sum(~q_cond)/N**3/len(times)
                omg_fraction[kkk,jjj] += np.sum(~pi_cond)*np.sum(~omg_cond)/N**3/len(times)
                div_h_fraction[kkk,jjj] += np.sum(~pi_cond)*np.sum(~div_h_cond)/N**3/len(times)
                omg_g_fraction[kkk,jjj] += np.sum(~pi_cond)*np.sum(~omg_g_cond)/N**3/len(times)
                
                
            
            pi_q_hist_high[kkk,ii] += np.histogram(pi[q_cond].ravel(), bins = pi_bins)[0]/len(times)
            pi_q_hist_low[kkk,ii] += np.histogram(pi[~q_cond].ravel(), bins = pi_bins)[0]/len(times)
            
            pi_div_h_hist_high[kkk,ii] += np.histogram(pi[div_h_cond].ravel(), bins = pi_bins)[0]/len(times)
            pi_div_h_hist_low[kkk,ii] += np.histogram(pi[~div_h_cond].ravel(), bins = pi_bins)[0]/len(times)
            
            pi_omg_hist_high[kkk,ii] += np.histogram(pi[omg_cond].ravel(), bins = pi_bins)[0]/len(times)
            pi_omg_hist_low[kkk,ii] += np.histogram(pi[~omg_cond].ravel(), bins = pi_bins)[0]/len(times)
            
            pi_omg_g_hist_high[kkk,ii] += np.histogram(pi[omg_g_cond].ravel(), bins = pi_bins)[0]/len(times)
            pi_omg_g_hist_low[kkk,ii] += np.histogram(pi[~omg_g_cond].ravel(), bins = pi_bins)[0]/len(times)
            
        
pi_bin_width = pi_bins[1] - pi_bins[0]        

pi_hist[:] = comm.allreduce(pi_hist, op = MPI.SUM)
pi_hist[:] = pi_hist/(pi_bin_width*pi_hist.sum(axis = 1))[:,None]

if pi_q_hist_high.shape != (10,3,600): 
    print(f"rank {rank} has odd shape. " )

pi_q_hist_high[:] = comm.allreduce(pi_q_hist_high, op = MPI.SUM)
pi_q_hist_low[:] = comm.allreduce(pi_q_hist_low, op = MPI.SUM)
pi_q_hist_high[:] = pi_q_hist_high/(pi_bin_width*pi_q_hist_high.sum(axis = 2))[:,:,None]
pi_q_hist_low[:] = pi_q_hist_low/(pi_bin_width*pi_q_hist_low.sum(axis = 2))[:,:,None]

pi_div_h_hist_high[:] = comm.allreduce(pi_div_h_hist_high, op = MPI.SUM)
pi_div_h_hist_low[:] = comm.allreduce(pi_div_h_hist_low, op = MPI.SUM)
pi_div_h_hist_high[:] = pi_div_h_hist_high/(pi_bin_width*pi_div_h_hist_high.sum(axis = 2))[:,:,None]
pi_div_h_hist_low[:] = pi_div_h_hist_low/(pi_bin_width*pi_div_h_hist_low.sum(axis = 2))[:,:,None]

pi_omg_hist_high[:] = comm.allreduce(pi_omg_hist_high, op = MPI.SUM)   
pi_omg_hist_low[:] = comm.allreduce(pi_omg_hist_low, op = MPI.SUM)
pi_omg_hist_high[:] = pi_omg_hist_high/(pi_bin_width*pi_omg_hist_high.sum(axis = 2))[:,:,None]
pi_omg_hist_low[:] = pi_omg_hist_low/(pi_bin_width*pi_omg_hist_low.sum(axis = 2))[:,:,None]

pi_omg_g_hist_high[:] = comm.allreduce(pi_omg_g_hist_high, op = MPI.SUM)
pi_omg_g_hist_low[:] = comm.allreduce(pi_omg_g_hist_low, op = MPI.SUM)
pi_omg_g_hist_high[:] = pi_omg_g_hist_high/(pi_bin_width*pi_omg_g_hist_high.sum(axis = 2))[:,:,None]
pi_omg_g_hist_low[:] = pi_omg_g_hist_low/(pi_bin_width*pi_omg_g_hist_low.sum(axis = 2))[:,:,None]

q_fraction[:] = comm.allreduce(q_fraction, op = MPI.SUM)
omg_fraction[:] = comm.allreduce(omg_fraction, op = MPI.SUM)
div_h_fraction[:] = comm.allreduce(div_h_fraction, op = MPI.SUM)
omg_g_fraction[:] = comm.allreduce(omg_g_fraction, op = MPI.SUM)

if rank ==0:
    print(f"pi_abs_max,omg_abs_max,q_abs_max,div_h_abs_max,omg_g_abs_max:{pi_abs_max,omg_abs_max,q_abs_max,div_h_abs_max,omg_g_abs_max}")
    np.savez_compressed(savePath/f"vol_fraction.npz",q_fraction = q_fraction, omg_fraction = omg_fraction, div_h_fraction = div_h_fraction, omg_g_fraction = omg_g_fraction,factors = factors)
    np.savez_compressed(savePath/f"pi_hist.npz",pi_hist = pi_hist, pi_bins = pi_bins)
    np.savez_compressed(savePath/f"pi_q_hist_high_low_factors.npz",pi_q_hist_high = pi_q_hist_high, pi_q_hist_low = pi_q_hist_low, pi_bins = pi_bins)
    np.savez_compressed(savePath/f"pi_div_h_hist_high_low_factors.npz",pi_div_h_hist_high = pi_div_h_hist_high, pi_div_h_hist_low = pi_div_h_hist_low, pi_bins = pi_bins)
    np.savez_compressed(savePath/f"pi_omg_hist_high_low_factors.npz",pi_omg_hist_high = pi_omg_hist_high, pi_omg_hist_low = pi_omg_hist_low, pi_bins = pi_bins)
    np.savez_compressed(savePath/f"pi_omg_g_hist_high_low_factors.npz",pi_omg_g_hist_high = pi_omg_g_hist_high, pi_omg_g_hist_low = pi_omg_g_hist_low, pi_bins = pi_bins)
    
    print(f"saved in {savePath}")