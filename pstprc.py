import numpy as np 
# from pyfftw.interfaces.scipy_fft import fft ,  ifft ,  irfft2 ,  rfft2 , irfftn ,  rfftn, fftfreq, dst, dct, idst, idct, rfft,  irfft
from scipy.fft import fft ,  ifft ,  irfft2 ,  rfft2 , irfftn ,  rfftn, fftfreq, dst, dct, idst, idct, rfft,  irfft
from mpi4py import MPI
from time import time
import pathlib
import os

curr_path = pathlib.Path(__file__).parent

## ---------------MPI things--------------
comm = MPI.COMM_WORLD
num_process =  comm.Get_size()
rank = comm.Get_rank()
## ---------------------------------------
## --------------- Params ----------------
nu = 1e-31
N = 3*128
ro = 0.1
omega = 1.7277
## ---------------------------------------


## ---------- rest of the parameters --------------
# loadPath = pathlib.Path("/home/rajpoot.rajendra/3D_Bous_codes/gaussian_vortex/local_test5/test_wave_vortex/seperate_test/Ro_0o1/neg_vortex/Ev_1o0Ew_rerun/omega_run_1_10/RESTART") #!  Rajendra's data
loadPath = pathlib.Path(f"/mnt/pfs/rajarshi.chattopadhyay/boussinesq/spectrum-development/nu_{nu}_N_{N}/Ro_{ro}/forcedTide_ring_{omega:.2f}_LW")
savePath = loadPath
savePlot = curr_path/"Plots/nu_{nu}_N_{N}/Ro_{ro}/forcedTide_ring_{omega:.2f}_LW"
# savePath = pathlib.Path(f"/mnt/pfs/rajarshi.chattopadhyay/data/nu_{nu}_N_{N}/tide") #! Rajendra's data

# savePath.mkdir(parents=True, exist_ok=True)
if rank == 0: 
    try:
        savePath.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        pass
comm.Barrier()
# paramfile = (loadPath/f"params.txt")
# if paramfile.exists():
#     ## ---------- Beginning from existing data -----------
#     """Load the parameters in the param file"""
#     with open((curr_path/loadPath/f"params.txt"),"r") as param_file:
#         param = eval(param_file.read()) 
#     paths = sorted([x for x in (curr_path/loadPath/f"").iterdir() if "time_" in str(x)], key=os.path.getmtime)
#     """The folder is paths[-1]"""
#     paths = paths[-1]
#     tlast = float(str(paths).split("time_")[-1])
    
#     ro = param["Rossby"]
#     lp = param["hyperviscous"] 
#     alph = param["Alpha"] 
#     num_slabs = param["Processes"] 
#     T = param["Final_time"]
#     dt = param["time_step"]
#     st = param["interval of saving indices"] 
    
#     del paramfile,param_file,param,paths
    
# else: 
#     MPI.Finalize()
#     raise ValueError("No data file found!")
    
    


num_slabs = 192
alph = 20
lp = 8

Ns = num_slabs//num_process
Nslab = N// num_slabs

if rank ==0: print(f"# slabs and Ns of the sim : {num_slabs},{Ns} ")
## -------------------------------------------------------


## -------------Defining the grid ---------------
PI = np.pi
TWO_PI = 2*PI
Nf = N//2 + 1
Np = N//num_process
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
u1_v = np.empty((3, Np, N, N), dtype= np.float64)
u1_w = np.empty((3, Np, N, N), dtype= np.float64)
p = u[0].copy()
b = u[0].copy()
b1 = np.empty_like(b)
b11 = np.empty_like(b)
b12 = np.empty_like(b)
b13 = np.empty_like(b)
b1_w = np.empty_like(b)
b1_v = np.empty_like(b)
dtQ = np.empty_like(b)
dtP = np.empty_like(b)
pv = np.empty_like(b)
pv_tide = np.empty_like(b)
e = np.empty_like(b)
Pi = np.ones_like(b)

uk = np.empty((3, N, Np, N), dtype= np.complex128)
u1k = np.empty((3, N, Np, N), dtype= np.complex128)
u1k_w = np.empty((3, N, Np, N), dtype= np.complex128)
u1k_v = np.empty((3, N, Np, N), dtype= np.complex128)
pk = uk[0].copy()
bk = pk.copy()
b1k = bk.copy()
b1k_w = bk.copy()
b1k_v = bk.copy()
pvk = bk.copy()
dtQk = bk.copy()
dtPk = bk.copy()
u_v  = np.empty((3,Np,N,N), dtype= np.float64)
b_v = u[0].copy()
p_v = b_v.copy()

uk_v = np.empty((3, N, Np, N), dtype= np.complex128)
bk_v = pk.copy()
pk_v = bk_v.copy()


u_w  = np.empty((3,Np,N,N), dtype= np.float64)
b_w = u[0].copy()

uk_w = np.empty((3,N,Np,N), dtype= np.complex128)
bk_w = pk.copy()

uk_w_z = np.empty((3,N,N,Np), dtype= np.complex128)
bk_w_z = np.empty((N,N,Np), dtype= np.complex128)

rhsu = np.empty_like(p)
rhsu1 = np.empty_like(p)
rhsu2 = np.empty_like(p)
rhsu3 = np.empty_like(p)
omgx = np.empty_like(p)
omgy = np.empty_like(p)
omgz = np.empty_like(p)
omg = np.empty_like(p)
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
div = p1.copy()
divH = p1.copy()
divH_tide = p1.copy()
divHk = pk.copy()
divHk_tide = np.empty((N,Np))



arr_temp_r = np.empty((Np, N, N),dtype = np.float64)
arr_temp_k = np.empty((N, Np, N),dtype= np.float64)
arr_temp_fr = np.empty((Np, N, N), dtype= np.complex128)      
arr_temp_ifr = np.empty((N, Np, N), dtype= np.complex128)
arr_mpi = np.empty((num_process,  Np,  Np, N), dtype= np.complex128)
arr_mpi_r = np.empty((num_process,  Np,  Np, N), dtype= np.float64)
arr_mpi_Z = np.empty((num_process,  Np, N,  Np), dtype= np.complex128)


ek = np.empty_like(pk,dtype = np.float64)
ek_tide = np.empty_like((N,Np),dtype = np.float64)
PIk = np.empty_like(ek,dtype = np.float64)
PIk_tide = np.empty_like(ek_tide,dtype = np.float64)
ek_w = np.empty_like(pk,dtype = np.float64)
ek_wtide = np.empty_like(ek_tide,dtype = np.float64)
PIk_w = np.empty_like(ek,dtype = np.float64)
PIk_wtide = np.empty_like(ek_tide,dtype = np.float64)
ek_v = np.empty_like(pk,dtype = np.float64)
ek_vtide = np.empty_like(ek_tide,dtype = np.float64)
PIk_v = np.empty_like(ek,dtype = np.float64)
PIk_vtide = np.empty_like(ek_tide,dtype = np.float64)
ek_avg = np.zeros_like(ek)
PIk_avg = np.zeros_like(PIk)
ek_v_avg = np.zeros_like(ek)
PIk_v_avg = np.zeros_like(PIk)
ek_w_avg = np.zeros_like(ek)
PIk_w_avg = np.zeros_like(PIk)
ekh = np.empty((N, Np),dtype= np.float64)
ekv = np.empty((N),dtype= np.float64)
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

def energy_flux(u,b,ek,PIk,uk,bk):
    """Calculate the 3D E_k from the given u,v,w,b and store return in ek

    Args:
        u (3*nd array): Velocity
        b (nd array): Buoyancy
        ek (nd array): Energy
        PI_k (nd array): Flux
    """
    
    
    u1[:],b1[:] = RHS(u,b,u1,b1)
    
    u1k[0,:] = np.conjugate(fft_cos(u1[0],u1k[0]))/(4*N**6/TWO_PI**3)**0.5
    u1k[1,:] = np.conjugate(fft_cos(u1[1],u1k[1]))/(4*N**6/TWO_PI**3)**0.5
    u1k[2,:] = np.conjugate(fft_sin(u1[2],u1k[2]))/(4*N**6/TWO_PI**3)**0.5
    b1k[:] = np.conjugate(fft_sin(b1,b1k))/(4*N**6/TWO_PI**3)**0.5
    
    uk[0,:] = fft_cos(u[0],uk[0])/(4*N**6/TWO_PI**3)**0.5
    uk[1,:] = fft_cos(u[1],uk[1])/(4*N**6/TWO_PI**3)**0.5
    uk[2,:] = fft_sin(u[2],uk[2])/(4*N**6/TWO_PI**3)**0.5
    bk[:] = fft_sin(b,bk)/(4*N**6/TWO_PI**3)**0.5
        
    uk[:2,kzc == 0] = 2**(-0.5)*uk[:2,kzc == 0]
    u1k[:2,kzc == 0] = 2**(-0.5)*u1k[:2,kzc == 0]
    
    uk[2,:] = sin_to_cos(uk[2])
    u1k[2,:] = sin_to_cos(u1k[2])
    bk[:] = sin_to_cos(bk)
    b1k[:] = sin_to_cos(b1k)
    
    ek[:] = 0.5*dkx*dky*dkz*(np.abs(uk[0])**2 + np.abs(uk[1])**2 + np.abs(uk[2])**2/alph**2 + np.abs(bk)**2)*dealias_cos
    PIk[:] = np.real(uk[0]*u1k[0]+uk[1]*u1k[1]+ uk[2]*u1k[2]/alph**2 + bk*b1k)*dkx*dky*dkz*dealias_cos
    
    return ek, PIk


## ---------------------------- Calculating total flux -------------------------------    
def all_energy_flux(u,b,u_w,b_w,u_v,b_v,ek,PIk,ek_w,PIk_w,ek_v,PIk_v):
    """Calculate the 3D E_k from the given u,v,w,b and store return in ek

    Args:
        u (3*nd array): Velocity
        b (nd array): Buoyancy
        ek (nd array): Energy
        PI_k (nd array): Flux
    """
    
    
    u1[:],b1[:] = RHS(u,b,u1,b1)
    
    
    """ Calculating dt Q* """
    dtQ[:] = diff_z_sin(b1,b11) - diff_y(u1[0],b12) + diff_x(u1[1],b13) ## b1k is cos 
    dtQk[:] = fft_cos(dtQ,dtQk)
    """ Calculating dtP* """
    dtPk[:] = invlapc*dtQk
    dtP[:] = ifft_cos(dtPk,dtP)
    
    """ Calculating dt balanced* """
    u1_v[0,:] = -diff_y(dtP,u1_v[0])
    u1_v[1,:] = diff_x(dtP, u1_v[1])
    u1_v[2,:] = 0.
    b1_v[:] = diff_z_cos(dtP,b1_v)
    
    u1_w[0,:] = u1[0] - u1_v[0]
    u1_w[1,:] = u1[1] - u1_v[1]
    u1_w[2,:] = u1[2] - u1_v[2]
    b1_w[:] = b1 - b1_v
    
    
    u1k_v[0,:] = np.conjugate(fft_cos(u1_v[0],u1k_v[0]))/(4*N**6/TWO_PI**3)**0.5
    u1k_v[1,:] = np.conjugate(fft_cos(u1_v[1],u1k_v[1])/(4*N**6/TWO_PI**3)**0.5)
    u1k_v[2,:] = np.conjugate(fft_sin(u1_v[2],u1k_v[2])/(4*N**6/TWO_PI**3)**0.5)
    b1k_v[:] = np.conjugate(fft_sin(b1_v,b1k_v)/(4*N**6/TWO_PI**3)**0.5)
    
    u1k_w[0,:] = np.conjugate(fft_cos(u1_w[0],u1k_w[0])/(4*N**6/TWO_PI**3)**0.5)
    u1k_w[1,:] = np.conjugate(fft_cos(u1_w[1],u1k_w[1])/(4*N**6/TWO_PI**3)**0.5)
    u1k_w[2,:] = np.conjugate(fft_sin(u1_w[2],u1k_w[2])/(4*N**6/TWO_PI**3)**0.5)
    b1k_w[:] = np.conjugate(fft_sin(b1_w,b1k_w)/(4*N**6/TWO_PI**3)**0.5)
    

    # """ Calculating dt Q* """
    # dtQk[:] = 1j*kzc*b1k - 1j*ky*u1k[0] + 1j*kx*u1k[1] ## b1k is cos 
    # """ Calculating dtP* """
    # dtPk[:] = invlapc*dtQk
    
    # """ Calculating dt balanced* """
    # u1k_v[0,:] = -1j*ky*dtPk
    # u1k_v[1,:] = 1j*kx*dtPk
    # u1k_v[2,:] = 0.
    # b1k_v[:] = 1j*kzc*dtPk ## This is already cos
    
    # """Calculating dt wave* """
    # u1k_w[0,:] = u1k[0] - u1k_v[0] 
    # u1k_w[1,:] = u1k[1] - u1k_v[1]
    # u1k_w[2,:] = u1k[2] - u1k_v[2] #Already cos
    # b1k_w[:] = b1k - b1k_v[:]      #Already cos
    
    
    u1k[0,:] = np.conjugate(fft_cos(u1[0],u1k[0]))/(4*N**6/TWO_PI**3)**0.5
    u1k[1,:] = np.conjugate(fft_cos(u1[1],u1k[1]))/(4*N**6/TWO_PI**3)**0.5
    u1k[2,:] = np.conjugate(fft_sin(u1[2],u1k[2]))/(4*N**6/TWO_PI**3)**0.5
    b1k[:] = np.conjugate(fft_sin(b1,b1k))/(4*N**6/TWO_PI**3)**0.5

    uk[0,:] = fft_cos(u[0],uk[0])/(4*N**6/TWO_PI**3)**0.5
    uk[1,:] = fft_cos(u[1],uk[1])/(4*N**6/TWO_PI**3)**0.5
    uk[2,:] = fft_sin(u[2],uk[2])/(4*N**6/TWO_PI**3)**0.5
    bk[:] = fft_sin(b,bk)/(4*N**6/TWO_PI**3)**0.5
    
    uk_w[0,:] = fft_cos(u_w[0],uk_w[0])/(4*N**6/TWO_PI**3)**0.5
    uk_w[1,:] = fft_cos(u_w[1],uk_w[1])/(4*N**6/TWO_PI**3)**0.5
    uk_w[2,:] = fft_sin(u_w[2],uk_w[2])/(4*N**6/TWO_PI**3)**0.5
    bk_w[:] = fft_sin(b_w,bk_w)/(4*N**6/TWO_PI**3)**0.5
    
    uk_v[0,:] = fft_cos(u_v[0],uk_v[0])/(4*N**6/TWO_PI**3)**0.5
    uk_v[1,:] = fft_cos(u_v[1],uk_v[1])/(4*N**6/TWO_PI**3)**0.5
    uk_v[2,:] = fft_sin(u_v[2],uk_v[2])/(4*N**6/TWO_PI**3)**0.5
    bk_v[:] = fft_sin(b_v,bk_v)/(4*N**6/TWO_PI**3)**0.5
    
    uk[2,:] = sin_to_cos(uk[2])
    uk_v[2,:] = sin_to_cos(uk_v[2])
    uk_w[2,:] = sin_to_cos(uk_w[2])
    bk[:] = sin_to_cos(bk)
    bk_w[:] = sin_to_cos(bk_w)
    bk_v[:] = sin_to_cos(bk_v)
    
    u1k[2,:] = sin_to_cos(u1k[2])
    b1k[:] = sin_to_cos(b1k)
    
    u1k_v[2,:] = sin_to_cos(u1k_v[2])
    b1k_v[:] = sin_to_cos(b1k_v)
    
    u1k_w[2,:] = sin_to_cos(u1k_w[2])
    b1k_w[:] = sin_to_cos(b1k_w)
    
    uk[:2,kzc == 0] = 2**(-0.5)*uk[:2,kzc == 0]
    uk_w[:2,kzc == 0] = 2**(-0.5)*uk_w[:2,kzc == 0]
    uk_v[:2,kzc == 0] = 2**(-0.5)*uk_v[:2,kzc == 0]
    u1k[:2,kzc == 0] = 2**(-0.5)*u1k[:2,kzc == 0]
    u1k_w[:2,kzc == 0] = 2**(-0.5)*u1k_w[:2,kzc == 0]
    u1k_v[:2,kzc == 0] = 2**(-0.5)*u1k_v[:2,kzc == 0]
    
    
    ek[:] = 0.5*dkx*dky*dkz*(np.abs(uk[0])**2 + np.abs(uk[1])**2 + np.abs(uk[2])**2/alph**2 + np.abs(bk)**2)*dealias_cos
    PIk[:] = np.real(uk[0]*u1k[0]+uk[1]*u1k[1]+ uk[2]*u1k[2]/alph**2 + bk*b1k)*dkx*dky*dkz*dealias_cos
    
    ek_w[:] = 0.5*dkx*dky*dkz*(np.abs(uk_w[0])**2 + np.abs(uk_w[1])**2 + np.abs(uk_w[2])**2/alph**2 + np.abs(bk_w)**2)*dealias_cos
    
    PIk_w[:] = np.real(uk_w[0]*u1k_w[0]+uk_w[1]*u1k_w[1]+ uk_w[2]*u1k_w[2]/alph**2 + bk_w*b1k_w)*dkx*dky*dkz*dealias_cos
    
    ek_v[:] = 0.5*dkx*dky*dkz*(np.abs(uk_v[0])**2 + np.abs(uk_v[1])**2 + np.abs(uk_v[2])**2/alph**2 + np.abs(bk_v)**2)*dealias_cos
    
    PIk_v[:] = np.real(uk_v[0]*u1k_v[0]+uk_v[1]*u1k_v[1]+ uk_v[2]*u1k_v[2]/alph**2 + bk_v*b1k_v)*dkx*dky*dkz*dealias_cos
    
    if rank ==0 : print(f"Time {t[i]}: Total - ( Balanced + Wave)  Flux max : {np.max(np.abs(PIk_v -PIk_w))}")
    
    return ek, PIk,ek_w, PIk_w,ek_v, PIk_v
    
## -----------------------------------------------------------------

## -------------------- Giving the RHS of the balanced flow --------    
    
def RHSv(u1,b1,u1_v,b1_v,u1_w,b1_w):
    dtQ[:] = diff_z_sin(b1,b11) - diff_y(u1[0],b12) + diff_x(u1[1],b13) ## b1k is cos 
    dtQk[:] = fft_cos(dtQ,dtQk)
    """ Calculating dtP* """
    dtPk[:] = invlapc*dtQk
    dtP[:] = ifft_cos(dtPk,dtP)

    """ Calculating dt balanced* """
    u1_v[0,:] = -diff_y(dtP,u1_v[0])
    u1_v[1,:] = diff_x(dtP, u1_v[1])
    u1_v[2,:] = 0.
    b1_v[:] = diff_z_cos(dtP,b1_v)

    u1_w[0,:] = u1[0] - u1_v[0]
    u1_w[1,:] = u1[1] - u1_v[1]
    u1_w[2,:] = u1[2] - u1_v[2]
    b1_w[:] = b1 - b1_v

    return u1_v, b1_v, u1_w, b1_w   

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
    
    

    ## The RHS term with the pressure 
    # u_t[0] = rhsu 
    # u_t[1] = rhsv 
    # u_t[2] = rhsw 
    u_t[0] = rhsu - diff_x(p, p1)
    u_t[1] = rhsv - diff_y(p,  p1)
    u_t[2] = rhsw - alph**2 * diff_z_cos(p, p1)
    
        
    return u_t, b_t
## -------------------------------------------------------


## ----------- making sin*sin arrays to cos -----------------
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
    
## -----------------------------------------------------------
## -------- decomposing fields into balance and waves --------
def decompose(u,b,u_v,b_v,u_w,b_w):
    
    pv[:] = diff_z_sin(b,b1) + diff_x(u[1],u1[1]) - diff_y(u[0],u1[0])
    pvk[:] = fft_cos(pv,pvk)
    
    pk_v[:] = invlapc*pvk 
    
    p_v[:] = ifft_cos(pk_v,p_v)
    
    u_v[0] =  -diff_y(p_v,u_v[0])
    u_v[1] =  diff_x(p_v,u_v[1])
    u_v[2,:] =  0.
    b_v[:] =  diff_z_cos(p_v,b_v)
    
    u_w[0] =  u[0] - u_v[0] 
    u_w[1] =  u[1] - u_v[1] 
    u_w[2] =  u[2] - u_v[2] 
    b_w[:] =  b - b_v
    
    return u_v,b_v,u_w,b_w

## -----------------------------------------------------------

## -------------------------------------------------------------------
## 
## -------------------------------------------------------------------


## ----------------- Loading the data ------------------------------
# if rank ==0: print(tlast)
interval = 5.
# t = np.arange(520.,tlast+interval,interval)
# t = np.arange(250.,250 + 10*np.pi,0.1)
# t = np.arange(250,250 + 80*np.pi,0.1)
# t = list(np.arange(69.9,100.1,1.0))
# t = np.arange(0.,240.1,5.0)
t = np.arange(650,660.1,1)
# t = [172.0]
# t = np.array(list(np.arange(0,2.4,0.1)) + list(np.arange(2.4,6,1.)))
# t = np.array(list(np.arange(0,24.6,1.0)) + list(np.arange(24.9,75.0,1.0)) + list(np.arange(75.3,105.9,1.0)))
e_arr = np.empty_like(t)
e_arr_w = np.empty_like(t)
e_arr_v = np.empty_like(t)

T_w = np.empty_like(t, dtype = np.float64)
T_v = np.empty_like(t, dtype = np.float64)
# tcheck = np.arange(257.0,565.0,1.0)
# t = np.arange(100,1600,100)
## -------------------------------------------------------------------
if rank ==0:
    fluxes = np.zeros(len(t))
    fluxes_w = np.zeros(len(t))
    fluxes_v = np.zeros(len(t))
    energies = np.zeros(len(t))
    energies_w = np.zeros(len(t))
    energies_v = np.zeros(len(t))
## -------------------------------------------------------------------
for i in range(len(t)):
    sum = 0.0
    if rank ==0: print(f"Time : {t[i]:.1f}")
    u[:],b[:] = load_data(Ns,Nslab,nu,N,u,b)
    # if rank ==0 : print(f'np.max(np.abs(u)) : {np.max(np.abs(u))}')
    # if rank ==0 : print(f'Loading Done')
    # if rank ==0 : print(f'u_w.shape : {u_w.shape}')
    
    
    u_v[:],b_v[:],u_w[:],b_w[:] = decompose(u,b,u_v,b_v,u_w,b_w)
    # omg[:] = diff_x(u[1],u1[1]) - diff_y(u[0],u1[0])
    # u1[:],b1[:] = RHS(u,b,u1,b1)
    savedir = savePath/f"time_{t[i]:.1f}"
    if rank == 0: 
        try:savedir.mkdir(parents=True, exist_ok=False)
        except FileExistsError: pass
    comm.Barrier()
    # np.savez_compressed(savePath/f"time_{t[i]:.1f}/RHSFields_{rank}",u1=u1[0],v1=u1[1],w1=u1[2],b1=b1)
    # u1_v[:],b1_v[:],u1_w[:],b1_w[:] = RHSv(u1,b1,u1_v,b1_v,u1_w,b1_w)
    # if rank >0:
    #     comm.send(np.sum(u1[0]*u[0] + u1[1]*u[1] + u1[2]*u[2]/alph**2 + b1*b),dest = 0,tag =99)
    #     comm.send(np.sum(u1_v[0]*u_v[0] + u1_v[1]*u_v[1] + u1_v[2]*u_v[2]/alph**2 + b1_v*b_v),dest = 0,tag =991)
    #     comm.send(np.sum(u1_w[0]*u_w[0] + u1_w[1]*u_w[1] + u1_w[2]*u_w[2]/alph**2 + b1_w*b_w),dest = 0,tag =992)
    #     comm.send(np.sum(u[0]*u[0] + u[1]*u[1] + u[2]*u[2]/alph**2 + b*b),dest = 0,tag =199)
    #     comm.send(np.sum(u_v[0]*u_v[0] + u_v[1]*u_v[1] + u_v[2]*u_v[2]/alph**2 + b_v*b_v),dest = 0,tag =1991)
    #     comm.send(np.sum(u_w[0]*u_w[0] + u_w[1]*u_w[1] + u_w[2]*u_w[2]/alph**2 + b_w*b_w),dest = 0,tag =1992)
    # else:
    #     flux = np.sum(u1[0]*u[0] + u1[1]*u[1] + u1[2]*u[2]/alph**2 + b1*b)
    #     flux_v = np.sum(u1_v[0]*u_v[0] + u1_v[1]*u_v[1] + u1_v[2]*u_v[2]/alph**2 + b1_v*b_v)
    #     flux_w = np.sum(u1_w[0]*u_w[0] + u1_w[1]*u_w[1] + u1_w[2]*u_w[2]/alph**2 + b1_w*b_w)
    #     energy = np.sum(u[0]*u[0] + u[1]*u[1] + u[2]*u[2]/alph**2 + b*b)
    #     energy_v = np.sum(u_v[0]*u_v[0] + u_v[1]*u_v[1] + u_v[2]*u_v[2]/alph**2 + b_v*b_v)
    #     energy_w = np.sum(u_w[0]*u_w[0] + u_w[1]*u_w[1] + u_w[2]*u_w[2]/alph**2 + b_w*b_w)
    #     for ii in range(1,num_process):
    #         flux += comm.recv(source = ii,tag = 99)
    #         flux_v += comm.recv(source = ii, tag = 991)
    #         flux_w += comm.recv(source = ii, tag = 992)
    #         energy += comm.recv(source = ii,tag = 199)
    #         energy_v += comm.recv(source = ii, tag = 1991)
    #         energy_w += comm.recv(source = ii, tag = 1992)
            
    #     fluxes[i] = flux
    #     fluxes_v[i] = flux_v
    #     fluxes_w[i] = flux_w        
        
    #     energies[i] = 0.5*energy
    #     energies_v[i] = 0.5*energy_v
    #     energies_w[i] = 0.5*energy_w
                    
    #     print(f"Flux is {flux}")
    #     print(f"Sum of fluxes is {flux_v + flux_w}")
    #     print(f"Balanced and wave fluxes are {flux_v} and {flux_w}")
    
    # div[:] = diff_x(u[0],u1[0]) + diff_y(u[1],u1[1]) + diff_z_sin(u[2],u1[2])
    # if rank ==0 : print(f"Rank {rank} has divergence {np.sum(div)}")
    
    # np.savez_compressed(loadPath/f"time_{t[i]:.1f}/spectralWaveFields_{rank}",uk=uk[0],vk=uk[1],wk=uk[2],bk=bk)
    
    # np.savez_compressed(loadPath/f"time_{t[i]:.1f}/FieldsNew_{rank}.npz", u=u,b=b)
    

    
    
    
    # uk[0,:] = fft_cos(u[0],uk[0])
    # uk[1,:] = fft_cos(u[1],uk[1])
    # uk[2,:] = fft_sin(u[2],uk[2])
    # bk[:] = fft_sin(b,bk)
    
    
    # uk[2,:] = sin_to_cos(uk[2,:])
    # bk[:] = sin_to_cos(bk)
    
    # arr_mpi_Z[:] = np.moveaxis(uk[0].reshape(N,Np,num_process,Np),[0,1,2],[2,1,0])
    # comm.Alltoall([arr_mpi_Z,  MPI.DOUBLE_COMPLEX], [uk_w_z[0],  MPI.DOUBLE_COMPLEX])
    
    # arr_mpi_Z[:] = np.moveaxis(uk[1].reshape(N,Np,num_process,Np),[0,1,2],[2,1,0])
    # comm.Alltoall([arr_mpi_Z,  MPI.DOUBLE_COMPLEX], [uk_w_z[1],  MPI.DOUBLE_COMPLEX])
    
    # arr_mpi_Z[:] = np.moveaxis(uk[2].reshape(N,Np,num_process,Np),[0,1,2],[2,1,0])
    # comm.Alltoall([arr_mpi_Z,  MPI.DOUBLE_COMPLEX], [uk_w_z[2],  MPI.DOUBLE_COMPLEX])
    
    # arr_mpi_Z[:] = np.moveaxis(bk.reshape(N,Np,num_process,Np),[0,1,2],[2,1,0])
    # comm.Alltoall([arr_mpi_Z,  MPI.DOUBLE_COMPLEX], [bk_w_z,  MPI.DOUBLE_COMPLEX])
    
    #! Remember that after the communication, the shape now is (ky,kx,kz)
    # if rank ==0: 
    #     print(f"Rank {rank} --> Max for u : {np.max(np.abs(uk_w_z[0]))}")
    #     print(f"Rank {rank} --> Max for v : {np.max(np.abs(uk_w_z[1]))}")
    #     print(f"Rank {rank} --> Max for w : {np.max(np.abs(uk_w_z[2]))}")
    #     print(f"Rank {rank} --> Max for b : {np.max(np.abs(bk_w_z))}")
    
    # if rank ==0: 
    #     print(f"Rank {rank} --> u : {np.abs(uk_w_z[0,4,3,2])}")
    #     print(f"Rank {rank} --> v : {np.abs(uk_w_z[1,8,6,9])}")
    #     print(f"Rank {rank} --> w : {np.abs(uk_w_z[2,0,5,6])}")
    #     print(f"Rank {rank} --> b : {np.abs(bk_w_z[4,3,7])}")
    # sleep(10)
    # ## -----------------------------------------------------  
    
    ## ------------------ test fields ----------------------  
    # u_w[0,:] = np.cos(5*x)*np.cos(1*z)*np.cos(t[i])
    # u_w[1,:] = np.cos(3*x)*np.sin(4*y)*np.cos(1*z)*np.cos(2*t[i])
    # u_w[2,:] = np.sin(5*x)*np.sin(1*z)*np.sin(3*t[i])
    # b_w[:] = np.sin(3*x)*np.cos(4*y)*np.sin(1*z)*np.sin(4*t[i])
    # if rank ==0: print(f"Reached test function")
    # comm.Barrier()
    # if rank ==0: print(f"Done test function")
    # uk_w[0,:] = fft_cos(u_w[0],uk_w[0])
    # uk_w[1,:] = fft_cos(u_w[1],uk_w[1])
    # uk_w[2,:] = fft_sin(u_w[2],uk_w[2])
    # bk_w[:] = fft_sin(b_w,bk_w)
    
    # uk_w[2,:] = sin_to_cos(uk_w[2,:])
    # bk_w[:] = sin_to_cos(bk_w)
    ## -----------------------------------------------------  
    
    #! Saving with swapped axes such that the shape is (kx,ky,kz)
    # np.savez_compressed(savePath/f"time_{t[i]:.1f}/spectralFields_{rank}",uk=np.moveaxis(uk_w_z[0],[0,1],[1,0]),vk=np.moveaxis(uk_w_z[1],[0,1],[1,0]),wk=np.moveaxis(uk_w_z[2],[0,1],[1,0]),bk=np.moveaxis(bk_w_z,[0,1],[1,0]))
    
    # if rank ==0: print(f"Reached saving fields")
    # comm.Barrier()
    # if rank ==0: print(f"Done saving fields")
    # divH[:] = diff_x(u_w[0],p11)  + diff_y(u_w[1],p12)
    # divHk[:] = fft_cos(divH,divHk)
    
    # div[:] = diff_x(u[0],u1[0]) + diff_y(u[1],u1[1]) + diff_z_sin(u[2],u1[2])
    # print(f"Rank {rank} has divergence {np.sum(div)}")
    
    # for j in range(Ns):
    #     e[j*Nslab:(j+1)*(Nslab),...] = np.load(loadPath/f"time_{t[i]:.1f}/e_{rank*Ns+j}.npy")
    
    e_arr[i] = comm.allreduce(np.sum(0.5*(u[0]**2 + u[1]**2 + u[2]**2/alph**2 + b**2)*dx*dy*dz),op = MPI.SUM)
    e_arr_w[i] = comm.allreduce(np.sum(0.5*(u_w[0]**2 + u_w[1]**2 + u_w[2]**2/alph**2 + b_w**2)*dx*dy*dz),op = MPI.SUM)
    e_arr_v[i] = comm.allreduce(np.sum(0.5*(u_v[0]**2 + u_v[1]**2 + u_v[2]**2/alph**2 + b_v**2)*dx*dy*dz),op = MPI.SUM)
    # continue
    # e[:] = 0.5*(u[0]**2 + u[1]**2 + u[2]**2/alph**2 + b**2)*dx*dy*dz
    
    # pv[:] = diff_z_sin(b,b1) + diff_x(u[1],u1[1]) - diff_y(u[0],u1[0])
    
    # Pi[:] = Flux(u,b,Pi)
    # ek[:],PIk[:] = energy_flux(u,b,ek,PIk,uk,bk)
    # ek_w[:],PIk_w[:]  = energy_flux(u_w,b_w,ek_w,PIk_w,uk_w,bk_w)
    # ek_v[:],PIk_v[:]  = energy_flux(u_v,b_v,ek_v,PIk_v,uk_v,bk_v)
    ek[:],PIk[:],ek_w[:],PIk_w[:],ek_v[:],PIk_v[:] = all_energy_flux(u,b,u_w,b_w,u_v,b_v,ek,PIk,ek_w,PIk_w,ek_v,PIk_v)
    ek_avg[:] = ek_avg + ek
    PIk_avg[:] = PIk_avg + PIk
    ek_w_avg[:] = ek_w_avg + ek_w
    PIk_w_avg[:] = PIk_w_avg + PIk_w
    ek_v_avg[:] = ek_v_avg + ek_v
    PIk_v_avg[:] = PIk_v_avg + PIk_v
    # b1[:] = diff_z_sin(b,b1)
    if rank ==0 : print(f"Time {i}: Total - ( Balanced + Wave)  Flux max : {np.max(np.abs(PIk -PIk_v -PIk_w))}")
    
    if rank ==0 : print(f"Time {t[i]}: Total - ( Balanced + Wave)  energy max : {np.max(np.abs(ek -ek_v -ek_w))}")
    T_w[i] = comm.allreduce(np.sum(PIk_w),op = MPI.SUM)
    T_v[i] = comm.allreduce(np.sum(PIk_v),op = MPI.SUM)
    ## ---------- Setting tide condition -----------------
    # cond = kzc ==1
    # pv_tide[:] = np.abs(pv[:,:,1])
    # divHk_tide[:] = np.abs(divHk[:,:,1])
    # divH_tide[:] = ifft_cos(np.abs(divHk[:,:,1],)divH_tide)
    # ek_tide[:] = np.abs(ek[:,:,1])
    # PIk_tide[:]  = np.abs(PIk[:,:,1])
    # ek_wtide[:] = np.abs(ek_w[:,:,1])
    # PIk_wtide[:]  = np.abs(PIk_w[:,:,1])
    # ek_vtide[:] = np.abs(ek_v[:,:,1])
    # PIk_vtide[:]  = np.abs(PIk_v[:,:,1])
    ## ---------------------------------------------------
    ## ---------------- Horizontal Ek and pseudo vertical Ek, summing over Ek in kz ---------------------
    # ekh = np.sum(ek,axis=2)
    # ekh_w = np.sum(ek_w,axis=2)
    # ekh_v = np.sum(ek_v,axis=2)
    # ekv = np.sum(ek,axis=(0,1))
    # ekv_w = np.sum(ek_w,axis=(0,1))
    # ekv_v = np.sum(ek_v,axis=(0,1))
    ## ----------------------------------------------------------------------------
        

    
    
    
    # ------------------ Saving the data ----------------------------------------
    # if t[i] >779.:
    #     if rank == 0: 
    #         create_dir(savePath/f"E_k/time_{t[i]:.1f}")
    #         create_dir(savePath/f"PIk/time_{t[i]:.1f}")
    #     #     # create_dir(savePath/f"Pi/time_{t[i]:.1f}")
    #     #     create_dir(savePath/f"PV/time_{t[i]:.1f}")
    #     #     # create_dir(savePath/f"e/time_{t[i]:.1f}")
    #     #     # create_dir(savePath/f"divH/time_{t[i]:.1f}")
    #     #     create_dir(savePath/f"divHk/time_{t[i]:.1f}")
    #     #     create_dir(savePath/f"zeta/time_{t[i]:.1f}")
    #     comm.Barrier()
        
    #     np.save(savePath/f"E_k/time_{t[i]:.1f}/e_{rank}",  ek)
    #     np.save(savePath/f"E_k/time_{t[i]:.1f}/eVortex_{rank}",  ek_v)
    #     np.save(savePath/f"E_k/time_{t[i]:.1f}/eWave_{rank}",  ek_w)
    #     np.save(savePath/f"PIk/time_{t[i]:.1f}/PI_{rank}",  PIk)
    #     np.save(savePath/f"PIk/time_{t[i]:.1f}/PIVortex_{rank}",  PIk_v)
    #     np.save(savePath/f"PIk/time_{t[i]:.1f}/PIWave_{rank}",  PIk_w)
        
        
    #     # # np.save(savePath/f"e/time_{t[i]:.1f}/e_{rank}",  e)
    #     # # np.save(savePath/f"e/time_{t[i]:.1f}/eVortex_{rank}",  e_v)
    #     # # np.save(savePath/f"e/time_{t[i]:.1f}/eWave_{rank}",  e_w)
    #     # np.save(savePath/f"Pi/time_{t[i]:.1f}/Pi_{rank}",  Pi)
    #     # np.save(savePath/f"E_kh/time_{t[i]:.1f}/ekh_{rank}",  ekh)
    #     # np.save(savePath/f"E_kh/time_{t[i]:.1f}/ekhVortex_{rank}",  ekh_v)
    #     # np.save(savePath/f"E_kh/time_{t[i]:.1f}/ekhWave_{rank}",  ekh_w)
    #     # np.save(savePath/f"divH/time_{t[i]:.1f}/divH_{rank}",  divH)
    #     # np.save(savePath/f"divHk/time_{t[i]:.1f}/divHkTide_{rank}",divHk_tide)
    #     # np.save(savePath/f"PV/time_{t[i]:.1f}/PV_{rank}",  pv)
    #     # np.save(savePath/f"zeta/time_{t[i]:.1f}/zeta_{rank}",  omg)
    #     # np.save(savePath/f"zeta/time_{t[i]:.1f}/bz_{rank}",  b1)
    #     ## ----------------------------------------------------------------------------
    #     # # if rank ==0 : print(f'Before barrier ')
    #     # # if rank ==0 : print(f'After barrier ')
    #     comm.Barrier()  
# if rank ==0:
#     saveData = savePath/f"timeRange_{t[0]:.2f}-{t[-1]:.2f}"
#     saveData.mkdir(parents=True,  exist_ok=True)
#     np.save(saveData/"energy",  energies)
#     np.save(saveData/"energy_wave",energies_w)
#     np.save(saveData/"energy_vortex",energies_v)
#     np.save(saveData/"flux",  fluxes)
#     np.save(saveData/"flux_wave",fluxes_w)
#     np.save(saveData/"flux_vortex",fluxes_v)

## ----------------- saving the average flux -------------- ## 
if rank == 0: 
    create_dir(savePath/f"E_k/")
    create_dir(savePath/f"PIk/")
#     # create_dir(savePath/f"Pi/time_{t[i]:.1f}")
#     create_dir(savePath/f"PV/time_{t[i]:.1f}")
#     # create_dir(savePath/f"e/time_{t[i]:.1f}")
#     # create_dir(savePath/f"divH/time_{t[i]:.1f}")
#     create_dir(savePath/f"divHk/time_{t[i]:.1f}")
#     create_dir(savePath/f"zeta/time_{t[i]:.1f}")
comm.Barrier()

# np.save(savePath/f"E_k/e_{rank}",  ek_avg/len(t))
# np.save(savePath/f"E_k/eVortex_{rank}",  ek_v_avg/len(t))
# np.save(savePath/f"E_k/eWave_{rank}",  ek_w_avg/len(t))
# np.save(savePath/f"PIk/PI_{rank}",  PIk_avg/len(t))
# np.save(savePath/f"PIk/PIVortex_{rank}",  PIk_v_avg/len(t))
# np.save(savePath/f"PIk/PIWave_{rank}",  PIk_w_avg/len(t))


# # np.save(savePath/f"e/time_{t[i]:.1f}/e_{rank}",  e)
# # np.save(savePath/f"e/time_{t[i]:.1f}/eVortex_{rank}",  e_v)
# # np.save(savePath/f"e/time_{t[i]:.1f}/eWave_{rank}",  e_w)
# np.save(savePath/f"Pi/time_{t[i]:.1f}/Pi_{rank}",  Pi)
# np.save(savePath/f"E_kh/time_{t[i]:.1f}/ekh_{rank}",  ekh)
# np.save(savePath/f"E_kh/time_{t[i]:.1f}/ekhVortex_{rank}",  ekh_v)
# np.save(savePath/f"E_kh/time_{t[i]:.1f}/ekhWave_{rank}",  ekh_w)
# np.save(savePath/f"divH/time_{t[i]:.1f}/divH_{rank}",  divH)
# np.save(savePath/f"divHk/time_{t[i]:.1f}/divHkTide_{rank}",divHk_tide)
# np.save(savePath/f"PV/time_{t[i]:.1f}/PV_{rank}",  pv)
# np.save(savePath/f"zeta/time_{t[i]:.1f}/zeta_{rank}",  omg)
# np.save(savePath/f"zeta/time_{t[i]:.1f}/bz_{rank}",  b1)
## ----------------------------------------------------------------------------
# # if rank ==0 : print(f'Before barrier ')
# # if rank ==0 : print(f'After barrier ')
comm.Barrier()  


## ------------ Plotting the timeseries --------------- ## 
# if rank ==0 :
#     import matplotlib.pyplot as plt
#     import matplotlib as mpl 
#     mpl.rc('text', usetex = True)

#     plt.figure(figsize = (8,6))
#     plt.xticks(fontsize = 18)
#     plt.yticks(fontsize = 18)
#     # plt.plot(times,lw = 4,eratio,color = "#fb8500",label = 'Ratio')
#     plt.plot(t,e_arr,lw = 4,color = "#001219",label = 'Net energy')
#     plt.plot(t,e_arr_w,lw = 4,color = "#fb8500",label = 'Wave energy')
#     plt.plot(t,e_arr_v,lw = 4,color = "#d90429",label = 'Balanced energy')
#     plt.axhline(y=0, color='black', linestyle='--', linewidth=2)
#     # plt.plot(times,lw = 4,ekvtot_arr,color = "#fb8500",label = 'Balanced energy')
#     # plt.plot(times,lw = 4,1 - ektide/ektide[0],color = "#ffb703",label = r'$k_z = 1$ Wave')
#     np.save(savePath/"time",t)
#     np.save(savePath/"totalenergy_timeseries",e_arr)
#     np.save(savePath/"waveenergy_timeseries",e_arr_w)
#     np.save(savePath/"balancedenergy_timeseries",e_arr_v)
    
#     # t = np.load(savePath/"time.npy")
#     # e_arr = np.load(savePath/"energy_timeseries.npy")
#     plt.xlabel(r"$t$",fontsize =22 )
#     plt.ylabel(r"$E$",fontsize =22,rotation = 0 )
#     # plt.ylim(1e-6,plt.ylim()[1])
#     # plt.title(fr"Energy timeseries",fontsize = 40)
#     # plt.tight_layout()
#     # plt.grid()
#     plt.legend(fontsize =22,fancybox = True, framealpha = 0.3)
#     try : savePlot.mkdir(parents=True, exist_ok=False)
#     except FileExistsError : pass
#     plt.savefig(savePlot/fr"energyTimeseries.jpg")#,transparent = True)
#     plt.close()
    
#     np.save(savePath/"time",t)
#     np.save(savePath/"wave_transfer",T_w)
#     np.save(savePath/"balanced_transfer",T_v)

#     plt.figure(figsize = (8,6))
#     plt.xticks(fontsize = 18)
#     plt.yticks(fontsize = 18)
#     # plt.plot(times,lw = 4,eratio,color = "#fb8500",label = 'Ratio')
#     plt.plot(t,T_w,lw = 4,color = "#fb8500",label = fr'$T_{{ub}}$')
#     plt.plot(t,T_v,lw = 4,color = "#d90429",label = fr'$T_{{b}}$')
#     plt.axhline(y=0, color='black', linestyle='--', linewidth=2)
#     # plt.plot(times,lw = 4,ekvtot_arr,color = "#fb8500",label = 'Balanced energy')
#     # plt.plot(times,lw = 4,1 - ektide/ektide[0],color = "#ffb703",label = r'$k_z = 1$ Wave')
#     # np.save(savePath/"time",t)
#     # np.save(savePath/"totalenergy_timeseries",e_arr)
#     # np.save(savePath/"waveenergy_timeseries",e_arr_w)
#     # np.save(savePath/"balancedenergy_timeseries",e_arr_v)
    
#     # t = np.load(savePath/"time.npy")
#     # e_arr = np.load(savePath/"energy_timeseries.npy")
#     plt.xlabel(r"$t$",fontsize =22 )
#     plt.ylabel(r"$T$",fontsize =22,rotation = 0 )
#     # plt.ylim(1e-6,plt.ylim()[1])
#     # plt.title(fr"Energy timeseries",fontsize = 40)
#     plt.tight_layout()
#     plt.grid()
#     plt.legend(fontsize =22,fancybox = True, framealpha = 0.3)
#     try : savePlot.mkdir(parents=True, exist_ok=False)
#     except FileExistsError : pass
#     plt.savefig(savePlot/fr"transferTimeseries.png")#,transparent = True)
#     plt.close()
# ## ---------------------------------------------------- ##

    
# """
# time nohup mpirun -n 128 python -u pv--energy.py > errors-outputs/pv--energy.out &
# """
    
    
