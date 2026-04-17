import numpy as np 
# from pyfftw.interfaces.scipy_fft import fft ,  ifft ,  irfft2 ,  rfft2 , irfftn ,  rfftn, fftfreq, dst, dct, idst, idct, rfft,  irfft
from scipy.fft import fft ,  ifft ,  irfft2 ,  rfft2 , irfftn ,  rfftn, fftfreq, dst, dct, idst, idct, rfft,  irfft
from mpi4py import MPI
from time import time
import pathlib, json, os,sys

curr_path = pathlib.Path(__file__).parent

## ---------------MPI things--------------
comm = MPI.COMM_WORLD
num_process =  comm.Get_size()
rank = comm.Get_rank()
## ---------------------------------------

## --------- Loading from the parameters file ------------
with open(curr_path/"parameters_OT.json") as f:
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
param["num_process"] = num_process
## -------------------------------------------------------

st = int(round(dt_save/dt))
st_r = int(round(dt_save_r/dt))
## ---------------------------------------
## -------------Defining the grid ---------------
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
## -----------------------------------------------




## --------- kx and ky for differentiation ---------    
# kx.shape = (Nf,  Np,  N )
# ky.shape = (Np,  Nf,  N)
kx_diff = kx[:Nf,  :, :].copy()
kx_diff[-1, :, :] = -kx_diff[-1, :, :]
ky_diff = np.swapaxes(kx_diff, 0, 1).copy()

## -------------------------------------------------

## ----------- Parameters ----------
sigPos_arr = np.array([omega])
sigNeg_arr = np.array([-omega])


## ---------------------------------
if low_wave: 
    savePath = pathlib.Path(f"/mnt/pfs/rajarshi.chattopadhyay/boussinesq/data_final/nu_{nu}_N_{N}/Ro_{ro}/forcedTide_ring_LW")
    e_w_thresh = 0.5
else: 
    savePath = pathlib.Path(f"/mnt/pfs/rajarshi.chattopadhyay/boussinesq/data_final/nu_{nu}_N_{N}/Ro_{ro}/forcedTide_ring")
    e_w_thresh = 4.0
if aa_v  == 0.:
    savePath = pathlib.Path(f"/mnt/pfs/rajarshi.chattopadhyay/boussinesq/data_final/nu_{nu}_N_{N}/Ro_{ro}/forcedTide_ring_OT")
# e_v_thresh = 4.0
e_v_thresh = 0.0 #! Avoiding balanced forcing
if rank == 0:
    try: savePath.mkdir(parents=True,  exist_ok=True)
    except FileExistsError: pass

## ------------Useful Operators-------------------
dealias_cos = (abs(kx)<N//3)*(abs(ky)<N//3)*(kzc<(N)//2)
dealias_sin = (abs(kx)<N//3)*(abs(ky)<N//3)*(kzs<(N)//2)

lapc = -(kx**2 + ky**2 + kzc**2 )
laps = -(kx**2 + ky**2 + kzs**2 )

kc = (-lapc )**0.5
ks = (-laps )**0.5
kh = (kx**2+ ky**2)**0.5
lappress = -(kx**2 + ky**2 + alph**2 * kzc**2)
invlapc = 1.0/np.where(lapc == 0, np.inf,  lapc)
invlaps = 1.0/laps
invpress = 1.0/np.where(lappress == 0,  np.inf ,  lappress)* dealias_cos
kcint = np.clip(np.round(kc), 0,  N//2).astype(int)
shells = np.arange(-0.5, N//2 + 1.5)
shells[0] = 0.
normalize = np.where((kzc ==0) + (kzc == N-1) ,0.5,1)/(2*N**6/(TWO_PI**2 * PI))
# Hyperviscous operators
vis_cos = nu*(kc)**(2*lp) ## This is in Fourier Space
vis_sin = nu*(ks)**(2*lp) ## This is in Fourier Space
## -------------------------------------------------


## -------------zeros arrays -----------------------
u  = np.zeros((3, Np, N, N), dtype= np.float64)
u_v  = np.zeros((3, Np, N, N), dtype= np.float64)
u_w  = np.zeros((3, Np, N, N), dtype= np.float64)
p = u[0].copy()
p_v = u[0].copy()
b = u[0].copy()
b_v = u[0].copy()
b_w = u[0].copy()


uk = np.zeros((3, N, Np, N), dtype= np.complex128)
uk_v = np.zeros((3, N, Np, N), dtype= np.complex128)
uk_w = np.zeros((3, N, Np, N), dtype= np.complex128)
pk = uk[0].copy()
ek = np.zeros_like(pk,dtype = np.float64)
pk_v = pk.copy()
bk = pk.copy()
bk_v = pk.copy()
bk_w = pk.copy()


theta2p = np.zeros((N, Np, N), dtype= np.complex128)
theta1p = np.zeros((N, Np, N), dtype= np.complex128)
theta2 = np.zeros((2,N, Np, N), dtype= np.complex128)
theta1 = np.zeros((2,N, Np, N), dtype= np.complex128)
theta1_z = np.zeros((2,N,N,Np))
theta2_z = np.zeros((2,N,N,Np))
arr_temp_theta = np.zeros((N,2,N,Np))
arr_temp_theta1 = np.zeros((N,N,Np))
arr_theta= np.zeros((num_process,Np,N,Np))
arr_theta_Z = np.zeros((num_process,Np,2,N,Np))
exptheta1w = np.zeros_like(pk)
exptheta2w = np.zeros_like(pk)
exptheta1b = np.zeros_like(pk)
exptheta2b = np.zeros_like(pk)
fk = np.zeros_like(uk)
fk_v = np.zeros_like(uk)
fk_w = np.zeros_like(uk)
fkb = np.zeros_like(bk)
fkb_v = np.zeros_like(bk)
fkb_w = np.zeros_like(bk)
ff = np.zeros_like(u)
fb = np.zeros_like(b)

rhsu = np.zeros_like(p)
rhsuk = np.zeros_like(pk)
rhsvk = np.zeros_like(pk)
rhswk = np.zeros_like(pk)
b_tk = np.zeros_like(pk)
rhsu1 = np.zeros_like(p)
rhsu2 = np.zeros_like(p)
rhsu3 = np.zeros_like(p)
rhsv = rhsu.copy()
rhsv2 = rhsu.copy()
rhsv1 = rhsu.copy()
rhsv3 = rhsu.copy()
rhsw = rhsu.copy()
rhsw1 = rhsu.copy()
rhsw2 = rhsu.copy()
rhsw3 = rhsu.copy()
# b_t = rhsu.copy()
b_t1 = rhsu.copy()
b_t2 = rhsu.copy()
b_t3 = rhsu.copy()
p1 = rhsu.copy()
p11 = rhsu.copy()
p12 = rhsu.copy()
p13 = rhsu.copy()
u_temp = p1.copy()
e = p1.copy()
pv = p1.copy()
pvk = pk.copy()
omg = u.copy()

pk_neg = np.empty_like(bk)
pk_pos = np.empty_like(bk)
cond = np.empty_like(bk)
pk_temp = np.empty_like(bk)
f1uk_temp = np.empty_like(uk)
f1bk_temp = np.empty_like(bk)


k1u = np.zeros((3, Np, N, N), dtype = np.float64)
k1b = np.zeros((Np, N, N), dtype = np.float64)
k2u = np.zeros((3, Np, N, N), dtype = np.float64)
k2b = np.zeros((Np, N, N), dtype = np.float64)
k3u = np.zeros((3, Np, N, N), dtype = np.float64)
k3b = np.zeros((Np, N, N), dtype = np.float64)
k4u = np.zeros((3, Np, N, N), dtype = np.float64)
k4b = np.zeros((Np, N, N), dtype = np.float64)

arr_temp_r = np.zeros((Np, N, N),dtype = np.float64)
arr_temp_k = np.zeros((N, Np, N),dtype= np.float64)
arr_temp_fr = np.zeros((Np, N, N), dtype= np.complex128)      
arr_temp_ifr = np.zeros((N, Np, N), dtype= np.complex128)
arr_mpi = np.zeros((num_process,  Np,  Np, N), dtype= np.complex128)
arr_mpi_r = np.zeros((num_process,  Np,  Np, N), dtype= np.float64)


## -----------------------------------------------------


## ------FFT + iFFT + derivative functions------- 
# def rfft_mpi(u, fu):
#     arr_temp_fr[:] = rfft2(u,  axes=(1, 2))
#     arr_mpi[:] = np.swapaxes(np.reshape(arr_temp_fr, (Np,  num_process,  Np, Nf)), 0, 1)
#     comm.Alltoall([arr_mpi,  MPI.DOUBLE_COMPLEX], [fu,  MPI.DOUBLE_COMPLEX])
#     fu[:] = fft(fu, axis = 0)
#     return fu

# def irfft_mpi(fu, u):
#     arr_temp_ifr = ifft(fu,  axis = 0)
#     comm.Alltoall([arr_temp_ifr,  MPI.DOUBLE_COMPLEX], [arr_mpi, MPI.DOUBLE_COMPLEX])
#     print()
#     arr_temp_fr[:] = np.reshape(np.swapaxes(arr_mpi,  0, 1), (Np,  N,  Nf))
#     u[:] = irfft2(arr_temp_fr, (N, N), axes = (1, 2))
#     return u    

def y_to_z(a,b):
    """reshapes any scalar array slabbed in z direction to an array slabbed in y direction

    Args:
        a (nd array): array slabbed in z direction
        c (nd array): array required in to pass in the Alltoall function
        b (nd array): array slabbed in y direction but z axis is in front
    Returns:
        (nd array): array slabbed in y direction
    """
    arr_theta[:] = np.moveaxis(a.reshape(N,Np,num_process,Np),[0,1,2,3],[2,3,0,1]) 
    comm.Alltoall([arr_theta,  MPI.DOUBLE], [arr_temp_theta1,  MPI.DOUBLE])
    # print(arr_temp_theta1.shape)
    b[:] = np.moveaxis(arr_temp_theta1,[0,1,2],[2,0,1])
    return b


def y_to_z_vec(a,bb,d = None):
    """reshapes any scalar array slabbed in z direction to an array slabbed in y direction

    Args:
        a (nd array): vector array slabbed in z direction
        c (nd array): vector array required in to pass in the Alltoall function
        b (nd array): vector array slabbed in y direction but z axis is in front
    Returns:
        (nd array): array slabbed in y direction
    """
    if d == None: d = 2
    
    arr_theta_Z[:] = np.moveaxis(a.reshape(d,N,Np,num_process,Np),[0,1,2,3,4],[2,3,4,0,1]) 
    comm.Alltoall([arr_theta_Z,  MPI.DOUBLE], [arr_temp_theta,  MPI.DOUBLE])
        
    bb[:] = np.moveaxis(arr_temp_theta,[0,1,2,3],[3,0,1,2])
    return bb

    
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

def cos_to_sin(x):
    """reshapes any dct transformed array to one in dst form (in axis =2 )

    Args:
        x (nd array): dst appropriate array

    Returns:
        (nd array) : dct appropriate array
    """
    
    
    x[:] = np.roll(x, -1, axis = -1)
    x[:, :, -1] = 0.
    return x
    
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
## -----------------------------------------------------
def decompose(u,b,u_v,b_v,u_w,b_w):
    
    pv[:] = diff_z_sin(b,b_w) + diff_x(u[1],u_w[1]) - diff_y(u[0],u_w[0]) #! u_w and b_w are use such that we do not need more memory.
    pvk[:] = fft_cos(pv,pvk)
    
    pk_v[:] = invlapc*pvk 
    
    p_v[:] = ifft_cos(pk_v,p_v)
    
    u_v[0] =  -diff_y(p_v,u_v[0])
    u_v[1] =  diff_x(p_v,u_v[1])
    u_v[2,:] =  0.
    b_v[:] =  diff_z_cos(p_v,b_v)
    
    #! u_w and b_w are properly written here. 
    u_w[0] =  u[0] - u_v[0] 
    u_w[1] =  u[1] - u_v[1] 
    u_w[2] =  u[2] - u_v[2] 
    b_w[:] =  b - b_v
    
    return u_v,b_v,u_w,b_w

## --------------- RHS of the equation -----------------
def RHS(t,u, b, u_t, b_t,e_v,e_w):
    # u_v[:],b_v[:],u_w[:],b_w[:] = decompose(u,b,u_v,b_v,u_w,b_w)
    ## The RHS terms of u, v and w excluding the pressure and the hypervisocsity term 
    fk_w[:],fkb_w[:] = f_tide(t,e_w,aa_w,e_w_thresh)
    # fk_v[:], fkb_v[:] = f_balanced(t,e_v,aa_v,e_v_thresh) 
    fk_v[:], fkb_v[:] = 0.0,0.0 #! Not forcing the balanced flow.
    
    # --------------- calculating the forcing contribution ----------------
    # ke_w = 0.5*comm.allreduce(np.sum((np.abs(fk_w[0])**2 + np.abs(fk_w[1])**2 + np.abs(sin_to_cos(fk_w[2]))**2 )*normalize), op = MPI.SUM)
    # pe_w = 0.5*comm.allreduce(np.sum((np.abs(sin_to_cos(fkb_w))**2)*normalize), op = MPI.SUM)
    # ke_v = 0.5*comm.allreduce(np.sum((np.abs(fk_v[0])**2 + np.abs(fk_v[1])**2 + np.abs(sin_to_cos(fk_v[2]))**2 )*normalize), op = MPI.SUM)
    # pe_v = 0.5*comm.allreduce(np.sum((np.abs(sin_to_cos(fkb_v))**2)*normalize), op = MPI.SUM)
    # if rank ==0: print(f"Forcing contribution at time {t} are : {ke_w}, {pe_w},{ke_v},{ pe_v}")
    
    # ---------------------------------------------------------------------
    fk[:] = fk_v + fk_w
    fkb[:] = fkb_v + fkb_w

    comm.Barrier()
    rhsu[:] = -ro*(u[0]*diff_x(u[0], rhsu1) + u[1]*diff_y(u[0], rhsu2) + u[2]*diff_z_cos(u[0], rhsu3)) + u[1] 
    
    rhsv[:] = -ro*(u[0]*diff_x(u[1], rhsv1) + u[1]*diff_y(u[1], rhsv2) + u[2]*diff_z_cos(u[1], rhsv3)) - u[0]
    
    rhsw[:] = -ro*(u[0]*diff_x(u[2], rhsw1) + u[1]*diff_y(u[2], rhsw2) + u[2]*diff_z_sin(u[2], rhsw3)) + alph**2*b
    
    b_t[:] = -ro*(u[0]*diff_x(b, b_t1) + u[1]*diff_y(b, b_t2) + u[2]*diff_z_sin(b, b_t3)) - u[2]
    comm.Barrier()
    
    ## The pressure term

    p1[:] =  diff_x(rhsu, p11) + diff_y(rhsv, p12) + diff_z_sin(rhsw, p13)
    pk[:] = invpress  * fft_cos(p1, pk)
    p[:] = ifft_cos(pk, p)
    comm.Barrier()
    
    

    ## The RHS term with the pressure   
    u_t[0] = ifft_cos(dealias_cos*(fk[0] + fft_cos(rhsu - diff_x(p, p1),rhsuk)),u_t[0])
    u_t[1] = ifft_cos(dealias_cos*(fk[1] + fft_cos(rhsv - diff_y(p,  p1),rhsuk)),u_t[1])
    u_t[2] = ifft_sin(dealias_sin*(fk[2] + fft_sin(rhsw - alph**2 * diff_z_cos(p, p1),rhswk)),u_t[2])
    b_t[:] = ifft_sin(dealias_sin*(fkb + fft_sin(b_t,b_tk)),b_t)
    comm.Barrier()
        
    return u_t, b_t


def tidePressure(kinit,p):
    Ky_p = Kx[:Nf]
    Ky_p[-1] = - Ky_p[-1]
    kx_p,ky_p,kz_p = np.meshgrid(Kx,  Ky_p,  Kzc[sx],  indexing = 'ij')
    kc2 = (kx_p**2 + ky_p**2+ kz_p**2)

    th = np.random.uniform(0, 2*TWO_PI,  kc2.shape)
    pinit = np.exp(1j*th)*(kx_p**2 + ky_p**2<kinit**2)*(kz_p == 1.0)

    pTemp1 = irfft2(pinit,(N,N),axes= (0,1))
    arr_mpi_temp = np.empty((num_process,  Np,  N, Np),dtype=np.float64)
    comm.Alltoall([pTemp1,  MPI.DOUBLE], [arr_mpi_temp, MPI.DOUBLE])
    arr_temp_fr_temp =np.swapaxes(np.swapaxes(arr_mpi_temp, 0,1),1,2) .reshape((Np,N,N))
    p[:] = idct(arr_temp_fr_temp,type = 2, axis =2)

    del pTemp1, pinit,arr_mpi_temp, arr_temp_fr_temp, kx_p,ky_p,kz_p,kc2

    return p
    
    
def dispersion(pk,uk_,bk_,sig):
    ## ----------- condition for tides --------------
    cond[:] = (np.abs(np.abs(kx)-1)<1e-5)* (np.abs(np.abs(ky)-1)<1e-5)
    # cond[:] = (kh <  alph*(sig**2 - 1)**0.5/(alph**2 - sig**2)**0.5 + 0.5)* ( kh>alph*(sig**2 - 1)**0.5/(alph**2 - sig**2)**0.5 - 0.5)
    pk_temp[:] = pk*cond
    
    uk_[0,:] = pk_temp* (1j*ky + kx*sig)/(sig**2 - 1)
    uk_[1,:] = pk_temp* (-1j*kx + ky*sig)/(sig**2 - 1)
    bk_[:] = -cos_to_sin(pk_temp)*(kh**2)/(1.*kzs *(sig**2 - 1))
    uk_[2,:] = 1j*sig*bk_
    

    return uk_, bk_
    
    


def initialize_forcing():
    global theta1p,theta2p,theta1_z,theta2_z,f1uk_temp,f1bk_temp,f2uk_temp,f2bk_temp,pinit
    
    # -------- initializing balanced theta ------ #
    theta1_z[0] = np.random.uniform(0,TWO_PI,(N,N,Np))
    theta2_z[0] = np.random.uniform(0,TWO_PI,(N,N,Np))
    
    theta1_z[0,N//2:-1,N//2:-1,:] = - theta1_z[0,1:N//2,1:N//2,:]
    theta2_z[0,N//2:-1,N//2:-1,:] = - theta2_z[0,1:N//2,1:N//2,:]
    
    # if rank ==0 : print(f"b4 communicating")
    comm.Barrier()
    theta1p[:] = y_to_z(theta1_z[0],theta1p)
    theta2p[:] = y_to_z(theta2_z[0],theta2p)
    comm.Barrier()
    
    theta1p[:] = np.where(((kzc >0)*(kc<6) ), theta1p, 1j*1e100)
    theta2p[:] = np.where(((kzc >0)*(kc<6) ), theta2p, 1j*1e100)
    # -------------------------------------------- #
    
    ## --------- Getting pressure with Kz =1 ----------
    # if rank ==0 : print(pinit)
    p[:] = tidePressure(kinit, p)
    pk[:] = fft_cos(p,pk) 
    a1 = 0.5 
    b1 = 0.5*1j
    pk_pos[:] = a1*pk + b1*pk.conjugate()
    pk_neg[:] = a1*pk - b1*pk.conjugate()

    ## ------------------------------------------------


    ## ---------------- tides --------------------------
    for sig in sigPos_arr:
        f1uk_temp[:], f1bk_temp[:] = dispersion(pk_pos,f1uk_temp,f1bk_temp,sig)

    for sig in sigNeg_arr:
        uk[:], bk[:] = dispersion(pk_neg,uk,bk,sig)
        f1uk_temp[:] = f1uk_temp + uk
        f1bk_temp[:] = f1bk_temp + bk
    ## ------------------------------------------------
    

    
    return None
    # --------------------------------------------- #
    

def f_balanced(t,e_v,aa,ec):
    """
    Input:
        u: 3D array of shape (3,Np,N,N)
        uk: 3D array of shape (3,N,Np,N)
    Output:
        f: 3D array of shape (3,Np,N,N)
    """
    #? Check if the energy is less than some critical energy. 
    if e_v <ec:
        comm.Barrier()
        # if rank ==0 : print(f"forcing balanced flow at t = {t:.4f} with energy = {e_v:.4f}")   
        fk_v[0] = -1j*ky*aa*(np.exp(1j*theta1p))*(kzc> 0)*(kc<6)
        fk_v[1] = 1j*kx* aa*(np.exp(1j*theta1p))*(kzc >0)*(kc<6)
        fk_v[2] = 0.
        fkb_v[:] = -kzs* aa*cos_to_sin(np.exp(1j*theta1p))*(kzs >0)*(ks<6)
    else:
        fk_v[:] = 0.
        fkb_v[:] = 0.
    return fk_v,fkb_v

def f_tide(t,e_w,aa,ec):
    """
    Forcing the frequencies at  omega = 2.
    But first it checks whether the wave energy has gotten less than the a crtical value.
    """
    
    if e_w<ec and t> 0.: 
        # if rank == 0: print(f"Forcing waves at time = {t:.4f} with energy = {e_w:.4f}")
        fk_w[0] = aa*(f1uk_temp[0] )*np.cos(omega*t)
        fk_w[1] = aa*(f1uk_temp[1] )*np.cos(omega*t)
        fk_w[2] = aa*(f1uk_temp[2] )*np.cos(omega*t)
        fkb_w[:] = aa*(f1bk_temp )*np.cos(omega*t)
        
    else:
        fk_w[:] = 0.
        fkb_w[:] = 0.
    return fk_w, fkb_w

def spectra(uk,bk):
    ek[:] = 0.5*(np.abs(uk[0])**2 + np.abs(uk[1])**2 + sin_to_cos(np.abs(uk[2])**2/alph**2) + sin_to_cos(np.abs(bk)**2))*normalize
    return comm.allreduce(np.histogram(kcint.ravel(), bins = shells, weights = ek.ravel())[0], op = MPI.SUM)
    
## -----------------------------------------------------------


## ---------------- Saving data + energy + Showing total energy ---------------------

def save(i,u,b,u_w,b_w,u_v,b_v):
    # ----------- ----------------------------
    #                 Saving the data (field + energy)
    # ----------- ----------------------------
    
    if t[i] % dt_save < dt*1.01 and begin:
        new_dir = savePath/f"time_{t[i]:.1f}"
        pv_dir = savePath/f"PV/time_{t[i]:.1f}"
        omg_dir = savePath/f"zeta/time_{t[i]:.1f}"
        try: 
            new_dir.mkdir(parents=True,  exist_ok=True)
            pv_dir.mkdir(parents=True,  exist_ok=True)
            omg_dir.mkdir(parents=True,  exist_ok=True)               
        except FileExistsError: pass
        # pv[:] = diff_z_sin(b,p11) + diff_x(u[1],p12) - diff_y(u[0],p13)
        # omg[:] = diff_x(u[1],p11) - diff_y(u[0],p12)
        
        np.savez_compressed(f"{new_dir}/Fields_{rank}",u = u[0],v = u[1],w = u[2],b = b)
        # np.save(pv_dir/f"PV_{rank}",pv)
        # np.save(omg_dir/f"zeta_{rank}",omg)
        comm.Barrier()
        
        
        # ----------- ----------------------------
        #          Calculating and printing
        # ----------- ----------------------------
        eng = comm.allreduce(np.sum(0.5*(u[0]**2 + u[1]**2 + u[2]**2/alph**2 + b**2)*dx*dy*dz), op = MPI.SUM)
        uk_v[0] = fft_cos(u_v[0],uk_v[0])
        uk_v[1] = fft_cos(u_v[1],uk_v[1])
        uk_v[2] = fft_sin(u_v[2],uk_v[2])
        bk_v[:] = fft_sin(b_v,bk_v)
        
        uk_w[0] = fft_cos(u_w[0],uk_w[0])
        uk_w[1] = fft_cos(u_w[1],uk_w[1])
        uk_w[2] = fft_sin(u_w[2],uk_w[2])
        bk_w[:] = fft_sin(b_w,bk_w)
        
        ek1d_v = spectra(uk_v,bk_v)
        ek1d_w = spectra(uk_w,bk_w)
        if rank ==0: np.savez_compressed(f"{new_dir}/Spectra",ek1d_v = ek1d_v,ek1d_w = ek1d_w)
        comm.Barrier()
        # e_v = comm.allreduce(np.sum(0.5*(u_v[0]**2 + u_v[1]**2 + u_v[2]**2/alph**2 + b_v**2)*dx*dy*dz), op = MPI.SUM)
        # e_w = comm.allreduce(np.sum(0.5*(u_w[0]**2 + u_w[1]**2 + u_w[2]**2/alph**2 + b_w**2)*dx*dy*dz), op = MPI.SUM)
        dissp = -nu*comm.allreduce(np.sum((kc**(2*lp)*(np.abs(uk[0])**2 + np.abs(uk[1])**2) +sin_to_cos( ks**(2*lp)*(np.abs(uk[2])**2/alph**2 + np.abs(bk)**2)))), op = MPI.SUM)
        # if rank == 0:
        #     print( "#----------------------------","\n",f"Energy at time {t[i]} is : {eng}","\n","#----------------------------")
        #     print( "#----------------------------","\n",f"Balanced Energy at time {t[i]} is : {np.sum(ek1d_v)}","\n","#----------------------------")
        #     print( "#----------------------------","\n",f"Wave Energy at time {t[i]} is : {np.sum(ek1d_w)}","\n","#----------------------------")
        #     print( "#----------------------------","\n",f"Total dissipation at time {t[i]} is : {dissp}","\n","#----------------------------")
            # print( "#----------------------------","\n",f" Rms field value at time {t[i]} is : {sqfld}","\n","#----------------------------")
    elif t[i]%dt_save_r < dt*1.01: 
        if rank ==0: 
            new_dir = savePath/f"time_{t[i]:.1f}"
            try: new_dir.mkdir(parents=True,  exist_ok=True)
            except FileExistsError: pass
            print(f"saving for omega plot at time {t[i]}")
            np.savez_compressed(f"{new_dir}/Fields_{rank}",u = u[0],v = u[1],w = u[2],b = b)                
    return "Done!"    
## -------------------------------------------------    
    
## ------------- Evolving the system ----------------- 
def evolve_and_save(t,  u, b): 
    global begin
    h = t[1] - t[0]
    hypervisc_sin = dealias_sin*(1. + h*vis_sin)**(-1)
    hypervisc_cos = dealias_cos*(1. + h*vis_cos)**(-1)
    initialize_forcing()
    # print(f"max of sin viscous term : {np.max(abs(hypervisc_sin))}")
    # print(f"max of cos viscous term : {np.max(abs(hypervisc_cos))}")
    unew = np.zeros_like(u)
    bnew = np.zeros_like(b)
    t3  = time()
    for i in range(t.size-1):
        # if rank == 0:  print("time", np.round(t[i], 4), end= '\r')
        if rank == 0:  print(f"step {i} in time {time() - t3}", end= '\r',file = sys.stderr)
        t3 = time()
        # u_v[:],b_v[:],u_w[:],b_w[:] = decompose(u,b,u_v,b_v,u_w,b_w)
        u_v[:],b_v[:],u[:],b[:] = decompose(u,b,u_v,b_v,u_w,b_w) #! Eliminating the geostrophic part

        e_w = 0.5*comm.allreduce(np.sum(u_w[0]**2 + u_w[1]**2 + u_w[2]**2/alph**2 + b_w**2), op = MPI.SUM)*dx*dy*dz
        e_v = 0.5*comm.allreduce(np.sum(u_v[0]**2 + u_v[1]**2 + u_v[2]**2/alph**2 + b_v**2), op = MPI.SUM)*dx*dy*dz
        if rank ==0 : print(f"t = {t[i]:.4f} bf: unbalanced energy = {e_w:.4f}, balanced energy = {e_v:.4f}")
        
        k1u[:], k1b[:] = RHS(t[i],u, b, k1u, k1b,e_v,e_w)
        comm.Barrier()
        k2u[:], k2b[:] = RHS(t[i]+h,u + h*k1u, b + h*k1b, k2u, k2b,e_v,e_w)
        # k2u[:], k2b[:] = RHS(t[i]+h/2.,u + h/2.*k1u, b + h/2.*k1b, k2u, k2b,e_v,e_w)
        # comm.Barrier()
        # k3u[:], k3b[:] = RHS(t[i]+h/2.,u + h/2.*k2u, b + h/2.*k2b, k3u, k3b,e_v,e_w)
        # comm.Barrier()
        # k4u[:], k4b[:] = RHS(t[i] + h,u + h*k3u, b + h*k3b, k4u, k4b,e_v,e_w)
        comm.Barrier()
        
        # unew[:] = u + h/6.0* ( k1u + 2*k2u + 2*k3u + k4u)
        # bnew[:] = b + h/6.0* ( k1b + 2*k2b + 2*k3b + k4b)
        
        unew[:] = u + h/2.0* ( k1u + k2u )
        bnew[:] = b + h/2.0* ( k1b + k2b )
        
        # u_v[:],b_v[:],u_w[:],b_w[:] = decompose(unew,bnew,u_v,b_v,u_w,b_w)
        # e_w = 0.5*comm.allreduce(np.sum(u_w[0]**2 + u_w[1]**2 + u_w[2]**2/alph**2 + b_w**2), op = MPI.SUM)*dx*dy*dz
        # e_v = 0.5*comm.allreduce(np.sum(u_v[0]**2 + u_v[1]**2 + u_v[2]**2/alph**2 + b_v**2), op = MPI.SUM)*dx*dy*dz
        # if rank ==0 : print(f"t = {t[i]:.4f} af : unbalanced energy = {e_w:.4f}, balanced energy = {e_v:.4f}")
        
        uk[0]  = hypervisc_cos * fft_cos(unew[0] , uk[0])
        uk[1]  = hypervisc_cos * fft_cos(unew[1] , uk[1])
        uk[2]  = hypervisc_sin * fft_sin(unew[2] , uk[2])
        bk[:]  = hypervisc_sin * fft_sin(bnew , bk)
        
        
        
        # u_v[:],b_v[:],u_w[:],b_w[:] = decompose(u,b,u_v,b_v,u_w,b_w)
                
        # u_v[:],b_v[:],u_w[:],b_w[:] = decompose(u,b,u_v,b_v,u_w,b_w)
                
        # corr = comm.allreduce(np.sum(np.real(fk[0]*np.conjugate(uk[0]) + fk[1]*np.conjugate(uk[1]) + fk[2]*np.conjugate(u[2])/alph**2 + fkb*np.conjugate(bk ))), op = MPI.SUM)
        # if rank == 0: print(f"Correlation is {corr}")
        # fk[:] = f_tide(t[i+1], uk,aa)            
        
        # corr = comm.allreduce(np.sum(np.real(fk[0]*np.conjugate(uk[0]) + fk[1]*np.conjugate(uk[1]) + fk[2]*np.conjugate(u[2])/alph**2 + fkb*np.conjugate(bk ))), op = MPI.SUM)
        # if rank == 0: print(f"Correlation is {corr}")
        # fk[:] = f_tide(t[i+1], uk,aa)            
        
                
        comm.Barrier()
        unew[0] = ifft_cos(uk[0], unew[0])
        unew[1] = ifft_cos(uk[1], unew[1])
        unew[2] = ifft_sin(uk[2], unew[2])
        bnew[:] = ifft_sin(bk, bnew)
        comm.Barrier()
        
        ##------ Saving the data after every one second ---------
        save(i,u,b,u_w,b_w,u_v,b_v)
        begin = True   
 
        ## -------------------------------------------------------
        if unew.max() > 100 or bnew.max() > 100: 
            print("Threshold exceeded at time", t[i+1], "Code about to be terminated")
            comm.Abort()
            # MPI.Finalize()
            # raise ValueError(f"Threshold exceeded at time {t[i+1]}. Code terminated")
        u[:], b[:] = unew, bnew
        
        
        comm.Barrier()
        
    ## ---------- Saving the final data ------------
    save(i+1, unew,bnew,u_w,b_w,u_v,b_v)
    ## -----------------------------------------------

    

## --------------- Initializing ---------------------
# Only the steady flow. 
# Only need to initialize pressure /q .


"""Structure 
If there exists a folder with the parameter names and has time folders in it. 
Load the parameters from parameters.txt
If the parameters match the current code parameters enter the last time folder.
Finally load the data.
If not start from scratch."""



begin = True ## It will change if we manage to load the data
# Does a folder exist with parameters file? 
paramfile = (savePath/f"parameters.json")
#! Modify the loading process!
if paramfile.exists() and not forcestart:
    ## ------------------------- Beginning from existing data -------------------------
    """Load the parameters in the param file"""
    with open(paramfile) as f:
        params_loaded = json.load(f)

    """ Things that need to match"""
    keys = ["Rossby","hyperviscous","Alpha","Initial balanced amplitude"]
    if rank ==0 : print("Paramfile Exists!")
    match = True
    for i in keys:
        if param[i] != params_loaded[i]: 
            if rank ==0: print(f"Parameter {i} does not match. Starting from scratch")
            match = False
        
    
    if match:
        if rank ==0 : print("Found existing simulation! Using last saved data.")
        begin = False
        """Loading the data from the last time  """    
        paths = sorted([x for x in (savePath).iterdir() if "time_" in str(x)], key=os.path.getmtime)
        """The folder is paths[-1]"""
        paths = paths[-1]
        tinit = float(str(paths).split("time_")[-1])
        tinit = 250.0 #! Manually setting restart time
        paths = savePath/f"time_{tinit:.1f}" 
        
        num_process_data = params_loaded["num_process"]
        if num_process_data >= num_process:
            Ns = num_process_data//num_process
            Nslab = N// num_process_data
            """loading the data"""            
            for j in range(Ns):
                field = np.load(paths/f"Fields_{rank*Ns+j}.npz")
                u[0,j*Nslab:(j+1)*(Nslab),...] = field["u"]
                u[1,j*Nslab:(j+1)*(Nslab),...] = field["v"]
                u[2,j*Nslab:(j+1)*(Nslab),...] = field["w"]
                b[j*Nslab:(j+1)*(Nslab),...] = field["b"]
        else: 
            Ns = num_process//num_process_data
            Nslab = N// num_process
            j = rank%2
            field = np.load(paths/f"Fields_{rank//2}.npz")
            u[0,...] = field["u"][j*Nslab:(j+1)*(Nslab)]
            u[1,...] = field["v"][j*Nslab:(j+1)*(Nslab)]
            u[2,...] = field["w"][j*Nslab:(j+1)*(Nslab)]
            b[...] = field["b"][j*Nslab:(j+1)*(Nslab)]
        if rank ==0: print(f"Data loaded for time {tinit}")
        del paths,num_process_data,Ns,Nslab
    del params_loaded,keys,match


if begin or forcestart:
    ## ---------------------- Beginning from start ----------------------------------
    Kx_p = Kx[:Nf]
    Kx_p[-1] = - Kx_p[-1]
    kx_p,ky_p,kz_p = np.meshgrid(Kx_p,  Ky,  Kzc[sx],  indexing = 'ij')
    kc2 = (kx_p**2 + ky_p**2+ kz_p**2)

    th = np.random.uniform(0, 2*TWO_PI,  kc2.shape)
    pinit = einit*np.exp(1j*th)*(kc2<kinit**2)

    pTemp1 = irfft2(pinit,(N,N),axes= (0,1))
    
    arr_mpi_temp = np.zeros((num_process,  Np,  N, Np),dtype=np.float64)
    comm.Alltoall([pTemp1,  MPI.DOUBLE], [arr_mpi_temp, MPI.DOUBLE])
    arr_temp_fr_temp =np.swapaxes(np.swapaxes(arr_mpi_temp, 0,1),1,2) .reshape((Np,N,N))
    p[:] = idct(arr_temp_fr_temp,type = 2, axis =2)

    del pTemp1, pinit,arr_mpi_temp, arr_temp_fr_temp, kx_p,ky_p,kz_p,kc2

    u[0, :] = -diff_y(p,  u[0])
    u[1, :] = diff_x(p,  u[1])
    u[2, :] = 0.
    b[:] = diff_z_cos(p,  b)
    tinit = 0.

        
if rank ==0: 
    with open(savePath/f"parameters.json","w") as jsonFile: json.dump(param, jsonFile,indent = 2)

##----------------- The initial energy ------------------
u_v[:],b_v[:],u_w[:],b_w[:] = decompose(u,b,u_v,b_v,u_w,b_w)
eng = comm.allreduce(0.5*dx*dy*dz*np.sum(u[0]**2 + u[1]**2 + (u[2]*u[2])/alph**2 + b*b), op = MPI.SUM)
eng_w = comm.allreduce(0.5*dx*dy*dz*np.sum(u_w[0]**2 + u_w[1]**2 + (u_w[2]*u_w[2])/alph**2 + b_w*b_w), op = MPI.SUM)
eng_v = comm.allreduce(0.5*dx*dy*dz*np.sum(u_v[0]**2 + u_v[1]**2 + (u_v[2]*u_v[2])/alph**2 + b_v*b_v), op = MPI.SUM)
if rank == 0: print(f"Total , wave and balanced Energy at time {tinit} is : {eng},{eng_w},{eng_v}",tinit, begin)
##-------------------------------------------------------

## --------------------------------------------------

## ----- executing the code -------------------------
    
t = np.arange(tinit,T+ dt*0.7, dt)
# print(len(t))
t1 = time()
evolve_and_save(t,u, b)
t2 = time() - t1 
# --------------------------------------------------
if rank ==0 : print(f"Time taken {t2}")
## --------- saving the calculation time -----------

## --------------------------------------------------