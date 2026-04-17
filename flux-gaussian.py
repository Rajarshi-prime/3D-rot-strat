import numpy as np 
from scipy.fft import fft ,  ifft ,  irfft2 ,  rfft2 , irfftn ,  rfftn, fftfreq, dst, dct, idst, idct, rfft,  irfft,fftshift
from mpi4py import MPI
from time import time
import pathlib,os,h5py
from sympy import LeviCivita

curr_path = pathlib.Path(__file__).parent
forcestart = True

## ---------------MPI things--------------
comm = MPI.COMM_WORLD
num_process =  comm.Get_size()
rank = comm.Get_rank()
isforcing = False
viscosity_integrator = "implicit"
# viscosity_integrator = "explicit"
# viscosity_integrator = "exponential"
if viscosity_integrator == "explicit": isexplicit = 1.
else : isexplicit = 0
## ---------------------------------------

## ------------- Time steps --------------
T = 0
dt =  0.001
dt_save = 1.0
st = round(dt_save/dt)
## ---------------------------------------

## -------------Defining the grid ---------------
PI = np.pi
TWO_PI = 2*PI
N = 256
Nf = N//2 + 1
Np = N//num_process
sx = slice(rank*Np ,  (rank+1)*Np)
L = TWO_PI
X = Y = Z = np.linspace(0, L, N, endpoint= False)
dx,dy,dz = X[1]-X[0], Y[1]-Y[0], Z[1]-Z[0]
x, y, z = np.meshgrid(X[sx], Y, Z, indexing='ij')

Kx = Ky = Kz = fftfreq(N,  1./N)*TWO_PI/L
dkx, dky, dkz = Kx[1]-Kx[0], Ky[1]-Ky[0], Kz[1]-Kz[0]
Kz = Kz[:Nf]
Kz[-1] = -Kz[-1]

kx,  ky,  kz = np.meshgrid(Kx,  Ky[sx],  Kz,  indexing = 'ij')
## -----------------------------------------------

## --------- kx and ky for differentiation ---------    
kx_diff = np.moveaxis(kz,[0,1,2],[2,1,0]).copy()
ky_diff = np.swapaxes(kx_diff, 0, 1).copy()
kz_diff = np.moveaxis(kz, [0,1], [1,0]).copy()

if rank ==0 : print(kx_diff.shape, ky_diff.shape, kz_diff.shape)

## -------------------------------------------------

## ---cd -------- Parameters ----------
nu = 0.  # Hyperviscosity
lp = 1 # Hyperviscosity power
einit = 0.8 # Initial velocity amplitude
shell_no = 1 # Fixing the energy of this shell with a particular profile
nshells = 1 # Number of consecutive shells to be forced


#----  Kolmogorov length scale - \eta \epsilon etc...---------

k_eta = N//3
f0    = k_eta**4.0*nu**3.0

# ------------------------------------------------------------

## ---------- rest of the parameters --------------


def create_dir(path):
    path = pathlib.Path(path)
    try:
        path.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        pass

if nu!= 0: 
    loadPath = pathlib.Path(f"/mnt/pfs/rajarshi.chattopadhyay/3D-DNS/data_new_new/forced_{isforcing}/N_{N}_Re_{1/nu:.1f}")
    pstprc_path = pathlib.Path(f"/mnt/pfs/rajarshi.chattopadhyay/codes/3D-DNS/pstprc_data/N_{N}_Re_{1/nu:.1f}")
else:
    loadPath = pathlib.Path(f"/mnt/pfs/rajarshi.chattopadhyay/3D-DNS/data_new_new/forced_{isforcing}/N_{N}_Re_inf")
    pstprc_path = pathlib.Path(f"/mnt/pfs/rajarshi.chattopadhyay/codes/3D-DNS/pstprc_data/N_{N}_Re_inf")
if rank ==0 : print(f"Load Path {loadPath}")
savePath = loadPath

create_dir(pstprc_path)
create_dir(savePath)
comm.Barrier()
paramfile = (loadPath/f"params.txt")


## -------------------------------------------------------



## --------- kx and ky for differentiation --------------------------

kx_diff = np.moveaxis(kz,[0,1,2],[2,1,0]).copy()
ky_diff = np.swapaxes(kx_diff, 0, 1).copy()
kz_diff = np.moveaxis(kz, [0,1], [1,0]).copy()

## ------------------------------------------------------------------

## ------------Useful Operators-------------------



lap = -(kx**2 + ky**2 + kz**2 )
k = (-lap)**0.5
kint = np.clip(np.round(k,0).astype(int),None,N//2)
kvec = np.array([kx,ky,kz])
invlap = 1.0/np.where(lap == 0, np.inf,  lap)
# dealias = (abs(kx)<N//3)*(abs(ky)<N//3)*(abs(kz)<N//3)
dealias = kint<=N/3 #! Spherical dealiasing


normalize = np.where((kz== 0) , 1/(N**6/TWO_PI**3),2/(N**6/TWO_PI**3))
shells = np.arange(-0.5,Nf, 1.)
shells[0] = 0.
# Hyperviscous operator
vis = nu*(-1*k**2)**(lp) ## This is in Fourier Space
## -------------------------------------------------


## -------------Empty arrays ----------------------------------------
epsilon = np.array([[[float(LeviCivita(i, j, k)) for k in range(3)] for j in range(3)] for i in range(3)])
u = np.empty((3, Np, N, N), dtype= np.float64)
omg = u.copy()
omg_filt = u.copy()
S = np.empty((3,3, Np, N, N), dtype= np.float64)
S_filt = S.copy()
tau_filt = S.copy()
u1 = np.empty((3, Np, N, N), dtype= np.float64)
p = u[0].copy()
vort_stretch = p.copy()
somg = p.copy()
filt = p.copy()
pi = p.copy()
pi_s1 = p.copy()
pi_omg1 = p.copy()
pi_s2 = p.copy()
pi_omg2 = p.copy()
pi_c = p.copy()

uk = np.empty((3, N, Np, Nf), dtype= np.complex128)
Sk = np.empty((3,3, N, Np, Nf), dtype= np.complex128)
omgk = uk.copy()
u1k = np.empty((3, N, Np, Nf), dtype= np.complex128)
pk = uk[0].copy()
filt_k = pk.copy()



rhsuk = np.empty_like(pk)
rhsvk = rhsuk.copy()
rhswk = rhsuk.copy()

rhsu = np.empty_like(p)
rhsv = rhsu.copy()
rhsw = rhsu.copy()

rhsu1 = np.empty_like(p)
rhsu2 = np.empty_like(p)
rhsu3 = np.empty_like(p)

div = p.copy()
dissip = p.copy()



arr_temp_k = np.empty((N, Np, N),dtype= np.float64)
arr_temp_fr = np.empty((Np, N, Nf), dtype= np.complex128)      
arr_temp_ifr = np.empty((N, Np, Nf), dtype= np.complex128)      
arr_mpi = np.empty((num_process,  Np,  Np, Nf), dtype= np.complex128)
arr_mpi_r = np.empty((num_process,  Np,  Np, N), dtype= np.float64)


ek = np.empty_like(pk,dtype = np.float64)
PIk = np.empty_like(ek,dtype = np.float64)
## ------------------------------------------------------------------


## ------------------------------------------------------------------
##                      Functions to use
## ------------------------------------------------------------------


## --------------------- FFT + diff fns ---------------------------
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
    arr_mpi_r[:] = np.swapaxes(np.reshape(u, (Np,  num_process,  Np,  N)), 0, 1)
    comm.Alltoall([arr_mpi_r,  MPI.DOUBLE], [arr_temp_k,  MPI.DOUBLE])
    arr_temp_k[:] = irfft(1j * kx_diff*rfft(arr_temp_k,  axis = 0), N,  axis=0)
    comm.Alltoall([arr_temp_k,  MPI.DOUBLE], [arr_mpi_r,  MPI.DOUBLE])
    u_x[:] = np.reshape(np.swapaxes(arr_mpi_r,  0, 1), (Np,  N, N))
    return u_x

def diff_y(u, u_y):
    u_y[:] = irfft(1j*ky_diff*rfft(u, axis= 1), N, axis= 1)
    return u_y
    
def diff_z(u, u_z):
    u_z[:] = irfft(1j*kz_diff*rfft(u, axis= 2), N, axis= 2)
    return u_z

    
## -----------------------------------------------------------------

## --------------------- Other functions ---------------------------
def load_data(t,u):
    
    load_num_slabs = len([i for i in os.listdir(savePath/f"time_{t:.1f}") if i.startswith("Fields")])
    data_per_rank = N//load_num_slabs
    rank_data = range(rank*Np,(rank + 1)*Np) # The rank contains these slices 
    slab_old = np.inf
    for lidx,j in enumerate(rank_data):
        slab = j//data_per_rank
        idx = j%data_per_rank
        
        if slab_old != slab:  Field = np.load(savePath/f"time_{t:.1f}/Fields_{slab}.npz")
        slab_old = slab
        u[0,lidx] = Field['u'][idx]
        u[1,lidx] = Field['v'][idx]
        u[2,lidx] = Field['w'][idx]

    comm.Barrier()
    return u

def energy_flux(uk,PIk):
    """Calculate the 3D E_k from the given u,v,w,b and store return in ek

    Args:
        u (3*nd array): Velocity
        ek (nd array): Energy
        PI_k (nd array): Flux
    """
    
    
    u1k[:] = RHS(uk,u1k)
    
    PIk[:] = np.real(np.conjugate(uk[0])*u1k[0]+np.conjugate(uk[1])*u1k[1]+ np.conjugate(uk[2])*u1k[2])*dkx*dky*dkz*normalize*dealias
    
    return PIk


## ------------------ RHS for Euler -----------------
def RHS(uk, uk_t):
    ## The RHS terms of u, v and w excluding the pressure and the hypervisocsity term 
    u[0] = irfft_mpi(uk[0], u[0])
    u[1] = irfft_mpi(uk[1], u[1])
    u[2] = irfft_mpi(uk[2], u[2])
    
    rhsu[:] = -(u[0]*diff_x(u[0], rhsu1) + u[1]*diff_y(u[0], rhsu2) + u[2]*diff_z(u[0], rhsu3)) 
    
    rhsv[:] = -(u[0]*diff_x(u[1], rhsu1) + u[1]*diff_y(u[1], rhsu2) + u[2]*diff_z(u[1], rhsu3)) 
    
    rhsw[:] = -(u[0]*diff_x(u[2], rhsu1) + u[1]*diff_y(u[2], rhsu2) + u[2]*diff_z(u[2], rhsu3)) 
    
    rhsuk[:]  = rfft_mpi(rhsu, rhsuk) *dealias # does not give proper flux
    rhsvk[:]  = rfft_mpi(rhsv, rhsvk) *dealias # does not give proper flux
    rhswk[:]  = rfft_mpi(rhsw, rhswk) *dealias # does not give proper flux
    
    ## The pressure term
    pk[:] = 1j*invlap  * (kx*rhsuk + ky*rhsvk + kz*rhswk)
    
    

    ## The RHS term with the pressure   
    uk_t[0] = rhsuk - 1j*kx*pk
    uk_t[1] = rhsvk - 1j*ky*pk
    uk_t[2] = rhswk - 1j*kz*pk
    

    
    # rhsu[:] = irfft_mpi(uk_t[0], rhsu)
    # rhsv[:] = irfft_mpi(uk_t[1], rhsv)
    # rhsw[:] = irfft_mpi(uk_t[2], rhsw)
        
    return uk_t


# def RHS(uk, uk_t):
#     ## The RHS terms of u, v and w excluding the pressure and the hypervisocsity term 
#     u[0] = irfft_mpi(uk[0], u[0])
#     u[1] = irfft_mpi(uk[1], u[1])
#     u[2] = irfft_mpi(uk[2], u[2])
    
#     omg[0] = irfft_mpi(1j*(ky*uk[2] - kz*uk[1]),omg[0])
#     omg[1] = irfft_mpi(1j*(kz*uk[0] - kx*uk[2]),omg[1])
#     omg[2] = irfft_mpi(1j*(kx*uk[1] - ky*uk[0]),omg[2])
    
#     rhsu[:] = (omg[2]*u[1] - omg[1]*u[2])
#     rhsv[:] = (omg[0]*u[2] - omg[2]*u[0])
#     rhsw[:] = (omg[1]*u[0] - omg[0]*u[1]) 
    
    
#     rhsuk[:]  = rfft_mpi(rhsu, rhsuk)*dealias 
#     rhsvk[:]  = rfft_mpi(rhsv, rhsvk)*dealias 
#     rhswk[:]  = rfft_mpi(rhsw, rhswk)*dealias 
    
#     ## The pressure term
#     pk[:] = 1j*invlap  * (kx*rhsuk + ky*rhsvk + kz*rhswk)
    
    

#     ## The RHS term with the pressure   
#     uk_t[0] = rhsuk - 1j*kx*pk 
#     uk_t[1] = rhsvk - 1j*ky*pk 
#     uk_t[2] = rhswk - 1j*kz*pk
    

        
#     return uk_t 
## -------------------------------------------------------


## ------------------ binning function ------------------- ## 
def e3d_to_e1d(x): #1 Based on whether k is 2D or 3D, it will bin the data accordingly. 
    return np.histogram(k.ravel(),bins = shells,weights=x.ravel())[0] 

## ------------------------------------------------------- ##
def tau(a,b, filt_k):
    """ Calculates the general filtered stress tensor given a particular filt_k matrix"""
    #! Using the rhsuk and rhsvk as temporary arrays
    rhsuk[:] = rfft_mpi(a,rhsuk)*filt_k
    rhsvk[:] = rfft_mpi(b,rhsvk)*filt_k
    rhswk[:] = rfft_mpi(a*b,rhswk)*filt_k
    
    rhsu[:] = irfft_mpi(rhsuk,rhsu)
    rhsv[:] = irfft_mpi(rhsvk,rhsv)
    rhsw[:] = irfft_mpi(rhswk,rhsw)
    
    return rhsw - rhsu*rhsv
    

def filterd(a,af,filt_k):
    """ Calculates the filtered field given a particular filter """
    af[:] = irfft_mpi(rfft_mpi(a,pk)*filt_k,af)
    return af


    # # Calculating the filtered stress and strain tensor

   
t = [4.0]

for i,time in enumerate(t):
    if rank ==0: print(f"Time : {t[i]:.1f}")
    
    u[:] = load_data(t[i],u)
    # comm.Barrier()
    div[:] = diff_x(u[0],u1[0]) + diff_y(u[1],u1[1]) + diff_z(u[2],u1[2])
    divmax = comm.allreduce(np.max(np.abs(div)),op = MPI.MAX)
    if rank ==0 : print(f"Max abs divergence {divmax}")
    # if rank ==0 : print(f'np.max(np.abs(u)) : {np.max(np.abs(u))}')
    # if rank ==0 : print(f'Loading Done')
    # if rank ==0 : print(f'u_w.shape : {u_w.shape}')
    
    uk[0,:] = rfft_mpi(u[0],uk[0])*dealias
    uk[1,:] = rfft_mpi(u[1],uk[1])*dealias
    uk[2,:] = rfft_mpi(u[2],uk[2])*dealias
    
    u1k[:] = RHS(uk,u1k)
    
    u1[0] = irfft_mpi(u1k[0],u1[0])
    u1[1] = irfft_mpi(u1k[1],u1[1])
    u1[2] = irfft_mpi(u1k[2],u1[2])
    
    tot_transfer = comm.allreduce(np.sum(u[0]*u1[0] + u[1]*u1[1] + u[2]*u1[2])*dx*dy*dz,op = MPI.SUM)
    if rank ==0 : print(f"Total transfer : {tot_transfer}")
    #? Just to compare the values of the filtered flux
    PIk[:] = energy_flux(uk,PIk)
    ek[:] = 0.5*normalize*(uk[0]*np.conjugate(uk[0]) + uk[1]*np.conjugate(uk[1]) + uk[2]*np.conjugate(uk[2])).real
    
    etot = comm.allreduce(np.sum(ek),op = MPI.SUM)/(TWO_PI**3)
    PIk_arr = comm.allreduce(e3d_to_e1d(PIk),op = MPI.SUM)
    # PIk_arr = -np.cumsum(PIk_arr)
    PIk_arr = np.cumsum(PIk_arr[::-1])[::-1]
    if rank ==0: print(f"PIk_arr : {PIk_arr}, etot = {etot}")
    # raise SystemExit()
    #? ------------------------------------------------
    
    Sk[:] = 1j*0.5*(np.einsum('i...,j...->ij...',kvec,uk) + np.einsum('i...,j...->ij...',uk,kvec))
    omgk[:] = 1j*np.einsum('ijk,j...,k...->i...',epsilon, kvec,uk)
    
        
    for ii in range(3):
        omg[ii] = irfft_mpi(omgk[ii],omg[ii])
        for jj in range(ii,3):
            S[ii,jj] = irfft_mpi(Sk[ii,jj],S[ii,jj])
            
            if jj > ii : S[jj,ii] = S[ii,jj]
    
    
    k_filts = 1.0*np.round(np.logspace(1,3,50))
    pi_tot = np.zeros_like(k_filts)
    # k_filts = np.append(k_filts, 0)
    for fidx,k_filt in enumerate(k_filts):
        #* Filter should vanish for l = inf
        l_filt = TWO_PI/k_filt if k_filt != 0 else np.inf
        #? Gaussian filter
        filt_k[:] =  np.exp(-k**2/(2.0*k_filt**2)) if k_filt != 0 else 0.0
        #? Cutoff filter
        # filt_k[:] = np.where(np.round(k)< k_filt, 1.0, 0.0)
        for ii in range(3):
            omg_filt[ii] = irfft_mpi(omgk[ii]*filt_k,omg_filt[ii])
            for jj in range(ii,3):
                S_filt[ii,jj] = irfft_mpi(Sk[ii,jj]*filt_k,S_filt[ii,jj])
                tau_filt[ii,jj] = tau(u[ii],u[jj],filt_k)
                
                if jj> ii :
                    S_filt[jj,ii] = S_filt[ii,jj]
                    tau_filt[jj,ii] = tau_filt[ii,jj]
        
        pi[:] = -np.einsum('ij...,ij...->...',S_filt,tau_filt) 
        pi_s1[:] = -l_filt**2*np.einsum('ij...,jk...,ki...->...',S_filt,S_filt,S_filt)
        pi_omg1[:] = 0.25*l_filt**2*np.einsum('i...,ik...,k...->...',omg_filt,S_filt,omg_filt)
        
        pi_tot[fidx] = comm.allreduce(np.sum(pi),op = MPI.SUM)*dx*dy*dz
        
    if rank ==0: 
        print(f"pi_tot : {pi_tot}")
        print(f"PIk_arr : {PIk_arr}")
        print(f"k_filts : {k_filts}")
    comm.Barrier()

            
    
    
    

    
