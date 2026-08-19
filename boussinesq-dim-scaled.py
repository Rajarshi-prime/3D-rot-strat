""" 
The code forces the dimensional boussinesq equations with constant power input to the waves with wavenunbers between 1 and 4 separately.
"""
#%%
import numpy as np 
from scipy.fft import fft ,  ifft ,  irfft2 ,  rfft2 , irfftn ,  rfftn,   rfft,  irfft,fftfreq
from scipy.linalg import expm
inv = np.linalg.inv
from mpi4py import MPI
from time import time
import pathlib,sys,h5py
curr_path = pathlib.Path(__file__).parent
if int(float(sys.argv[-1])) == 0:
    forcestart = True
    omg_save = False
elif int(float(sys.argv[-1])) == 1: 
    forcestart = False
    omg_save = True
elif int(float(sys.argv[-1])) == 2: 
    forcestart = True
    omg_save = True
else : 
    forcestart = False
    omg_save = False
    

# idx = int(float(sys.argv[-1]))
idx = 1
# forcestart = bool(float(sys.argv[-1]))
#%%

## ---------------MPI things--------------
comm = MPI.COMM_WORLD
num_process =  comm.Get_size()
rank = comm.Get_rank()
#%%
if rank ==0: print(f"Forcestart:{forcestart}, omg_save :{omg_save}")
isforcing = False
viscosity_integrator = "implicit" 
# viscosity_integrator = "explicit" #! Do not use this for hyperviscous simulations or cases with high resolution simulations.
# viscosity_integrator = "exponential"
if viscosity_integrator == "explicit": isexplicit = 1.
else : isexplicit = 0.
## ---------------------------------------
#%%

## ------------- Time steps --------------
N = 64
dt = 0.256/N   #! Such that increasing resolution will decrease the dt
f_corr = float(sys.argv[-2])
# N_bs = [15,200]
N_b = float(sys.argv[-3])
T = 120 if not omg_save else 31.4/f_corr
dt_save = dt if not omg_save else round(2/N_b,int(np.log10(N_b)))
# saveint = int(np.log10(1/dt_save))
saveint = 3
st = round(dt_save/dt)


## ---------------------------------------
#%%

## -------------Defining the grid ---------------
alpha = 20 # Width/Height
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
# lp = int(float(sys.argv[-1])) # Hyperviscosity power
lp = 8 # Hyperviscosity power
nu0 = 0.59 #! Viscosity for N = 1
# m = 1.5 #! Desired kmax*eta
# nu = nu0*(3*m/(N*2**0.5))**(2*(lp - 1/3))  #? scaling with resolution. For 512, nu = 0.002 #! Need to add scaling for hyperviscosity
# m = [10,20,5,1,50,100,1000][int(float(sys.argv[-1]))] # Dissipation strength at the highest kmax. 
m = 1000 # Dissipation strength at the highest kmax. 
nu = m/((2**0.5*N)//3)**(2*lp) # Because boussinesq does not follow Kolmogorov scaling.

re = np.inf if nu==0 else 1/nu

fbyN = f_corr/N_b if N_b != 0 else 0.0
Nbyf = N_b/f_corr if f_corr !=0 else np.inf

#%%
# param = dict()
# param["nu"] = nu
# param["hyperviscous"] = lp
# param["Initial energy"] = einit
# param["Gridsize"] = N
# param["Processes"] = num_process
# param["Final_time"] = T
# param["time_step"] = dt
# param["interval of saving indices"] = st

## ---------------------------------
#%%

#%%


if nu!= 0: savePath = pathlib.Path(f"./data/bsnq_scaled/f_{f_corr:.1f}_Nb_{N_b:.1f}/tide_test_forced_{isforcing}/N_{N}_Re_{re:.1f}")
else: savePath = pathlib.Path(f"./data/bsnq_scaled/tide_test_forced_{isforcing}/N_{N}_Re_inf")

if rank == 0:
    print(savePath)
    try: savePath.mkdir(parents=True,  exist_ok=True)
    except FileExistsError: pass

## ------------Useful Operators-------------------
#%%

lap = -1.0*(kx**2 + ky**2 + kz**2 )
lap_press = -1.0*(kx**2 + ky**2 + alpha**2 * kz**2 ) #! Laplacian operator modified with alpha
k_act = (-lap_press)**0.5
k_act_max = comm.allreduce(k_act.max(),op = MPI.MAX)
shells_act = np.arange(-0.5,int(k_act_max)+1, 1.)
shells_act[0] = 0.0
k = (-lap)**0.5
kint = np.clip(np.round(k,0).astype(int),None,N//2)
kh = (kx**2 + ky**2)**0.5
# cond  = (kh>=shell_no[0])*(kh<=shell_no[-1])*(kz > 1)*(kz<6)
cond = (np.abs(np.abs(kx) -1.0) < 0.5)*(np.abs(np.abs(ky) -1.0) < 0.5)*(np.abs(np.abs(kz) - 1.0) < 0.5)

# dealias = kint<=N/3 #! Spherical dealiasing
# dealias = (abs(kx)<N//3)*(abs(ky)<N//3)*(abs(kz)<N//3) #! Cubic 2/3 dealiasing
# dealias = np.exp(-(N//10)*((1.0*kx/N)**(N//10) +(1.0*kz/N)**(N//10) + (1.0*kz/N)**(N//10))) #! Exponential dealiasing a la. Sanjay for 360^3.
dealias = kint < 2**0.5*N/3 #! phase shifted dealiasing
phase_k = np.exp(1j*(kx*dx/2 + ky*dy/2 + kz*dz/2))*dealias
conjphase_k = np.conjugate(phase_k)*dealias

invlap = dealias/np.where(lap == 0, np.inf,  lap)
invlap_press = dealias/np.where(lap_press == 0, np.inf,  lap_press)
lapwv = -1.0*(kx**2 + ky**2 + (fbyN)**2*alpha**2*kz**2 )
invlapwv = dealias/np.where(lapwv == 0, np.inf,lapwv)

# Hyperviscous operator
vis = nu*(-lap)**(lp) ## This is in Fourier Space

normalize = np.where((kz== 0) + (kz == N//2) , 1/(N**6/TWO_PI**3),2/(N**6/TWO_PI**3))
shells = np.arange(-0.5,Nf, 1.)
shells[0] = 0.

shell_no = np.sort(np.unique(np.concatenate(comm.allgather(np.unique(kint[cond].ravel())))))
# shell_no = np.arange(4,4+nshells) # the shells to be forced 
nshells = len(shell_no) # Number of consecutive shells to be forced
#%%

#----  Kolmogorov length scale - \eta \epsilon etc...---------

f0 = (nu0)**3*TWO_PI**3/ nshells #! Total power input at each shells
einit = 0.0*TWO_PI**3 if f_corr> 0 or N_b > 0 else 1*TWO_PI**3# Initial energy

# f0 = 0.02 /(N_b**2)*nshells#! Total power input at each shells
if rank ==0 : print(f" Power input  : {nshells*f0} \n Viscosity : {nu}, Re : {re},dt : {dt}")
# def create_G_half(G_half ,f = f_corr, Nb = N_b, alpha = alpha,invlap_press = invlap_press,dealias = dealias): #!old one
#     # f = 1.0/np.where(kz == 0.0, np.inf, kz)/alpha
#     # Nb = 1.0/np.where(kh == 0.0, np.inf, kh)
#     sig = (-(kh**2*Nb**2 +f**2 *kz**2*alpha**2 )/np.where(lap_press == 0, np.inf,  lap_press))**0.5 

#     Delta_t = -0.5*dt
#     G_half += ((kh < 0.5)*(kz >0.5))[None,None,:]*np.array([
#         [np.cos(Delta_t*f)*np.ones_like(k), - np.sin(Delta_t*f)*np.ones_like(k),0*np.ones_like(k),0*np.ones_like(k)],
#         [np.sin(Delta_t*f)*np.ones_like(k),  np.cos(Delta_t*f)*np.ones_like(k),0*np.ones_like(k),0*np.ones_like(k)],
#         [0*np.ones_like(k),0*np.ones_like(k),1*np.ones_like(k),0*np.ones_like(k)],
#         [0*np.ones_like(k),0*np.ones_like(k),Nb*Delta_t/alpha*np.ones_like(k), 1*np.ones_like(k)]
        
#     ]) #! The kh = 0 mode
    
#     invkh = 1/np.where(kh < 0.5, np.inf,kh)
#     invkz = 1/np.where(kz < 0.5, np.inf,kz)
    
#     G_half += ((kh > 0.5)*(kz <0.5))[None,None,:]*np.array([
#         [1- Delta_t*f*kx*ky*invkh**2,-Delta_t*f*ky**2*invkh**2,0*np.ones_like(k),0*np.ones_like(k)],
#         [Delta_t*f*kx**2*invkh**2, 1+ Delta_t*f*kx*ky*invkh**2,0*np.ones_like(k),0*np.ones_like(k)],
#         [0*np.ones_like(k),0*np.ones_like(k),np.cos(Delta_t*Nb)*np.ones_like(k),-alpha*np.sin(Delta_t*Nb)*np.ones_like(k)],
#         [0*np.ones_like(k),0*np.ones_like(k),np.sin(Delta_t*Nb)/alpha*np.ones_like(k),np.cos(Delta_t*Nb)*np.ones_like(k)]
        
        
#     ]) #! The kz = 0 mode
    
    
    
#     G_half += ((kh < 0.5)*(kz <0.5))[None,None,:]*np.array([
#         [np.cos(Delta_t*f)*np.ones_like(k), - np.sin(Delta_t*f)*np.ones_like(k),0*np.ones_like(k),0*np.ones_like(k)],
#         [np.sin(Delta_t*f)*np.ones_like(k),   np.cos(Delta_t*f)*np.ones_like(k),0*np.ones_like(k),0*np.ones_like(k)],
#         [0*np.ones_like(k),0*np.ones_like(k),np.cos(Delta_t*Nb)*np.ones_like(k),-alpha*np.sin(Delta_t*Nb)*np.ones_like(k)],
#         [0*np.ones_like(k),0*np.ones_like(k),np.sin(Delta_t*Nb)/alpha*np.ones_like(k),np.cos(Delta_t*Nb)*np.ones_like(k)]
        
#     ]) #! The kz = 0, kh = 0 mode
    
#     if f == 0.0 and Nb>0.0:
#         if rank ==0: print(f'Creating G_half for zero f and non-zero Nb')
        
#         kappa = kh*(-invlap_press)**0.5
#         G_half += ((kh>0.5)*(kz>0.5))[None,None,...]*np.array([
#             [1*np.ones_like(k),0*np.ones_like(k),kx*kz*invkh**2*(1- np.cos(Delta_t*Nb*kappa)),alpha*kx*kz*kappa*invkh**2*np.sin(Delta_t*Nb*kappa)],
#             [0*np.ones_like(k),1*np.ones_like(k),ky*kz*invkh**2*(1- np.cos(Delta_t*Nb*kappa)),alpha*ky*kz*kappa*invkh**2*np.sin(Delta_t*Nb*kappa)],
#             [0*np.ones_like(k),0*np.ones_like(k),np.cos(Delta_t*Nb*kappa),-alpha*kappa*np.sin(Delta_t*Nb*kappa)],
#             [0*np.ones_like(k),0*np.ones_like(k),np.sin(Delta_t*Nb*kappa)/(alpha*np.where(kappa ==0.0, np.inf,kappa)),np.cos(Delta_t*Nb*kappa)]
#         ]) 
#         del kappa
        
#     elif Nb == 0.0 and f>0.0:
#         if rank ==0: print(f'Creating G_half for non-zero f and zero Nb')
        
#         gamma = alpha*kz*(-invlap_press)**0.5
#         G_half += ((kh>0.5)*(kz>0.5))[None,None,...]*np.array([
#             [np.cos(gamma*Delta_t*f) - kx*ky*gamma*(invkz/alpha)**2*np.sin(gamma*Delta_t*f),-(ky**2 + alpha**2*kz**2)*gamma*(invkz/alpha)**2*np.sin(gamma*Delta_t*f),0*np.ones_like(k),0*np.ones_like(k)],
#             [(kx**2 + alpha**2*kz**2)*gamma*(invkz/alpha)**2*np.sin(gamma*Delta_t*f),np.cos(gamma*Delta_t*f) + kx*ky*gamma*(invkz/alpha)**2*np.sin(gamma*Delta_t*f),0*np.ones_like(k),0*np.ones_like(k)],
#             [(kx*(1-np.cos(gamma*Delta_t*f)) - gamma*ky*np.sin(gamma*Delta_t*f))*invkz,(gamma*kx*np.sin(gamma*Delta_t*f) + ky*(1- np.cos(gamma*Delta_t*f)))*invkz,0*np.ones_like(k),0*np.ones_like(k)],
#             [0*np.ones_like(k),0*np.ones_like(k),0*np.ones_like(k),1*np.ones_like(k)]
#         ])
#         del gamma
        
#     elif f> 0.0 and Nb > 0.0:
#         if rank ==0: print(f'Creating G_half for non-zero f and Nb')
#         denom = np.where(f**2 * kx**2 + sig**2*ky**2 == 0.0,np.inf, f**2 * kx**2 + sig**2*ky**2)
#         Sm = np.array([
#             [-ky*Nb*invkz/(alpha*f),kx*Nb*invkz/(alpha*f**2),alpha*kz*(f*ky + 1j*kx*sig)*invkh**2/Nb, alpha*kz*(f*ky - 1j*kx*sig)*invkh**2/Nb],
#             [kx*Nb*invkz/(alpha*f),ky*N*invkz/(alpha*f**2),-alpha*kz*(f**2*(kx**2 + alpha**2 *kz**2) + ky**2*Nb**2)*(-invlap_press)*(f*kx - 1j*ky*sig)/(Nb*denom),-alpha*kz*(f**2*(kx**2 + alpha**2 *kz**2) + ky**2*Nb**2)*(-invlap_press)*(f*kx + 1j*ky*sig)/(Nb*denom)],
#             [0*np.ones_like(k),alpha/Nb*np.ones_like(k),-1j*alpha*sig/Nb,1j*alpha*sig/Nb],
#             [1*np.ones_like(k),0*np.ones_like(k),1*np.ones_like(k),1*np.ones_like(k)]
#         ]) +((kh<0.5)+(kz<0.5))[None, None,:]*np.identity((4))[...,None,None,None]
        
#         # det = np.linalg.det(np.moveaxis(Sm,[0,1,2,3,4],[3,4,0,1,2]))
#         # print((det ==0.0).sum(),((kh<0.5)+(kz<0.5)).sum())
#         # raise SystemExit
#         G_half += ((kh>0.5)*(kz>0.5))[None,None,...]*np.einsum('ij...,jk...->ik...',np.einsum('ij...,jk...->ik...',Sm,np.array([
#             [1*np.ones_like(k),Delta_t*np.ones_like(k),0*np.ones_like(k),0*np.ones_like(k)],
#             [0*np.ones_like(k),1*np.ones_like(k),0*np.ones_like(k),0*np.ones_like(k)],
#             [0*np.ones_like(k),0*np.ones_like(k),np.exp(-1j*sig*Delta_t),0*np.ones_like(k)],
#             [0*np.ones_like(k),0*np.ones_like(k),0*np.ones_like(k),np.exp(1j*sig*Delta_t)]
#         ])),np.moveaxis(inv(np.moveaxis(Sm,[0,1,2,3,4],[3,4,0,1,2])),[0,1,2,3,4],[2,3,4,0,1]))
        
        
#         del Sm,denom

#     else:
#         if rank ==0: print(f'Creating G_half for zero f and Nb')
        
#         G_half += (np.identity((4))[...,None,None,None])*((kh>0.5)*(kz>0.5))[None,None,...]
    
#     del invkh,invkz,sig 
#     return G_half*dealias[None,None,:]


def create_G_half(G_half ,f = f_corr, Nb = N_b, alpha = alpha,invlap_press = invlap_press,dealias = dealias):
    # f = 1.0/np.where(kz == 0.0, np.inf, kz)/alpha
    # Nb = 1.0/np.where(kh == 0.0, np.inf, kh)
    sig = (-(kh**2*Nb**2 +f**2 *kz**2*alpha**2 )*invlap_press)**0.5 

    Delta_t = -0.5*dt

    Lmat = np.array([
        [-f*kx*ky*(-invlap_press),                    f*(kx**2*(-invlap_press) - 1),           0.0*invlap_press,      alpha*kx*kz*Nb*(-invlap_press)],
        [f*(1-ky**2*(-invlap_press) ),              f*kx*ky*(-invlap_press),                 0.0*invlap_press,      alpha*ky*kz*Nb*(-invlap_press)],
        [-alpha**2*f*ky*kz*(-invlap_press),           alpha**2*f*kx*kz*(-invlap_press),        0*invlap_press,     -alpha*Nb*(kx**2 + ky**2)*(-invlap_press)],
        [0*invlap_press,                             0*invlap_press,                          Nb/alpha + 0*invlap_press, 0*invlap_press]
    ])*((kh>0.5)+(kz>0.5))[None,None,...] + ((kh<0.5)*(kz<0.5))[None,None,...]*np.array([
        [0.*k,                   -f + 0.*k,           0.0*k,      0.*k],
        [f + 0.*k,                    0.*k,           0.0*k,      0.*k],
        [0.0*k,                   0.0*k,           0.0*k,     -alpha*Nb +  0.0*k],
        [0.0*k,                   0.0*k,           Nb/alpha + 0.0*k,     0.0*k]
        
    ])
    invsig = 1.0/np.where(sig ==0.0, np.inf, sig)

    G_half += np.moveaxis(expm(np.moveaxis(Lmat ,[0,1,2,3,4],[3,4,0,1,2])* Delta_t),[0,1,2,3,4],[2,3,4,0,1])      #! The kz \neq 0, kh \neq 0 mode
        
    del Lmat,sig,invsig
    return G_half*dealias[None,None,:]


G_half = 0.0*np.ones((4,4,N,Np,Nf),dtype = np.float64)

G_half = create_G_half(G_half)
comm.Barrier()
# print(np.abs(eigL.imag).min())
# r1,r2,r3 = np.random.randint(0,N),  np.random.randint(0,Np), np.random.randint(0,Nf)
# eigG_half = np.linalg.eigvals(np.moveaxis(G_half,[0,1,2,3,4],[3,4,0,1,2]))
# # print(np.abs(G_half.imag).max(),np.max(np.abs(eigG_half)**2),np.min(np.abs(eigG_half)**2)) 
# print(np.abs(G_half.imag).max(),eigG_half[r1,r2,r3],(r1,r2,r3),np.max(np.abs(eigG_half)**2),np.min(np.abs(eigG_half)**2)) 


# error = np.argmax(np.abs(4 - np.sum(np.abs(eigG_half)**2,axis =-1)))
# print(f"K max error: {kx.ravel()[error], ky.ravel()[error],kz.ravel()[error]}")
# raise SystemExit
# check_G = np.einsum('ij...,jk...-> ik...',G_half,G_half)
# maxerror = comm.allreduce(np.abs(G).max(),op = MPI.MAX)
# if rank ==0: 
#     print(f"Error in G {maxerror}")
# raise SystemExit
## -------------------------------------------------


## -------------zeros arrays -----------------------
u  = np.zeros((3, Np, N, N), dtype= np.float64)
b = np.zeros_like(u[0])
b1 = b.copy()
b2 = b.copy()
b3 = b.copy()
omg= np.zeros((3, Np, N, N), dtype= np.float64)
temp_4 = np.zeros((4,N,Np,Nf),dtype = np.complex128)

uk = np.zeros((3, N, Np, Nf), dtype= np.complex128)
uk_w = uk.copy()
uk_v = uk.copy()
pk = uk[0].copy()
pv = uk[0].copy()
bk = pk.copy()
theta = pk.copy()
sig = pk.copy()
denom1 = pk.copy()
denom2 = pk.copy()
bk_w = bk.copy()
bk_v = bk.copy()

ek = np.zeros_like(pk, dtype = np.float64)
Pik = np.zeros_like(pk, dtype = np.float64)
ek_arr = np.zeros(Nf)
ek_act = np.zeros(shells_act.size -1)
Pik_arr = np.zeros(Nf)
ekh_arr = np.zeros(Nf)
Pikh_arr = np.zeros(Nf)
ekz_arr = np.zeros(Nf)
Pikz_arr = np.zeros(Nf)
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

temp_k2d = np.zeros((N,Np),dtype = np.complex128)
temp_2d = np.zeros((Np,N),dtype = np.float64)

arr_temp_k = np.zeros((N, Np, N),dtype= np.float64)
arr_temp_fr = np.zeros((Np, N, Nf), dtype= np.complex128)      
arr_temp_fr_2d = np.zeros((Np, N), dtype= np.complex128)      
arr_temp_ifr = np.zeros((N, Np, Nf), dtype= np.complex128)      
arr_temp_ifr_2d = np.zeros((N, Np), dtype= np.complex128)      
arr_mpi = np.zeros((num_process,  Np,  Np, Nf), dtype= np.complex128)
arr_mpi_2d = np.zeros((num_process,  Np,  Np), dtype= np.complex128)
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


def fft_mpi_xy(u2d,fu2d):
    arr_temp_fr_2d[:] = fft(u2d,axis = -1)
    arr_mpi_2d[:] = np.swapaxes(np.reshape(arr_temp_fr_2d, (Np,  num_process,  Np)), 0, 1)
    comm.Alltoall([arr_mpi_2d,  MPI.DOUBLE_COMPLEX], [fu2d,  MPI.DOUBLE_COMPLEX])
    fu2d[:] = fft(fu2d, axis = 0)
    return fu2d

def ifft_mpi_xy(fu2d, u2d):
    arr_temp_ifr_2d[:] = ifft(fu2d,  axis = 0)
    comm.Alltoall([arr_temp_ifr_2d,  MPI.DOUBLE_COMPLEX], [arr_mpi_2d, MPI.DOUBLE_COMPLEX])
    arr_temp_fr_2d[:] = np.reshape(np.swapaxes(arr_mpi_2d,  0, 1), (Np,  N))
    u2d[:] = ifft(arr_temp_fr_2d, axis = -1).real
    return u2d    

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

def e3d_to_e1d(x,k = k,shells = shells): #1 Based on whether k is 2D or 3D, it will bin the data accordingly. 
    return np.histogram(k.ravel(),bins = shells,weights=x.ravel())[0] 

def ensure_reality(uk):
    temp_2d[:] = ifft_mpi_xy(uk[...,0],temp_2d)
    temp_k2d[:] = fft_mpi_xy(temp_2d,temp_k2d)
    uk[...,0] = temp_k2d.copy()
    return uk
def ensure_div_free(fk):
    pk[:] = invlap  * (kx*fk[0] + ky*fk[1] + kz*fk[2])*dealias
    
    fk[0] +=  kx*pk
    fk[1] +=  ky*pk
    fk[2] +=  kz*pk
    
    return fk*dealias[None,...]
 
def vortex(uk,bk,alpha = alpha): 
    global uk_v,bk_v
    """
    Projects the velocity and buoyancy in the vortical modes. 
    """
    pv[:] = 1j*(kx*uk[1] - ky*uk[0] +    alpha*kz*bk*(fbyN))
    uk_v[0] = -1j*ky*pv*invlapwv
    uk_v[1] = 1j*kx*pv*invlapwv
    uk_v[2] = 0. + 0.j
    bk_v = 1j*alpha*kz*pv*invlapwv*(fbyN)
    
    return uk_v, bk_v
    
def u_dot_grad(u,f):
    return u[0]*diff_x(f,b1) + u[1]*diff_y(f,b2) + u[2]*diff_z(f,b3)

def inner_product_k(a1k,a2k, b1k,b2k,alpha = alpha):
    """Calculates quantities like energy in spectral space with factor of alpha in the vertical componennt"""
    val = comm.allreduce(
        np.sum(
            normalize*(np.conjugate(a1k[0])*b1k[0]+np.conjugate(a1k[1])*b1k[1]+np.conjugate(a1k[2])*b1k[2]/alpha**2 + np.conjugate(a2k)*b2k).real
        ) 
    , op = MPI.SUM)
    return val
def inner_product(a1,a2,b1,b2,alpha = alpha):
    """Calculates quantities like energy in position space with factor of alpha in the vertical componennt"""
    val = comm.allreduce(
        np.sum(
            a1[0]*b1[0]+a1[1]*b1[1]+a1[2]*b1[2]/alpha**2 + a2*b2
        ) 
    , op = MPI.SUM)
    return val


if N_b == 0.0 and f_corr == 0.0:
    if rank ==0: print(f"Initializing linear forcing")
    def forcing(uk,bk):
        """
        Calculates the net dissipation of the flow and injects that amount into larges scales of the horizontal flow
        """
        global fk, fkb, factor3d, factor, ek_arr,kint
        # uk_v[:],bk_v[:] = vortex(uk,bk)
        # uk_w[:],bk_w[:] = uk - uk_v, bk - bk_v
        
        
        # ek[:] = 0.5*(np.abs(uk_w[0])**2 + np.abs(uk_w[1])**2 + np.abs(uk_w[2])**2 + np.abs(bk_w)**2)*dealias*normalize*(kh>0.5)*(kz > 0.5) #! This is the 3D ek array of waves
        ek[:] = 0.5*(np.abs(uk[0])**2 + np.abs(uk[1])**2 + np.abs(uk[2])**2/alpha**2 + np.abs(bk)**2)*dealias*normalize #! This is the 3D ek array of waves
        
        # ek_arr[:] = comm.allreduce(e3d_to_e1d(ek*cond),op = MPI.SUM) #! This is the shell-summed ek array.
        #? Only if you are forcing one or two shells 
        ek_arr[:] = 0.0
        for shell in shell_no:
            ek_arr[shell] = comm.allreduce(np.sum(ek*cond*(kint>= shell-0.5)*(kint< shell +0.5)),op = MPI.SUM)
        ek_arr[:] = np.where(np.abs(ek_arr)< 1e-10,np.inf, ek_arr)
        """Change forcing starts here"""
        # Const Power Input
        factor[:] = 0.0
        factor[shell_no] = f0/(2*ek_arr[shell_no])
        factor3d[:] = factor[kint]*dealias*cond
        
        
        # # Constant shell energy
        # factor[:] = np.tanh(np.where(np.abs(ek_arr0) < 1e-10, 0, (ek_arr0/ek_arr)**0.5 - 1)) #! The factors for each shell is calculated
        # factor3d[:] = factor[kint]

        
        fk[0] = factor3d*uk[0]
        fk[1] = factor3d*uk[1]
        fk[2] = factor3d*uk[2]
        fkb[:] = factor3d*bk

        """Change forcing ends here here"""
        
        pk[:] = invlap  * (kx*fk[0] + ky*fk[1] + kz*fk[2])*dealias
        
        fk[0] = fk[0] + kx*pk
        fk[1] = fk[1] + ky*pk
        fk[2] = fk[2] + kz*pk
        
        return fk*isforcing*dealias, fkb*isforcing*dealias
else: 
    if rank ==0 : print(f"Initializing random forcing")
    def forcing(tt,uk,bk,f0 = f0*nshells,cond = cond ,h = dt,theta= theta,pk = pk,f1uk = f1k,f1bk = f1bk,f2uk = f2k, f2bk = f2bk,fk = fk, fkb = fkb,denom1 = denom1, denom2 = denom2,alpha = alpha):
        # ------------------- negative frequency ------------------- #
        theta[:] = np.random.uniform(0,TWO_PI,(N,Np,Nf))
        pk[:] = ensure_reality(np.exp(1j*theta)*cond*N**3*dealias)
        
        
        sig[:] = -(-invlap_press*(kh**2*N_b**2 +f_corr**2 *kz**2*alpha**2 ))**0.5
        denom1[:] = dealias/np.where(sig**2 - f_corr**2 == 0., np.inf, sig**2 - f_corr**2)
        denom2[:] = dealias/np.where(N_b**2 - sig**2 == 0., np.inf, N_b**2 - sig**2)
        
        f1uk[0,:] = (pk* (1j*ky*f_corr - kx*sig)*denom1)*np.exp(1j*sig*tt)
        f1uk[1,:] = -(pk* (1j*kx + ky*sig)*denom1)*np.exp(1j*sig*tt)
        f1uk[2,:] = (alpha**2*kz*sig*pk*denom2)*np.exp(1j*sig*tt)
        f1bk[:] = (1j*alpha*kz*N_b*pk*denom2)*np.exp(1j*sig*tt)


        
        # neg_corr = comm.allreduce(np.sum(normalize*(np.einsum('i...,i...->...',np.conjugate(uk),f1uk) + np.conjugate(bk)*f1bk).real), op = MPI.SUM)
        neg_corr = inner_product_k(uk,bk,f1uk,f1bk)
        # ---------------------------------------------------------- #
        # ------------------- positive frequency ------------------- #
        theta[:] = np.random.uniform(0,TWO_PI,(N,Np,Nf))
        pk[:] = ensure_reality(np.exp(1j*theta)*cond*N**3*dealias)

        
        sig[:] = (-invlap_press*(kh**2*N_b**2 +f_corr**2 *kz**2*alpha**2 ))**0.5
        denom1[:] = dealias/np.where(sig**2 - f_corr**2 == 0., np.inf, sig**2 - f_corr**2)
        denom2[:] = dealias/np.where(N_b**2 - sig**2 == 0., np.inf, N_b**2 - sig**2)
        
        f2uk[0,:] = (pk* (1j*ky*f_corr - kx*sig)*denom1)*np.exp(1j*sig*tt)
        f2uk[1,:] = -(pk* (1j*kx + ky*sig)*denom1)*np.exp(1j*sig*tt)
        f2uk[2,:] = (alpha**2*kz*sig*pk*denom2)*np.exp(1j*sig*tt)
        f2bk[:] = (1j*alpha*kz*N_b*pk*denom2)*np.exp(1j*sig*tt)
        



        
        # pos_corr = comm.allreduce(np.sum(normalize*(np.einsum('i...,i...->...',np.conjugate(uk),f2uk) + np.conjugate(bk)*f2bk).real), op = MPI.SUM)
        pos_corr = inner_product_k(uk,bk,f2uk,f2bk)
        # ---------------------------------------------------------- #
        # norm = comm.allreduce(np.sum(normalize*(np.einsum('i...,i...->...',np.conjugate(f1uk),f1uk) + np.conjugate(f1bk)*f1bk).real),op =MPI.SUM)**0.5
        norm = inner_product_k(f1uk,f1bk,f1uk,f1bk)**0.5

        if np.abs(neg_corr) > 1e-10*(2*f0*h)**0.5*norm: 
            beta = -pos_corr/neg_corr
            fk[:] = beta*f1uk  + f2uk
            fkb[:] = beta*f1bk  + f2bk

        else: 
            beta = 1.0
            fk[:] = beta*f1uk  
            fkb[:] = beta*f1bk 
            
        # aa = (2.0*f0/comm.allreduce(np.sum(normalize*(np.einsum('i...,i...->...',np.conjugate(fk),fk) + np.conjugate(fkb)*fkb).real),op = MPI.SUM)/h)**0.5
        aa = (2.0*f0/inner_product_k(fk,fkb, fk,fkb)/h)**0.5

        
        
        return aa*fk*isforcing*dealias, aa*fkb*isforcing*dealias

def ABC_flow_rhs(a,b,c):
    factor = a * (alpha**2 - 1) / (alpha**2 + 1)

    result = 1.0*factor*np.array([
        - b * np.cos(x) * np.cos(z),
        c * np.sin(y) * np.sin(z),
        (c * np.cos(y) * np.cos(z) - b * np.sin(x) * np.sin(z))
    ])

    return result

def ABC_flow(a,b,c):
    Fx = c*np.cos(y) + a*np.sin(z)
    Fy = a*np.cos(z) + b*np.sin(x)
    Fz = b*np.cos(x) + c*np.sin(y)
    return 1.0*np.array([Fx,Fy,Fz])

def G_prod(uk,bk,G_half = G_half):
    return np.einsum('ij...,jk...,k...->i...',G_half,G_half,np.concatenate((uk,bk[None,...]),axis = 0))
    
def G_half_prod(uk,bk,G_half = G_half):
    return np.einsum('ij...,j...->i...',G_half,np.concatenate((uk,bk[None,...]),axis = 0))


def RHS(uk, bk,uk_t,bk_t,visc = 1,forc = 1,rhsuk = rhsuk, rhsvk = rhsvk, rhswk = rhswk, rhsbk = rhsbk,alpha = alpha):
    ## The RHS terms of u, v and w excluding the forcing and the hypervisocsity term 
    if N_b ==0.0 and f_corr== 0.0 and forc == 1:
        fk[:],fkb[:] = forcing(uk, bk)
    else: forc = 0
    
    u[0] = irfft_mpi(uk[0]*dealias, u[0])
    u[1] = irfft_mpi(uk[1]*dealias, u[1])
    u[2] = irfft_mpi(uk[2]*dealias, u[2])
    b[:] = irfft_mpi(bk*dealias, b)
    
    # omg[0] = irfft_mpi(1j*(ky*uk[2] - kz*uk[1])*dealias,omg[0])
    # omg[1] = irfft_mpi(1j*(kz*uk[0] - kx*uk[2])*dealias,omg[1])
    # omg[2] = irfft_mpi(1j*(kx*uk[1] - ky*uk[0])*dealias,omg[2])
    
    rhsu[:] = -(u_dot_grad(u,u[0])) 
    rhsv[:] = -(u_dot_grad(u,u[1])) 
    rhsw[:] = -(u_dot_grad(u,u[2])) 
    rhsb[:] = - (u_dot_grad(u,b)) 
    

    
    rhsuk[:]  = 0.5*(rfft_mpi(rhsu, rhsuk))*dealias 
    rhsvk[:]  = 0.5*(rfft_mpi(rhsv, rhsvk))*dealias 
    rhswk[:]  = 0.5*(rfft_mpi(rhsw, rhswk))*dealias 
    rhsbk[:] = 0.5*(rfft_mpi(rhsb,rhsbk))*dealias
    
    
    
    u[0] = irfft_mpi(uk[0]*phase_k, u[0])
    u[1] = irfft_mpi(uk[1]*phase_k, u[1])
    u[2] = irfft_mpi(uk[2]*phase_k, u[2])
    b[:] = irfft_mpi(bk*phase_k, b)
    
    # omg[0] = irfft_mpi(1j*(ky*uk[2] - kz*uk[1])*phase_k,omg[0])
    # omg[1] = irfft_mpi(1j*(kz*uk[0] - kx*uk[2])*phase_k,omg[1])
    # omg[2] = irfft_mpi(1j*(kx*uk[1] - ky*uk[0])*phase_k,omg[2])
    
    # rhsu[:] = (omg[2]*u[1] - omg[1]*u[2]) 
    # rhsv[:] = (omg[0]*u[2] - omg[2]*u[0]) 
    # rhsw[:] = (omg[1]*u[0] - omg[0]*u[1]) 
    rhsu[:] = -(u_dot_grad(u,u[0])) 
    rhsv[:] = -(u_dot_grad(u,u[1])) 
    rhsw[:] = -(u_dot_grad(u,u[2])) 
    rhsb[:] = - (u_dot_grad(u,b)) 
    

    
    rhsuk  += (0.5*rfft_mpi(rhsu, pk)*conjphase_k )*dealias 
    rhsvk  += (0.5*rfft_mpi(rhsv, pk)*conjphase_k )*dealias 
    rhswk  += (0.5*rfft_mpi(rhsw, pk)*conjphase_k  )*dealias 
    rhsbk += (0.5*rfft_mpi(rhsb,pk)*conjphase_k)*dealias
    
    # rhsuk  += (0.5*rfft_mpi(rhsu, pk)*conjphase_k + f_corr*uk[1] )*dealias 
    # rhsvk  += (0.5*rfft_mpi(rhsv, pk)*conjphase_k - f_corr*uk[0] )*dealias 
    # rhswk  += (0.5*rfft_mpi(rhsw, pk)*conjphase_k + alpha*bk*N_b )*dealias 
    # rhsbk += (0.5*rfft_mpi(rhsb,pk)*conjphase_k - N_b*uk[2]/alpha)*dealias
    
    ## The pressure term
    pk[:] = 1j*invlap_press  * (kx*rhsuk + ky*rhsvk + kz*rhswk)
    
    

    ## The RHS term with the pressure   
    uk_t[0] = rhsuk - 1j*kx*pk - nu*((-lap)**lp)*uk[0]*isexplicit * visc + forc*fk[0]*dealias
    uk_t[1] = rhsvk - 1j*ky*pk - nu*((-lap)**lp)*uk[1]*isexplicit * visc + forc*fk[1]*dealias
    uk_t[2] = rhswk - 1j*alpha**2*kz*pk - nu*((-lap)**lp)*uk[2]*isexplicit * visc + forc*fk[2]*dealias
    bk_t[:] = rhsbk - nu*((-lap)**lp)*bk*isexplicit * visc + forc*fkb*dealias

        
    return uk_t, bk_t




## -----------------------------------------------------------


## ---------------- Saving data + energy + Showing total energy ---------------------
def load_sanjay():
    load_num_slabs = 128
    data_per_rank = N//load_num_slabs
    rank_data = range(rank*Np,(rank + 1)*Np) # The rank contains these slices 
    slab_old = np.inf
    for lidx,j in enumerate(rank_data):
        slab = j//data_per_rank
        idx = j%data_per_rank
        if slab_old != slab:  
            Field = np.load(f"/home/sanjay.cp/Wave_Kinematics/Bous_Dyn_eps_1/T_19900/U_{slab}.npz")
        slab_old = slab
        u[0,lidx] = Field['ufull'][idx]
        u[1,lidx] = Field['vfull'][idx]
        u[2,lidx] = Field['wfull'][idx]*(fbyN)
        b[lidx] = Field['bfull'][idx]
    return u,b


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
    


def add_group(file,group_name):
    if group_name not in file:
        file.create_group(group_name)
def add_dataset(file, dataset_name):
    if dataset_name not in file:
        f["/energy_timeseries/"].create_dataset("tot_energy" ,data = np.array([np.sum(ek_arr)]),maxshape = (None,),chunks = True)


def save(i,uk,bk,alpha = alpha,saveint = saveint):
    # return None
    global ek,k1u,k1b,ek_arr,Pik,Pik_arr
    
    k1u[:],k1b[:] = RHS(uk,bk, k1u,k1b,visc = 0,forc = 0)
    
    ek[:] = 0.5*(np.abs(uk[0])**2 + np.abs(uk[1])**2 + np.abs(uk[2])**2 /alpha**2+ np.abs(bk)**2)*normalize #! This is the 3D ek array
    ek_arr[:] = 0.0
    ek_arr[:] = comm.allreduce(e3d_to_e1d(ek),op = MPI.SUM) #! This is the shell-summed ek array.
    ek_act[:] = comm.allreduce(e3d_to_e1d(ek,k_act,shells_act),op = MPI.SUM)
    ekh_arr[:] = comm.allreduce(e3d_to_e1d(ek,kh),op = MPI.SUM) #! This is the shell-summed ek array.
    ekz_arr[:] = comm.allreduce(e3d_to_e1d(ek,kh),op = MPI.SUM) #! This is the shell-summed ek array.
    Pik[:] = np.real(np.conjugate(uk[0])*k1u[0]+np.conjugate(uk[1])*k1u[1]+ np.conjugate(uk[2])*k1u[2]/alpha**2 + np.conjugate(bk)*k1b)*dealias*normalize
    
    Pik_arr[:] = comm.allreduce(e3d_to_e1d(Pik),op = MPI.SUM)
    Pik_arr[:] = np.cumsum(Pik_arr[::-1])[::-1]
    
    Pik_act = comm.allreduce(e3d_to_e1d(Pik,k_act,shells_act),op = MPI.SUM)
    Pik_act[:] = np.cumsum(Pik_act[::-1])[::-1]
    
    Pikh_arr[:] = comm.allreduce(e3d_to_e1d(Pik ,kh),op = MPI.SUM)
    Pikh_arr[:] = np.cumsum(Pikh_arr[::-1])[::-1]
    
    Pikz_arr[:] = comm.allreduce(e3d_to_e1d(Pik,kz),op = MPI.SUM)
    Pikz_arr[:] = np.cumsum(Pikz_arr[::-1])[::-1]
    
    
    
    uk_v[:],bk_v[:] = vortex(uk,bk)
    uk_w[:],bk_w[:] = uk - uk_v, bk -bk_v

    ek_v = 0.5*(np.abs(uk_v[0])**2 + np.abs(uk_v[1])**2 + np.abs(uk_v[2])**2/alpha**2 + np.abs(bk_v)**2)*normalize
    ek_w = 0.5*(np.abs(uk_w[0])**2 + np.abs(uk_w[1])**2 + np.abs(uk_w[2])**2/alpha**2 + np.abs(bk_w)**2)*normalize

    ek_v_val = comm.allreduce(ek_v.sum(),op = MPI.SUM)
    ek_w_val = comm.allreduce(ek_w.sum(),op = MPI.SUM)

    u[0] = irfft_mpi(uk[0], u[0])
    u[1] = irfft_mpi(uk[1], u[1])
    u[2] = irfft_mpi(uk[2], u[2])
    b[:] = irfft_mpi(bk, b)
    omg[2] = irfft_mpi(1j*(kx*uk[1] - ky*uk[0])*dealias,omg[2])
    zeta_rms = (comm.allreduce(np.sum(omg[2]**2), op =MPI.SUM )/N**3)**0.5
    # ----------- ----------------------------
    #                 Saving the data (field)
    # ----------- ----------------------------
    # new_dir = savePath/f"time_{t[i]:.1f}"
    new_dir = savePath/f"time_{t[i]:.{saveint}f}" if omg_save else savePath/f"last"
    try: new_dir.mkdir(parents=True,  exist_ok=True)
    except FileExistsError: pass
    comm.Barrier()
    # np.savez_compressed(f"{new_dir}/Fields_{rank}.npz",uhat = uk)
    # np.savez_compressed(f"{new_dir}/Fields_{rank}",u = u[0],v = u[1],w = u[2])
    
    np.savez_compressed(f"{new_dir}/Fields_k_{rank}",uk = uk[0],vk = uk[1],wk = uk[2],bk = bk)
    if rank ==0: 
        if omg_save: new_dir = new_dir.parent
        with h5py.File(new_dir/'spectra_flux.hdf5', 'a') as f:
            if 'Energy_Spectra' not in f:
                f.create_group('Energy_Spectra')
            if 'Flux_Spectra' not in f:
                f.create_group('Flux_Spectra')
            if 'unscaled_Energy_Spectra' not in f:
                f.create_group('unscaled_Energy_Spectra')
            if 'unscaled_Flux_Spectra' not in f:
                f.create_group('unscaled_Flux_Spectra')
            if 'Hor_Energy_Spectra' not in f:
                f.create_group('Hor_Energy_Spectra')
            if 'Hor_Flux_Spectra' not in f:
                f.create_group('Hor_Flux_Spectra')
            if 'Ver_Energy_Spectra' not in f:
                f.create_group('Ver_Energy_Spectra')
            if 'Ver_Flux_Spectra' not in f:
                f.create_group('Ver_Flux_Spectra')
            if 'energy_timeseries' not in f:
                f.create_group('energy_timeseries')
            if "tot_energy" not in f["/energy_timeseries/"]:
                f["/energy_timeseries/"].create_dataset("tot_energy" ,data = np.array([np.sum(ek_arr)]),maxshape = (None,),chunks = True)
            else:
                f["/energy_timeseries/tot_energy"].resize((f["/energy_timeseries/tot_energy"].shape[0] + 1,))
                f["/energy_timeseries/tot_energy"][-1] = np.sum(ek_arr)
            if "bal_energy" not in f["/energy_timeseries/"]:
                f["/energy_timeseries/"].create_dataset("bal_energy",data = np.array([ek_v_val]),maxshape = (None,),chunks = True)
            else:
                f["/energy_timeseries/bal_energy"].resize((f["/energy_timeseries/bal_energy"].shape[0] + 1,))
                f["/energy_timeseries/bal_energy"][-1] = ek_v_val
            if "ubal_energy" not in f["/energy_timeseries/"]:

                f["/energy_timeseries/"].create_dataset("ubal_energy",data = np.array([ek_w_val]),maxshape = (None,),chunks = True)
            else:    
                f["/energy_timeseries/ubal_energy"].resize((f["/energy_timeseries/ubal_energy"].shape[0] + 1,))
                f["/energy_timeseries/ubal_energy"][-1] = ek_w_val
                
            if 'zeta_rms_timeseries' not in f:
                f.create_dataset('zeta_rms_timeseries',data = np.array([zeta_rms]),maxshape = (None,),chunks = True)
            else:                 
                f['zeta_rms_timeseries'].resize((f["zeta_rms_timeseries"].shape[0] + 1,))
                f['zeta_rms_timeseries'][-1] = zeta_rms
                
                
            try: f[f"unscaled_Energy_Spectra/time_{t[i]:.{saveint}f}"][...] = ek_act
            except KeyError: f[f"unscaled_Energy_Spectra/time_{t[i]:.{saveint}f}"] = ek_act
            try : f[f"unscaled_Flux_Spectra/time_{t[i]:.{saveint}f}"][...]=  Pik_act
            except KeyError: f[f"unscaled_Flux_Spectra/time_{t[i]:.{saveint}f}"]=  Pik_act
            
            try: f[f"Energy_Spectra/time_{t[i]:.{saveint}f}"][...] = ek_arr
            except KeyError: f[f"Energy_Spectra/time_{t[i]:.{saveint}f}"] = ek_arr
            try : f[f"Flux_Spectra/time_{t[i]:.{saveint}f}"][...]=  Pik_arr
            except KeyError: f[f"Flux_Spectra/time_{t[i]:.{saveint}f}"]=  Pik_arr
            
            try: f[f"Hor_Energy_Spectra/time_{t[i]:.{saveint}f}"][...] = ekh_arr
            except KeyError: f[f"Hor_Energy_Spectra/time_{t[i]:.{saveint}f}"] = ekh_arr
            try : f[f"Hor_Flux_Spectra/time_{t[i]:.{saveint}f}"][...]=  Pikh_arr
            except KeyError: f[f"Hor_Flux_Spectra/time_{t[i]:.{saveint}f}"]=  Pikh_arr
            
            try: f[f"Ver_Energy_Spectra/time_{t[i]:.{saveint}f}"][...] = ekz_arr
            except KeyError: f[f"Ver_Energy_Spectra/time_{t[i]:.{saveint}f}"] = ekz_arr
            try : f[f"Ver_Flux_Spectra/time_{t[i]:.{saveint}f}"][...]=  Pikz_arr
            except KeyError: f[f"Ver_Flux_Spectra/time_{t[i]:.{saveint}f}"]=  Pikz_arr
            
            
    # np.savez_compressed(f"{new_dir}/Energy_spectrum",ek = ek_arr)
    # np.savez_compressed(f"{new_dir}/Flux_spectrum",Pik = Pik_arr)
    
    
    # u_temp = rfftn(u, axes = (-2,-1))[...,cond_ky, :N//3+1] #! Will only save the values in x k_x and k_y plane for the dealiased mode. 
    # b_temp = rfftn(b,axes = (-2,-1))[...,cond_ky,:N//3 + 1]
    # np.savez_compressed(f"{new_dir}/Fields_cmp_{rank}",u = u_temp[0],v = u_temp[1],w = u_temp[2],b = b_temp)
    # np.savez_compressed(f"{new_dir}/Energy_spectrum",ek = ek_arr)
    # np.savez_compressed(f"{new_dir}/Flux_spectrum",Pik = Pik_arr)
    
    comm.Barrier()
    
    # ----------- ----------------------------
    #          Calculating and printing
    # ----------- ----------------------------
    eng1 = comm.allreduce(np.sum(0.5*(u[0]**2 + u[1]**2 + u[2]**2/alpha**2 + b**2)*dx*dy*dz), op = MPI.SUM)
    eng2 = np.sum(ek_arr)
    divmax = comm.allreduce(np.max(np.abs(diff_x(u[0],  rhsu) + diff_y(u[1],rhsv) + diff_z(u[2],rhsw))),op = MPI.MAX)
    #! Needs to be changed 
    # # dissp = -nu*comm.allreduce(np.sum((kc**(2*lp)*(np.abs(uk[0])**2 + np.abs(uk[1])**2) +sin_to_cos( ks**(2*lp)*(np.abs(uk[2])**2/alph**2 + np.abs(bk)**2)))), op = MPI.SUM)
    if rank == 0:
        print( "#----------------------------","\n",f"Energy at time {t[i]:.{saveint}f} is : {eng1}, {eng2}","\n","#----------------------------")
        print(f"Maximum divergence {divmax}")
        print(f"Vortex energy {ek_v_val},wave energy {ek_w_val}")
        # print( "#----------------------------","\n",f"Total dissipation at time {t[i]} is : {dissp}","\n","#----------------------------")
    return "Done!"    

## -------------------------------------------------    
    
## ------------- Evolving the system ----------------- 
def evolve_and_save(t,  uk,bk,uknew=uknew, bknew = bknew,temp_4 = temp_4,alpha = alpha,dt_save = dt_save): 
    global begin
    h = t[1] - t[0]
    if viscosity_integrator == "implicit": hypervisc= dealias*(1. + h*vis)**(-1)
    else: hypervisc = 1.
    
    # if  viscosity_integrator == "exponential": 
    #     semi_G =  np.exp(-nu*(k**(2*lp))*h)
    #     semi_G_half =  semi_G**0.5
    # else: semi_G = semi_G_half = 1.
    
    t3  = time()
    calc_time = 0
    for i in range(t.size-1):
        calc_time += time() - t3
        if rank == 0:  print(f"step {i} in time {time() - t3}", end= '\r',file = sys.stderr)
        ## ------------- saving the data -------------------- ##
        if abs(np.sin(t[i]/dt_save*PI)) - np.sin(0.5*h/dt_save*PI) < 1e-12:
        #     save_hdf5(i,uk,bk)
            save(i,uk,bk)
        begin = True   
        ## -------------------------------------------------- ##
        t3 = time()
        
        # fk[:] = forcing(uknew,fk)
        # fk[:],fkb[:] = forcing(t[i],uk,bk)
        temp_4[:] = np.concatenate((uk,bk[None,:]),axis = 0)
        k1u[:],k1b[:] = RHS(temp_4[:3],temp_4[3], k1u,k1b)
        # k2u[:],k2b[:] = RHS(uk + h*k1u,bk + h*k1b ,k2u,k2b) #! Only for RK2
        # k2u[:],k2b[:] = RHS(semi_G_half*(uk + h/2.*k1u) ,semi_G_half*(bk + h/2.*k1b),k2u,k2b)
        temp_4[:] = G_half_prod(uk + h/2.*k1u,bk + h/2.*k1b)
        k2u[:],k2b[:] = RHS(temp_4[:3],temp_4[3],k2u,k2b)
        # k3u[:],k3b[:] = RHS(semi_G_half*uk + h/2.*k2u, semi_G_half*bk + h/2.*k2b,k3u,k3b)
        temp_4[:] = G_half_prod(uk ,bk ) + h/2.*np.concatenate((k2u,k2b[None,:]),axis = 0)
        k3u[:],k3b[:] = RHS(temp_4[:3],temp_4[3],k3u,k3b)
        # k4u[:],k4b[:] = RHS(semi_G*uk + semi_G_half*h*k3u,semi_G*bk + semi_G_half*h*k3b, k4u,k4b)
        temp_4[:] = G_prod(uk, bk) + h*G_half_prod(k3u,k3b)
        k4u[:],k4b[:] = RHS(temp_4[:3],temp_4[3], k4u,k4b)
        
        # uknew[:] = uk + h/2.0* ( k1u + k2u )  
        # uknew[:] = (semi_G*uk + h/6.0* ( semi_G*k1u + 2*semi_G_half*(k2u + k3u) + k4u)  )*hypervisc 
        # bknew[:] = (semi_G*bk + h/6.0* ( semi_G*k1b + 2*semi_G_half*(k2b + k3b) + k4b)  )*hypervisc 
        temp_4[:] =  hypervisc[None,:]*(G_prod(uk,bk) + h/6.0*(G_prod(k1u,k1b) + 2*G_half_prod(k2u + k3u, k2b + k3b) + np.concatenate((k4u,k4b[None,:]),axis = 0)))
        uknew[:] = temp_4[:3]
        bknew[:] = temp_4[3]
        
        # check_div = comm.allreduce(np.abs(1j*(kx*uknew[0] + ky *uknew[1] + kz*uknew[2])*normalize**0.5).max(),op = MPI.SUM)
        # if rank ==0: 
        #     print(f"max div after RK4 {check_div}")
        if f_corr > 0.0 or N_b > 0.0:
            fk[:],fkb[:] = forcing(t[i],uknew,bknew)
            forc = 1
        else: forc = 0 
        
        # corr = inner_product_k(uknew,bknew,fk,fkb)
        # feng = inner_product_k(fk,fkb,fk,fkb)*0.5*h
        # if rank ==0: print(f"Energy injected : {feng}, correlation :{corr}, alpha = {alpha}")
        
        # uk_v[:], bk_v[:] = vortex(fk,fkb)
        # feng = inner_product_k(uk_v,bk_v,bk_v,bk_v)*0.5
        # if rank ==0: print(f"Balance forcing energy {feng}")
        
        
        uknew[:] = (uknew  + fk*h*dealias*isforcing *forc)
        bknew[:] = (bknew  + fkb*h*dealias*isforcing*forc)
        # uknew[:] = (semi_G*uk + h/6.0* ( semi_G*k1u + 2*semi_G_half*(k2u + k3u) + k4u)  + h*fk)*hypervisc
        # uknew[:] = (uknew + h*fk)
        
        
        
        
        
        """ Enforcing the reality condition """
        uk[0] = ensure_reality(uknew[0])
        uk[1] = ensure_reality(uknew[1])
        uk[2] = ensure_reality(uknew[2])
        bk[:] = ensure_reality(bknew)
        
        """Enforcing div free conditon"""
        uk[:] = ensure_div_free(uk)
        # uk[:] = uknew
        # bk[:] = bknew
  
        #! Although RHS should obey the above two conditions, the rfft adds dependent degrees of freedom for kz = 0 that is evolved separately. Therefore, in some extreme cases, numerical errors can build up. We add the two projections to avoid them.
        # ------------------------------------- #
        
        
        
 
        ## -------------------------------------------------------
        if uk.max() > 100*N**3 : 
            # save(i+1, uk,bk)
            print("Threshold exceeded at time", t[i+1], "Code about to be terminated")
            comm.Abort()
        
        
        comm.Barrier()
        
    ## ---------- Saving the final data ------------
    save(i+1, uk,bk)
    if rank ==0: print(f"average calculation time per step {calc_time/(t.size-1)}")
    ## ---------------------------------------------

    

## --------------- Initializing ---------------------


"""Structure 
If there exists a folder with the parameter names and has time folders in it. 
Load the parameters from parameters.txt
If the parameters match the current code parameters enter the last time folder.
Finally load the data.
If not start from scratch."""



#! Modify the loading process!

if not forcestart:
    ## ------------------------- Beginning from existing data -------------------------
    if rank ==0 : print("Found existing simulation! Using last saved data.")
    """Loading the data from the last time  """    
    # paths = sorted([x for x in pathlib.Path("/mnt/pfs/rajarshi.chattopadhyay/codes/HIT_3D/(-lap_press)ata/forced_True/N_512_Re_500.0").iterdir() if "time_" in str(x)], key=os.path.getmtime)
    
    
    # paths = sorted([x for x in (savePath).iterdir() if "time_" in str(x)], key=os.path.getmtime)
    # """The folder is paths[-1]"""
    # paths = paths[-2]
    paths = savePath/f"last"
    tinit = 0.0

    if rank ==0 : print(f"Loading data from {paths}")
    # tinit = float(str(paths).split("time_")[-1])
    
    uk,bk = load_npz(paths,uk,bk)
    u[0] = irfft_mpi(uk[0],u[0])  
    u[1] = irfft_mpi(uk[1],u[1])  
    u[2] = irfft_mpi(uk[2],u[2])  
    b[:] = irfft_mpi(bk,b)   
    # u = load_hdf5(paths,u) 
    del paths
    # tinit = 0.0   
    # u[:],b[:] = load_sanjay()
    comm.Barrier()
    if rank ==0: print("Data loaded successfully")
     
    uk[0] = rfft_mpi(u[0], uk[0])*dealias
    uk[1] = rfft_mpi(u[1], uk[1])*dealias
    uk[2] = rfft_mpi(u[2], uk[2])*dealias
    bk[:] = rfft_mpi(b, bk)*dealias
    
    trm = (kx*uk[0]  + ky*uk[1] + kz*uk[2])
    uk[0] = uk[0] + invlap*kx*trm
    uk[1] = uk[1] + invlap*ky*trm
    uk[2] = uk[2] + invlap*kz*trm
    
    u[0] = irfft_mpi(uk[0],u[0])
    u[1] = irfft_mpi(uk[1],u[1])
    u[2] = irfft_mpi(uk[2],u[2])

if forcestart:
    ## ---------------------- Beginning from start ----------------------------------

    kinit = 31 # Wavenumber of maximum non-zero initial pressure mode.    
    thu = np.random.uniform(0, TWO_PI,  k.shape)
    thv = np.random.uniform(0, TWO_PI,  k.shape)
    thw = np.random.uniform(0, TWO_PI,  k.shape)

    # eprofile = 1/np.where(kint ==0, np.inf,kint**(2.0))/normalize
    eprofile = k_act**2*np.exp(-k_act**2/2)/normalize
    
    
    amp = (eprofile/np.where(k_act == 0, np.inf, k_act**2))**0.5
    
    uk[0] = amp*np.exp(1j*thu)*(k_act**2<kinit**2)*(k_act>0)*dealias
    uk[1] = amp*np.exp(1j*thv)*(k_act**2<kinit**2)*(k_act>0)*dealias
    uk[2] = amp*np.exp(1j*thw)*(k_act**2<kinit**2)*(k_act>0)*dealias/alpha
    
    u[0] = irfft_mpi(uk[0], u[0])
    u[1] = irfft_mpi(uk[1], u[1])
    u[2] = irfft_mpi(uk[2], u[2])
    
    uk[0] = rfft_mpi(u[0],uk[0])
    uk[1] = rfft_mpi(u[1],uk[1])
    uk[2] = rfft_mpi(u[2],uk[2])
    bk[:] = 0.
    
    trm = (kx*uk[0]  + ky*uk[1] + kz*uk[2])
    uk[0] = uk[0] + invlap*kx*trm
    uk[1] = uk[1] + invlap*ky*trm
    uk[2] = uk[2] + invlap*kz*trm
    
    ek[:] = 0.5*(np.abs(uk[0])**2 + np.abs(uk[1])**2 + np.abs(uk[2])**2/alpha**2 + np.abs(bk)**2)*normalize #! This is the 3D ek array
    ek_arr0 = comm.allreduce(e3d_to_e1d(ek),op = MPI.SUM) #! This is the shell-summed ek a
    # if rank ==0: print(ek_arr0, np.sum(ek_arr0))
    e0 = np.sum(ek_arr0)
    uk[0] = uk[0] *(einit/e0)**0.5
    uk[1] = uk[1] *(einit/e0)**0.5
    uk[2] = uk[2] *(einit/e0)**0.5
    bk[:] = bk  *(einit/e0)**0.5
    
    u[0] = irfft_mpi(uk[0], u[0])
    u[1] = irfft_mpi(uk[1], u[1])
    u[2] = irfft_mpi(uk[2], u[2])
    b[:] = irfft_mpi(bk,b)
    
    
    # uk_v, bk_v = vortex(uk,bk)
    # uk,bk = uk - uk_v, bk - bk_v
    tinit = 0.


# aalpha,beta,gamma = 10,12,13
# u[:] = ABC_flow(aalpha,beta,gamma)
# uk[0] = rfft_mpi(u[0],uk[0])
# uk[1] = rfft_mpi(u[1],uk[1])
# uk[2] = rfft_mpi(u[2],uk[2])
# uk[2] *= 1/alpha

ek[:] = 0.5*(np.abs(uk[0])**2 + np.abs(uk[1])**2 + np.abs(uk[2])**2/alpha**2 + np.abs(bk)**2)*normalize #! This is the 3D ek array
ek_arr0 = comm.allreduce(e3d_to_e1d(ek),op = MPI.SUM) #! This is the shell-summed ek a
if rank ==0: print(ek_arr0, np.sum(ek_arr0))


uk_v[:],bk_v[:] = vortex(uk,bk)
uk_w[:],bk_w[:] = uk - uk_v,bk - bk_v


ek_cross = ((np.einsum('ipqr,ipqr->',np.conjugate(uk_w),uk_v*normalize)) + np.einsum('pqr,pqr->', np.conjugate(bk_w),bk_v*normalize)).real

ek_v = 0.5*(np.abs(uk_v[0])**2 + np.abs(uk_v[1])**2 + np.abs(uk_v[2])**2/alpha**2 + np.abs(bk_v)**2)*normalize
ek_w = 0.5*(np.abs(uk_w[0])**2 + np.abs(uk_w[1])**2 + np.abs(uk_w[2])**2/alpha**2 + np.abs(bk_w)**2)*normalize


ek_cross_val = comm.allreduce(ek_cross,op = MPI.SUM)
ek_v_val = comm.allreduce(ek_v.sum(),op = MPI.SUM)
ek_w_val = comm.allreduce(ek_w.sum(),op = MPI.SUM)

if rank == 0: print(f"Intial vortex energy {ek_v_val},wave energy {ek_w_val}, cross energy {ek_cross_val}")


# raise SystemExit

# ek_arr0[0:shell_no[0]] = 0.
# ek_arr0[shell_no[-1] + 1:] = 0.


divmax = comm.allreduce(np.max(np.abs( diff_x(u[0],  rhsu) + diff_y(u[1],rhsv) + diff_z(u[2],rhsw))),op = MPI.MAX)
if rank ==0 : print(f" max divergence {divmax}")

#----------------- The initial energy ------------------
e0 = comm.allreduce(0.5*dx*dy*dz*np.sum(u[0]**2 + u[1]**2 + (u[2]**2)/alpha**2 + b**2 ),op = MPI.SUM)
if rank ==0 : print(f"Initial Physical space energy: {np.sum(e0)}")
#-------------------------------------------------------

#----------------- testing const. power input ----------
# temp_4[:] = G_half_prod(uk,bk)
# k1u[:],k1b[:] = RHS(temp_4[:3],temp_4[3],k1u,k1b)
# ucalc = 0.0*u
# ucalc[0] = irfft_mpi(k1u[0],ucalc[0])
# ucalc[1] = irfft_mpi(k1u[1],ucalc[1])
# ucalc[2] = irfft_mpi(k1u[2],ucalc[2])

# uanltc = ABC_flow_rhs(aalpha,beta,gamma)

# maxerror = comm.allreduce(np.abs(ucalc- uanltc).max(),op = MPI.MAX)
# if rank ==0 : print(f"Max error in ABC Rhs {maxerror}")
# # fk[:],fkb[:] = forcing(uk,bk)
# # pwrinpt = comm.allreduce(np.sum(np.real(np.einsum('i...,i...->...',np.conjugate(fk[:2]),uk[:2]) + np.conjugate(fk[2])*uk[2]/alpha**2 + np.conjugate(fkb)*bk)*dealias*normalize),op = MPI.SUM)
# # if rank ==0: 
# #     print(f" Prescribed power input: {nshells*f0}, calculated:{pwrinpt}")
# # comm.Barrier()
# raise SystemExit
#-------------------------------------------------------

# --------------------------------------------------
## ----- executing the code -------------------------
t = np.arange(tinit,T+ 0.5*dt, dt)
t1 = time()
evolve_and_save(t,uk,bk)
t2 = time() - t1 
# --------------------------------------------------
if rank ==0: print(t2)
## --------- saving the calculation time -----------
if rank ==0: 
    with open(savePath/f"calcTime.txt","a") as f:
        f.write(str({f"time taken to run from {tinit} to {T} is": t2}))
## --------------------------------------------------



# %%
