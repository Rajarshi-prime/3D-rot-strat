
# #%%
# from google.colab import drive
# drive.mount('/content/drive')
#%%
import numpy as npp
import math
import jax.numpy as np
import jax.numpy.fft as fft
import jax,gc
from time import time
# import h5py
import pathlib,json,os
from tqdm import tqdm
from sympy import LeviCivita
curr_path = pathlib.Path("./data")
curr_path.mkdir(exist_ok=True, parents = True)
os.listdir(curr_path)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# np.cuda.runtime.setDevice(0)
key = jax.random.PRNGKey(9231472983719)
#%%
jax.config.update("jax_enable_x64", True)
#%%
forcestart = True
idx = 3
# forcestart = bool(float(sys.argv[-1]))
isforcing = True
viscosity_integrator = "implicit"
# viscosity_integrator = "explicit" #! Do not use this for hyperviscous simulations or cases with high resolution simulations.
# viscosity_integrator = "exponential"
if viscosity_integrator == "explicit": isexplicit = 1.
else : isexplicit = 0.
#%%
rank = 0
num_process = 1
#%%
## ------------- Time steps --------------
N = 256
dt = 0.256/N * (2*np.pi)**1.5  #! Such that increasing resolution will decrease the dt
T = 10000
dt_save = 1.0
st = round(dt_save/dt)
f_corr = 1.0
N_bs = [5,10,15,20]
N_b = N_bs[idx]
#%%
fvec = np.array([0.,0.,f_corr])
N_bvec = np.array([0,0,N_b])
## ---------------------------------------
#%%
""
## -------------Defining the grid ---------------
PI = np.pi
TWO_PI = 2*PI
Nf = N//2 + 1
Np = N//num_process
sx = slice(0,N)
L = TWO_PI
X = Y = Z = np.linspace(0, L, N, endpoint= False)
dx,dy,dz = X[1]-X[0], Y[1]-Y[0], Z[1]-Z[0]
x, y, z = np.meshgrid(X[sx], Y, Z, indexing='ij')

Kx = Ky = fft.fftfreq(N,  1./N)*TWO_PI/L
Kz = np.abs(Ky[:Nf])

kx,  ky,  kz = np.meshgrid(Kx,  Ky[sx],  Kz,  indexing = 'ij')

kvec = np.array([kx,ky,kz])
## -----------------------------------------------

#%%


## --------- kx and ky for differentiation ---------
# kx_diff = np.moveaxis(kz,[0,1,2],[2,1,0]).copy()
# ky_diff = np.swapaxes(kx_diff, 0, 1).copy()
# kz_diff = np.moveaxis(kz, [0,1], [1,0]).copy()

# if rank ==0 : print(kx_diff.shape, ky_diff.shape, kz_diff.shape)

## -------------------------------------------------
#%%

#%%
## It is best to define the function which returns the real part of the iifted function as ifft.
@jax.jit
def irfft(x):
    return fft.irfftn(x,(N,N,N),axes = (-3,-2,-1))

@jax.jit
def rfft(x):
    return fft.rfftn(x,axes = (-3,-2,-1))



lp = 8 # Hyperviscosity power
nu0 = 0.5 #! Viscosity for N = 1
m = 1 #! Desired kmax*eta
nu = nu0*(3*m/N)**(2*(lp - 1/3))  #? scaling with resolution. For 512, nu = 0.002 #! Need to add scaling for hyperviscosity

fbyN = f_corr/N_b if N_b != 0 else 0.0
einit = 1 # Initial energy
nshells = 4 # Number of consecutive shells to be forced
shell_no = np.arange(1,1+nshells) # the shells to be forced
#%%

f0 = 0.02 /(N_b**2)*nshells#! Total power input at each shells
re = np.inf if nu==0 else 1/nu
if rank ==0 : print(f" Power input  : {nshells*f0} \n Viscosity : {nu}, Re : {re},dt : {dt}")
#%%
nu,f0*nshells
#%%
""


if nu!= 0: savePath = curr_path/f"data/bsnq/f_{f_corr:.1f}_Nb_{N_b:.1f}/forced_{isforcing}/N_{N}_Re_{re:.1f}"
else: savePath = curr_path/f"data/bsnq/forced_{isforcing}/N_{N}_Re_inf"

if rank == 0:
    print(savePath)
    try: savePath.mkdir(parents=True,  exist_ok=True)
    except FileExistsError: pass

lap = -1.0*(kx**2 + ky**2 + kz**2 )
k = np.linalg.norm(kvec, axis = 0 )
kint = np.clip(np.round(k,0).astype(int),None,N//2+1)
kh = (kx**2 + ky**2)**0.5
dealias = kint<=N/3 #! Spherical dealiasing
# dealias = (abs(kx)<N//3)*(abs(ky)<N//3)*(abs(kz)<N//3) #! Cubic 2/3 dealiasing
# dealias = np.exp(-(N//10)*((1.0*kx/N)**(N//10) +(1.0*kz/N)**(N//10) + (1.0*kz/N)**(N//10))) #! Exponential dealiasing a la. Sanjay for 360^3.
invlap = dealias/np.where(lap == 0, np.inf,  lap)
lapwv = -1.0*(kx**2 + ky**2 + (fbyN)**2*kz**2 )
invlapwv = dealias/np.where(lapwv == 0, np.inf,lapwv)
#%%
# Hyperviscous operator
vis = nu*(k)**(2*lp) ## This is in Fourier Space

normalize = np.where((kz== 0) + (kz == N//2) , 1/(N**6/TWO_PI**3),2/(N**6/TWO_PI**3))
shells = np.arange(-0.5,Nf, 1.)
shells = shells.at[0].set(0.)

cond_kx = np.abs(np.round(Kx))<=N//3
cond_ky = np.abs(np.round(Ky))<=N//3
cond_kz = np.abs(np.round(Kz))<=N//3
## -------------------------------------------------
#%%

epsilon = np.array([[[float(LeviCivita(i, j,kk)) for kk in range(3)] for j in range(3)] for i in range(3)])
## ----------------------------------------------------
epsilon.shape
del kx,ky
#%%
# u  = np.zeros((3, Np, N, N), dtype= np.float64)
# b = np.zeros_like(u[0])

# omg= np.zeros((3, Np, N, N), dtype= np.float64)


# uk = np.zeros((3, N, Np, Nf), dtype= np.complex128)
# uk_w = uk.copy()
# uk_v = uk.copy()
# pk = uk[0].copy()

# bk = pk.copy()
# bk_w = bk.copy()
# bk_v = bk.copy()

# ek = np.zeros_like(pk, dtype = np.float64)
# Pik = np.zeros_like(pk, dtype = np.float64)
# ek_arr = np.zeros(Nf)
# Pik_arr = np.zeros(Nf)
# factor = np.zeros(Nf)
# factor3d = np.zeros_like(pk,dtype= np.float64)


# fk = np.zeros_like(uk)
# fkb = np.zeros_like(bk)

@jax.jit
def e3d_to_1d(x):
    return (np.histogram(k.ravel(),bins = shells,weights=x.ravel())[0]).real
## --------------------------------------------------------------

# @jax.jit
# def diff_x(u,kx_diff = kx_diff):
#     return fft.irfft(1j*kx_diff*fft.rfft(u, axis= -3), N, axis= -3)
# @jax.jit
# def diff_y(u,ky_diff = ky_diff):
#     return fft.irfft(1j*ky_diff*fft.rfft(u, axis= -2), N, axis= -2)
# @jax.jit
# def diff_z(u,kz_diff = kz_diff):
#     return fft.irfft(1j*kz_diff*fft.rfft(u, axis= -1), N, axis= -1)

@jax.jit
def vortex(uk,bk, fbyN = fbyN):

    """
    Projects the velocity and buoyancy in the vortical modes.
    """
    uk_v = 0.0*uk.copy()
    pv = 1j*(kvec[0]*uk[1] - kvec[1]*uk[0] + kvec[2]*bk*(fbyN))
    uk_v = uk_v.at[0].set( -1j*kvec[1]*pv*invlapwv)
    uk_v = uk_v.at[1].set( 1j*kvec[0]*pv*invlapwv)
    bk_v = 1j*kvec[2]*pv*invlapwv*(fbyN)

    return uk_v, bk_v


@jax.jit
def forcing(uk,bk):
    """
    Calculates the net dissipation of the flow and injects that amount into larges scales of the horizontal flow
    """
    # global fk, fkb, factor3d, factor, ek_arr,kint
    uk_v,bk_v = vortex(uk,bk)
    uk_w,bk_w = uk - uk_v, bk - bk_v


    ek = 0.5*(np.abs(uk_w[0])**2 + np.abs(uk_w[1])**2 + np.abs(uk_w[2])**2 + np.abs(bk_w)**2)*dealias*normalize*(kh>0.5)*(kz > 0.5) #! This is the 3D ek array of waves

    ek_arr = e3d_to_1d(ek) #! This is the shell-summed ek array.
    #? Only if you are forcing 1 or two shells
    # for shell in shell_no:
    #     ek_arr[shell] = comm.allreduce(np.sum(ek*(kint>= shell-0.5)*(kint< shell +0.5)),op = MPI.SUM)

    ek_arr = np.where(np.abs(ek_arr)< 1e-10,1e20, ek_arr)
    """Change forcing starts here"""
    # Const Power Input
    factor = 0.*ek_arr.copy()
    factor = factor.at[shell_no].set(f0/(2*ek_arr[shell_no]))
    factor3d = factor[kint]*dealias*(kh>0.5)*(kz > 0.5)


    # # Constant shell energy
    # factor[:] = np.tanh(np.where(np.abs(ek_arr0) < 1e-10, 0, (ek_arr0/ek_arr)**0.5 - 1)) #! The factors for each shell is calculated
    # factor3d[:] = factor[kint]

    fk = factor3d[None,...]*uk_w
    fkb = factor3d*bk_w

    """Change forcing ends here here"""

    pk = invlap  * np.einsum('i...,i...-> ...',kvec,fk)*dealias

    fk = fk + kvec*pk[None,...]
    # fk[0] = fk[0] + kx*pk
    # fk[1] = fk[1] + ky*pk
    # fk[2] = fk[2] + kz*pk


    return fk*isforcing*dealias, fkb*isforcing*dealias
# @jax.jit
# def diff_x(x,h = dx):
#     return (np.roll(x,-1,axis = 0) - np.roll(x,1,axis = 0))/(2*h)

# @jax.jit
# def diff_y(x,h = dy):
#     return (np.roll(x,-1,axis = 1) - np.roll(x,1,axis = 1))/(2*h)

# @jax.jit
# def diffn_x(x,n):
#     y = x
#     for i in range(n):
#         y = diff_x(y)
#     return y

# @jax.jit
# def diffn_y(x,n):
#     y = x
#     for i in range(n):
#         y = diff_y(y)
#     return y
#%%

@jax.jit
def RHS(uk, bk,visc = 1,forc = 1,isexplicit = isexplicit,invlap = invlap,fvec = fvec, N_bvec = N_bvec):
    ## The RHS terms of u, v and w excluding the forcing and the hypervisocsity term
    fk,fkb = forcing(uk,bk)
    fk *= forc
    fkb *= forc

    u = irfft(uk)
    # b = irfft(bk)

    omg = irfft(1j*np.einsum('ijk,j...,k...->i...',epsilon,kvec,uk))

    # rhs = np.einsum('ijk,j...,k...->i...', epsilon, u,(omg + fvec[:,None,None,None])) + N_bvec[:,None,None,None]*b[None, ...]

    # rhsb = - np.einsum('i...,i...->...',u,irfft(1j*kvec*bk[None,...])) - N_b*u[2]


    rhsk = (rfft(np.einsum('ijk,j...,k...->i...', epsilon, u,(omg + fvec[:,None,None,None]))) + N_bvec[:,None,None,None]*bk[None, ...] + fk)*dealias[None,...]
    rhsbk = (-1j*np.einsum('i...,i...->...',kvec,rfft(u*irfft(bk)[None,...])) - N_b*uk[2] + fkb)*dealias

    ## The pressure term
    pk = 1j*invlap  * np.einsum('i...,i...->...',kvec, rhsk)

    # uk_t = rhsk - 1j*kvec*pk[None,...] - (nu*((-lap)**lp)*isexplicit * visc)[None,...] * uk

    # ## The RHS term with the pressure
    # bk_t = rhsbk - nu*((-lap)**lp)*bk*isexplicit * visc


    # return uk_t, bk_t
    return rhsk - 1j*kvec*pk[None,...] - (nu*((-lap)**lp)*isexplicit * visc)[None,...] * uk, rhsbk - nu*((-lap)**lp)*bk*isexplicit * visc

# AA,BB,CC = 1, 1, 1
# u0 = np.array([
#     AA*np.sin(z) + CC * np.cos(y),BB*np.sin(x) + AA*np.cos(z), CC*np.sin(y) + BB*np.cos(x)
# ])
# b0 = np.cos(x)
# uk0 = rfft(u0)
# bk0 = rfft(b0)
# u1k, b1k = RHS(uk0,bk0, visc=0, forc = 0)
# u1 =irfft(u1k)
# b1 =irfft(b1k)
# # np.allclose(u1[2], -N_b*np.cos(x) )
# np.allclose(b1, -N_b*(np.cos(x) +  np.sin(y)) + np.sin(x)*(np.cos(y) + np.sin(z)) )
# # np.allclose(u1[1], -np.sin(z))
#%%
# @jax.jit
def RK4(i,h,ti,uk,bk,semi_G_half, semi_G, hypervisc):

    # k1u,k1b = RHS(uk,bk)
    # # k2u,k2b = RHS(uk + h*k1u,bk + h*k1b ,k2u,k2b) #! Only for RK2
    # k2u,k2b = RHS(semi_G_half*(uk + h/2.*k1u) ,semi_G_half*(bk + h/2.*k1b))
    # k3u,k3b = RHS(semi_G_half*uk + h/2.*k2u, semi_G_half*bk + h/2.*k2b)
    # k4u,k4b = RHS(semi_G*uk + semi_G_half*h*k3u,semi_G*bk + semi_G_half*h*k3b)

    # # uknew = uk + h/2.0* ( k1u + k2u )
    # uknew = (semi_G*uk + h/6.0* ( semi_G*k1u + 2*semi_G_half*(k2u + k3u) + k4u)  )*hypervisc
    # bknew = (semi_G*bk + h/6.0* ( semi_G*k1b + 2*semi_G_half*(k2b + k3b) + k4b)  )*hypervisc


    ku,kb = RHS(uk,bk)
    uknew = semi_G*uk + h/6.0* semi_G*ku
    bknew = semi_G*bk + h/6.0* semi_G*kb

    ku,kb =  RHS(semi_G_half*(uk + h/2.*ku) ,semi_G_half*(bk + h/2.*kb))
    uknew += h/6.0*2*semi_G_half*ku
    bknew += h/6.0*2*semi_G_half*kb

    ku,kb = RHS(semi_G_half*uk + h/2.*ku, semi_G_half*bk + h/2.*kb)
    uknew += h/6.0*2*semi_G_half*ku
    bknew += h/6.0*2*semi_G_half*kb

    ku,kb = RHS(semi_G*uk + semi_G_half*h*ku,semi_G*bk + semi_G_half*h*kb)
    uknew += h/6.0*ku
    bknew += h/6.0*kb


    """ Enforcing the reality condition """
    u = irfft(uknew*hypervisc)
    b = irfft(bknew*hypervisc)

    uk = rfft(u)
    bnew = rfft(b)


    """Enforcing div free conditon"""
    pk = invlap  * np.einsum('i...,i...->...',kvec,uk)
    uknew = uk + kvec*pk[None,...]


    return uknew,bknew
#%%
# AA,BB,CC = 1, 1, 1
# u0 = np.array([
#     AA*np.sin(z) + CC * np.cos(y),BB*np.sin(x) + AA*np.cos(z), CC*np.sin(y) + BB*np.cos(x)
# ])
# h = 1e-6
# b0 = np.cos(x)
# uk0 = rfft(u0)
# bk0 = rfft(b0)
# u1k, b1k = RK4(0,h,0,uk0,bk0,semi_G_half = 1.0, semi_G = 1.0,)
# u1 =irfft(u1k)
# b1 =irfft(b1k)
# # np.allclose(u1[2], -N_b*np.cos(x) )
# np.allclose(b1, b0 + h*( -N_b*(np.cos(x) +  np.sin(y)) + np.sin(x)*(np.cos(y) + np.sin(z))) )
# np.allclose(u1[1], u0[1] + h*(-np.sin(z)))
#%%
def save(t,uk,bk):
    ku,kb = RHS(uk,bk,visc = 0,forc = 0)


    ek_arr = e3d_to_1d(.5*(np.abs(uk[0])**2 + np.abs(uk[1])**2 + np.abs(uk[2])**2 + np.abs(bk)**2)*normalize) #! This is the shell-summed ek array.

    Pik_arr = e3d_to_1d(np.real(np.conjugate(uk[0])*ku[0]+np.conjugate(uk[1])*ku[1]+ np.conjugate(uk[2])*ku[2] + np.conjugate(bk)*kb)*dealias*normalize)
    Pik_arr = np.cumsum(Pik_arr[::-1])[::-1]


    uk_v,bk_v = vortex(uk,bk)
    uk_w,bk_w = uk - uk_v, bk -bk_v

    ek_v_val = np.sum(0.5*(np.abs(uk_v[0])**2 + np.abs(uk_v[1])**2 + np.abs(uk_v[2])**2 + np.abs(bk_v)**2)*normalize)
    ek_w_val = np.sum(0.5*(np.abs(uk_w[0])**2 + np.abs(uk_w[1])**2 + np.abs(uk_w[2])**2 + np.abs(bk_w)**2)*normalize)


    u = irfft(uk)
    b = irfft(bk)
    # ----------- ----------------------------
    #                 Saving the data (field)
    # ----------- ----------------------------
    # new_dir = savePath/f"time_{t[i]:.1f}"
    new_dir = savePath/f"last"
    try: new_dir.mkdir(parents=True,  exist_ok=True)
    except FileExistsError: pass
    # np.savez_compressed(f"{new_dir}/Fields_{rank}.npz",uhat = uk)
    # np.savez_compressed(f"{new_dir}/Fields_{rank}",u = u[0],v = u[1],w = u[2])

    # u_temp = uk[:,cond_kx,cond_ky,:N//3+1] #! Will only save the values in x k_x and k_y plane for the dealiased mode.
    # b_temp = bk[cond_kx,cond_ky,:N//3 + 1]


    np.savez(f"{new_dir}/Fields_cmp.npz",u = u[0],v = u[1],w =u[2],b = b)
    np.savez(f"{new_dir}/Energy_spectrum",ek = ek_arr)
    np.savez(f"{new_dir}/Flux_spectrum",Pik = Pik_arr)


    # ----------- ----------------------------
    #          Calculating and printing
    # ----------- ----------------------------
    eng1 = np.sum(0.5*(u[0]**2 + u[1]**2 + u[2]**2 + b**2)*dx*dy*dz)
    eng2 = np.sum(ek_arr)
    print( "#----------------------------","\n",f"Energy at time {t} is : {eng1}, {eng2}","\n","#----------------------------")
    print(f"Maximum divergence {np.max(np.abs(np.einsum('i...,i...->...',kvec,uk)))}")
    print(f"Vortex energy {ek_v_val},wave energy {ek_w_val}")
    return "Done!"
#%%
# save(0,uk0,bk0)

#! Needs work
# def load_trunc(t):
#     loadPath = savePath/f"last"
#     uk = np.zeros((3, N,N,Nf),dtype = np.complex128)
#     bk = np.zeros((N,N,Nf),dtype = np.complex128)
#     data= np.load(loadPath/f"Fields_cmp.npz")
#     uk = uk.at[0,cond_kx,cond_ky,:N//3 + 1].set(data["u"])
#     uk = uk.at[1,cond_kx,cond_ky,:N//3 + 1].set(data["v"])
#     uk = uk.at[2,cond_kx,cond_ky,:N//3 + 1].set(data["w"])
#     bk = bk.at[cond_kx,cond_ky,:N//3 + 1].set(data["b"])
#     return uk, bk

#%%
def load(t,path):
    uk = np.zeros((3, N,N,Nf),dtype = np.complex128)

    data= np.load(path/f"Fields_cmp.npz")
    uk  = uk.at[0].set(rfft(data["u"]))
    uk  = uk.at[1].set(rfft(data["v"]))
    uk  = uk.at[2].set(rfft(data["w"]))
    bk = rfft(data["b"])
    return uk, bk
#%%
# ukload, bkload = load_trunc(0)
# np.abs(uk0-ukload).max(), np.abs(bk0-bkload).max()

# @jax.jit
def evolve_and_save(t,uk,bk):
    global begin
    h = t[1] - t[0]

    if viscosity_integrator == "implicit": hypervisc= dealias*(1. + h*vis)**(-1)
    else: hypervisc = 1.

    if  viscosity_integrator == "exponential":
        semi_G =  np.exp(-nu*(k**(2*lp))*h)
        semi_G_half =  semi_G**0.5
    else: semi_G = semi_G_half = 1.

    # t3  = time()
    # calc_time = 0
    for i in tqdm(range(len(t)-1)):
        # print(ifft2(x_old).std())
        ti = t[i]
        if i % st ==0 :save(ti,uk,bk)


        uk,bk = RK4(i,h,ti,uk,bk,semi_G_half,semi_G,hypervisc)
    # field_save(i,t[-1], x_old,cs_old)
    # particle_save(i,t[-1],xprtcl)

    # save(t[-1],x_old, cs_old, xprtcl)

    return uk,bk

kint.shape
#%%
if not forcestart:
    ## ------------------------- Beginning from existing data -------------------------
    if rank ==0 : print("Found existing simulation! Using last saved data.")
    """Loading the data from the last time  """
    # paths = sorted([x for x in pathlib.Path("/mnt/pfs/rajarshi.chattopadhyay/codes/HIT_3D/data/forced_True/N_512_Re_500.0").iterdir() if "time_" in str(x)], key=os.path.getmtime)


    # paths = sorted([x for x in (savePath).iterdir() if "time_" in str(x)], key=os.path.getmtime)
    # """The folder is paths[-1]"""
    # paths = paths[-2]
    paths = savePath/f"last"

    if rank ==0 : print(f"Loading data from {paths}")
    # tinit = float(str(paths).split("time_")[-1])
    tinit = 0.0
    uk,bk = load(tinit,paths)

    trm = (kx*uk[0]  + ky*uk[1] + kz*uk[2])
    uk = uk + (invlap*trm)[None,...]*kvec
    u = irfft(uk)
    b = irfft(bk)
#%%
if forcestart:
    kinit = 31 # Wavenumber of maximum non-zero initial pressure mode.
    th= jax.random.uniform(key, shape=kvec.shape, minval=0, maxval=TWO_PI)
    # eprofile = 1/np.where(kint ==0, np.inf,kint**(2.0))/normalize
    eprofile = kint**2*np.exp(-kint**2/2)/normalize


    amp = (eprofile/np.where(kint == 0, np.inf, kint**2))**0.5

    uk = (amp*(kint**2<kinit**2)*(kint>0)*dealias)[None,...]*np.exp(1j*th)

    print(uk.shape)

    u = irfft(uk)


    uk = rfft(u)

    bk = 0.*uk[0].copy()

    pk = np.einsum('i...,i...->...',kvec,uk)
    uk += (invlap*pk)[None,...]* kvec






    uk_v, bk_v = vortex(uk,bk)
    uk,bk = uk - uk_v, bk - bk_v
    tinit = 0.

    ek_arr0 = e3d_to_1d(0.5*(np.abs(uk[0])**2 + np.abs(uk[1])**2 + np.abs(uk[2])**2 + np.abs(bk)**2)*normalize) #! This is the shell-summed ek a
    # if rank ==0: print(ek_arr0, np.sum(ek_arr0))

    e0 = np.sum(ek_arr0)
    uk *= (einit/e0)**0.5
    bk *= (einit/e0)**0.5

    u = irfft(uk)
    b = irfft(bk)




    ek_arr0 = e3d_to_1d( 0.5*(np.abs(uk[0])**2 + np.abs(uk[1])**2 + np.abs(uk[2])**2 + np.abs(bk)**2)*normalize ) #! This is the shell-summed ek a
    print(ek_arr0, np.sum(ek_arr0))


    uk_v,bk_v = vortex(uk,bk)
    uk_w,bk_w = uk - uk_v,bk - bk_v
    del th,amp

ek_cross = ((np.einsum('ipqr,ipqr->',np.conjugate(uk_w),uk_v*normalize)) + np.einsum('pqr,pqr->', np.conjugate(bk_w),bk_v*normalize)).real

ek_v = np.sum(0.5*(np.abs(uk_v[0])**2 + np.abs(uk_v[1])**2 + np.abs(uk_v[2])**2 + np.abs(bk_v)**2)*normalize)
ek_w = np.sum(0.5*(np.abs(uk_w[0])**2 + np.abs(uk_w[1])**2 + np.abs(uk_w[2])**2 + np.abs(bk_w)**2)*normalize)

ek_arr0 = ek_arr0.at[0:shell_no[0]].set(0.)
ek_arr0 = ek_arr0.at[shell_no[-1] + 1:].set(0.)

divmax = np.max(np.abs(np.einsum('i...,i...->...',kvec,uk)))
print(divmax)

print(f"Initial Physical space energy: {np.sum(0.5*dx*dy*dz*np.sum(u[0]**2 + u[1]**2 + (u[2]**2) + b**2 ))}")
#%%
ek_cross,ek_v,ek_w
#%%
gc.collect()
# jax.block_until_ready()
#jax.clear_caches()
#%%

#%%
h = dt

if viscosity_integrator == "implicit": hypervisc= dealias*(1. + h*vis)**(-1)
else: hypervisc = 1.

if  viscosity_integrator == "exponential":
    semi_G =  np.exp(-nu*(k**(2*lp))*h)
    semi_G_half =  semi_G**0.5
else: semi_G = semi_G_half = 1.
RK4(0,h,0,uk,bk,semi_G_half, semi_G, hypervisc)
#%%
#t1= time()
#c_steps = 10
#for i in tqdm(range(c_steps)):
#    ti = 0
#     RK4(i,h,ti,uk,bk,semi_G_half,semi_G,hypervisc)
# uk0, bk0 = uk.copy(),bk.copy()
#t2 = time() 
#print(f"{c_steps} took {t2 -t1} seconds")
# save(0,uk0,bk0)


## ----- executing the code -------------------------
t = np.arange(tinit,T+ 0.5*dt, dt)
# t = np.arange(0,200, dt)
t1 = time()
uk,bk = evolve_and_save(t,uk,bk)
t2 = time() - t1
# --------------------------------------------------
#%%
# u = irfft(uk)
# u0 = irfft(uk0)

