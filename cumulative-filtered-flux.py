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
low_wave = False
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
    factor = 1/0.7
    loadPathdata = pathlib.Path(f"/mnt/pfs/rajarshi.chattopadhyay/boussinesq/spectrum-development/nu_{nu}_N_{N}/Ro_{ro}/forcedTide_ring_{omega:.2f}_LW")
    savePlot = pathlib.Path(f"/mnt/pfs/rajarshi.chattopadhyay/codes/boussinesq/Plots/nu_{nu}_N_{N}/Ro_{ro}/forcedTide_ring_{omega:.2f}_LW/filtered_flux")
else: 
    factor = 1.
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
qz = np.zeros((N,N,Np))
div_hz = np.zeros((N,N,Np))
omg_gz = np.zeros((N,N,Np))

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
    
# pih_max = 2.0
# pih_min = -2.0
# piv_max = 2.0
# piv_min = -2.0
# pib_max = 2.0
# pib_min = -2.0
# pi_max = 2.0
# pi_min = -2.0

k_filts = np.array([20,40,60])
# factor = 1.0 #! Change this for LW simulation.

omg_max = 20/factor
omg_min = -0/factor

q_max = 20/factor
q_min = -0/factor

div_h_max = 20/factor
div_h_min = -0/factor

omg_g_max = 20/factor
omg_g_min = -0/factor
    
# pih_bins = np.linspace(pih_min,pih_max,601)
# piv_bins = np.linspace(piv_min,piv_max,601)
# pib_bins = np.linspace(pib_min,pib_max,601)
# pi_bins = np.linspace(pi_min,pi_max,601)
omg_bins = np.linspace(omg_min,omg_max,501)
q_bins = np.linspace(q_min,q_max,501)
div_h_bins = np.linspace(div_h_min,div_h_max,501)
omg_g_bins = np.linspace(omg_g_min,omg_g_max,501)
# pih_hist = np.zeros((len(k_filts),len(pih_bins[:-1])))
# piv_hist = np.zeros((len(k_filts),len(piv_bins[:-1])))
# pib_hist = np.zeros((len(k_filts),len(pib_bins[:-1])))
# pi_hist = np.zeros((len(k_filts),len(pi_bins[:-1])))
pih_omg_hist = np.zeros((len(k_filts),len(omg_bins[:-1])))
piv_omg_hist = np.zeros((len(k_filts),len(omg_bins[:-1])))
pib_omg_hist = np.zeros((len(k_filts),len(omg_bins[:-1])))
pi_omg_hist = np.zeros((len(k_filts),len(omg_bins[:-1])))
pih_q_hist = np.zeros((len(k_filts),len(q_bins[:-1])))
piv_q_hist = np.zeros((len(k_filts),len(q_bins[:-1])))
pib_q_hist = np.zeros((len(k_filts),len(q_bins[:-1])))
pi_q_hist = np.zeros((len(k_filts),len(q_bins[:-1])))
pih_div_h_hist = np.zeros((len(k_filts),len(div_h_bins[:-1])))
piv_div_h_hist = np.zeros((len(k_filts),len(div_h_bins[:-1]))) 
pib_div_h_hist = np.zeros((len(k_filts),len(div_h_bins[:-1])))
pi_div_h_hist = np.zeros((len(k_filts),len(div_h_bins[:-1])))
pih_omg_g_hist = np.zeros((len(k_filts),len(omg_g_bins[:-1])))
piv_omg_g_hist = np.zeros((len(k_filts),len(omg_g_bins[:-1])))
pib_omg_g_hist = np.zeros((len(k_filts),len(omg_g_bins[:-1])))
pi_omg_g_hist = np.zeros((len(k_filts),len(omg_g_bins[:-1])))

pih_omg_cuml = np.zeros_like(pih_omg_hist)
piv_omg_cuml = np.zeros_like(pih_omg_hist)
pib_omg_cuml = np.zeros_like(pih_omg_hist)
pi_omg_cuml = np.zeros_like(pih_omg_hist)

pih_q_cuml = np.zeros_like(pih_q_hist)
piv_q_cuml = np.zeros_like(pih_q_hist)
pib_q_cuml = np.zeros_like(pih_q_hist)
pi_q_cuml = np.zeros_like(pih_q_hist)

pih_div_h_cuml = np.zeros_like(pih_div_h_hist)
piv_div_h_cuml = np.zeros_like(pih_div_h_hist)
pib_div_h_cuml = np.zeros_like(pih_div_h_hist)
pi_div_h_cuml = np.zeros_like(pih_div_h_hist)

pih_omg_g_cuml = np.zeros_like(pih_omg_g_hist)
piv_omg_g_cuml = np.zeros_like(pih_omg_g_hist)
pib_omg_g_cuml = np.zeros_like(pih_omg_g_hist)
pi_omg_g_cuml = np.zeros_like(pih_omg_g_hist)


tflux = np.zeros(len(k_filts))
# t = 600.0
times = np.arange(500,600.1,1.)
div_omg_g_hist = 0.
for t in times:
    if rank ==0: print(t, num_process,Ns)
    loadPath = loadPathdata/f"time_{t:.1f}"
    for qq in range(Ns):
        Fields = np.load(loadPath/f"Fields_{(rank*Ns + qq):.0f}.npz")
        u[0,qq*n_slab:(qq + 1)*n_slab]  = Fields["u"]
        u[1,qq*n_slab:(qq + 1)*n_slab]  = Fields["v"]
        u[2,qq*n_slab:(qq + 1)*n_slab]  = Fields["w"]
        b[qq*n_slab:(qq + 1)*n_slab]  = Fields["b"]
        q[qq*n_slab:(qq + 1)*n_slab] = np.load(loadPathdata/f"PV/time_{t:.1f}/PV_{(rank*Ns + qq):.0f}.npy")
        omg[qq*n_slab:(qq + 1)*n_slab] = np.load(loadPathdata/f"zeta/time_{t:.1f}/zeta_{(rank*Ns + qq):.0f}.npy")
    
    comm.Barrier()
    div_h[:] = diff_x(u[0],u_temp) + diff_y(u[1],v_temp)
    omg_g[:] = -ifft_cos(kh**2*invlapc*fft_cos(q,pk),omg_g)
    if rank ==0: print("Calculated")

    # k_filts = np.arange(1,100)

    # k_filts = 1.0*np.array([20,40,60,int(N/(3**0.5)) +1])
    div_omg_g_hist = div_omg_g_hist + np.histogram2d(div_h.ravel(),omg_g.ravel(),bins = [div_h_bins,omg_g_bins])[0]/(N**3) 
    for ii,k_filt in enumerate(k_filts):
        l_filt = TWO_PI/k_filt
        cond_c = (kcint<= k_filt)
        cond_s = (ksint<= k_filt)
        
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

        # levels = np.linspace(-5,5,200)
        # plt.figure(figsize = (8,6))

        # # dat = np.clip(u_filt[...,N//2].T,None,None) # clipped data between vmin and vmax
        # dat = u_filt[2,...,N//2].T
        # p1 = plt.contourf(X,Y,dat ,200,cmap = "RdYlBu_r",extend = "both")
        # cbar = plt.colorbar(p1)#min = -5,vmax = 5)
        # # cbar.set_clim(-5,5)
        # plt.xlabel(r"$x$")
        # plt.ylabel(r"$y$",rotation = 0)


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
                    A[i,j] = (del_cos(j,u_filt[i],u_temp))
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
            pi_v[:] = pi_v + tau[i,2]*A[i,2] + 1/alph**2 *tau[2,i]*A[2,i]

        pi_v[:] = pi_v + 1/alph**2 *tau[2,2]*A[2,2]
            

        pi_b[:] = 0.
        for i in range(3):
            pi_b[:] = pi_b + B[i]*grdB[i]


        ## ------ corresponding with the notation --------------- ##
        pi_h[:] = - pi_h
        pi_v[:] = - pi_v
        pi_b[:] = - pi_b
        ## ------------------------------------------------------ ##

        totflux = comm.allreduce(np.sum(pi_h + pi_b + pi_v)*(TWO_PI/N)**3, op = MPI.SUM)
        tflux[ii] += totflux
        tothflux = comm.allreduce(np.sum(pi_h)*(TWO_PI/N)**3, op = MPI.SUM)
        totvflux = comm.allreduce(np.sum(pi_v)*(TWO_PI/N)**3, op = MPI.SUM)
        totbflux = comm.allreduce(np.sum(pi_b)*(TWO_PI/N)**3, op = MPI.SUM)
        if rank == 0: print(f"total flux, k_filt {k_filt} : {totflux},{tflux[ii]}")
        # levels = np.linspace(-5,5,200)    
        if k_filt not in [20,40,60]: continue
        ## ---------------------- converting to slabbed in z direction ----------- ## 

        pih[:] = x_to_z(pi_h,pih)
        piv[:] = x_to_z(pi_v,piv)
        pib[:] = x_to_z(pi_b,pib)
        
        omgz[:] = x_to_z(omg,omgz)
        qz[:] = x_to_z(q,qz)
        div_hz[:] = x_to_z(div_h,div_hz)
        omg_gz[:] = x_to_z(omg_g,omg_gz)
        
        pi[:] = pih + piv + pib
        
        ## ---------------------- calculating flux histograms -------------------- ##
        
        # omg_gz[:] = 1.
        # omgz[:]  = 1.5
        # qz[:] = 2.
        # div_hz[:] = 3.
        
        # pih_hist = pih_hist + np.histogram(pih.ravel(),bins = pih_bins)[0]/(N**3)
        # piv_hist = piv_hist + np.histogram(piv.ravel(),bins = piv_bins)[0]/(N**3)
        # pib_hist = pib_hist + np.histogram(pib.ravel(),bins = pib_bins)[0]/(N**3)
        # pi_hist = pi_hist + np.histogram(pi.ravel(),bins = pi_bins)[0]/(N**3)
        
        # pih_omg_hist[ii] +=  np.histogram(np.abs(omgz).ravel(), bins = omg_bins, weights = pih.ravel())[0]/(N**3)
        # piv_omg_hist[ii] +=  np.histogram(np.abs(omgz).ravel(), bins = omg_bins, weights = piv.ravel())[0]/(N**3)
        # pib_omg_hist[ii] +=  np.histogram(np.abs(omgz).ravel(), bins = omg_bins, weights = pib.ravel())[0]/(N**3)
        # pi_omg_hist[ii] +=  np.histogram(np.abs(omgz).ravel(), bins = omg_bins, weights = pi.ravel())[0]/(N**3)
        
        # pih_q_hist[ii] +=  np.histogram(np.abs(qz).ravel(), bins = q_bins, weights = pih.ravel())[0]/(N**3)
        # piv_q_hist[ii] +=  np.histogram(np.abs(qz).ravel(), bins = q_bins, weights = piv.ravel())[0]/(N**3)
        # pib_q_hist[ii] +=  np.histogram(np.abs(qz).ravel(), bins = q_bins, weights = pib.ravel())[0]/(N**3)
        # pi_q_hist[ii] +=  np.histogram(np.abs(qz).ravel(), bins = q_bins, weights = pi.ravel())[0]/(N**3)
        
        pih_div_h_hist[ii] +=  np.histogram(np.abs(div_hz).ravel(), bins = div_h_bins, weights = pih.ravel())[0]/(N**3)
        piv_div_h_hist[ii] +=  np.histogram(np.abs(div_hz).ravel(), bins = div_h_bins, weights = piv.ravel())[0]/(N**3)
        pib_div_h_hist[ii] +=  np.histogram(np.abs(div_hz).ravel(), bins = div_h_bins, weights = pib.ravel())[0]/(N**3)
        pi_div_h_hist[ii] +=  np.histogram(np.abs(div_hz).ravel(), bins = div_h_bins, weights = pi.ravel())[0]/(N**3)
        
        pih_omg_g_hist[ii] +=  np.histogram(np.abs(omg_gz).ravel(), bins = omg_g_bins, weights = pih.ravel())[0]/(N**3)
        piv_omg_g_hist[ii] +=  np.histogram(np.abs(omg_gz).ravel(), bins = omg_g_bins, weights = piv.ravel())[0]/(N**3)
        pib_omg_g_hist[ii] +=  np.histogram(np.abs(omg_gz).ravel(), bins = omg_g_bins, weights = pib.ravel())[0]/(N**3)
        pi_omg_g_hist[ii] +=  np.histogram(np.abs(omg_gz).ravel(), bins = omg_g_bins, weights = pi.ravel())[0]/(N**3)
        

        # pih_omg_hist = np.sum(0.5*(pih_bins[1:] + pih_bins[:-1])[:,None]*np.histogram2d(pih.ravel(),omgz.ravel(),bins = [pih_bins,omg_bins])[0],axis = 0)*(TWO_PI/N**3)
        # piv_omg_hist = np.sum(0.5*(piv_bins[1:] + piv_bins[:-1])[:,None]*np.histogram2d(piv.ravel(),omgz.ravel(),bins = [piv_bins,omg_bins])[0],axis = 0)*(TWO_PI/N**3)
        # pib_omg_hist = np.sum(0.5*(pib_bins[1:] + pib_bins[:-1])[:,None]*np.histogram2d(pib.ravel(),omgz.ravel(),bins = [pib_bins,omg_bins])[0],axis = 0)*(TWO_PI/N**3)
        # pi_omg_hist = np.sum(0.5*(pi_bins[1:] + pi_bins[:-1])[:,None]*np.histogram2d(pi.ravel(),omgz.ravel(),bins = [pi_bins,omg_bins])[0],axis = 0)*(TWO_PI/N**3)    
        
        # pih_q_hist = np.sum(0.5*(pih_bins[1:] + pih_bins[:-1])[:,None]*np.histogram2d(pih.ravel(),qz.ravel(),bins = [pih_bins,q_bins])[0],axis = 0)*(TWO_PI/N**3)
        # piv_q_hist = np.sum(0.5*(piv_bins[1:] + piv_bins[:-1])[:,None]*np.histogram2d(piv.ravel(),q.ravel(),bins = [piv_bins,q_bins])[0],axis = 0)*(TWO_PI/N**3)
        # pib_q_hist = np.sum(0.5*(pib_bins[1:] + pib_bins[:-1])[:,None]*np.histogram2d(pib.ravel(),qz.ravel(),bins = [pib_bins,q_bins])[0],axis = 0)*(TWO_PI/N**3)
        # pi_q_hist = np.sum(0.5*(pi_bins[1:] + pi_bins[:-1])[:,None]*np.histogram2d(pi.ravel(),qz.ravel(),bins = [pi_bins,q_bins])[0],axis = 0)*(TWO_PI/N**3)
        
        # pih_div_h_hist = np.sum(0.5*(pih_bins[1:] + pih_bins[:-1])[:,None]*np.histogram2d(pih.ravel(),div_hz.ravel(),bins = [pih_bins,div_h_bins])[0],axis = 0)*(TWO_PI/N**3)
        # piv_div_h_hist = np.sum(0.5*(piv_bins[1:] + piv_bins[:-1])[:,None]*np.histogram2d(piv.ravel(),div_hz.ravel(),bins = [piv_bins,div_h_bins])[0],axis = 0)*(TWO_PI/N**3)
        # pib_div_h_hist = np.sum(0.5*(pib_bins[1:] + pib_bins[:-1])[:,None]*np.histogram2d(pib.ravel(),div_hz.ravel(),bins = [pib_bins,div_h_bins])[0],axis = 0)*(TWO_PI/N**3)
        # pi_div_h_hist = np.sum(0.5*(pi_bins[1:] + pi_bins[:-1])[:,None]*np.histogram2d(pi.ravel(),div_hz.ravel(),bins = [pi_bins,div_h_bins])[0],axis = 0)*(TWO_PI/N**3)
        
        # pih_omg_g_hist = np.sum(0.5*(pih_bins[1:] + pih_bins[:-1])[:,None]*np.histogram2d(pih.ravel(),omg_gz.ravel(),bins = [pih_bins,omg_g_bins])[0],axis = 0)*(TWO_PI/N**3)
        # piv_omg_g_hist = np.sum(0.5*(piv_bins[1:] + piv_bins[:-1])[:,None]*np.histogram2d(piv.ravel(),omg_gz.ravel(),bins = [piv_bins,omg_g_bins])[0],axis = 0)*(TWO_PI/N**3)
        # pib_omg_g_hist = np.sum(0.5*(pib_bins[1:] + pib_bins[:-1])[:,None]*np.histogram2d(pib.ravel(),omg_gz.ravel(),bins = [pib_bins,omg_g_bins])[0],axis = 0)*(TWO_PI/N**3)
        # pi_omg_g_hist = np.sum(0.5*(pi_bins[1:] + pi_bins[:-1])[:,None]*np.histogram2d(pi.ravel(),omg_gz.ravel(),bins = [pi_bins,omg_g_bins])[0],axis = 0)*(TWO_PI/N**3)
        

        # if rank == 0: print(f"total flux, k_filt {k_filt} : {np.sum(pi_q_hist)}, {np.sum(pi_omg_hist)}, {np.sum(pi_div_h_hist)}, {np.sum(pi_omg_g_hist)}")
        
        
        
        

        
        
        

        ## ---------------------- plotting the fluxes --------------------------- ##
        # zslices = np.array([N//3,N//4,N//5, 2*N//3, 2*N//4, 2*N//5])
        # # if k_filt < 10. or k_filt > 100. : continue
        # if rank in zslices: 
        #     plt.figure(figsize = (16,12))
        #     plt.suptitle(fr"$z = ${Z[rank]:.2f},$k_{{\ell}} = $ {k_filt}",fontsize = 30)

        #     # dat = q.T # clipped data between vmin and vmax
        #     # dat = np.clip(u_filt[...,N//2].T,None,None) # clipped data between vmin and vmax
        #     plt.subplot(2,2,1)
        #     vmin = -2*pi.std() + pi.mean()
        #     vmax = 2*pi.std() + pi.mean()
        #     dat = np.clip(pi[...,0].T,vmin,vmax)
        #     norm = colors.TwoSlopeNorm(vcenter = 0,vmin = vmin,vmax = vmax )
        #     p1 = plt.contourf(np.linspace(0,2*np.pi,N),np.linspace(0,2*np.pi,N),dat ,200,cmap = "RdYlBu_r",extend = "both",norm = norm)
        #     cbar = plt.colorbar(p1)
        #     # cbar.set_clim(-5,5)
        #     plt.xlabel(r"$x$")
        #     plt.ylabel(r"$y$",rotation = 0)
        #     plt.title(r"$\Pi_{\ell}$")
            
            
        #     plt.subplot(2,2,2)
        #     vmin = - 2*omg.std() + omg.mean()
        #     vmax = 2*omg.std() + omg.mean()
        #     dat = np.clip(omgz[...,0].T,vmin,vmax)
        #     norm = colors.TwoSlopeNorm(vcenter = 0,vmin = vmin,vmax = vmax )
        #     p1 = plt.contourf(np.linspace(0,2*np.pi,N),np.linspace(0,2*np.pi,N),dat ,200,cmap = "RdYlBu_r",extend = "both",norm = norm)
        #     cbar = plt.colorbar(p1)
        #     # cbar.set_clim(-5,5)
        #     plt.xlabel(r"$x$")
        #     plt.ylabel(r"$y$",rotation = 0)
        #     plt.title(r"$\zeta$")
            
            
        #     plt.subplot(2,2,3)
        #     vmin = - 2*q.std() + q.mean()
        #     vmax = 2*q.std() + q.mean()
        #     dat = np.clip(omg_gz[...,0].T,vmin,vmax)
        #     norm = colors.TwoSlopeNorm(vcenter = 0,vmin = vmin,vmax = vmax )
        #     p1 = plt.contourf(np.linspace(0,2*np.pi,N),np.linspace(0,2*np.pi,N),dat ,200,cmap = "RdYlBu_r",extend = "both",norm = norm)
        #     cbar = plt.colorbar(p1)
        #     # cbar.set_clim(-5,5)
        #     plt.xlabel(r"$x$")
        #     plt.ylabel(r"$y$",rotation = 0)
        #     plt.title(r"$\zeta_G$")
            
            
        #     plt.subplot(2,2,4)
        #     vmin = - 2*div_hz.std() + div_h.mean()
        #     vmax = 2*div_h.std() + div_h.mean()
        #     dat = np.clip(div_hz[...,0].T,vmin,vmax)
        #     norm = colors.TwoSlopeNorm(vcenter = 0,vmin = vmin,vmax = vmax )
        #     p1 = plt.contourf(np.linspace(0,2*np.pi,N),np.linspace(0,2*np.pi,N),dat ,200,cmap = "RdYlBu_r",extend = "both",norm = norm)
        #     cbar = plt.colorbar(p1)
        #     # cbar.set_clim(-5,5)
        #     plt.xlabel(r"$x$")
        #     plt.ylabel(r"$y$",rotation = 0)
        #     plt.title(r"$\nabla . u$")
            
            
            
        #     plt.tight_layout()
            
        # plt.grid()    
        # plt.savefig(savePlot/f"flux_slab_{rank:.0f}_t_{t:.2f}_k_{k_filt:.2f}.png")
        #     plt.close()


        
        
    # ## ---------------- plotting the net physical flux ------------------------- ##
    # if rank ==0: 
    #     print(tflux[:20])
    #     plt.figure(figsize = (8,6))
    #     plt.plot(k_filts,tflux,label = r"$\Pi$")
    #     plt.legend(fontsize = 10)
    #     plt.xlabel(r"$k_{\ell}$")
    #     plt.ylabel(r"$\Pi$")
    #     plt.xscale("log")
    #     # plt.yscale("log")
    #     plt.title(fr"Net Physical Flux")
    #     plt.tight_layout()
    # plt.grid()    
    # plt.savefig(savePlot/f"Net_physical_flux_t_{t:.2f}.png")
    #     plt.close()


## ---------------------- averaging the histograms ----------------------- ##     
# div_h_omg_g_hist = comm.allreduce(div_omg_g_hist,op = MPI.SUM)/(len(times))
# pih_hist = comm.allreduce(pih_hist,op = MPI.SUM)/(len(times))
# piv_hist = comm.allreduce(piv_hist,op = MPI.SUM)/(len(times))
# pib_hist = comm.allreduce(pib_hist,op = MPI.SUM)/(len(times))
# pi_hist = comm.allreduce(pi_hist,op = MPI.SUM)/(len(times))

    ## ---------------------- plotting the histograms ----------------------- ##
for ii,k_filt in enumerate(k_filts):
    # pih_omg_hist[ii] = comm.allreduce(pih_omg_hist[ii],op = MPI.SUM)/(tflux[ii]/TWO_PI**3)
    # piv_omg_hist[ii] = comm.allreduce(piv_omg_hist[ii],op = MPI.SUM)/(tflux[ii]/TWO_PI**3)
    # pib_omg_hist[ii] = comm.allreduce(pib_omg_hist[ii],op = MPI.SUM)/(tflux[ii]/TWO_PI**3)
    # pi_omg_hist[ii] = comm.allreduce(pi_omg_hist[ii],op = MPI.SUM)/(tflux[ii]/TWO_PI**3)
    # pih_q_hist[ii] = comm.allreduce(pih_q_hist[ii],op = MPI.SUM)/(tflux[ii]/TWO_PI**3)
    # piv_q_hist[ii] = comm.allreduce(piv_q_hist[ii],op = MPI.SUM)/(tflux[ii]/TWO_PI**3)
    # pib_q_hist[ii] = comm.allreduce(pib_q_hist[ii],op = MPI.SUM)/(tflux[ii]/TWO_PI**3)
    # pi_q_hist[ii] = comm.allreduce(pi_q_hist[ii],op = MPI.SUM)/(tflux[ii]/TWO_PI**3)
    pih_div_h_hist[ii] = comm.allreduce(pih_div_h_hist[ii],op = MPI.SUM)/(tflux[ii]/TWO_PI**3)
    piv_div_h_hist[ii] = comm.allreduce(piv_div_h_hist[ii],op = MPI.SUM)/(tflux[ii]/TWO_PI**3)
    pib_div_h_hist[ii] = comm.allreduce(pib_div_h_hist[ii],op = MPI.SUM)/(tflux[ii]/TWO_PI**3)
    pi_div_h_hist[ii] = comm.allreduce(pi_div_h_hist[ii],op = MPI.SUM)/(tflux[ii]/TWO_PI**3)
    pih_omg_g_hist[ii] = comm.allreduce(pih_omg_g_hist[ii],op = MPI.SUM)/(tflux[ii]/TWO_PI**3)
    piv_omg_g_hist[ii] = comm.allreduce(piv_omg_g_hist[ii],op = MPI.SUM)/(tflux[ii]/TWO_PI**3)
    pib_omg_g_hist[ii] = comm.allreduce(pib_omg_g_hist[ii],op = MPI.SUM)/(tflux[ii]/TWO_PI**3)
    pi_omg_g_hist[ii] = comm.allreduce(pi_omg_g_hist[ii],op = MPI.SUM)/(tflux[ii]/TWO_PI**3)
    
    # pih_omg_cuml[ii] = np.cumsum(pih_omg_hist[ii])
    # piv_omg_cuml[ii] = np.cumsum(piv_omg_hist[ii])
    # pib_omg_cuml[ii] = np.cumsum(pib_omg_hist[ii])
    # pi_omg_cuml[ii] = np.cumsum(pi_omg_hist[ii])
    
    # pih_q_cuml[ii] = np.cumsum(pih_q_hist[ii])
    # piv_q_cuml[ii] = np.cumsum(piv_q_hist[ii])
    # pib_q_cuml[ii] = np.cumsum(pib_q_hist[ii])
    # pi_q_cuml[ii] = np.cumsum(pi_q_hist[ii])
    
    pih_div_h_cuml[ii] = np.cumsum(pih_div_h_hist[ii])
    piv_div_h_cuml[ii] = np.cumsum(piv_div_h_hist[ii])
    pib_div_h_cuml[ii] = np.cumsum(pib_div_h_hist[ii])
    pi_div_h_cuml[ii] = np.cumsum(pi_div_h_hist[ii])
    
    pih_omg_g_cuml[ii] = np.cumsum(pih_omg_g_hist[ii])
    piv_omg_g_cuml[ii] = np.cumsum(piv_omg_g_hist[ii])
    pib_omg_g_cuml[ii] = np.cumsum(pib_omg_g_hist[ii])
    pi_omg_g_cuml[ii] = np.cumsum(pi_omg_g_hist[ii])
    
    tflux[ii] = tflux[ii]/len(times)
    if rank ==0:
        # plt.figure(figsize = (8,6))
        # plt.pcolor(div_h_bins,omg_g_bins,div_h_omg_g_hist.T,cmap = "RdBu_r")
        # plt.ylabel(rf"$\zeta_g$",rotation = 0)
        # plt.xlabel(r"$\nabla . u$")
        # plt.tight_layout()
        # # plt.xscale("symlog",linthresh = 1e-2)
        # plt.grid()
        # plt.savefig(savePlot/f"2D_histogram.png")
        # plt.close()
        
        # plt.figure(figsize = (8,6))
        # plt.plot(pih_bins[1:],pih_hist,label = r"$\Pi_h$")
        # plt.plot(piv_bins[1:],piv_hist,label = r"$\Pi_v$")
        # plt.plot(pib_bins[1:],pib_hist,label = r"$\Pi_b$")
        # plt.plot(pi_bins[1:],pi_hist,label = r"$\Pi$")
        # plt.axvline(totflux,ls = "--",color = "black")
        # plt.legend(fontsize = 10)
        # plt.ylabel(rf"$P(\Pi)_{{\ell}}$",rotation = 0)
        # plt.xlabel(r"$\Pi$")
        # plt.title(fr"$k_{{\ell}} = $ {k_filt}")
        # plt.tight_layout()
        # plt.yscale("log")
        # # plt.xscale("symlog",linthresh = 1e-2)
        # plt.grid()
        # plt.savefig(savePlot/f"histogram_averaged_k_{k_filt:.2f}.png")
        # plt.close()
        
        # plt.figure(figsize = (8,6))
        # plt.plot(omg_bins[1:]/omg_bins[-1],pih_omg_cuml[ii],label = r"$\Pi_h$")
        # plt.plot(omg_bins[1:]/omg_bins[-1],piv_omg_cuml[ii],label = r"$\Pi_v$")
        # plt.plot(omg_bins[1:]/omg_bins[-1],pib_omg_cuml[ii],label = r"$\Pi_b$")
        # plt.plot(omg_bins[1:]/omg_bins[-1],pi_omg_cuml[ii],label = r"$\Pi$")
        # # plt.axvline(totflux,ls = "--",color = "black")
        # plt.legend(fontsize = 10)
        # plt.xlabel(rf"$\frac{\left|{\zeta}}{{\zeta_{max}}}\right|$")
        # plt.ylabel(r"$\frac{\sum\limits_{\zeta} \Pi}{\langle \Pi \rangle}$",rotation = 0,labelpad = 20)
        # plt.title(fr"$k_{{\ell}} = $ {k_filt}")
        # plt.tight_layout()
        # # plt.yscale("log")
        # # plt.xscale("symlog",linthresh = 1e-2)
        # plt.grid()
        # plt.savefig(savePlot/f"cumulative_vorticity_histogram_k_{k_filt:.2f}.png")
        # plt.close()
        
        # plt.figure(figsize = (8,6))
        # plt.plot(q_bins[1:]/q_bins[-1],pih_q_cuml[ii],label = r"$\Pi_h$")
        # plt.plot(q_bins[1:]/q_bins[-1],piv_q_cuml[ii],label = r"$\Pi_v$")
        # plt.plot(q_bins[1:]/q_bins[-1],pib_q_cuml[ii],label = r"$\Pi_b$")
        # plt.plot(q_bins[1:]/q_bins[-1],pi_q_cuml[ii],label = r"$\Pi$")
        # # plt.axvline(totflux,ls = "--",color = "black")
        # plt.legend(fontsize = 10)
        # plt.xlabel(rf"$q$")
        # plt.ylabel(r"$\frac{\sum\limits_{q} \Pi}{\langle \Pi \rangle}$",rotation = 0,labelpad = 20)
        # plt.title(fr"$k_{{\ell}} = $ {k_filt}")
        # plt.tight_layout()
        # # plt.yscale("symlog",linthresh = 1e-3)
        # # plt.yscale("log")
        # # plt.xscale("symlog",linthresh = 1e-2)
        # plt.grid()
        # plt.savefig(savePlot/f"cumulative_potential_vorticity_histogram_k_{k_filt:.2f}.png")
        # plt.close()

        plt.figure(figsize = (8,6))
        plt.plot(div_h_bins[1:]/div_h_bins[-1],pih_div_h_cuml[ii],label = r"$\Pi_h$")
        plt.plot(div_h_bins[1:]/div_h_bins[-1],piv_div_h_cuml[ii],label = r"$\Pi_v$")
        plt.plot(div_h_bins[1:]/div_h_bins[-1],pib_div_h_cuml[ii],label = r"$\Pi_b$")
        plt.plot(div_h_bins[1:]/div_h_bins[-1],pi_div_h_cuml[ii],label = r"$\Pi$")
        # plt.axvline(totflux,ls = "--",color = "black")
        plt.legend(fontsize = 10)
        plt.xlabel(r"$\left|\frac{\nabla . u}{(\nabla . u)_{max}}\right|$")
        plt.ylabel(r"$\frac{\sum\limits_{(\nabla . u)} \Pi}{\langle \Pi \rangle}$",rotation = 0,labelpad = 20)
        plt.title(fr"$k_{{\ell}} = $ {k_filt}")
        plt.tight_layout()
        # plt.yscale("symlog",linthresh = 1e-3)
        # plt.yscale("log")
        plt.grid()
        plt.savefig(savePlot/f"cumulative_div_h_histogram_k_{k_filt:.2f}.png")
        plt.close()
        
        plt.figure(figsize = (8,6))
        plt.plot(omg_g_bins[1:]/omg_g_bins[-1],pih_omg_g_cuml[ii],label = r"$\Pi_h$")
        plt.plot(omg_g_bins[1:]/omg_g_bins[-1],piv_omg_g_cuml[ii],label = r"$\Pi_v$")
        plt.plot(omg_g_bins[1:]/omg_g_bins[-1],pib_omg_g_cuml[ii],label = r"$\Pi_b$")
        plt.plot(omg_g_bins[1:]/omg_g_bins[-1],pi_omg_g_cuml[ii],label = r"$\Pi$")
        # plt.plot(omg_g_bins[1:],np.cumsum(pih_omg_g_hist),label = r"$\Pi_h$")
        # plt.plot(omg_g_bins[1:],np.cumsum(piv_omg_g_hist),label = r"$\Pi_v$")
        # plt.plot(omg_g_bins[1:],np.cumsum(pib_omg_g_hist),label = r"$\Pi_b$")
        # plt.plot(omg_g_bins[1:],np.cumsum(pi_omg_g_hist),label = r"$\Pi$")
        # plt.axvline(totflux,ls = "--",color = "black")
        plt.legend(fontsize = 10)
        plt.xlabel(r"$\left|\frac{\zeta_G}{(\zeta_G)_{max}}\right|$")
        plt.ylabel(r"$\frac{\sum\limits_{\zeta_G} \Pi}{\langle \Pi \rangle}$",rotation = 0,labelpad = 20)
        plt.title(fr"$k_{{\ell}} = $ {k_filt}")
        plt.tight_layout()
        # plt.yscale("symlog",linthresh = 1e-3)
        # plt.yscale("log")
        # plt.ylim()
        # plt.xscale("symlog",linthresh = 1e-2)
        plt.grid()
        plt.savefig(savePlot/f"cumulative_omg_g_histogram_k_{k_filt:.2f}.png")
        plt.close()
            
        ## ---------------------------------------------------------------------- ##
        
## ------------------ total flux for different k_filt --------------- ##

if rank == 0:
    plt.figure(figsize = (8,6))
    for ii, k_filt in enumerate(k_filts):
        plt.plot(omg_g_bins[1:]/omg_g_bins[-1],pi_omg_g_cuml[ii],label = fr"$k_{{\ell}} = {k_filt}$")
    plt.legend(fontsize = 10)
    plt.xlabel(r"$\left|\frac{\zeta_G}{(\zeta_G)_{max}}\right|$")
    plt.ylabel(r"$\frac{\sum\limits_{\zeta_G} \Pi}{\langle \Pi \rangle}$",rotation = 0,labelpad = 20)
    plt.tight_layout()
    plt.grid()
    plt.savefig(savePlot/f"cumulative_omg_g_total_flux.png")
    plt.close()
    
    # plt.figure(figsize = (8,6))
    # for ii, k_filt in enumerate(k_filts):
    #     plt.plot(q_bins[1:]/q_bins[-1],pi_q_cuml[ii],label = fr"$k_{{\ell}} = {k_filt}$")
    # plt.legend(fontsize = 10)
    # plt.xlabel(rf"$q$")
    # plt.ylabel(r"$\frac{\sum\limits_{q} \Pi}{\langle \Pi \rangle}$",rotation = 0,labelpad = 20)
    # plt.tight_layout()
    # plt.grid()
    # plt.savefig(savePlot/f"cumulative_q_total_flux.png")
    # plt.close()
    
    plt.figure(figsize = (8,6))
    for ii, k_filt in enumerate(k_filts):
        plt.plot(div_h_bins[1:]/div_h_bins[-1],pi_div_h_cuml[ii],label = fr"$k_{{\ell}} = {k_filt}$")
    plt.legend(fontsize = 10)
    plt.xlabel(r"$\left|\frac{\nabla . u}{(\nabla . u)_{max}}\right|$")
    plt.ylabel(r"$\frac{\sum\limits_{(\nabla . u)} \Pi}{\langle \Pi \rangle}$",rotation = 0,labelpad = 20)
    plt.tight_layout()
    plt.grid()
    plt.savefig(savePlot/f"cumulative_div_h_total_flux.png")
    plt.close()
    
    # plt.figure(figsize = (8,6))
    # for ii, k_filt in enumerate(k_filts):
    #     plt.plot(omg_bins[1:]/omg_bins[-1],pi_omg_cuml[ii],label = fr"$k_{{\ell}} = {k_filt}$")
    # plt.legend(fontsize = 10)
    # plt.xlabel(rf"$\zeta$")
    # plt.ylabel(r"$\frac{\sum\limits_{\zeta} \Pi}{\langle \Pi \rangle}$",rotation = 0,labelpad = 20)
    # plt.tight_layout()
    # plt.grid()
    # plt.savefig(savePlot/f"cumulative_vorticity_total_flux.png")
    # plt.close()
    