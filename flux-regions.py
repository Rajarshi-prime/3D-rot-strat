import numpy as np
from scipy.fft import fftn,ifftn, fft ,  ifft ,  irfft2 ,  rfft2 , irfftn ,  rfftn, fftfreq, dst, dct, idst, idct, rfft,  irfft
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from mpi4py import MPI
import pathlib,os,sys
import matplotlib as mpl 
import matplotlib.colors as colors
mpl.rc('text', usetex = True)
mpl.rcParams['font.size'] = 25

curr_path = pathlib.Path(__file__).parent

## -------------- cao et al colormap ----------------
hexVals = ['#700000','#8e0308','#c3060c','#ef1119','#ff3928','#ff6b46','#ff9271','#ffbdab','#ffe1e1','#ffffff','#d4f5ff','#92dde6','#41c1d8','#00a1cf','#0093ca','#0072b7','#00599f','#002c6d','#001847']
hexVals.reverse()
custom_cm=colors.LinearSegmentedColormap.from_list('custom',hexVals)
cmap=custom_cm

custom_cm1 = colors.LinearSegmentedColormap.from_list('custom',['#ffffff','#303030'])
cmap_c = custom_cm1
## --------------------------------------------------

## ---------------MPI things--------------
comm = MPI.COMM_WORLD
num_process =  comm.Get_size()
rank = comm.Get_rank()
low_wave = [True,False]
low_wave = low_wave[int(sys.argv[-2])]
## ---------------------------------------

N = 384
Np = N// num_process
ro = 0.1*float(sys.argv[-1])
nu = 1e-31
alph = 20
num_slabs = 384
Ns = num_slabs// num_process
n_slab = N//num_slabs
flux_thresh = 0.9

lw = "_LW" if low_wave else ""
factor = 1.0/0.5 if low_wave else 1.0
factor1 = 10.0 if low_wave else 1.0
omega = 1.7277
loadPath = pathlib.Path(f"/mnt/pfs/rajarshi.chattopadhyay/boussinesq/spectrum-development/nu_{nu}_N_{N}/Ro_{ro}/forcedTide_ring_{omega:.2f}{lw}")

savePath = loadPath
savePlot = curr_path/f"Plots/nu_{nu}_N_{N}/Ro_{ro:.1f}/forcedTide_ring_{omega:.2f}{lw}/"

loadPathdata = pathlib.Path(f"/mnt/pfs/rajarshi.chattopadhyay/boussinesq/spectrum-development/nu_{nu}_N_{N}/Ro_{ro}/forcedTide_ring_{omega:.2f}{lw}")
savePlot = curr_path/f"Plots/nu_{nu}_N_{N}/Ro_{ro:.1f}/forcedTide_ring_{omega:.2f}{lw}/filtered_flux_data"


savePath = loadPathdata/f"filtered_flux_data/"
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
b_temp = np.zeros((Np,N,N))
s = np.zeros((Np,N,N))
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
region_omg = np.zeros((N,N,Np))
region_omg_c = region_omg.copy()
region_q = np.zeros((N,N,Np))
region_q_c = region_q.copy()
region_div_h = np.zeros((N,N,Np))
region_div_h_c = region_div_h.copy()
region_omg_g = np.zeros((N,N,Np))
region_omg_g_c = region_omg_g.copy()

vmin,vmax = -2,2
norm = mpl.colors.TwoSlopeNorm(vmin =vmin, vmax = vmax,vcenter = 0.0)
# cmap = "RdBu"
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
    


k_filts = [20,40,60]
tflux = np.zeros(len(k_filts))
# t = 600.0
times = np.arange(1100,1100.1,1.)
flux_mean = np.zeros((len(k_filts),len(times)))
flux_std = np.zeros((len(k_filts),len(times)))
div_omg_g_hist = 0.
# for jj,t in enumerate(times):
t = 1100.0
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
div_h[:] = diff_x(u[0],u_temp) + diff_y(u[1],v_temp)
omg_g[:] = -ifft_cos(kh**2*invlapc*fft_cos(q,pk),omg_g)
omgz[:] = x_to_z(omg,omgz)
qz[:] = x_to_z(q,qz)
div_hz[:] = x_to_z(div_h,div_hz)
omg_gz[:] = x_to_z(omg_g,omg_gz)
if rank ==0: print("Calculated")

plot_slabs = [N//6,2*N//6, 3*N//6, 4*N//6, N//5,2*N//5, 3*N//5, 4*N//5, N//4, 2*N//4, 3*N//4, 4*N//4 ]
pi_c = 0.1
for ii,k_filt in enumerate(k_filts):      
    

    
    cuml_fluxes = np.load(savePath/f"pi_cuml_{k_filt:.2f}.npz")
    pi_omg_g = cuml_fluxes["pi_omg_g_cuml"]
    pi_q = cuml_fluxes["pi_q_cuml"]
    pi_div_h = cuml_fluxes["pi_div_h_cuml"]
    pi_omg = cuml_fluxes["pi_omg_cuml"]
    
    
    vals = np.load(savePath/f"cuml_bins.npz")
    omg_g_vals = 0.5*(vals["omg_g_bins"][:-1] + vals["omg_g_bins"][1:])
    q_vals = 0.5*(vals["q_bins"][:-1] + vals["q_bins"][1:])
    div_h_vals = 0.5*(vals["div_h_bins"][:-1] + vals["div_h_bins"][1:])
    omg_vals = 0.5*(vals["omg_bins"][:-1] + vals["omg_bins"][1:])
    
    # print(omg_g_vals.shape, pi_omg_g.shape)
    if rank ==0 : 
        print(f"Max min flux {pi_omg_g.min()}, {pi_omg_g.max()}")
        print(f"Max min flux {pi_q.min()}, {pi_q.max()}")
        print(f"Max min flux {pi_div_h.min()}, {pi_div_h.max()}")
        print(f"Max min flux {pi_omg.min()}, {pi_omg.max()}")
    
    omg_interp = interp1d(pi_omg,omg_vals)
    q_interp = interp1d(pi_q,q_vals)
    div_h_interp = interp1d(pi_div_h,div_h_vals)
    omg_g_interp = interp1d(pi_omg_g,omg_g_vals)
    
    omg_thresh = omg_interp(flux_thresh)
    q_thresh = q_interp(flux_thresh) 
    div_h_thresh = div_h_interp(flux_thresh)
    omg_g_thresh = omg_g_interp(flux_thresh)
    
    
    region_omg[:] = np.where(np.abs(omgz)< omg_thresh,omgz,0.0)
    region_q[:] = np.where(np.abs(qz)< q_thresh,qz,0.0)
    region_div_h[:] = np.where(np.abs(div_hz)< div_h_thresh,div_hz,0.0)
    region_omg_g[:] = np.where(np.abs(omg_gz)< omg_g_thresh,omg_gz,0.0)
    
    region_omg_c[:] = np.where(np.abs(omgz)< omg_thresh,np.nan,1.0)
    region_q_c[:] = np.where(np.abs(qz)< q_thresh,np.nan,1.0)
    region_div_h_c[:] = np.where(np.abs(div_hz)< div_h_thresh,np.nan,1.0)
    region_omg_g_c[:] = np.where(np.abs(omg_gz)< omg_g_thresh,np.nan,1.0)
    
    # region_omg[:] = np.where(mask,omgz,np.nan)
    # region_q[:] = np.where(mask,qz,np.nan)
    # region_div_h[:] = np.where(mask,div_hz,np.nan)
    # region_omg_g[:] = np.where(mask,omg_gz,np.nan)
    
    
    
    if (rank+1) in plot_slabs:
        np.savez_compressed(savePath/f"90_percent_flux_fields_{k_filt:.2f}_zslab_{rank}.npz",region_omg = region_omg,region_q = region_q,region_div_h = region_div_h,region_omg_g = region_omg_g, region_omg_c = region_omg_c,region_q_c = region_q_c,region_div_h_c = region_div_h_c,region_omg_g_c = region_omg_g_c)
        
        plt.figure(figsize = (20,20))
        plt.suptitle(fr"k = {k_filt:.2f}, $z = \Pi/{N/(rank+1):.2f}$")
        
        plt.subplot(221)
        region_omg[:] = np.clip(region_omg,vmin,vmax)
        p1 = plt.contourf(X,Y,region_omg[:,:,0].T,200, cmap = cmap,extend = "both")
        plt.colorbar(p1)
        plt.contourf(X,Y,region_omg_c[:,:,0].T,20, cmap = cmap_c,extend = "both",alpha = 0.5)
        plt.title(r"$\omega$")
        
        plt.subplot(222)
        region_div_h[:] = np.clip(region_div_h,vmin,vmax)
        p2 = plt.contourf(X,Y,region_div_h[:,:,0].T,200, cmap = cmap,extend = "both")
        plt.colorbar(p2)
        plt.contourf(X,Y,region_div_h_c[:,:,0].T,20, cmap = cmap_c,extend = "both",alpha = 0.5)
        plt.title(r"$\nabla \cdot u$")
        
        plt.subplot(223)
        region_q[:] = np.clip(region_q,vmin,vmax)
        p3 = plt.contourf(X,Y,region_q[:,:,0].T,200, cmap = cmap,extend = "both")
        plt.colorbar(p3)
        plt.contourf(X,Y,region_q_c[:,:,0].T,20, cmap = cmap_c,extend = "both",alpha = 0.5)
        plt.title(r"$q$")
        
        plt.subplot(224)
        region_omg_g[:] = np.clip(region_omg_g,vmin,vmax)
        p4 = plt.contourf(X,Y,region_omg_g[:,:,0].T,200, cmap = cmap,extend = "both")
        plt.colorbar(p4)
        plt.contourf(X,Y,region_omg_g_c[:,:,0].T,20, cmap = cmap_c,extend = "both",alpha = 0.5)
        plt.title(r"$\omega_g$")
        
        plt.savefig(savePlot/f"90_percent_flux_field_regions_{k_filt:.2f}_{rank+1}.png")
        plt.close()
        
        
    flux_pdfs = np.load(savePath/f"pi_hist_{k_filt:.2f}.npz")
    pi_pdf = flux_pdfs["pi"]
    
    pi_bins = np.load(savePath/f"bins_{k_filt:.2f}.npz")["pi_bins"]
    
    pi_vals = 0.5*(pi_bins[:-1] + pi_bins[1:])
    
    pi_mean = np.sum(pi_pdf*pi_vals)
    
    if rank ==0  : print(f"Mean flux = {pi_mean}")
    
    pi_val_max = np.max(pi_vals)
    
    Deltas = np.linspace(pi_val_max/100,pi_val_max,100)
    
    conds = np.array([np.abs(pi_vals)<= delta for delta in Deltas])
    
    pi_Delta_means = np.array([np.sum(pi_vals * pi_pdf * cond) for cond in conds])
    
    if rank ==0 : print(pi_mean, pi_Delta_means/pi_mean)
    
    delta_interp = interp1d(pi_Delta_means/pi_mean,Deltas)
    
    delta_c = delta_interp(pi_c)
    
    if rank ==0: print(f"Delta_c = {delta_c}")
    
    
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
    pi_h[:] = - ro*pi_h
    pi_v[:] = - ro*pi_v
    pi_b[:] = - ro*pi_b


    pih[:] = x_to_z(pi_h,pih)
    piv[:] = x_to_z(pi_v,piv)
    pib[:] = x_to_z(pi_b,pib)
    pi[:] = pih + piv + pib
    
    mask = np.abs(pi) > delta_c
        
    region_omg[:] = np.where(mask,omgz,0.0)
    region_q[:] = np.where(mask,qz,0.0)
    region_div_h[:] = np.where(mask,div_hz,0.0)
    region_omg_g[:] = np.where(mask,omg_gz,0.0)
    
    region_omg_c[:] = np.where(mask,np.nan,1.0)
    region_q_c[:] = np.where(mask,np.nan,1.0)
    region_div_h_c[:] = np.where(mask,np.nan,1.0)
    region_omg_g_c[:] = np.where(mask,np.nan,1.0)
    
    
    if (rank+1) in plot_slabs:
        np.savez_compressed(savePath/f"90_percent_flux_regions_{k_filt:.2f}_zslab_{rank}.npz",region_omg = region_omg,region_q = region_q,region_div_h = region_div_h,region_omg_g = region_omg_g, region_omg_c = region_omg_c,region_q_c = region_q_c,region_div_h_c = region_div_h_c,region_omg_g_c = region_omg_g_c)
        
        plt.figure(figsize = (20,20))
        plt.suptitle(fr"k = {k_filt:.2f}, $z = \Pi/{N/(rank+1):.2f}$")
        
        plt.subplot(221)
        region_omg[:] = np.clip(region_omg,vmin,vmax)
        p1 = plt.contourf(X,Y,region_omg[:,:,0].T,200, cmap = cmap,extend = "both")
        plt.colorbar(p1)
        plt.contourf(X,Y,region_omg_c[:,:,0].T,20, cmap = cmap_c,extend = "both",alpha = 0.5)
        plt.title(r"$\omega$")
        
        plt.subplot(222)
        region_div_h[:] = np.clip(region_div_h,vmin,vmax)
        p2 = plt.contourf(X,Y,region_div_h[:,:,0].T,200, cmap = cmap,extend = "both")
        plt.colorbar(p2)
        plt.contourf(X,Y,region_div_h_c[:,:,0].T,20, cmap = cmap_c,extend = "both",alpha = 0.5)
        plt.title(r"$\nabla \cdot u$")
        
        plt.subplot(223)
        region_q[:] = np.clip(region_q,vmin,vmax)
        p3 = plt.contourf(X,Y,region_q[:,:,0].T,200, cmap = cmap,extend = "both")
        plt.colorbar(p3)
        plt.contourf(X,Y,region_q_c[:,:,0].T,20, cmap = cmap_c,extend = "both",alpha = 0.5)
        plt.title(r"$q$")
        
        plt.subplot(224)
        region_omg_g[:] = np.clip(region_omg_g,vmin,vmax)
        p4 = plt.contourf(X,Y,region_omg_g[:,:,0].T,200, cmap = cmap,extend = "both")
        plt.colorbar(p4)
        plt.contourf(X,Y,region_omg_g_c[:,:,0].T,20, cmap = cmap_c,extend = "both",alpha = 0.5)
        plt.title(r"$\omega_g$")
        
        plt.savefig(savePlot/f"90_percent_flux_regions_{k_filt:.2f}_{rank+1}.png")
        plt.close()
        


#* Load the prbability distribution. 
#* Starting from zero, calculate the area under the mean as a function of distance from the origin.
#* 
        

        
        