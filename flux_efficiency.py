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
low_wave = bool(int(sys.argv[-2]))
filt = 3
## ---------------------------------------
sname = {0:"",1:"kh_filt/",2:"kz_filt/",3:"khkz_filt/"}[filt]

N = 384
Np = N// num_process
ro = 0.1*float(sys.argv[-1])
nu = 1e-31
alph = 20
num_slabs = 384
Ns = num_slabs// num_process
n_slab = N//num_slabs

if low_wave:
    factor = 1./0.7
    loadPathdata = pathlib.Path(f"/mnt/lustre/icts_user3/boussinesq/data_final/nu_{nu}_N_{N}/Ro_{ro}/forcedTide_ring_LW")
    savePlot = pathlib.Path(f"/mnt/lustre/icts_user3/boussinesq/Plots/nu_{nu}_N_{N}/Ro_{ro}/forcedTide_ring_LW/filtered_flux")
else: 
    factor = 1.0
    loadPathdata = pathlib.Path(f"/mnt/lustre/icts_user3/boussinesq/data_final/nu_{nu}_N_{N}/Ro_{ro}/forcedTide_ring")
    savePlot = pathlib.Path(f"/mnt/lustre/icts_user3/boussinesq/Plots/nu_{nu}_N_{N}/Ro_{ro}/forcedTide_ring/filtered_flux")

savePath = loadPathdata/f"filtered_flux/{sname}"
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
khint = np.clip(np.round(kh,0),0,N//2)
invlapc = -1/np.where(kc == 0, np.inf, kc**2)
dealias_cos = (abs(kx)<N//3)*(abs(ky)<N//3)*(kzc<(2*N)//3)
lappress = -(kx**2 + ky**2 + alph**2 * kzc**2)
invpress = 1.0/np.where(lappress == 0,  np.inf ,  lappress)* dealias_cos


kx_diff = kx[:Nf,  :, :].copy()
kx_diff[-1, :, :] = -kx_diff[-1, :, :]
ky_diff = np.swapaxes(kx_diff, 0, 1).copy()



## –--------------- empty arrays for mpi communication ----------------- ##
u_temp = np.zeros((Np,N,N))
v_temp = np.zeros((Np,N,N))
b_temp = np.zeros((Np,N,N))
p_temp = np.zeros((Np,N,N))
p = np.zeros((Np,N,N))

rhsu = np.zeros((Np,N,N))
rhsv = np.zeros((Np,N,N))
rhsw = np.zeros((Np,N,N))
b_t = np.zeros((Np,N,N))


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
e_filt = np.zeros_like(b)
p_filt = np.zeros_like(b)

tau = np.zeros((3,3,Np,N,N))
A = np.ones((3,3,Np,N,N))
S = np.zeros((2,2,Np,N,N))
B = np.zeros((3,Np,N,N))
grdB = np.zeros((3,Np,N,N))


pi_h = np.zeros((Np,N,N))
pi_v = np.zeros((Np,N,N))
pi_b = np.zeros((Np,N,N))
pi_ = np.zeros((Np,N,N))

adv_h = np.zeros((Np,N,N))
adv_v = np.zeros((Np,N,N))
adv_b = np.zeros((Np,N,N))
adv_mat = np.zeros((Np,N,N))


pih = np.zeros((N,N,Np))
piv = np.zeros((N,N,Np))
pib = np.zeros((N,N,Np))
pi = np.zeros((N,N,Np))
omgz = np.zeros((N,N,Np))
qz = np.zeros((N,N,Np))
div_hz = np.zeros((N,N,Np))
omg_gz = np.zeros((N,N,Np))
sig_s = np.zeros((Np,N,N))
sig_n = np.zeros((Np,N,N))
sig = np.zeros((Np,N,N))
sigz = np.zeros((N,N,Np))

trtau = np.zeros((Np,N,N))
trS = np.zeros((Np,N,N))
dettau = np.zeros((Np,N,N))
detS = np.zeros((Np,N,N))
t3tau = np.zeros((Np,N,N))
t3A = np.zeros((Np,N,N))
t4tau = np.zeros((Np,N,N))
t4A = np.zeros((Np,N,N))
t1B = np.zeros((Np,N,N))
t1grdB = np.zeros((Np,N,N))
eff_mat = np.zeros((Np,N,N))

t1 = np.zeros((Np,N,N))
t2 = np.zeros((Np,N,N))
t3 = np.zeros((Np,N,N))
t4 = np.zeros((Np,N,N))

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

def pressure(u, b, p):
    ## calculates the pressure term
    rhsu[:] = -ro*(u[0]*diff_x(u[0], u_temp) + u[1]*diff_y(u[0], v_temp) + u[2]*diff_z_cos(u[0], b_temp)) + u[1]
    
    rhsv[:] = -ro*(u[0]*diff_x(u[1], u_temp) + u[1]*diff_y(u[1], v_temp) + u[2]*diff_z_cos(u[1], b_temp)) - u[0]
    
    rhsw[:] = -ro*(u[0]*diff_x(u[2], u_temp) + u[1]*diff_y(u[2], v_temp) + u[2]*diff_z_sin(u[2], b_temp)) + alph**2*b
    
    b_t[:] = -ro*(u[0]*diff_x(b, u_temp) + u[1]*diff_y(b, v_temp) + u[2]*diff_z_sin(b, b_temp)) - u[2]
    
    ## The pressure term
    p_temp[:] =  diff_x(rhsu, u_temp) + diff_y(rhsv, v_temp) + diff_z_sin(rhsw, b_temp)
    pk[:] = invpress  * fft_cos(p_temp, pk)
    p[:] = ifft_cos(pk, p)
    
    return p
    
    
    
    
  
        
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
    
pih_max = 0.4
pih_min = -0.4
piv_max = 0.4
piv_min = -0.4
pib_max = 0.4
pib_min = -0.4
pi_max = 0.4
pi_min = -0.4

k_filts = np.array([20,40,60])

div_h_max = 20/factor
div_h_min = -20/factor

omg_g_max = 20/factor
omg_g_min = -20/factor
    
pih_bins = np.linspace(pih_min,pih_max,601)
piv_bins = np.linspace(piv_min,piv_max,601)
pib_bins = np.linspace(pib_min,pib_max,601)
pi_bins = np.linspace(pi_min,pi_max,601)
div_h_bins = np.linspace(div_h_min,div_h_max,501)
omg_g_bins = np.linspace(omg_g_min,omg_g_max,501)

pih_hist = np.zeros((len(k_filts),len(pih_bins[:-1])))
piv_hist = np.zeros((len(k_filts),len(piv_bins[:-1])))
pib_hist = np.zeros((len(k_filts),len(pib_bins[:-1])))
pi_hist = np.zeros((len(k_filts),len(pi_bins[:-1])))
pih_div_h_hist = np.zeros((len(k_filts),len(div_h_bins[:-1])))
piv_div_h_hist = np.zeros((len(k_filts),len(div_h_bins[:-1]))) 
pib_div_h_hist = np.zeros((len(k_filts),len(div_h_bins[:-1])))
pi_div_h_hist = np.zeros((len(k_filts),len(div_h_bins[:-1])))
pih_omg_g_hist = np.zeros((len(k_filts),len(omg_g_bins[:-1])))
piv_omg_g_hist = np.zeros((len(k_filts),len(omg_g_bins[:-1])))
pib_omg_g_hist = np.zeros((len(k_filts),len(omg_g_bins[:-1])))
pi_omg_g_hist = np.zeros((len(k_filts),len(omg_g_bins[:-1])))

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
# eff_rms = np.zeros(len(k_filts))
pi_rms = np.zeros(len(k_filts))
adv_rms = np.zeros(len(k_filts))
# t = 600.0
times = np.arange(1000,1100.1,1.)
div_omg_g_hist = 0.
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
    p[:] = pressure(u,b,p)
    if rank ==0: print("Calculated")

    # k_filts = np.arange(1,100)

    # k_filts = 1.0*np.array([20,40,60,int(N/(3**0.5)) +1])
    div_omg_g_hist = div_omg_g_hist + np.histogram2d(div_h.ravel(),omg_g.ravel(),bins = [div_h_bins,omg_g_bins])[0]/(N**3) 
    for ii,k_filt in enumerate(k_filts):
        l_filt = TWO_PI/k_filt
        if filt == 0 : 
            cond_c = (kcint< k_filt)
            cond_s = (ksint< k_filt)
        elif filt == 1 :
            cond_c = (khint < k_filt)
            cond_s = (khint < k_filt)
        elif filt == 2 :
            cond_c = (kzc < k_filt)
            cond_s = (kzs < k_filt)
        elif filt == 3 :
            cond_c = (khint < k_filt)*(kzc < k_filt)
            cond_s = (khint < k_filt)*(kzs < k_filt)
        
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
        p_filt[:] = filtered_cos(p,p_filt)

        e_filt[:] = 0.5*(u_filt[0]**2 + u_filt[1]**2 + u_filt[2]**2/alph**2 + b_filt**2)
        # levels = np.linspace(-5,5,200)
        # plt.figure(figsize = (8,6))

        # # dat = np.clip(u_filt[...,N//2].T,None,None) # clipped data between vmin and vmax
        # dat = u_filt[2,...,N//2].T
        # p1 = plt.contourf(X,Y,dat ,200,cmap = "RdYlBu_r",extend = "both")
        # cbar = plt.colorbar(p1)#min = -5,vmax = 5)
        # # cbar.set_clim(-5,5)
        # plt.xlabel(r"$x$")
        # plt.ylabel(r"$y$",rotation = 0)


        # # Calculating the filtered stress and strSin tensor
        

        count = 0.
        
        adv_b[:] = 0.
        adv_h[:] = 0.
        adv_v[:] = 0.
        for i in range(3):
            for j in range(i+1):
                if i < 2:
                    tau[i,j] = -filtered_cos(u[i]*u[j],u_temp) + u_filt[i]*u_filt[j]
                    A[i,j] = (del_cos(j,u_filt[i],u_temp))
                    if i!= j:
                        A[j,i] = (del_cos(i,u_filt[j],v_temp))
                    count = count + 1
                elif i == 2 and j <2:
                    tau[i,j] = -filtered_sin(u[i]*u[j],u_temp) + u_filt[i]*u_filt[j]
                    A[i,j] = (del_cos(j,u_filt[i],u_temp))
                    A[j,i] = (del_cos(i,u_filt[j],v_temp))
                    count = count + 1
                    
                else : 
                    tau[i,j] = -filtered_cos(u[i]*u[j],u_temp) + u_filt[i]*u_filt[j]
                    A[i,j] = (del_sin(j,u_filt[i],u_temp))
                    count = count + 1
                
                tau[j,i] = tau[i,j]
            
            if i<2: 
                # adv_h += ro*u_filt[i]*del_cos(i,e_filt + p_filt/ro,b_temp) 
                adv_h += ro*u_filt[i]*del_cos(i,e_filt ,b_temp) 
                B[i] = -filtered_sin(b*u[i],u_temp) + u_filt[i]* b_filt
                adv_b += - ro*del_sin(i,b_filt*B[i],b_temp)
            else: 
                # adv_v += ro*u_filt[i]*del_cos(i,e_filt+ p_filt/ro,b_temp)
                adv_v += ro*u_filt[i]*del_cos(i,e_filt,b_temp)
                B[i] = -filtered_cos(b*u[i],u_temp) + u_filt[i]* b_filt
                adv_b += - ro*del_cos(i,b_filt*B[i],b_temp)
                
            grdB[i] = del_sin(i,b_filt,u_temp)

        mxtrc = comm.allreduce(np.max(np.abs(A[1,1] + A[2,2]+ A[0,0])),op = MPI.MAX)
        if rank ==0 : print(count, mxtrc)

         # # Calculating different flux components
        #! Add the calculation of the eigen values and the trSces. 
        pi_h[:] = 0.
    
        for i in range(2):
            for j in range(2):
                    S[i,j] = 0.5*(A[i,j] + A[j,i])
                    pi_h[:] = pi_h + tau[i,j]*A[j,i]
                    adv_h += - ro*del_cos(j,tau[i,j]*u_filt[i],b_temp)
                    
        
        pi_v[:] = 0.
        for i in range(2):
            pi_v[:] = pi_v + tau[i,2]*A[i,2] + 1/alph**2 *tau[2,i]*A[2,i]
            adv_v += - ro*del_sin(2,u_filt[i]*tau[i,2],u_temp) - 1/alph**2 *ro*del_cos(i,u_filt[2]*tau[2,i],v_temp)
            
        pi_v[:] = pi_v + 1/alph**2 *tau[2,2]*A[2,2]
        adv_v += - 1/alph**2*ro*del_sin(2,u_filt[2]*tau[2,2],u_temp)
        

        pi_b[:] = 0.
        for i in range(3):
            pi_b[:] = pi_b + B[i]*grdB[i]
        
        pi_[:] = ro*(pi_h + pi_v + pi_b)
        pi_rms[ii] = pi_rms[ii] + np.sum(pi_**2)/(N**3)
        adv_mat[:] = (adv_h + adv_v + adv_b) 
        adv_rms[ii] = adv_rms[ii] + np.sum(adv_mat**2)/(N**3)
        
        

for ii, k_filt in enumerate(k_filts):
    # eff_rms[ii] = comm.allreduce(eff_rms[ii],op = MPI.SUM)/len(times)
    pi_rms[ii] = comm.allreduce(pi_rms[ii],op = MPI.SUM)/len(times)
    adv_rms[ii] = comm.allreduce(adv_rms[ii],op = MPI.SUM)/len(times)
    if rank ==0 : print(f"k_filt {k_filt} eff_rms",(pi_rms[ii]/adv_rms[ii])**0.5)
eff_rms = (pi_rms/adv_rms)**0.5

if rank ==0:
    np.savez_compressed(savePath/f"flux_efficiency_wo_pressure.npz",eff_rms = eff_rms,k_filts = k_filts)
    # np.savez_compressed(savePath/f"flux_efficiency_new.npz",eff_rms = eff_rms,k_filts = k_filts)