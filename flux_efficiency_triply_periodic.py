import numpy as np
from scipy.fft import fftn,ifftn, fft ,  ifft ,  irfft2 ,  rfft2 , irfftn ,  rfftn, fftfreq, dst, dct, idst, idct, rfft,  irfft
import matplotlib.pyplot as plt
from mpi4py import MPI
import pathlib,os,sys,h5py
import matplotlib as mpl 
import matplotlib.colors as colors
mpl.rc('text', usetex = True)
mpl.rcParams['font.size'] = 25

## ---------------MPI things--------------
comm = MPI.COMM_WORLD
num_process =  comm.Get_size()
rank = comm.Get_rank()
# low_wave = bool(int(sys.argv[-2]))
low_wave = False
filt = 0
## ---------------------------------------
sname = {0:"",1:"kh_filt/",2:"kz_filt/",3:"khkz_filt/"}[filt]

N = 384
iso = True
Np = N// num_process
# ro = 0.1*float(sys.argv[-1])
ro = 1
ro = ro if not iso else 1.0
lp = 8 # Hyperviscosity power
if iso:
    # nu0 = 8.192 #! Viscosity for N = 1
    # nu0 = 4.714 #! Viscosity from Pope's 256 run 
    nu0 = 0.8 #! Viscosity for N = 1
    m = 2.0 #! Desired kmax*eta
    nu = nu0*(3*m/N)**(2*(lp - 1/3))  #? scaling with resolution. For 512, nu = 0.002 #! Need to add scaling for hyperviscosity
alph = 1 if iso else 20 
num_slabs = 384
Ns = num_slabs// num_process
n_slab = N//num_slabs


# if iso:
#     isforcing = True
#     if N!= 1024:
#         loadPathdata = pathlib.Path(f"/mnt/pfs/rajarshi.chattopadhyay/codes/HIT_3D/data/forced_{isforcing}/N_{N}_Re_{1/nu:.1f}")
#     elif N == 1024:
#         loadPathdata = pathlib.Path(f"/mnt/pfs/ritwik.mukherjee/HIT_3D/data/forced_True/N_1024_Re_1259.9/")
#         nu = 1/1259.9
#         lp = 1
#     savePlot = pathlib.Path(f"/mnt/pfs/rajarshi.chattopadhyay/codes/boussinesq/Plots/nu_{nu}_N_{N}/Ro_{ro}/isotropic_N_{N}_Re_{1/nu:.1f}/filtered_flux")
    
    
# else:
#     if low_wave:
#         factor = 1./0.7
#         loadPathdata = pathlib.Path(f"/mnt/pfs/rajarshi.chattopadhyay/codes/boussinesq/data_final/nu_{nu}_N_{N}/Ro_{ro}/forcedTide_ring_LW")
#         savePlot = pathlib.Path(f"/mnt/pfs/rajarshi.chattopadhyay/codes/boussinesq/Plots/nu_{nu}_N_{N}/Ro_{ro}/forcedTide_ring_LW/filtered_flux")
#     else: 
#         factor = 1.0
#         loadPathdata = pathlib.Path(f"/mnt/pfs/rajarshi.chattopadhyay/codes/boussinesq/data_final/nu_{nu}_N_{N}/Ro_{ro}/forcedTide_ring")
#         savePlot = pathlib.Path(f"/mnt/pfs/rajarshi.chattopadhyay/codes/boussinesq/Plots/nu_{nu}_N_{N}/Ro_{ro}/forcedTide_ring/filtered_flux")


loadPathdata = pathlib.Path(f"/mnt/pfs/rajarshi.chattopadhyay/codes/boussinesq/data/forced_True/N_{N}_Re_{1/nu:.1f}")
savePlot = pathlib.Path(f"/mnt/pfs/rajarshi.chattopadhyay/codes/boussinesq/Plots/nu_{nu}_N_{N}/Ro_{ro}/isotropic_N_{N}_Re_{1/nu:.1f}/filtered_flux")
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
cond_ky = np.abs(np.round(Ky))<=N//3
cond_kz = np.abs(np.round(Kz))<=N//3
## -----------------------------------------------




## --------- kx and ky for differentiation ---------    
kx_diff = np.moveaxis(kz,[0,1,2],[2,1,0]).copy()
ky_diff = np.swapaxes(kx_diff, 0, 1).copy()
kz_diff = np.moveaxis(kz, [0,1], [1,0]).copy()

lap = -(kx**2 + ky**2 + kz**2 )
k = (-lap)**0.5
kint = np.clip(np.round(k,0).astype(int),None,N//2)
# kh = (kx**2 + ky**2)**0.5
dealias = kint<=N/3 #! Spherical dealiasing
# dealias = (abs(kx)<N//3)*(abs(ky)<N//3)*(abs(kz)<N//3)
invlap = dealias/np.where(lap == 0, np.inf,  lap)

kh = (kx**2 + ky**2 )**0.5
khint = np.clip(np.round(kh,0),0,N//2)


lappress = -(kx**2 + ky**2 + alph**2 * kz**2)
invpress = 1.0/np.where(lappress == 0,  np.inf ,  lappress)* dealias


kx_diff = np.moveaxis(kz,[0,1,2],[2,1,0]).copy()
ky_diff = np.swapaxes(kx_diff, 0, 1).copy()
kz_diff = np.moveaxis(kz, [0,1], [1,0]).copy()


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


pk = np.zeros((N,Np,Nf),dtype = np.complex128)



arr_temp_r = np.zeros((Np, N, N),dtype = np.float64)
arr_temp_k = np.zeros((N, Np, N),dtype= np.float64)
arr_temp_fr = np.zeros((Np, N, Nf), dtype= np.complex128)      
arr_temp_ifr = np.zeros((N, Np, Nf),dtype= np.complex128)      
arr_mpi = np.zeros((num_process,  Np,  Np, Nf), dtype= np.complex128)
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

def del_i(i,u,u_i):
    if i ==0 : 
        return diff_x(u,u_i)
    elif i ==1 : 
        return diff_y(u,u_i)
    elif i == 2: 
        return diff_z(u,u_i)
    


def pressure(u, b, p):
    ## calculates the pressure term
    rhsu[:] = -ro*(u[0]*diff_x(u[0], u_temp) + u[1]*diff_y(u[0], v_temp) + u[2]*diff_z(u[0], b_temp)) + u[1]
    
    rhsv[:] = -ro*(u[0]*diff_x(u[1], u_temp) + u[1]*diff_y(u[1], v_temp) + u[2]*diff_z(u[1], b_temp)) - u[0]*(not iso)
    
    rhsw[:] = -ro*(u[0]*diff_x(u[2], u_temp) + u[1]*diff_y(u[2], v_temp) + u[2]*diff_z(u[2], b_temp)) + alph**2*b
    
    b_t[:] = -ro*(u[0]*diff_x(b, u_temp) + u[1]*diff_y(b, v_temp) + u[2]*diff_z(b, b_temp)) - u[2]*(not iso)
    
    ## The pressure term
    p_temp[:] =  diff_x(rhsu, u_temp) + diff_y(rhsv, v_temp) + diff_z(rhsw, b_temp)
    pk[:] = invpress  * rfft_mpi(p_temp, pk)
    p[:] = irfft_mpi(pk, p)
    
    return p
    
    

def load_trunc(x):
    x1 = np.zeros((*x.shape[:-2],N,Nf),dtype = np.complex128)
    x1[...,cond_ky,:x.shape[-1]] = x.copy()
    return irfftn(x1,(N,N), axes = (-2,-1))   
    
def load_npz(paths,u,b):
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
            Field = np.load(paths/f"Fields_cmp_{slab}.npz")
        slab_old = slab
        u[0,lidx] = load_trunc(Field['u'][idx])
        u[1,lidx] = load_trunc(Field['v'][idx])
        u[2,lidx] = load_trunc(Field['w'][idx])
        if not iso:
            b[idx] = load_trunc(Field['b'][idx])
        else: 
            b[idx] = 0.0

        
        
        """Loading the OG data"""
        # if slab_old != slab:  Field = np.load(paths/f"Fields_{slab}.npz")
        # slab_old = slab
        # u[0,lidx] = Field['u'][idx]
        # u[1,lidx] = Field['v'][idx]
        # u[2,lidx] = Field['w'][idx]
        
    return u,b
        
# k_filts = np.array([20,40,60])
k_filts = np.arange(1,N//3+1)
    
# pih_max = 0.4
# pih_min = -0.4
# piv_max = 0.4
# piv_min = -0.4
# pib_max = 0.4
# pib_min = -0.4
# pi_max = 0.4
# pi_min = -0.4


# div_h_max = 20/factor
# div_h_min = -20/factor

# omg_g_max = 20/factor
# omg_g_min = -20/factor
    
# pih_bins = np.linspace(pih_min,pih_max,601)
# piv_bins = np.linspace(piv_min,piv_max,601)
# pib_bins = np.linspace(pib_min,pib_max,601)
# pi_bins = np.linspace(pi_min,pi_max,601)
# div_h_bins = np.linspace(div_h_min,div_h_max,501)
# omg_g_bins = np.linspace(omg_g_min,omg_g_max,501)

# pih_hist = np.zeros((len(k_filts),len(pih_bins[:-1])))
# piv_hist = np.zeros((len(k_filts),len(piv_bins[:-1])))
# pib_hist = np.zeros((len(k_filts),len(pib_bins[:-1])))
# pi_hist = np.zeros((len(k_filts),len(pi_bins[:-1])))
# pih_div_h_hist = np.zeros((len(k_filts),len(div_h_bins[:-1])))
# piv_div_h_hist = np.zeros((len(k_filts),len(div_h_bins[:-1]))) 
# pib_div_h_hist = np.zeros((len(k_filts),len(div_h_bins[:-1])))
# pi_div_h_hist = np.zeros((len(k_filts),len(div_h_bins[:-1])))
# pih_omg_g_hist = np.zeros((len(k_filts),len(omg_g_bins[:-1])))
# piv_omg_g_hist = np.zeros((len(k_filts),len(omg_g_bins[:-1])))
# pib_omg_g_hist = np.zeros((len(k_filts),len(omg_g_bins[:-1])))
# pi_omg_g_hist = np.zeros((len(k_filts),len(omg_g_bins[:-1])))

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
times = np.arange(15,15.1,1.)
div_omg_g_hist = 0.
for jj,t in enumerate(times):
    if rank ==0: print(t, num_process,Ns)
    loadPath = loadPathdata/f"time_{t:.1f}"
    u,b = load_npz(loadPath,u,b) 
    # for qq in range(Ns):
    #     Fields = np.load(loadPath/f"Fields_{(rank*Ns + qq):.0f}.npz")
    #     u[0,qq*n_slab:(qq + 1)*n_slab]  = Fields["u"]
    #     u[1,qq*n_slab:(qq + 1)*n_slab]  = Fields["v"]
    #     u[2,qq*n_slab:(qq + 1)*n_slab]  = Fields["w"]
    #     b[qq*n_slab:(qq + 1)*n_slab]  = Fields["b"]
    div_h[:] = diff_x(u[0],u_temp) + diff_y(u[1],v_temp) + diff_z(u[2],b_temp)
    divmax = comm.allreduce(np.max(np.abs(div_h)),op = MPI.MAX)

    # raise SystemExit(f"Loaded the data max div: {divmax}")
    comm.Barrier()
    q[:] = diff_x(u[1],v_temp) - diff_y(u[0],u_temp) + diff_z(b,b_temp)
    omg[:] = diff_x(u[1],v_temp) - diff_y(u[0],u_temp)
    sig_s[:] = diff_x(u[1],u_temp) + diff_y(u[0],v_temp)
    div_h[:] = diff_x(u[0],u_temp) + diff_y(u[1],v_temp)
    sig_n[:] = diff_x(u[0],u_temp) - diff_y(u[1],v_temp)
    sig[:] = (sig_s**2 + sig_n**2)**0.5
    omg_g[:] = -irfft_mpi(kh**2*invlap*rfft_mpi(q,pk),omg_g)
    p[:] = pressure(u,b,p)
    if rank ==0: print("Calculated")

    # k_filts = np.arange(1,100)

    # k_filts = 1.0*np.array([20,40,60,int(N/(3**0.5)) +1])
    # div_omg_g_hist = div_omg_g_hist + np.histogram2d(div_h.ravel(),omg_g.ravel(),bins = [div_h_bins,omg_g_bins])[0]/(N**3) 
    for ii,k_filt in enumerate(k_filts):
        l_filt = TWO_PI/k_filt
        if filt == 0 : 
            cond = (kint< k_filt)
            # cond = (ksint< k_filt)
        elif filt == 1 :
            cond = (khint < k_filt)
            # cond = (khint < k_filt)
        elif filt == 2 :
            cond = (kz < k_filt)
            # cond = (kzs < k_filt)
        elif filt == 3 :
            cond = (khint < k_filt)*(kz < k_filt)
            # cond = (khint < k_filt)*(kzs < k_filt)
        
        def filtered(u,uf):
            uf[:] = irfft_mpi(rfft_mpi(u,pk)*cond,uf)
            return uf

        u_filt[0] = filtered(u[0],u_filt[0])
        u_filt[1] = filtered(u[1],u_filt[1])
        u_filt[2] = filtered(u[2],u_filt[2])
        b_filt[:] = filtered(b,b_filt)
        p_filt[:] = filtered(p,p_filt)

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
                    tau[i,j] = -filtered(u[i]*u[j],u_temp) + u_filt[i]*u_filt[j]
                    A[i,j] = (del_i(j,u_filt[i],u_temp))
                    if i!= j:
                        A[j,i] = (del_i(i,u_filt[j],v_temp))
                    count = count + 1
                elif i == 2 and j <2:
                    tau[i,j] = -filtered(u[i]*u[j],u_temp) + u_filt[i]*u_filt[j]
                    A[i,j] = (del_i(j,u_filt[i],u_temp))
                    A[j,i] = (del_i(i,u_filt[j],v_temp))
                    count = count + 1
                    
                else : 
                    tau[i,j] = -filtered(u[i]*u[j],u_temp) + u_filt[i]*u_filt[j]
                    A[i,j] = (del_i(j,u_filt[i],u_temp))
                    count = count + 1
                
                tau[j,i] = tau[i,j]
            
            if i<2: 
                # adv_h += ro*u_filt[i]*del_i(i,e_filt + p_filt/ro,b_temp) 
                adv_h += ro*u_filt[i]*del_i(i,e_filt ,b_temp) 
                B[i] = -filtered(b*u[i],u_temp) + u_filt[i]* b_filt
                adv_b += - ro*del_i(i,b_filt*B[i],b_temp)
            else: 
                # adv_v += ro*u_filt[i]*del_i(i,e_filt+ p_filt/ro,b_temp)
                adv_v += ro*u_filt[i]*del_i(i,e_filt,b_temp)
                B[i] = -filtered(b*u[i],u_temp) + u_filt[i]* b_filt
                adv_b += - ro*del_i(i,b_filt*B[i],b_temp)
                
            grdB[i] = del_i(i,b_filt,u_temp)

        mxtrc = comm.allreduce(np.max(np.abs(A[1,1] + A[2,2]+ A[0,0])),op = MPI.MAX)
        if rank ==0 : print(count, mxtrc)

         # # Calculating different flux components
        #! Add the calculation of the eigen values and the trSces. 
        pi_h[:] = 0.
    
        for i in range(2):
            for j in range(2):
                    S[i,j] = 0.5*(A[i,j] + A[j,i])
                    pi_h[:] = pi_h + tau[i,j]*A[j,i]
                    adv_h += - ro*del_i(j,tau[i,j]*u_filt[i],b_temp)
                    
        
        pi_v[:] = 0.
        for i in range(2):
            pi_v[:] = pi_v + tau[i,2]*A[i,2] + 1/alph**2 *tau[2,i]*A[2,i]
            adv_v += - ro*del_i(2,u_filt[i]*tau[i,2],u_temp) - 1/alph**2 *ro*del_i(i,u_filt[2]*tau[2,i],v_temp)
            
        pi_v[:] = pi_v + 1/alph**2 *tau[2,2]*A[2,2]
        adv_v += - 1/alph**2*ro*del_i(2,u_filt[2]*tau[2,2],u_temp)
        

        pi_b[:] = 0.
        for i in range(3):
            pi_b[:] = pi_b + B[i]*grdB[i]
        
        pi_[:] = ro*(pi_h + pi_v + pi_b)
        tflux[ii] += np.mean(pi_)/len(times)
        pi_rms[ii] = pi_rms[ii] + np.sum(pi_**2)/(N**3)
        adv_mat[:] = (adv_h + adv_v + adv_b)
        adv_rms[ii] = adv_rms[ii] + np.sum(adv_mat**2)/(N**3)
        
        

for ii, k_filt in enumerate(k_filts):
    # eff_rms[ii] = comm.allreduce(eff_rms[ii],op = MPI.SUM)/len(times)
    pi_rms[ii] = comm.allreduce(pi_rms[ii],op = MPI.SUM)/len(times)
    adv_rms[ii] = comm.allreduce(adv_rms[ii],op = MPI.SUM)/len(times)
    if rank ==0 : print(f"k_filt {k_filt} eff_rms",(pi_rms[ii]/adv_rms[ii])**0.5)
eff_rms = (pi_rms/adv_rms)**0.5

def h5data(f,dsetname,data):
    if dsetname in f:
        del dsetname
    f.create_dataset(dsetname, data = data, dtype = np.float64, compression = "gzip")
    
if rank ==0:
    print(eff_rms)
    with h5py.File(f"/mnt/pfs/rajarshi.chattopadhyay/codes/boussinesq/efficiencies.hdf5", "a") as f:
        
        try: 
            grp = f.create_group(f"N_{N}_ro_{ro}_Re_{1/nu:.1f}")
        
        except ValueError: pass
        grp = f[f"N_{N}_ro_{ro}_Re_{1/nu:.1f}"]
            
        h5data(grp,"eff_rms", eff_rms)
        h5data(grp,"k_filts", k_filts)
        h5data(grp,"mean_flux", tflux)
    # np.savez_compressed(savePath/f"flux_efficiency_wo_pressure.npz",eff_rms = eff_rms,k_filts = k_filts)
    # np.savez_compressed(savePath/f"flux_efficiency_new.npz",eff_rms = eff_rms,k_filts = k_filts)