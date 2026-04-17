#%%
import numpy as np
from scipy.fft import fftn,ifftn, fft ,  ifft ,  irfft2 ,  rfft2 , irfftn ,  rfftn, fftfreq, dst, dct, idst, idct, rfft,  irfft
import matplotlib.pyplot as plt
from mpi4py import MPI
import pathlib,os,sys,h5py
import matplotlib as mpl 
import matplotlib.colors as colors
mpl.rc('text', usetex = True)
mpl.rcParams['font.size'] = 25
#%%
## ---------------MPI things--------------
comm = MPI.COMM_WORLD
num_process =  comm.Get_size()
rank = comm.Get_rank()
# low_wave = bool(int(sys.argv[-2]))
low_wave = False
filt =0
## ---------------------------------------
sname = {0:"",1:"kh_filt/",2:"kz_filt/",3:"khkz_filt/"}[filt]

N = 128
kend_pow = 9 if N == 1024 else 8 if N == 512 else 7
iso = False
Np = N// num_process
# ro = 0.1*float(sys.argv[-1])
ro = 1
ro = ro if not iso else 1.0
lp = 8 # Hyperviscosity power


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
kvec = np.array([kx,ky,kz])
cond_ky = np.abs(np.round(Ky))<=N//3
cond_kz = np.abs(np.round(Kz))<=N//3
## -----------------------------------------------




## --------- kx and ky for differentiation ---------    
kx_diff = np.moveaxis(kz,[0,1,2],[2,1,0]).copy()
ky_diff = np.swapaxes(kx_diff,0, 1).copy()
kz_diff = np.moveaxis(kz, [0,1], [1,0]).copy()

lap = -(kx**2 + ky**2 + kz**2 )
k = (-lap)**0.5
kint = np.clip(np.round(k,0).astype(int),None,N//2)
# kh = (kx**2 + ky**2)**0.5
dealias = kint<=N/3 #! Spherical dealiasing
# dealias = (abs(kx)<N//3)*(abs(ky)<N//3)*(abs(kz)<N//3)
invlap = dealias/np.where(lap ==0, np.inf,  lap)

kh = (kx**2 + ky**2 )**0.5
khint = np.clip(np.round(kh,0),0,N//2)


# lappress = -(kx**2 + ky**2 + alph**2 * kz**2)
# invpress = 1.0/np.where(lappress ==0,  np.inf ,  lappress)* dealias


## –--------------- empty arrays for mpi communication ----------------- ##
u_temp = np.zeros((Np,N,N))
v_temp = np.zeros((Np,N,N))
b_temp = np.zeros((Np,N,N))
p_temp = np.zeros((Np,N,N))
p = np.zeros((Np,N,N))
Pi = p.copy()
eff = p.copy()
rhsu = np.zeros((Np,N,N))
rhsv = np.zeros((Np,N,N))
rhsw = np.zeros((Np,N,N))
b_t = np.zeros((Np,N,N))


pk = np.zeros((N,Np,Nf),dtype = np.complex128)
uk = np.zeros((3,N,Np,Nf),dtype = np.complex128)
Ak = np.zeros((3,3,N,Np,Nf),dtype = np.complex128)
Sk = Ak.copy()
rhsuk = np.empty_like(pk)
rhsvk = rhsuk.copy()
rhswk = rhsuk.copy()

arr_temp_r = np.zeros((Np, N, N),dtype = np.float64)
arr_temp_k = np.zeros((N, Np, N),dtype= np.float64)
arr_temp_fr = np.zeros((Np, N, Nf), dtype= np.complex128)      
arr_temp_ifr = np.zeros((N, Np, Nf),dtype= np.complex128)      
arr_mpi = np.zeros((num_process,  Np,  Np, Nf), dtype= np.complex128)
arr_mpi_r = np.zeros((num_process,  Np,  Np, N), dtype= np.float64)




tau_filt = np.zeros((3,3,Np,N,N))
S_filt = np.zeros((3,3,Np,N,N))

t1 = np.zeros((Np,N,N))
t2 = np.zeros((Np,N,N))
t3 = np.zeros((Np,N,N))
t4 = np.zeros((Np,N,N))

## --------------------------------------------------------------------- ##
def rfft_mpi(u, fu):
    arr_temp_fr[:] = rfft2(u,  axes=(1, 2))
    arr_mpi[:] = np.swapaxes(np.reshape(arr_temp_fr, (Np,  num_process,  Np, Nf)),0, 1)
    comm.Alltoall([arr_mpi,  MPI.DOUBLE_COMPLEX], [fu,  MPI.DOUBLE_COMPLEX])
    fu[:] = fft(fu, axis =0)
    return fu

def irfft_mpi(fu, u):
    arr_temp_ifr[:] = ifft(fu,  axis =0)
    comm.Alltoall([arr_temp_ifr,  MPI.DOUBLE_COMPLEX], [arr_mpi, MPI.DOUBLE_COMPLEX])
    arr_temp_fr[:] = np.reshape(np.swapaxes(arr_mpi, 0, 1), (Np,  N,  Nf))
    u[:] = irfft2(arr_temp_fr, (N, N), axes = (1, 2))
    return u    


def diff_x(u,  u_x):
    arr_mpi_r[:] = np.moveaxis(np.reshape(u, (Np,  num_process,  Np,  N)),[0,1], [1,0])
    comm.Alltoall([arr_mpi_r,  MPI.DOUBLE], [arr_temp_k,  MPI.DOUBLE])
    arr_temp_k[:] = irfft(1j * kx_diff*rfft(arr_temp_k,  axis =0), N,  axis=0)
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



def filtered(a,af,filt_k):
    """ Calculates the filtered field given a particular filter """
    af[:] = irfft_mpi(rfft_mpi(a,pk)*filt_k,af)
    return af


def energy_flux(u,Sk,filt_k):
    """Calculates the 3D energy flux in physical space"""    
    for i in range(3):
        for j in range(i,3):
            tau_filt[i,j] = tau(u[i],u[j],filt_k)
            
            S_filt[i,j] = irfft_mpi(Sk[i,j]*filt_k,S_filt[i,j])
        
        if j> i: 
            S_filt[j,i] = S_filt[i,j].copy()
            tau_filt[j,i] = tau_filt[i,j].copy()
    
    
    Pi[:] = -np.einsum('ij...,ji...->...',tau_filt,S_filt)
    eff[:] = Pi/(np.einsum('ij...,ij...->...',tau_filt, tau_filt) *np.einsum('ij...,ij...->...',S_filt, S_filt))**0.5
    return Pi,eff


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
            b[lidx] = load_trunc(Field['b'][idx])
        else: 
            b[lidx] =0.0

        
        
        """Loading the OG data"""
        # if slab_old != slab:  Field = np.load(paths/f"Fields_{slab}.npz")
        # slab_old = slab
        # u[0,lidx] = Field['u'][idx]
        # u[1,lidx] = Field['v'][idx]
        # u[2,lidx] = Field['w'][idx]
        
    return u,b


def h5data(f,dsetname,data):
    if dsetname in f:
        del dsetname
    f.create_dataset(dsetname, data = data, dtype = np.float64, compression = "gzip")
        
# k_filts = np.array([20,40,60])

def calculate_efficiency(times,lasttrue = False, rot_strat = ''):
    # raise SystemExit

    for jj,t in enumerate(times):
        if rank ==0: print(t, num_process,Ns)
        if not lasttrue: loadPath = loadPathdata/f"time_{t:.1f}"
        else: loadPath = loadPathdata/f"last"
        u[:],b[:] = load_npz(loadPath,u,b)  
        
        for i in range(3):
            uk[i] = rfft_mpi(u[i],uk[i])
            
        divmax = comm.allreduce(np.abs(1j*np.einsum('i...,i...->...',kvec,uk)).max(),op = MPI.MAX)
        if rank == 0: print(f"Max div u : {divmax/N**3}")
        Ak[:] =1j* np.einsum('i...,j...->ij...',uk,kvec)
        Sk[:] =0.5*(Ak + np.moveaxis(Ak,[0,1],[1,0]))
        
        k_filts = 1.0*np.round(np.logspace(2,kend_pow,20, base = 2))
        mean_eff = np.zeros(len(k_filts))
        mean_flux = np.zeros(len(k_filts))
        for fidx,k_filt in enumerate(k_filts):
            if rank ==0 : print(f"k_filt : {k_filt}")
            #* Filter should vanish for l = inf
            l_filt = TWO_PI/k_filt if k_filt !=0 else np.inf
            #? Gaussian filter
            # filt_k =  np.exp(-k**2/(2.0*k_filt**2)) if k_filt !=0 else 0.0
            #? Cutoff filter
            filt_k = np.where(np.round(k)< k_filt, 1.0, 0.0)
            
            Pi[:],eff[:] = energy_flux(u,Sk,filt_k)
            mean_eff[fidx] = comm.allreduce(np.sum(eff), op = MPI.SUM)/N**3
            mean_flux[fidx] = comm.allreduce(np.sum(Pi), op = MPI.SUM)/N**3
        
        if rank ==0 : print(f"Time {t}, k : {k_filts}, eff : {mean_eff}, flux : {mean_flux}")
        


    if rank ==0:
        with h5py.File(f"/mnt/pfs/rajarshi.chattopadhyay/codes/boussinesq/{rot_strat}spectral_efficiencies.hdf5", "a") as f:
            if rot_strat == '':
                grpname = f"N_{N}_ro_{ro}_Re_{1/nu:.1f}"
            else: 
                grpname = f"N_{N}_f_{fcorr}_Nb_{Nb}_Re_{1/nu:.1f}"
                
            if grpname in f:
                grp = f[grpname]
            else :
                grp = f.create_group(grpname)
                
            # h5data(grp,"eff_rms", eff_rms)
            h5data(grp, "eff_mean",mean_eff)
            h5data(grp, "flux_mean",mean_flux)
            h5data(grp,"k_filts", k_filts)



#%%
if iso:
    # nu0 = 8.192 #! Viscosity for N = 1
    # nu0 = 4.714 #! Viscosity from Pope's 256 run 
    # nu0 = 0.8 #! Viscosity for N = 1
    # m = 1.0 #! Desired kmax*eta
    # nu = nu0*(3*m/N)**(2*(lp - 1/3))
    if N == 512: re = 509796569554719017322718167040.0
    elif N == 384: re = 6189700196426916767658409984.0
    elif N == 1024: re = 1259.9
    elif N == 256: re = 184.3
    nu = 1/re  #? scaling with resolution. For 512, nu = 0.002 #! Need to add scaling for hyperviscosity
    
    isforcing = True
    if N!= 1024:
        loadPathdata = pathlib.Path(f"/mnt/pfs/rajarshi.chattopadhyay/codes/HIT_3D/data/forced_{isforcing}/N_{N}_Re_{1/nu:.1f}")
    elif N == 1024:
        loadPathdata = pathlib.Path(f"/mnt/pfs/ritwik.mukherjee/HIT_3D/data/forced_True/N_1024_Re_1259.9/")
        nu = 1/1259.9
        lp = 1
    savePlot = pathlib.Path(f"/mnt/pfs/rajarshi.chattopadhyay/codes/boussinesq/Plots/nu_{nu}_N_{N}/Ro_{ro}/isotropic_N_{N}_Re_{1/nu:.1f}/filtered_flux")
    
    tlast = os.listdir(loadPathdata)
    if "last" in tlast: 
        times = [-1]
        lasttrue = True
        num_slabs = len([i for i in os.listdir(loadPathdata/f"last/") if "Fields_" in i ])

    else:
        times = [float(i.split("time_")[1]) for i in tlast if "time_" in i]
        times.sort()
        lasttrue = False
        times = [times[-1]]
        
        num_slabs = len([i for i in os.listdir(loadPathdata/f"time_{times[-1]:.1f}") if "Fields_" in i ])
        
    Ns = num_slabs// num_process
    n_slab = N//num_slabs
    if rank ==0: print(num_slabs)

    savePath = loadPathdata/f"spectral_filtered_flux/{sname}"
    try: savePlot.mkdir(parents = True,exist_ok = True)
    except FileExistsError: pass
    try: savePath.mkdir(parents = True,exist_ok = True)
    except FileExistsError: pass
    if rank ==0: print("saved in " ,str(savePath))
    calculate_efficiency(times,lasttrue = lasttrue, rot_strat = '')
        
        
    
    
    
else: 
    paths = open("/mnt/pfs/rajarshi.chattopadhyay/codes/boussinesq/lastpaths.txt","r")
    pathlist = [i.split("\n")[0] for i in paths.readlines() if "/mnt" in i]
    Nbs = [float(i.split("Nb_")[1].split("/")[0]) for i in pathlist]
    fcorrs = [float(i.split("f_")[1].split("_Nb_")[0]) for i in pathlist]
    N_s = [int(i.split("N_")[1].split("_Re_")[0]) for i in pathlist]
    Res = [float(i.split("Re_")[1]) for i in pathlist]
    nus = 1/np.array(Res)
    
    isforcing = True
    for i,path in enumerate(pathlist): 
        Ndat = N_s[i]
        if Ndat != N: continue
        nu = nus[i]
        re = Res[i]
        fcorr = fcorrs[i]
        Nb = Nbs[i]
        
        if rank ==0 :print(f'Iteration {i+1} with same resolution')
        loadPathdata = pathlib.Path(path)
        savePlot = pathlib.Path(f"/mnt/pfs/rajarshi.chattopadhyay/codes/boussinesq/Plots/nu_{nu}_N_{N}/f_{fcorr}_Nb_{Nb}/filtered_flux")
        
        tlast = os.listdir(loadPathdata)
        
        if "last" in tlast: 
            times = [-1]
            lasttrue = True
            num_slabs = len([i for i in os.listdir(loadPathdata/f"last/") if "Fields_" in i ])

        else:
            times = [float(i.split("time_")[1]) for i in tlast if "time_" in i]
            times.sort()
            lasttrue = False
            times = [times[-1]]
            
            num_slabs = len([i for i in os.listdir(loadPathdata/f"time_{times[-1]:.1f}") if "Fields_" in i ])
            
        Ns = num_slabs// num_process
        n_slab = N//num_slabs
        if rank ==0: print(num_slabs)

        savePath = loadPathdata/f"spectral_filtered_flux/{sname}"
        try: savePlot.mkdir(parents = True,exist_ok = True)
        except FileExistsError: pass
        try: savePath.mkdir(parents = True,exist_ok = True)
        except FileExistsError: pass
        if rank ==0: print("saved in " ,str(savePath))
        calculate_efficiency(times,lasttrue = lasttrue, rot_strat = 'rot_strat')

# if loadPathdata.exists() is False: print(loadPathdata,"\n", "/mnt/pfs/rajarshi.chattopadhyay/codes/HIT_3D/data/forced_True/N_256_Re_184.3")

#%%    

# loadPathdata.exists(),str(loadPath)
# len(os.listdir(loadPath))

