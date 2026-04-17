import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftfreq,fftn, ifftn,dct,dst
# from pyfftw.interfaces.scipy_fft import fftfreq,fftn, ifftn,dct,dst
import pathlib
import matplotlib as mpl 
mpl.rc('text', usetex = True)
import os
from mpi4py import MPI
# from pyevtk.hl import imageToVTK

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

## ------------ Paths --------------------
loadPath =   pathlib.Path(f"/mnt/pfs/rajarshi.chattopadhyay/boussinesq/spectrum-development/nu_{nu}_N_{N}/Ro_{ro}/forcedTide_ring_{omega:.2f}")
savePath = curr_path/f"Plots/nu_{nu}_N_{N}/Ro_{ro}/forcedTide_ring_{omega:.2f}"
savePath.mkdir(parents=True,  exist_ok=True)
## ---------------------------------------

## ---------- rest of the parameters --------------
# paramfile = (loadPath/f"params.txt")
# if paramfile.exists():
#     ## ---------- Beginning from existing data -----------
#     """Load the parameters in the param file"""
#     with open((loadPath/f"params.txt"),"r") as param_file:
#         param = eval(param_file.read()) 
#     times = sorted([float(str(x).split("time_")[-1]) for x in (loadPath/f"E_k").iterdir() if "time_" in str(x)])
#     num_slabs = len([x for x in (loadPath/f"E_k/time_100.0").iterdir() if "e_" in str(x)])

#     # print(num_slabs)
#     ro = param["Rossby"]
#     lp = param["hyperviscous"] 
#     alph = param["Alpha"] 
#     T = param["Final_time"]
#     dt = param["time_step"]
#     st = param["interval of saving indices"] 
    
#     del paramfile,param_file,param
#     # print(times)
    
# else: 
#     raise ValueError("No data file found!")
# if rank == 0:
#     print(num_slabs)
#     print(len(times))



# times = np.arange(0,1001,5)
num_slabs = 96
Ns = N//num_slabs
# Ntimes = len(times)
# times_o = np.copy(times)
# time_range = Ntimes//num_process
# # print(f"times_range Rank: {time_range}")
# if rank ==  num_process -1 : 
#     times = times[rank*(time_range):]
# else :
#     times = times[rank*(time_range):(rank+1)*time_range]
# times = [rank*2]
# print(f"rank {rank} : {times}")
## ------------------------------------------------

## ------------------ Grid ------------------------
Nf = N//2 + 1
PI = np.pi
TWO_PI = 2*PI
L = TWO_PI
Lz = PI
X = Y = np.linspace(0, L, N, endpoint= False)
Z = np.linspace(0+ Lz/(2*N), Lz + Lz/(2*N), N, endpoint= False)
dx = X[1] - X[0]
dy = Y[1] - Y[0]
dz = Z[1] - Z[0]
Kx = Ky = fftfreq(N,  1./N)*TWO_PI/L
Kzc = np.arange(N)* PI/Lz

Kz = Kx
# Kz = Kzc.copy()
# Kz[N//2:] = Kz[N//2: ] - N
kx,  ky,  kz = np.meshgrid(Kx,  Ky,  Kz ,  indexing = 'ij')
K = np.arange(Nf)
k = (kx**2 + ky**2 + kz**2 )**0.5
kh = np.mean((kx**2 + ky**2)**0.5,axis=2)
dkx = Kx[1] -Kx[0]
dky = Ky[1] -Ky[0]
dkz = Kzc[1] -Kzc[0]
## ------------------------------------------------

## -------------- creating functions --------------
# def e3d_to_e1d(e,erate):
# def e3d_to_e1d(e,e1df):
#     # e = np.ones_like(k)
        
#     # erate1d_f = np.empty_like(Kzc)             
#     for i in range(Nf):
#         cond = (k> (K[i] -0.5)) *(k < (K[i] +0.5))
#         e1df[i]  = np.sum(e[cond])
#         # erate1d_f[i]  = np.sum(erate[cond])
#     # return e1d_f,erate1d_f
#     return e1df
    
def e3d_to_e1d(x): #1 Based on whether k is 2D or 3D, it will bin the data accordingly. 
    return np.bincount(np.round(k).ravel().astype(int),weights=x.ravel())[:Nf]    
    
# def e2d_to_e1d(e,e1df):
#     # e = np.ones_like(k)
        
#     # erate1d_f = np.empty_like(Kzc)             
#     for i in range(Nf):
#         cond = (kh> (K[i] -0.5)) *(kh <( K[i] +0.5))
#         e1df[i]  = np.sum(e[cond])
#         # erate1d_f[i]  = np.sum(erate[cond])
#     # return e1d_f,erate1d_f
#     return e1df

def e2d_to_e1d(x): #1 Based on whether k is 2D or 3D, it will bin the data accordingly. 
    return np.bincount(np.round(kh).ravel().astype(int),weights=x.ravel())[:Nf]
    
## ------------------------------------------------

## ---------- Empty arrays -------------

u = np.empty((3,N,N,N))
b = np.empty((N,N,N))

uk = np.empty((3,N,N,N),dtype = np.complex128)
bk = np.empty((N,N,N),dtype=np.complex128)

ek = np.empty((N,N,N))
ekTemp = np.empty((N,N,N))
ek_w = np.empty((N,N,N))
ek_v = np.empty((N,N,N))
e1dk = np.empty(Nf)
e1dk_w = np.empty(Nf)
e1dk_v = np.empty(Nf)
e = np.empty((N,N,N))
eTemp = np.empty((N,N,N))
e_w = np.empty((N,N,N))
e_v = np.empty((N,N,N))
Pi = np.ones((N,N,N))
Pi_w = np.ones((N,N,N))
Pi_v = np.ones((N,N,N))

PIk = np.empty((N,N,N))
PIk_w = np.empty((N,N,N))
PIk_v = np.empty((N,N,N))
PI1dk = np.empty(Nf)
PI1dk_w = np.empty(Nf)
PI1dk_v = np.empty(Nf)
PI1dk_avg = np.empty(Nf)
PI1dk_avg_w = np.empty(Nf)
PI1dk_avg_v = np.empty(Nf)

ekh = np.empty((N,N))
ekh_w = np.empty((N,N))
ekh_v = np.empty((N,N))
e1dkh = np.empty(Nf)
e1dkh_w = np.empty(Nf)
e1dkh_v = np.empty(Nf)

e1dkh_avg = np.empty(Nf)
e1dkh_avg_w = np.empty(Nf)
e1dkh_avg_v = np.empty(Nf)    

ekv = np.empty(N)
ekv_w = np.empty(N)
ekv_v = np.empty(N)

ekv_avg = np.empty(N)
ekv_avg_w = np.empty(N)
ekv_avg_v = np.empty(N)

pv = np.empty((N,N,N))
# if rank == 0:
#     E = np.empty(len(times_o))
#     ek_t = np.empty((Ntimes,N))
#     ekh_t = np.empty((Ntimes,N))
#     ekv_t = np.empty((Ntimes,N))

## -------------------------------------

## ------ Creating directories ----------

# Net PV
# new_dir_name = loadPath/f"3D_PV"
# new_dir = pathlib.Path(curr_path,  new_dir_name)
# new_dir.mkdir(parents=True,  exist_ok=True)

# Plots 
# new_dir_name = savePath/f"PV"
# new_dir = pathlib.Path(curr_path,  new_dir_name)
# new_dir.mkdir(parents=True,  exist_ok=True)

new_dir_name = savePath/f"E_k-k"
new_dir = pathlib.Path(curr_path,  new_dir_name)
new_dir.mkdir(parents=True,  exist_ok=True)
new_dir_name = savePath/f"PI_k-k"
new_dir = pathlib.Path(curr_path,  new_dir_name)
new_dir.mkdir(parents=True,  exist_ok=True)
new_dir_name = savePath/f"E_kv-kv"
new_dir = pathlib.Path(curr_path,  new_dir_name)
new_dir.mkdir(parents=True,  exist_ok=True)
new_dir_name = savePath/f"E_kh-kh"
new_dir = pathlib.Path(curr_path,  new_dir_name)
new_dir.mkdir(parents=True,  exist_ok=True)
## --------------------------------------
# times_o = np.arange(100,1600,100)
# times_o = np.arange(0,40,2)
# times = np.array(list(np.arange(0,2.4,0.2)) + list(np.arange(2.4,6,1.)))
# times = [times_o[rank]]
# times = [182.20]
# times = np.arange(0.,26.1,2.0)
# times = np.arange(0.,200.,1)
times = [1000.0]
# ektot = np.empty(len(times))
# etot = np.empty(len(times))
# eratio = np.empty(len(times))
# ekwtot = np.empty(len(times))
# ekvtot = np.empty(len(times))
# ektide = np.empty(len(times))
for jj,time in enumerate(times):
    print(f"Time {time:.1f}")
    ekv[:] = 0.
    et = 0.
    for j in range(num_slabs):
        print(f"slab {j}")
        PIk[:,j*Ns:(j+1)*Ns,:] = np.load(loadPath/f"PIk/PI_{j}.npy")
        PIk_v[:,j*Ns:(j+1)*Ns,:] = np.load(loadPath/f"PIk/PIVortex_{j}.npy")
        PIk_w[:,j*Ns:(j+1)*Ns,:] = np.load(loadPath/f"PIk/PIWave_{j}.npy")
        
        # Pi[j*Ns:(j+1)*Ns,:] = np.load(loadPath/f"Pi/time_{time:.1f}/Pi_{j}.npy")
        
        ek[:,j*Ns:(j+1)*Ns,:] = np.load(loadPath/f"E_k/e_{j}.npy")
        ek_w[:,j*Ns:(j+1)*Ns,:] = np.load(loadPath/f"E_k/eWave_{j}.npy")
        ek_v[:,j*Ns:(j+1)*Ns,:] = np.load(loadPath/f"E_k/eVortex_{j}.npy")
        # ekh[:,j*Ns:(j+1)*Ns] = np.load(loadPath/f"E_kh/time_{time:.1f}/ekh_{j}.npy")
        # ekh_w[:,j*Ns:(j+1)*Ns] = np.load(loadPath/f"E_kh/time_{time:.1f}/ekhWave_{j}.npy")
        # ekh_v[:,j*Ns:(j+1)*Ns] = np.load(loadPath/f"E_kh/time_{time:.1f}/ekhVortex_{j}.npy")
        
        # e[j*Ns:(j+1)*Ns,:] = np.load(loadPath/f"time_{time:.1f}/e_{j}.npy")
        # u[0,j*Ns:(j+1)*Ns,:] = np.load(loadPath/f"time_{time:.1f}/u_{j}.npy")
        # u[1,j*Ns:(j+1)*Ns,:] = np.load(loadPath/f"time_{time:.1f}/v_{j}.npy")
        # u[2,j*Ns:(j+1)*Ns,:] = np.load(loadPath/f"time_{time:.1f}/w_{j}.npy")
        # b[j*Ns:(j+1)*Ns,:] = np.load(loadPath/f"time_{time:.1f}/b_{j}.npy")
        # pv[j*Ns:(j+1)*Ns,...] = np.load(loadPath/f"PV/time_{time:.1f}/PV_{j}.npy")
        # ----------------------- Loading total energy -------------------------------------
    print(f"Loading for time {time:.1f}: Done!")
    # print(f"Max for ekv_w at time {time:.1f}: {np.max(ekv_w)}")
    # print(f"Max for ekv_v at time {time:.1f}: {np.max(ekv_v)}")
    ekh[:] = np.sum(ek,axis=2)
    ekh_w[:] = np.sum(ek_w,axis=2)
    ekh_v[:] = np.sum(ek_v,axis=2)
    
    ekv[:] = np.sum(ek,axis=(0,1))
    ekv_w[:] = np.sum(ek_w,axis=(0,1))
    ekv_v[:] = np.sum(ek_v,axis=(0,1))
    print(f"Total flux : {np.sum(PIk)}")
    print(f"Max Balanced flux : {np.max(PIk_v)}")
    print(f"Max Wave flux : {np.max(PIk_w)}")
    print(f"Time/ {time}: Total - ( Balanced + Wave)  Flux max : {np.max(np.abs(PIk - PIk_v -PIk_w))}")
    # print(f"Time {time}: Total - ( Balanced + Wave)  energy max : {np.max(np.abs(ek - ek_v -ek_w))}")

    
    
    # eTemp[:] = 0.5*(u[0]**2 +u[1]**2+u[2]**2/alph**2 + b**2 )*dx*dy*dz
    # ekTemp[:] = 0.5*(np.abs(uk[0])**2 +np.abs(uk[1])**2+np.abs(uk[2])**2/alph**2 + np.abs(bk)**2 )*dkx*dky*dkz
    # print(np.sum(eTemp),np.sum(ekTemp),np.sum(e),np.sum(ek))
    # eratio[jj] = np.sum(ekTemp)/np.sum(eTemp)
    # ektot[jj] = np.sum(ek)
    # etot[jj] = np.sum(e)
    # ekwtot[jj] = np.sum(ek_w)
    # ekvtot[jj] = np.sum(ek_v)
    # print(f"Diff for Net : {np.sum(ek) -np.sum(ekh)}")
    # print(f"Diff for Balanced : {np.sum(ek_v) -np.sum(ekh_v)}")
    # print(f"Diff for Wave : {np.sum(ek_w) -np.sum(ekh_w)}")
    # ektide[jj] = np.sum(ek_w[:,:,1])
    
    
    # # # ---------------- Saving the total PV ------------------------
    # # np.save(loadPath/f"3D_PV/time_{time:.1f}",pv)
    # # imageToVTK(loadPath/f"3D_PV/time_{time:.1f}", pointData={"data": pv})
    # # print("Done!")
    # # # -------------------------------------------------------------
    
    
    PI1dk[:] = e3d_to_e1d(PIk) 
    PI1dk_w[:] = e3d_to_e1d(PIk_w) 
    PI1dk_v[:] = e3d_to_e1d(PIk_v) 
    PI1dk[:] = np.cumsum(PI1dk[::-1])[::-1]
    PI1dk_w[:] = np.cumsum(PI1dk_w[::-1])[::-1]
    PI1dk_v[:] = np.cumsum(PI1dk_v[::-1])[::-1]
    
    PI1dk_avg[:] = PI1dk_avg + PI1dk
    PI1dk_avg_w[:] = PI1dk_avg_w + PI1dk_w
    PI1dk_avg_v[:] = PI1dk_avg_v + PI1dk_v
    e1dk[:] = e3d_to_e1d(ek)
    e1dk_w[:] = e3d_to_e1d(ek_w)
    e1dk_v[:] = e3d_to_e1d(ek_v)
    
    ekv_avg[:] = ekv_avg + ekv
    ekv_avg_w[:] = ekv_avg_w + ekv_w
    ekv_avg_v[:] = ekv_avg_v + ekv_v
    print(f"Time {time}: Total - ( Balanced + Wave)  energy max : {np.max(np.abs(e1dk) - np.abs(e1dk_v +e1dk_w))}")
    print(f"Time {time}: Total energy max : {np.max(np.abs(e1dk) )}")
    e1dkh[:] = e2d_to_e1d(ekh)
    e1dkh_w[:] = e2d_to_e1d(ekh_w)
    e1dkh_v[:] = e2d_to_e1d(ekh_v)
    
    e1dkh_avg[:] = e1dkh_avg + e1dkh
    e1dkh_avg_w[:] = e1dkh_avg_w + e1dkh_w
    e1dkh_avg_v[:] = e1dkh_avg_v + e1dkh_v
    print(f"Energy for time {time:.1f}: Done!")
    print(f"Balanced + Waves - Net : {np.max(np.abs(e1dk_w + e1dk_v- e1dk))}")
    

    
    ## --------------- Flux plot -----------------------
    plt.figure(figsize = (8,6))
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    plt.plot(K[1:],PI1dk[1:],'-',lw = 4,color = "#001219",label = "Net Flux")
    plt.plot(K[1:],PI1dk_w[1:],'-',lw = 4,color = "#fb8500",label = "Unbalanced Flux")
    plt.plot(K[1:],PI1dk_v[1:],'-',lw = 4,color = "#d90429",label = "Balanced Flux")
    plt.xscale("log",base =10)
    # plt.yscale("symlog",linthresh = 1e-6)
    plt.xlabel(r"$k$",fontsize =22 )
    plt.ylabel(r"$\Pi_k$",fontsize =22 ,rotation = 0,labelpad = 20)
    plt.xlim(1,Nf/3**0.5)
    
    # plt.ylim(-1e-1,1e-1)
    plt.xlim(0,200)
    # plt.title(fr"$\Pi_k$ v $k$ time {time:.1f}",fontsize = 18)
    # plt.tight_layout()
    plt.legend(fontsize = 22,fancybox = True, framealpha = 0.3)
    plt.grid()
    plt.tight_layout()
    plt.savefig(savePath/fr"PI_k-k/PIk-v-k-avg.jpg")#,transparent = True)
    plt.close()
    
    ## -------------------------------------------------

        
        
    
    # ## --------------- Total energy plot -----------------------
    # plt.figure(figsize = (8,6))
    # plt.xticks(fontsize = 18)
    # plt.yticks(fontsize = 18)
    # plt.loglog(K,e1dk_w,color = "#fb8500",label = "Wave")
    # plt.loglog(K,e1dk,color = "#001219",label = "Net")
    # plt.loglog(K,e1dk_v,color = "#d90429",label = "Balanced")
    # plt.xlabel(r"$k$",fontsize =22 )
    # plt.ylabel(r"$E_k$",fontsize =22 ,rotation = 0)
    # plt.ylim(1e-6,plt.ylim()[1])
    # plt.title(fr"$E_k$ v $k$ time {time:.1f}",fontsize = 30)
    # plt.tight_layout()
    # plt.grid()
    # plt.legend(fontsize = 22,fancybox = True, framealpha = 0.3)
    # plt.savefig(savePath/fr"E_k-k/Ek-v-k-time_{time:.1f}.png")#,transparent = True) 
    # plt.close()        
    # ## ---------------------------------------------------------
    
    ## ----------- Horizontal energy spectrum ----------------
    plt.figure(figsize = (8,6))
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    plt.loglog(K,e1dkh_w,color = "#fb8500",label = 'Unbalanced',lw = 4.)
    plt.loglog(K,e1dkh,color = "#001219",label = 'Net',lw = 4.)
    plt.loglog(K,e1dkh_v,color = "#d90429", label = "Balanced",lw = 4.)
    plt.loglog(K[8:50],((0.8*K)**(-3))[8:50],'--',color = "#000000",lw = 4.)
    plt.loglog(K[6:60],((0.3*K)**(-2))[6:60],'--',color = "#000000",lw = 4.)
    plt.xlabel(r"$k_h$",fontsize =22 )
    plt.ylabel(r"$E_{k_h}$",fontsize =22 ,rotation = 0,labelpad = 20)
    plt.ylim(1e-6,plt.ylim()[1])
    # plt.title(fr"$E_{{k_h}}$ v $k_h$ time {time:.1f}",fontsize = 22)
    plt.axvline(1.0,color = "#181818",linestyle = '-.',lw = 1.)
    plt.axvline(6.0,color = "#1f1f1f",linestyle = '-.',lw = 1.)
    plt.text(25,3e-2,r"$k^{-2}$",fontsize = 22)
    plt.text(10,1e-4,r"$k^{-3}$",fontsize = 22)
    # plt.tight_layout()
    plt.grid()
    plt.legend(fontsize =22,fancybox = True, framealpha = 0.3)
    plt.tight_layout()
    plt.savefig(savePath/f"E_kh-kh/Ekh-v-kh-avg.jpg")#,transparent = True)
    plt.close()
    ## ---------------------------------------------------------
    
    ## ---------------- Vertical energy spectrum ---------------
    plt.figure(figsize = (8,6))
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    plt.loglog(Kzc,ekv_w,color = "#fb8500",label = 'Wave',lw = 4.)
    plt.loglog(Kzc,ekv,color = "#001219",label = 'Net',lw = 4.)
    plt.loglog(Kzc,ekv_v,color = "#d90429",label = 'Balanced',lw = 4.)
    plt.loglog(Kzc[5:70],((1.0*Kzc)**(-2))[5:70],'--',color = "#000000",lw = 4.)
    plt.loglog(Kzc[5:70],((1.5*Kzc)**(-3))[5:70],'--',color = "#000000",lw = 4.)
    plt.xlabel(r"$k_z$",fontsize =22 )
    plt.ylabel(r"$E_{k_z}$",fontsize =22 ,rotation = 0,labelpad = 40)
    plt.ylim(1e-6,plt.ylim()[1])
    # plt.title(fr"$E_{{k_z}}$ v $k_z$ time {time:.1f}",fontsize = 22)
    plt.text(10,2e-2,r"$k^{-2}$",fontsize = 22)
    plt.text(10,1e-4,r"$k^{-3}$",fontsize = 22)
    plt.tight_layout()
    plt.grid()
    plt.legend(fontsize =22,fancybox = True, framealpha = 0.3)
    plt.savefig(savePath/fr"E_kv-kv/Ekv-v-kv-avg.png")#,transparent = True)
    plt.close()
    ## ---------------------------------------------------------

# ## --------------- Avg Flux plot -----------------------
# PI1dk_avg[:] = PI1dk_avg/len(times)
# PI1dk_avg_w[:] = PI1dk_avg_w/len(times)
# PI1dk_avg_v[:] = PI1dk_avg_v/len(times)
# plt.figure(figsize = (8,6))
# plt.xticks(fontsize = 18)
# plt.yticks(fontsize = 18)
# plt.plot(K,PI1dk_avg,'-',lw = 4,color = "#001219",label = "Net Flux")
# plt.plot(K,PI1dk_avg_w,'-',lw = 4,color = "#fb8500",label = "Wave Flux")
# plt.plot(K,PI1dk_avg_v,'-',lw = 4,color = "#d90429",label = "Balanced Flux")
# plt.xlabel(r"$k$",fontsize =22 )
# plt.ylabel(r"$\Pi_k$",fontsize =22 ,rotation = 0,labelpad = 20)
# # plt.ylim(-1e-1,1e-1)
# plt.xlim(0,160)
# plt.title(fr"Time averaged $\Pi_k$ v $k$ ",fontsize = 18)
# plt.tight_layout()
# plt.legend(fontsize = 22,fancybox = True, framealpha = 0.3)
# plt.grid()
# plt.tight_layout()
# plt.savefig(savePath/fr"PI_k-k/PIk-v-k-avg.jpg")#,transparent = True)
# plt.close()

# ## -------------------------------------------------
# ## --------------- Avg Ekh plot -----------------------
# e1dkh_avg[:] = e1dkh_avg/len(times)
# e1dkh_avg_w[:] = e1dkh_avg_w/len(times)
# e1dkh_avg_v[:] = e1dkh_avg_v/len(times)

# plt.figure(figsize = (8,6))
# plt.xticks(fontsize = 18)
# plt.yticks(fontsize = 18)
# plt.loglog(K,e1dkh_avg_w,color = "#fb8500",label = 'Wave',lw = 4.)
# plt.loglog(K,e1dkh_avg,color = "#001219",label = 'Net',lw = 4.)
# plt.loglog(K,e1dkh_avg_v,color = "#d90429", label = "Balanced",lw = 4.)
# plt.loglog(K[8:50],((1.1*K)**(-3))[8:50],'--',color = "#000000",lw = 4.)
# plt.loglog(K[10:80],((0.7*K)**(-2))[10:80],'--',color = "#000000",lw = 4.)
# plt.xlabel(r"$k_h$",fontsize =22 )
# plt.ylabel(r"$E_{k_h}$",fontsize =22 ,rotation = 0,labelpad = 20)
# plt.ylim(1e-6,plt.ylim()[1])
# plt.title(fr"$E_{{k_h}}$ v $k_h$ Averaged",fontsize = 22)
# plt.text(30,4e-3,r"$k^{-2}$",fontsize = 22)
# plt.text(10,7e-5,r"$k^{-3}$",fontsize = 22)
# # plt.tight_layout()
# plt.grid()
# plt.legend(fontsize =22,fancybox = True, framealpha = 0.3)
# plt.tight_layout()
# plt.savefig(savePath/f"E_kh-kh/Ekh-v-kh-avg.jpg")#,transparent = True)
# plt.close()
# ## -------------------------------------------------
# ## --------------- Avg Ekv plot -----------------------
# ekv_avg[:] = ekv_avg/len(times)
# ekv_avg_w[:] = ekv_avg_w/len(times)
# ekv_avg_v[:] = ekv_avg_v/len(times)

# plt.figure(figsize = (8,6))
# plt.xticks(fontsize = 18)
# plt.yticks(fontsize = 18)
# plt.loglog(Kzc,ekv_avg_w,color = "#fb8500",label = 'Wave',lw = 4.)
# plt.loglog(Kzc,ekv_avg,color = "#001219",label = 'Net',lw = 4.)
# plt.loglog(Kzc,ekv_avg_v,color = "#d90429",label = 'Balanced',lw = 4.)
# plt.loglog(Kzc[5:70],((1.0*Kzc)**(-2))[5:70],'--',color = "#000000",lw = 4.)
# plt.loglog(Kzc[5:70],((1.5*Kzc)**(-3))[5:70],'--',color = "#000000",lw = 4.)
# plt.xlabel(r"$k_z$",fontsize =22 )
# plt.ylabel(r"$E_{k_z}$",fontsize =22 ,rotation = 0,labelpad = 40)
# plt.ylim(1e-6,plt.ylim()[1])
# plt.title(fr"$E_{{k_z}}$ v $k_z$ averaged",fontsize = 22)
# plt.text(10,2e-2,r"$k^{-2}$",fontsize = 22)
# plt.text(10,1e-4,r"$k^{-3}$",fontsize = 22)
# plt.tight_layout()
# plt.grid()
# plt.legend(fontsize =22,fancybox = True, framealpha = 0.3)
# plt.savefig(savePath/fr"E_kv-kv/Ekv-v-kv-avg.png")#,transparent = True)
# plt.close()

# ## -------------------------------------------------

# print(f"Plotting for time {time:.1f}: Done!")

# epostot = np.sum(e)
# ektot = np.sum(ek)
# etot_w = np.sum(ek_w)
# etot_v = np.sum(ek_v)
# etot = etot_w  + etot_v
# comm.Barrier()

# if rank > 0: 
#     print(f"Sending from rank {rank}")
#     comm.send(etot,dest = 0, tag = 100)
#     comm.send(etot_w,dest = 0, tag = 101)
#     comm.send(etot_v,dest = 0, tag = 102)
#     # comm.send(epostot,dest = 0, tag = 103)
#     comm.send(ektot,dest = 0, tag = 104)
#     # comm.send(etot_w,dest = 0, tag = 105)
#     # comm.send(etot_v,dest = 0, tag = 106)
# else:
#     change = lambda e1: (e1/e1[0]-1)
#     ektot_arr = np.empty(len(times_o))
#     ektot_act_arr = np.empty(len(times_o))
#     ekwtot_arr = np.empty(len(times_o))
#     ekvtot_arr = np.empty(len(times_o))
#     etot_arr = np.empty(len(times_o))
#     etotw_arr = np.empty(len(times_o))
#     etotv_arr = np.empty(len(times_o))
#     # ektide = np.empty(len(times_o))
#     # etot_arr[0] = epostot
#     # etotw_arr[0] = etot_w
#     # etotv_arr[0] = etot_v
#     ektot_arr[0] = etot
#     ekwtot_arr[0] = etot_w
#     ekvtot_arr[0] = etot_v
#     ektot_act_arr[0] = ektot
#     for j in range(1,len(times_o)):
#         print(f"Received from rank {j}")
#         ektot_arr[j] = comm.recv(source=j, tag =100)
#         ekwtot_arr[j] = comm.recv(source=j, tag =101)
#         ekvtot_arr[j] = comm.recv(source=j, tag =102)
#         # etot_arr[j] = comm.recv(source=j, tag =103)
#         ektot_act_arr[j] = comm.recv(source=j, tag =104)
#         # etotw_arr[j] = comm.recv(source=j, tag =105)
#         # etotv_arr[j] = comm.recv(source=j, tag =106)
    
#     np.save(loadPath/"ektot_arr.npy",ektot_act_arr)
#     np.save(loadPath/"ekwtot_arr.npy",ekwtot_arr)
#     np.save(loadPath/"ekvtot_arr.npy",ekvtot_arr)
#     # plt.figure(figsize = (8,6))
#     # plt.xticks(fontsize = 18)
#     # plt.yticks(fontsize = 18)
#     # # plt.plot(times,eratio,color = "#fb8500",label = 'Ratio')
#     # # plt.plot(times_o,change(ektot_arr),'-',lw = 4,color = "#001219",label = 'spectral energy(summed)')
#     # plt.plot(times_o,change(ektot_act_arr),'.',color = "#001219",label = 'Net energy')
#     # # plt.plot(times_o,change(etot_arr),color = "#fb8500",label = 'total energy ')
#     # plt.plot(times_o,change(etotw_arr),color = "#fb8500",label = 'Wave energy')
#     # plt.plot(times_o,change(etotv_arr),color = "#d90429",label = 'Balanced energy')
#     # # plt.plot(times,ekwtot_arr,color = "#d90429",label = 'Wave energy')
#     # # plt.plot(times,ekvtot_arr,color = "#fb8500",label = 'Balanced energy')
#     # # plt.plot(times,1 - ektide/ektide[0],color = "#ffb703",label = r'$k_z = 1$ Wave')
#     # plt.xlabel(r"$t$",fontsize =22 )
#     # plt.ylabel(r"$\frac{\Delta E}{E}$",fontsize =22,rotation = 0 )
#     # # plt.ylim(1e-6,plt.ylim()[1])
#     # plt.title(fr"Energy Change",fontsize = 30)
#     # # plt.tight_layout()
#     # plt.grid()
#     # plt.legend(fontsize =20,fancybox = True, framealpha = 0.3)
#     # # plt.savefig(fr"{savePath}/energyChangeTimeseries.png")#,transparent = True)
#     # plt.close()
    
#     plt.figure(figsize = (8,6))
#     plt.xticks(fontsize = 18)
#     plt.yticks(fontsize = 18)
#     # plt.plot(times,eratio,color = "#fb8500",label = 'Ratio')
#     plt.plot(times_o,ektot_act_arr,color = "#001219",label = 'Net energy')
#     plt.plot(times_o,ekwtot_arr,color = "#fb8500",label = 'Wave energy')
#     plt.plot(times_o,ekvtot_arr,color = "#d90429",label = 'Balanced energy')
#     # plt.plot(times,ekvtot_arr,color = "#fb8500",label = 'Balanced energy')
#     # plt.plot(times,1 - ektide/ektide[0],color = "#ffb703",label = r'$k_z = 1$ Wave')
#     plt.xlabel(r"$t$",fontsize =22 )
#     plt.ylabel(r"$E$",fontsize =22,rotation = 0 )
#     # plt.ylim(1e-6,plt.ylim()[1])
#     plt.title(fr"Energy",fontsize = 40)
#     # plt.tight_layout()
#     plt.grid()
#     plt.legend(fontsize =22,fancybox = True, framealpha = 0.3)
#     plt.savefig(fr"{savePath}/energyTimeseries.png")#,transparent = True)
#     plt.close()


# # """
# #     time nohup mpirun -n 100 python -u plotting.py > errors-outputs/plotting.out &
# # """