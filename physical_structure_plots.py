# %%
import numpy as np
from scipy.fft import fft ,  ifft ,  irfft2 ,  rfft2 , irfftn ,  rfftn, fftfreq, dst, dct, idst, idct, rfft,  irfft
import matplotlib.pyplot as plt
import pathlib,os
import matplotlib as mpl 
mpl.rc('text', usetex = True)
mpl.rcParams['font.size'] = 25

hexVals = ['#700000','#8e0308','#c3060c','#ef1119','#ff3928','#ff6b46','#ff9271','#ffbdab','#ffe1e1','#ffffff','#d4f5ff','#92dde6','#41c1d8','#00a1cf','#0093ca','#0072b7','#00599f','#002c6d','#001847']
hexVals.reverse()
custom_cm=mpl.colors.LinearSegmentedColormap.from_list('custom',hexVals)
cmap=custom_cm


# %%
N = 384
ro = 0.1
nu = 1e-31
omega = 1.7277

xindex = 384//2
num_slabs = 192
slab = xindex/(N//num_slabs)
slab

# %%
t = 60.0

# %%
# os.listdir(f"/mnt/lustre/icts_user3/boussinesq/data_final/nu_1e-32_N_384/Ro_0.1/forcedTide_ring_LW")
# os.listdir(f"Y:/boussinesq/data/nu_{nu}_N_{N}/Ro_{ro}/")

# %%
loadPath = pathlib.Path(f"/mnt/pfs/rajarshi.chattopadhyay/boussinesq/spectrum-development/nu_{nu}_N_{N}/Ro_{ro}/forcedTide_ring_{omega:.2f}/time_{t:.1f}")
loadPathdata = pathlib.Path(f"/mnt/pfs/rajarshi.chattopadhyay/boussinesq/spectrum-development/nu_{nu}_N_{N}/Ro_{ro}/forcedTide_ring_{omega:.2f}/")
print(loadPath.exists(),loadPathdata.exists(),str(loadPath))

# # %% [markdown]
# # # YZ slices of the fields

# # %%
# Fields = np.load(loadPath/f"Fields_{slab:.0f}.npz")
# u = Fields['u']
# v = Fields['v']
# w = Fields['w']
# b = Fields['b']
# PV = np.load(loadPathdata/f"PV/time_{t:.1f}/PV_{slab:.0f}.npy")
# omg = np.load(loadPathdata/f"zeta/time_{t:.1f}/zeta_{slab:.0f}.npy")


# # %%
# plt.figure(figsize = (16,12))
# p1 = plt.pcolor(np.arange(1,N+1),np.arange(1,N+1), u[0,:,:].T,cmap = "seismic")
# plt.colorbar(p1)
# plt.xlabel(r"$y$")
# plt.ylabel(r"$z$",rotation = 0)
# plt.title(r"$u$")

# # %%
# p1 = plt.pcolor(np.arange(1,N+1),np.arange(1,N+1), v[0,:,:].T,cmap = "seismic")
# plt.colorbar(p1)
# plt.xlabel(r"$y$")
# plt.ylabel(r"$z$",rotation = 0)
# plt.title(r"$v$")

# # %%
# p1 = plt.pcolor(np.arange(1,N+1),np.arange(1,N+1), w[0,:,:].T,cmap = "seismic")
# plt.colorbar(p1)
# plt.xlabel(r"$y$")
# plt.ylabel(r"$z$",rotation = 0)
# plt.title(r"$w$")

# # %%
# p1 = plt.pcolor(np.arange(1,N+1),np.arange(1,N+1), b[0,:,:].T,cmap = "seismic")
# plt.colorbar(p1)
# plt.xlabel(r"$y$")
# plt.ylabel(r"$z$",rotation = 0)
# plt.title(r"$b$")

# # %%
# p1 = plt.pcolor(np.arange(1,N+1),np.arange(1,N+1), PV[0,:,:].T,cmap = "seismic")
# plt.colorbar(p1)
# plt.xlabel(r"$y$")
# plt.ylabel(r"$z$",rotation = 0)
# plt.title(r"$PV$")

# # %%
# p1 = plt.pcolor(np.arange(1,N+1),np.arange(1,N+1), omg[0,:,:].T,cmap = "seismic")
# plt.colorbar(p1)
# plt.xlabel(r"$y$")
# plt.ylabel(r"$z$",rotation = 0)
# plt.title(r"$\omega$")

# # %%
# np.max(np.abs(w[:,:,0]))

# # %% [markdown]
# # # XY slice

# # %%
# u = np.zeros((N,N))
# v = np.zeros((N,N))
# w = np.zeros((N,N))
# b = np.zeros((N,N))
# PV = np.empty((N,N))
# omg = np.empty((N,N))

# # %%
# for i in range(num_slabs):
#     print(i,end = "\r")
#     Fields = np.load(loadPath/f"Fields_{i:.0f}.npz")
#     u[i*(N//num_slabs): (i+1)*(N//num_slabs),:]  = Fields["u"][:,:,N//2]
#     v[i*(N//num_slabs): (i+1)*(N//num_slabs),:]  = Fields["v"][:,:,N//2]
#     w[i*(N//num_slabs): (i+1)*(N//num_slabs),:]  = Fields["w"][:,:,N//2]
#     b[i*(N//num_slabs): (i+1)*(N//num_slabs),:]  = Fields["b"][:,:,N//2]
#     PV[i*(N//num_slabs): (i+1)*(N//num_slabs),:] = np.load(loadPathdata/f"PV/time_{t:.1f}/PV_{i:.0f}.npy")[:,:,N//2]
#     omg[i*(N//num_slabs): (i+1)*(N//num_slabs),:] = np.load(loadPathdata/f"zeta/time_{t:.1f}/zeta_{i:.0f}.npy")[:,:,N//2]

# # %%
# plt.figure(figsize = (16,12))
# p1 = plt.pcolor(np.arange(1,N+1),np.arange(1,N+1), u.T,cmap = "seismic")
# plt.colorbar(p1)
# plt.xlabel(r"$x$")
# plt.ylabel(r"$y$",rotation = 0)
# plt.title(r"u")

# # %%
# plt.figure(figsize = (16,12))
# p1 = plt.pcolor(np.arange(1,N+1),np.arange(1,N+1), v.T,cmap = "seismic")
# plt.colorbar(p1)
# plt.xlabel(r"$x$")
# plt.ylabel(r"$y$",rotation = 0)
# plt.title(r"v")
# plt.close()

# # %%
# plt.figure(figsize = (16,12))
# p1 = plt.pcolor(np.arange(1,N+1),np.arange(1,N+1), w.T,cmap = "seismic")
# plt.colorbar(p1)
# plt.xlabel(r"$x$")
# plt.ylabel(r"$y$",rotation = 0)
# plt.title(r"w")
# # plt.close()

# # %%
# plt.figure(figsize = (16,12))
# p1 = plt.pcolor(np.arange(1,N+1),np.arange(1,N+1), b.T,cmap = "seismic")
# plt.colorbar(p1)
# plt.xlabel(r"$x$")
# plt.ylabel(r"$y$",rotation = 0)
# plt.title(r"b")
# plt.close()

# # %%
# # levels = np.linspace(-5,5,200)
# plt.figure(figsize = (8,6))

# # c_PV = PV.T # clipped data between vmin and vmax
# c_PV = np.clip(PV.T,-5,5) # clipped data between vmin and vmax
# p1 = plt.contourf(np.linspace(0,2*np.pi,N),np.linspace(0,2*np.pi,N),c_PV ,200,cmap = "seismic",extend = "both")
# cbar = plt.colorbar(p1)#min = -5,vmax = 5)
# # cbar.set_clim(-5,5)
# plt.xlabel(r"$x$")
# plt.ylabel(r"$y$",rotation = 0)
# plt.title(r"$pv$",fontsize = 50)

# # %%
# # levels = np.linspace(-5,5,200)
# plt.figure(figsize = (8,6))
# c_vort = np.clip(omg.T,-5,5)

# p1 = plt.contourf(np.linspace(0,2*np.pi,N),np.linspace(0,2*np.pi,N), c_vort,200,cmap = "seismic",extend = "both")
# plt.colorbar(p1)
# plt.xlabel(r"$x$")
# plt.ylabel(r"$y$",rotation = 0)
# plt.title(r"$\zeta$",fontsize = 50)
# # plt.savefig(f"/home/rajarshi.chattopadhyay/python/fluid/boussinesq/Rajarshi/Plots/nu_{nu}_N_{N}/Ro_{ro:.1f}/QG/zeta.png",dpi = 300)

# # %%
# # plt.plot(times,e,'o-')
# # plt.xlabel(r"$t$")
# # plt.ylabel(r"$E$",rotation = 0)
# # plt.grid()


# # %%
# fields_path  = pathlib.Path(f"/mnt/lustre/icts_uswwer3/boussinesq/data/nu_{nu}_N_{N}/Ro_{ro}/forcedinertial/time_0.0/")

# # %%
# ubar = np.zeros(N)
# vbar = np.zeros(N)

# # %%
# for i in range(num_slabs):
#     field = np.load(fields_path/f"Fields_{i}.npz")
#     ubar = ubar + np.sum(field["u"],axis = (1,2))
#     vbar = vbar + np.sum(field["v"],axis = (1,2))

# # %%
# print(ubar.min(),ubar.max())

# # %%
# print

# %% [markdown]
# # Full Data

# %%
u = np.zeros((N,N,N))
v = np.zeros((N,N,N))
w = np.zeros((N,N,N))
b = np.zeros((N,N,N))
PV = np.empty((N,N,N))
omg = np.empty((N,N,N))
times = [float(i.split("_")[-1]) for i in os.listdir(loadPathdata/f"PV/")]
times.sort()
print(len(times))
# raise ValueError("Stop here")
two_slope_norm_hor = mpl.colors.TwoSlopeNorm(vmin = -1.5, vmax = 2,vcenter = 0) #! For the colors
two_slope_norm_ver = mpl.colors.TwoSlopeNorm(vmin = -5, vmax = 5,vcenter = 0) #! For the colors
savePlotPath = pathlib.Path(f"/mnt/pfs/rajarshi.chattopadhyay/codes/boussinesq/Plots/nu_{nu}_N_{N}/Ro_{ro}/forcedTide_ring_{omega:.2f}/PV_plots")
savePlotPath.mkdir(parents = True,exist_ok = True)
print(f"Rossby number is {ro}")
for t in times[::5]:
    print(t)
    if t>=0.0:
# t = 61.9
# %%
        for i in range(num_slabs):
            # print(i,end = "\r")
            Fields = np.load(loadPath/f"Fields_{i:.0f}.npz")
            # u[i*(N//num_slabs): (i+1)*(N//num_slabs),:]  = Fields["u"]
            # v[i*(N//num_slabs): (i+1)*(N//num_slabs),:]  = Fields["v"]
            # w[i*(N//num_slabs): (i+1)*(N//num_slabs),:]  = Fields["w"]
            # b[i*(N//num_slabs): (i+1)*(N//num_slabs),:]  = Fields["b"]
            PV[i*(N//num_slabs): (i+1)*(N//num_slabs),:] = np.load(loadPathdata/f"PV/time_{t:.1f}/PV_{i:.0f}.npy")
            # omg[i*(N//num_slabs): (i+1)*(N//num_slabs),:] = np.load(loadPathdata/f"zeta/time_{t:.1f}/zeta_{i:.0f}.npy")

        # %%
        # levels = np.linspace(-5,5,200)
        plt.figure(figsize = (8,6))

        # c_PV = PV.T # clipped data between vmin and vmax
        c_PV = np.clip(PV[:,:,N//2].T,-5,5) # clipped data between vmin and vmax
        p1 = plt.contourf(np.linspace(0,2*np.pi,N),np.linspace(0,2*np.pi,N),c_PV ,200,cmap = cmap,extend = "both",norm = two_slope_norm_hor)
        cbar = plt.colorbar(p1)#min = -5,vmax = 5)
        # cbar.set_clim(-5,5)
        plt.xlabel(r"$x$")
        plt.ylabel(r"$y$",rotation = 0)
        # plt.title(r"$q$",fontsize = 50)
        PV.max(),PV.min(),np.sum(PV**2/N**3)**0.5
        plt.tight_layout()
        plt.savefig(savePlotPath/f"PV_hor_{t:.1f}.png")
        plt.close()
        
        # plt.figure(figsize = (8,6))

        # c_PV = PV.T # clipped data between vmin and vmax
        # c_PV = np.clip(PV[N//2,:,:].T,-5,5) # clipped data between vmin and vmax
        # p1 = plt.contourf(np.linspace(0,2*np.pi,N),np.linspace(0,2*np.pi,N),c_PV ,200,cmap = cmap,extend = "both",norm = two_slope_norm_ver)
        # cbar = plt.colorbar(p1)#min = -5,vmax = 5)
        # # cbar.set_clim(-5,5)
        # plt.xlabel(r"$y$")
        # plt.ylabel(r"$z$",rotation = 0)
        # # plt.title(r"$q$",fontsize = 50)
        # PV.max(),PV.min(),np.sum(PV**2/N**3)**0.5
        # plt.tight_layout()
        # plt.savefig(savePlotPath/f"PV_ver_{t:.1f}.png")
        # plt.close()
        # %%
        # levels = np.linspace(-5,5,200)
        # plt.figure(figsize = (8,6))
        # c_vort = np.clip(omg[:,:,N//2].T,-1.5,1.5)

        # # p1 = plt.contourf(np.linspace(0,2*np.pi,N),np.linspace(0,2*np.pi,N), c_vort,200,cmap = "seismic",extend = "both")
        # # plt.colorbar(p1)
        # # plt.xlabel(r"$x$")
        # # plt.ylabel(r"$y$",rotation = 0)
        # # plt.title(r"$\zeta$",fontsize = 50)
        # # plt.savefig(f"/mnt/lustre/icts_user3/boussinesq/Plots/nu_1e-32_N_384/tests/zeta_{t:.1f}.png")
        # # # plt.savefig(f"/home/rajarshi.chattopadhyay/python/fluid/boussinesq/Rajarshi/Plots/nu_{nu}_N_{N}/Ro_{ro:.1f}/QG/zeta.png",dpi = 300)
        # # omg.max(),omg.min(),np.sum(omg**2/N**3)**0.5
        # # plt.close()
        # # # %%
        # # np.sum(omg**2)**0.5/(N**1.5)

    # %% [markdown]
    # ## Calculating Divergence

    # %%
raise ValueError("Stop here")

PI = np.pi
TWO_PI = 2*PI
Nf = N//2 + 1
Np = N
sx = slice(0,N)
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

# %%
kx_diff = kx[:Nf,  :, :].copy()
kx_diff[-1, :, :] = -kx_diff[-1, :, :]
ky_diff = np.swapaxes(kx_diff, 0, 1).copy()

# %%
u_temp = np.zeros((N,N,N))
u1 = np.zeros_like(u)
v1 = np.zeros_like(v)
w1 = np.zeros_like(w)


# %%
def diff_x(u,  u_x):
    u_x[:] = irfft(1j*kx_diff*rfft(u, axis= 0), N, axis= 0)
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

# %%
# div_h = diff_x(u,u1) + diff_y(v,v1) 
div = diff_x(u,u1) + diff_y(v,v1) + diff_z_sin(w,w1)
print(div.abs().max())
# %%
# # levels = np.linspace(-5,5,200)
# plt.figure(figsize = (8,6))

# # c_PV = PV.T # clipped data between vmin and vmax
# c_PV = np.clip(div_h[:,:,N//2].T,-5,5) # clipped data between vmin and vmax
# p1 = plt.contourf(np.linspace(0,2*np.pi,N),np.linspace(0,2*np.pi,N),c_PV ,200,cmap = "seismic",extend = "both")
# cbar = plt.colorbar(p1)#min = -5,vmax = 5)
# # cbar.set_clim(-5,5)
# plt.xlabel(r"$x$")
# plt.ylabel(r"$y$",rotation = 0)
# plt.title(r"Horizontal Divergence",fontsize = 50)
# div_h.max(),div_h.min(),np.sum(div_h**2/N**3)**0.5

# # %%



