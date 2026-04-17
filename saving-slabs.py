import numpy as np
from scipy.fft import fftn,ifftn, fft ,  ifft ,  irfft2 ,  rfft2 , irfftn ,  rfftn, fftfreq, dst, dct, idst, idct, rfft,  irfft
# import matplotlib.pyplot as plt
from mpi4py import MPI
import pathlib,os
# import matplotlib as mpl 
# import matplotlib.colors as colors
# from mpl_toolkits.mplot3d import Axes3D
from time import time
# import h5py
# import math
# import array
# from matplotlib.colors import LightSource
#plt.style.use('ggplot')
#from palettable.cartocolors.diverging import Geyser_7
# from matplotlib import cm
# from matplotlib.colors import TwoSlopeNorm
# mpl.rcParams['axes.labelpad'] = -8
# mpl.rcParams['font.size'] = 10
# mpl.rc('text', usetex = True)




## ---------------MPI things--------------
comm = MPI.COMM_WORLD
num_process =  comm.Get_size()
rank = comm.Get_rank()
## ---------------------------------------

N = 384
Np = N// num_process
ro = 0.1
nu = 1e-31
alph = 20
num_slabs = 192
Ns = num_slabs// num_process
n_slab = N//num_slabs
omega = 1.7277


loadPathdata = pathlib.Path(f"/mnt/pfs/rajarshi.chattopadhyay/boussinesq/spectrum-development/nu_{nu}_N_{N}/Ro_{ro}/forcedTide_ring_{omega:.2f}")
savePlot = pathlib.Path(f"/mnt/pfs/rajarshi.chattopadhyay/codes/boussinesq/Plots/nu_{nu}_N_{N}/Ro_{ro}/forcedTide_ring_{omega:.2f}/3D/")
try: savePlot.mkdir(parents = True,exist_ok = True)
except FileExistsError: pass

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
    


times = np.arange(0,600.1,2.)
# times = np.arange(400,600.1,1.)
for t in times:
    if rank ==0: print(t)
    loadPath = pathlib.Path(f"/mnt/pfs/rajarshi.chattopadhyay/boussinesq/spectrum-development/nu_{nu}_N_{N}/Ro_{ro}/forcedTide_ring_{omega:.2f}/time_{t:.1f}")
    for qq in range(Ns):
        Fields = np.load(loadPath/f"Fields_{(rank*Ns + qq):.0f}.npz")
        u[0,qq*n_slab:(qq + 1)*n_slab]  = Fields["u"]
        u[1,qq*n_slab:(qq + 1)*n_slab]  = Fields["v"]
        u[2,qq*n_slab:(qq + 1)*n_slab]  = Fields["w"]
        b[qq*n_slab:(qq + 1)*n_slab]  = Fields["b"]
        q[qq*n_slab:(qq + 1)*n_slab] = np.load(loadPathdata/f"PV/time_{t:.1f}/PV_{rank:.0f}.npy")
        omg[qq*n_slab:(qq + 1)*n_slab] = np.load(loadPathdata/f"zeta/time_{t:.1f}/zeta_{rank:.0f}.npy")   
    div_h[:] = diff_x(u[0],u_temp) + diff_y(u[1],v_temp)
    omg_g[:] = -ifft_cos(kh**2*invlapc*fft_cos(q,pk),omg_g)


    """ 
    We need three slices
    1. Last slab in the z direction. 
    2. First slab in the y direction. 
    3. Last slab in the x direction. 
    Given that we need the last slab in x direction, the last rank will collect all the data and plot.
    """
    
    if rank < num_process - 1:
        #? Sending the last x slab is unecessary. 
        #? Send the first y slab
        comm.send(q[:,0,:],dest = num_process - 1,tag = 0)
        comm.send(omg[:,0,:],dest = num_process - 1,tag = 1)
        comm.send(div_h[:,0,:],dest = num_process - 1,tag = 2)
        comm.send(omg_g[:,0,:],dest = num_process - 1,tag = 3)
        
        #? Send the last z slab
        comm.send(q[:,:,-1],dest = num_process - 1,tag = 4)
        comm.send(omg[:,:,-1],dest = num_process - 1,tag = 5)
        comm.send(div_h[:,:,-1],dest = num_process - 1,tag = 6)
        comm.send(omg_g[:,:,-1],dest = num_process - 1,tag = 7)
        
    else:
        q_xslab = q[-1]
        omg_xslab = omg[-1]
        div_h_xslab = div_h[-1]
        omg_g_xslab = omg_g[-1]
        
        #? Receiving the first y slab and last z slab
        q_yslab = np.zeros((N,N))
        omg_yslab = np.zeros((N,N))
        div_h_yslab = np.zeros((N,N))
        omg_g_yslab = np.zeros((N,N))
        
        q_zslab = np.zeros((N,N))
        omg_zslab = np.zeros((N,N))
        div_h_zslab = np.zeros((N,N))
        omg_g_zslab = np.zeros((N,N))
        
        q_yslab[ rank*Np:(rank+1)*Np,:] = q[:,0,:]
        omg_yslab[ rank*Np:(rank+1)*Np,:] = omg[:,0,:]
        div_h_yslab[ rank*Np:(rank+1)*Np,:] = div_h[:,0,:]
        omg_g_yslab[ rank*Np:(rank+1)*Np,:] = omg_g[:,0,:]
        
        q_zslab[ rank*Np:(rank+1)*Np,:] = q[:,:,-1]
        omg_zslab[ rank*Np:(rank+1)*Np,:] = omg[:,:,-1]
        div_h_zslab[ rank*Np:(rank+1)*Np,:] = div_h[:,:,-1]
        omg_g_zslab[ rank*Np:(rank+1)*Np,:] = omg_g[:,:,-1]
        
        for ii in range(num_process - 1):
            q_yslab[ ii*Np:(ii+1)*Np,:] = comm.recv(source = ii,tag = 0)
            omg_yslab[ ii*Np:(ii+1)*Np,:] = comm.recv(source = ii,tag = 1)
            div_h_yslab[ ii*Np:(ii+1)*Np,:] = comm.recv(source = ii,tag = 2)
            omg_g_yslab[ ii*Np:(ii+1)*Np,:] = comm.recv(source = ii,tag = 3)
            
            q_zslab[ ii*Np:(ii+1)*Np,:] = comm.recv(source = ii,tag = 4)
            omg_zslab[ ii*Np:(ii+1)*Np,:] = comm.recv(source = ii,tag = 5)
            div_h_zslab[ ii*Np:(ii+1)*Np,:] = comm.recv(source = ii,tag = 6)
            omg_g_zslab[ ii*Np:(ii+1)*Np,:] = comm.recv(source = ii,tag = 7)
            
        ## --------------- Saving the slabs ----------------- ##
        savePath = loadPathdata/f"Slabs/time_{t:.1f}"
        try: savePath.mkdir(parents = True,exist_ok = True)
        except FileExistsError: pass
        np.save(savePath/f"q_yslab_0.npy",q_yslab)    
        np.save(savePath/f"omg_yslab_0.npy",omg_yslab)
        np.save(savePath/f"div_h_yslab_0.npy",div_h_yslab)
        np.save(savePath/f"omg_g_yslab_0.npy",omg_g_yslab)
        
        np.save(savePath/f"q_zslab_383.npy",q_zslab)
        np.save(savePath/f"omg_zslab_383.npy",omg_zslab)
        np.save(savePath/f"div_h_zslab_383.npy",div_h_zslab)
        np.save(savePath/f"omg_g_zslab_383.npy",omg_g_zslab)
        
        np.save(savePath/f"q_xslab_383.npy",q_xslab)
        np.save(savePath/f"omg_xslab_383.npy",omg_xslab)
        np.save(savePath/f"div_h_xslab_383.npy",div_h_xslab)
        np.save(savePath/f"omg_g_xslab_383.npy",omg_g_xslab)
        
        print(f"Saved the slabs at time {t:.1f}")
        
            
        ## ------------------ Plotting--------------- ##
        
        # q_min = -1.
        # q_max = 1.
        # q_center = 0.
        
        # omg_min = -1.
        # omg_max = 1.
        # omg_center = 0.
        
        # div_h_min = -1
        # div_h_max = 1
        # div_h_center = 0.
        
        # omg_g_min = -1.
        # omg_g_max = 1.
        # omg_g_center = 0.
        
        # q_zslab[:] = np.clip(q_zslab,q_min,q_max)
        # q_yslab[:] = np.clip(q_yslab,q_min,q_max)
        # q_xslab[:] = np.clip(q_xslab,q_min,q_max)
        
        # omg_zslab[:] = np.clip(omg_zslab,omg_min,omg_max)
        # omg_yslab[:] = np.clip(omg_yslab,omg_min,omg_max)
        # omg_xslab[:] = np.clip(omg_xslab,omg_min,omg_max)
        
        # div_h_zslab[:] = np.clip(div_h_zslab,div_h_min,div_h_max)
        # div_h_yslab[:] = np.clip(div_h_yslab,div_h_min,div_h_max)
        # div_h_xslab[:] = np.clip(div_h_xslab,div_h_min,div_h_max)
        
        # omg_g_zslab[:] = np.clip(omg_g_zslab,omg_g_min,omg_g_max)
        # omg_g_yslab[:] = np.clip(omg_g_yslab,omg_g_min,omg_g_max)
        # omg_g_xslab[:] = np.clip(omg_g_xslab,omg_g_min,omg_g_max)
        
         
        # cset = [[],[],[]]
        # x1, y1 = np.meshgrid(X, Y)
        # x2, z2 = np.meshgrid(X, Z)
        # y3, z3 = np.meshgrid(Y, Z)
        
        # t1 = time()
        # cmap = 'RdYlBu_r'
        # q_norm = TwoSlopeNorm(vmin=q_min, vmax=q_max, vcenter=q_center)
        # omg_norm = TwoSlopeNorm(vmin=omg_min, vmax=omg_max, vcenter=omg_center)
        # div_h_norm = TwoSlopeNorm(vmin=div_h_min, vmax=div_h_max, vcenter=div_h_center)
        # omg_g_norm = TwoSlopeNorm(vmin=omg_g_min, vmax=omg_g_max, vcenter=omg_g_center)
        
        
        # fig = plt.figure(figsize=plt.figaspect(3/4))  
        
        # ax1 = fig.add_subplot(2,2,1,projection='3d')
        # cset[0] = ax1.contourf(x1, y1, q_zslab.T,zdir = 'z',offset = Z[-1],cmap=cmap,norm=q_norm,levels = 50)
        # cset[1] = ax1.contourf(x2, q_yslab.T, z2,zdir = 'y',offset = 0,cmap=cmap,norm=q_norm,levels = 50)
        # cset[2] = ax1.contourf(q_xslab.T,y3,z3,zdir = 'x',offset = X[-1],cmap=cmap,norm=q_norm,levels = 50)
        # ax1.set_xlim3d(X[0], X[-1])
        # ax1.set_ylim3d(Y[0], Y[-1])
        # ax1.set_zlim3d(Z[0], Z[-1])

        # # ax1.axis('off')  # Remove axis lines


        # # colb=fig.colorbar(cset[1],shrink = 0.4)
        # # colb.formatter.set_powerlimits((0, 0))
        # # colb.ax.tick_params(labelsize=8)
        # # tick_locator = ticker.MaxNLocator(nbins=3)
        # # colb.locator = tick_locator
        # # colb.update_ticks()


        # ax1.set_xlabel(r'$x$')
        # ax1.set_ylabel(r'$y$')
        # ax1.set_zlabel(r'$z$')
        # # 
        # ax1.set_xticks([])
        # ax1.set_yticks([])
        # ax1.set_zticks([])

        # ax1.xaxis.line.set_linewidth(0)
        # ax1.yaxis.line.set_linewidth(0)
        # ax1.zaxis.line.set_linewidth(0)


        # ax1.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # ax1.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # ax1.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

        # # Rest of your code...

        # ax1.grid(False)
        # ax1.set_xticklabels([])
        # ax1.set_yticklabels([])
        # ax1.set_zticklabels([])
        # # ax1.axis('off')
        
        # ax2 = fig.add_subplot(2,2,2,projection='3d')
        # cset[0] = ax2.contourf(x1, y1, omg_zslab.T,zdir = 'z',offset = Z[-1],cmap=cmap,norm=omg_norm,levels = 50)
        # cset[1] = ax2.contourf(x2, omg_yslab.T, z2,zdir = 'y',offset = 0,cmap=cmap,norm=omg_norm,levels = 50)
        # cset[2] = ax2.contourf(omg_xslab.T,y3,z3,zdir = 'x',offset = X[-1],cmap=cmap,norm=omg_norm,levels = 50)
        # ax2.set_xlim3d(X[0], X[-1])
        # ax2.set_ylim3d(Y[0], Y[-1])
        # ax2.set_zlim3d(Z[0], Z[-1])
        
        # # colb=fig.colorbar(cset[1],shrink = 0.4)
        # # colb.formatter.set_powerlimits((0, 0))
        # # colb.ax.tick_params(labelsize=8)
        
        # ax2.set_xlabel(r'$x$')
        # ax2.set_ylabel(r'$y$')
        # ax2.set_zlabel(r'$z$')
        
        # ax2.set_xticks([])
        # ax2.set_yticks([])
        # ax2.set_zticks([])
        
        # ax2.xaxis.line.set_linewidth(0)
        # ax2.yaxis.line.set_linewidth(0)
        # ax2.zaxis.line.set_linewidth(0)
        
        # ax2.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # ax2.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # ax2.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        
        # ax2.grid(False)
        # ax2.set_xticklabels([])
        # ax2.set_yticklabels([])
        # ax2.set_zticklabels([])
        
        # ax3 = fig.add_subplot(2,2,3,projection='3d')
        # cset[0] = ax3.contourf(x1, y1, div_h_zslab.T,zdir = 'z',offset = Z[-1],cmap=cmap,norm=div_h_norm,levels = 50)
        # cset[1] = ax3.contourf(x2, div_h_yslab.T, z2,zdir = 'y',offset = 0,cmap=cmap,norm=div_h_norm,levels = 50)
        # cset[2] = ax3.contourf(div_h_xslab.T,y3,z3,zdir = 'x',offset = X[-1],cmap=cmap,norm=div_h_norm,levels = 50)
        # ax3.set_xlim3d(X[0], X[-1])
        # ax3.set_ylim3d(Y[0], Y[-1])
        # ax3.set_zlim3d(Z[0], Z[-1])
        
        # # colb=fig.colorbar(cset[1],shrink = 0.4)
        # # colb.formatter.set_powerlimits((0, 0))
        # # colb.ax.tick_params(labelsize=8)
        
        # ax3.set_xlabel(r'$x$')
        # ax3.set_ylabel(r'$y$')
        # ax3.set_zlabel(r'$z$')
        
        # ax3.set_xticks([])
        # ax3.set_yticks([])
        # ax3.set_zticks([])
        
        # ax3.xaxis.line.set_linewidth(0)
        # ax3.yaxis.line.set_linewidth(0)
        # ax3.zaxis.line.set_linewidth(0)
        
        # ax3.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # ax3.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # ax3.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        
        # ax3.grid(False)
        # ax3.set_xticklabels([])
        # ax3.set_yticklabels([])
        # ax3.set_zticklabels([])
        
        # ax4 = fig.add_subplot(2,2,4,projection='3d')
        # cset[0] = ax4.contourf(x1, y1, omg_g_zslab.T,zdir = 'z',offset = Z[-1],cmap=cmap,norm=omg_g_norm,levels = 50)
        # cset[1] = ax4.contourf(x2, omg_g_yslab.T, z2,zdir = 'y',offset = 0,cmap=cmap,norm=omg_g_norm,levels = 50)
        # cset[2] = ax4.contourf(omg_g_xslab.T,y3,z3,zdir = 'x',offset = X[-1],cmap=cmap,norm=omg_g_norm,levels = 50)
        # ax4.set_xlim3d(X[0], X[-1])
        # ax4.set_ylim3d(Y[0], Y[-1])
        # ax4.set_zlim3d(Z[0], Z[-1])
        
        # # colb=fig.colorbar(cset[1],shrink = 0.4)
        # # colb.formatter.set_powerlimits((0, 0))
        # # colb.ax.tick_params(labelsize=8)
        
        # ax4.set_xlabel(r'$x$')
        # ax4.set_ylabel(r'$y$')
        # ax4.set_zlabel(r'$z$')
        
        # ax4.set_xticks([])
        # ax4.set_yticks([])
        # ax4.set_zticks([])
        
        # ax4.xaxis.line.set_linewidth(0)
        # ax4.yaxis.line.set_linewidth(0)
        # ax4.zaxis.line.set_linewidth(0)
        
        # ax4.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # ax4.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # ax4.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        
        # ax4.grid(False)
        # ax4.set_xticklabels([])
        # ax4.set_yticklabels([])
        # ax4.set_zticklabels([])
                
        # ax1.view_init(elev=40., azim=-45)
        # ax2.view_init(elev=40., azim=-45)
        # ax3.view_init(elev=40., azim=-45)
        # ax4.view_init(elev=40., azim=-45)
        
        
        # # fig.view_init(elev=40., azim=-45,ax = [ax1,ax2,ax3,ax4])
            
        # ax1.set_title(r'$q$')
        # ax2.set_title(r'$\omega$')
        # ax3.set_title(r'$\nabla \cdot \mathbf{u}$')
        # ax4.set_title(r'$\omega_G$')
        # cbar = fig.colorbar(cset[1], ax = [ax1,ax2,ax3,ax4],shrink  = 0.6,orientation = 'vertical')
        # cbar.formatter.set_powerlimits((0,0))
        # cbar.ax.tick_params(labelsize=7)
        
        # # plt.tight_layout()
        # plt.savefig(savePlot/f"All3D_{t:.1f}.png",dpi = 300)
        # plt.close()
        # t2 = time()
        # print(f"time taken to plot {(t2-t1)/60:.2f} mins")
        