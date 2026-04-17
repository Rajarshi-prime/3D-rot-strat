import numpy as np
from scipy.fft import fftn,ifftn, fft ,  ifft ,  irfft2 ,  rfft2 , irfftn ,  rfftn, fftfreq, dst, dct, idst, idct, rfft,  irfft
import matplotlib.pyplot as plt
from mpi4py import MPI
import pathlib,os
import matplotlib as mpl 
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
from time import time
# import h5py
# import math
# import array
# from matplotlib.colors import LightSource
#plt.style.use('ggplot')
#from palettable.cartocolors.diverging import Geyser_7
# from matplotlib import cm
from matplotlib.colors import TwoSlopeNorm
mpl.rcParams['axes.labelpad'] = -8
mpl.rcParams['font.size'] = 10
mpl.rc('text', usetex = True)




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
num_slabs = 384
Ns = num_slabs// num_process
n_slab = N//num_slabs
omega = 1.7277

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

loadPath = pathlib.Path(f"/mnt/pfs/rajarshi.chattopadhyay/boussinesq/spectrum-development/nu_{nu}_N_{N}/Ro_{ro}/forcedTide_ring_{omega:.2f}_LW/")
savePlot = pathlib.Path(f"/mnt/pfs/rajarshi.chattopadhyay/codes/boussinesq/Plots/nu_{nu}_N_{N}/Ro_{ro}/forcedTide_ring_{omega:.2f}_LW/3D/")
try: savePlot.mkdir(parents = True,exist_ok = True)
except FileExistsError: pass

# loadPathdata.exists(),str(loadPath)
# len(os.listdir(loadPath))
q_yslab = np.zeros((N,N))
omg_yslab = np.zeros((N,N))
div_h_yslab = np.zeros((N,N))
omg_g_yslab = np.zeros((N,N))

q_zslab = np.zeros((N,N))
omg_zslab = np.zeros((N,N))
div_h_zslab = np.zeros((N,N))
omg_g_zslab = np.zeros((N,N))


div_h_min = -1.1
div_h_max = 1.1
div_h_center = 0.

omg_g_min = -1.1
omg_g_max = 1.1
omg_g_center = 0.

q_xslab = np.zeros((N,N))
omg_xslab = np.zeros((N,N))
div_h_xslab = np.zeros((N,N))
omg_g_xslab = np.zeros((N,N))

x1, y1 = np.meshgrid(X, Y)
x2, z2 = np.meshgrid(X, Z)
y3, z3 = np.meshgrid(Y, Z)
cmap = 'RdBu_r'
# omg_norm = TwoSlopeNorm(vmin=omg_min, vmax=omg_max, vcenter=omg_center)
div_h_norm = TwoSlopeNorm(vmin=div_h_min, vmax=div_h_max, vcenter=div_h_center)
omg_g_norm = TwoSlopeNorm(vmin=omg_g_min, vmax=omg_g_max, vcenter=omg_g_center)
div_h_levels = np.linspace(div_h_min,div_h_max,100)
omg_g_levels = np.linspace(omg_g_min,omg_g_max,100)
edges_kw = dict(color='0.4', linewidth=1, zorder=1e3)


for i in range(3):
    t = 2*(rank + i*num_process)
    print("plotting for time",rank + num_process*i)
  
    try:
        loadData = loadPath/f"Slabs/time_{t:.1f}"
        # q_yslab = np.load(loadData/f"q_yslab_0.npy")    
        # omg_yslab = np.load(loadData/f"omg_yslab_0.npy")
        div_h_yslab = np.load(loadData/f"div_h_yslab_0.npy")
        omg_g_yslab = np.load(loadData/f"omg_g_yslab_0.npy")
        
        # q_zslab = np.load(loadData/f"q_zslab_383.npy")
        # omg_zslab = np.load(loadData/f"omg_zslab_383.npy")
        div_h_zslab = np.load(loadData/f"div_h_zslab_383.npy")
        omg_g_zslab = np.load(loadData/f"omg_g_zslab_383.npy")
        
        # q_xslab = np.load(loadData/f"q_xslab_383.npy")
        # omg_xslab = np.load(loadData/f"omg_xslab_383.npy")
        div_h_xslab = np.load(loadData/f"div_h_xslab_383.npy")
        omg_g_xslab = np.load(loadData/f"omg_g_xslab_383.npy")
    except FileNotFoundError:
        print("File not found for time",t)
        continue
            
    # if np.isnan(q_yslab + omg_yslab + div_h_yslab + omg_g_yslab + q_zslab + omg_zslab + div_h_zslab + omg_g_zslab + q_xslab + omg_xslab + div_h_xslab + omg_g_xslab).any(): print("nan values found for time",t)
    ## ------------------ Plotting--------------- ##
    
    # q_min = -0.1
    # q_max = 0.1
    # q_center = 0.
    
    # omg_min = q_min
    # omg_max = q_max
    # omg_center = 0.
    
    
    # q_zslab[:] = np.clip(q_zslab,q_min,q_max)
    # q_yslab[:] = np.clip(q_yslab,q_min,q_max)
    # q_xslab[:] = np.clip(q_xslab,q_min,q_max)
    
    # omg_zslab[:] = np.clip(omg_zslab,omg_min,omg_max)
    # omg_yslab[:] = np.clip(omg_yslab,omg_min,omg_max)
    # omg_xslab[:] = np.clip(omg_xslab,omg_min,omg_max)
    
    div_h_zslab[:] = np.clip(div_h_zslab,div_h_min,div_h_max)
    div_h_yslab[:] = np.clip(div_h_yslab,div_h_min,div_h_max)
    div_h_xslab[:] = np.clip(div_h_xslab,div_h_min,div_h_max)
    
    omg_g_zslab[:] = np.clip(omg_g_zslab,omg_g_min,omg_g_max)
    omg_g_yslab[:] = np.clip(omg_g_yslab,omg_g_min,omg_g_max)
    omg_g_xslab[:] = np.clip(omg_g_xslab,omg_g_min,omg_g_max)
    
        
    # cset = [[],[],[]]
    
    t1 = time()
    # q_norm = TwoSlopeNorm(vmin=q_min, vmax=q_max, vcenter=q_center)
    
    fig = plt.figure(figsize=plt.figaspect(2/4),dpi = 200)  
    
    # ax1 = fig.add_subplot(1,2,1,projection='3d')
    # _ = ax1.contourf(x1, y1, q_zslab.T,zdir = 'z',offset = Z[-1],cmap=cmap,norm=q_norm,levels = 100)
    # cset[1] = ax1.contourf(x2, q_yslab.T, z2,zdir = 'y',offset = 0,cmap=cmap,norm=q_norm,levels = 100)
    # cset[2] = ax1.contourf(q_xslab.T,y3,z3,zdir = 'x',offset = X[-1],cmap=cmap,norm=q_norm,levels = 100)
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
    # cset[0] = ax2.contourf(x1, y1, omg_zslab.T,zdir = 'z',offset = Z[-1],cmap=cmap,norm=omg_norm,levels = 100)
    # cset[1] = ax2.contourf(x2, omg_yslab.T, z2,zdir = 'y',offset = 0,cmap=cmap,norm=omg_norm,levels = 100)
    # cset[2] = ax2.contourf(omg_xslab.T,y3,z3,zdir = 'x',offset = X[-1],cmap=cmap,norm=omg_norm,levels = 100)
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
    
    ax3 = fig.add_subplot(1,2,1,projection='3d')
    _ = ax3.contourf(x1, y1, div_h_zslab.T,zdir = 'z',offset = Z[-1],cmap=cmap,norm=div_h_norm,levels = 100)
    _ = ax3.contourf(x2, div_h_yslab.T, z2,zdir = 'y',offset = 0,cmap=cmap,norm=div_h_norm,levels = 100)
    C = ax3.contourf(div_h_xslab.T,y3,z3,zdir = 'x',offset = X[-1],cmap=cmap,norm=div_h_norm,levels = 100)
    ax3.set_xlim3d(X[0], X[-1])
    ax3.set_ylim3d(Y[0], Y[-1])
    ax3.set_zlim3d(Z[0], Z[-1])
    ax3.plot([X[-1], X[-1]], [Y[0], Y[-1]], Z[-1], **edges_kw)
    ax3.plot([X[0], X[-1]], [Y[0], Y[0]], Z[-1], **edges_kw)
    ax3.plot([X[-1], X[-1]], [Y[0], Y[0]], [Z[0], Z[-1]], **edges_kw)
    
    # colb=fig.colorbar(cset[1],shrink = 0.4)
    # colb.formatter.set_powerlimits((0, 0))
    # colb.ax.tick_params(labelsize=8)
    
    ax3.set_xlabel(r'$x$')
    ax3.set_ylabel(r'$y$')
    ax3.set_zlabel(r'$z$')
    
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_zticks([])
    
    ax3.xaxis.line.set_linewidth(0)
    ax3.yaxis.line.set_linewidth(0)
    ax3.zaxis.line.set_linewidth(0)
    
    ax3.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax3.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax3.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    
    ax3.grid(False)
    ax3.set_xticklabels([])
    ax3.set_yticklabels([])
    ax3.set_zticklabels([])
    
    ax4 = fig.add_subplot(1,2,2,projection='3d')
    _ = ax4.contourf(x1, y1, omg_g_zslab.T,zdir = 'z',offset = Z[-1],cmap=cmap,norm=omg_g_norm,levels = 100)
    _ = ax4.contourf(x2, omg_g_yslab.T, z2,zdir = 'y',offset = 0,cmap=cmap,norm=omg_g_norm,levels = 100)
    C = ax4.contourf(omg_g_xslab.T,y3,z3,zdir = 'x',offset = X[-1],cmap=cmap,norm=omg_g_norm,levels = 100)
    ax4.set_xlim3d(X[0], X[-1])
    ax4.set_ylim3d(Y[0], Y[-1])
    ax4.set_zlim3d(Z[0], Z[-1])
    ax4.plot([X[-1], X[-1]], [Y[0], Y[-1]], Z[-1], **edges_kw)
    ax4.plot([X[0], X[-1]], [Y[0], Y[0]], Z[-1], **edges_kw)
    ax4.plot([X[-1], X[-1]], [Y[0], Y[0]], [Z[0], Z[-1]], **edges_kw)
    
    # colb=fig.colorbar(cset[1],shrink = 0.4)
    # colb.formatter.set_powerlimits((0, 0))
    # colb.ax.tick_params(labelsize=8)
    
    ax4.set_xlabel(r'$x$')
    ax4.set_ylabel(r'$y$')
    ax4.set_zlabel(r'$z$')
    
    ax4.set_xticks([])
    ax4.set_yticks([])
    ax4.set_zticks([])
    
    ax4.xaxis.line.set_linewidth(0)
    ax4.yaxis.line.set_linewidth(0)
    ax4.zaxis.line.set_linewidth(0)
    
    ax4.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax4.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax4.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    
    ax4.grid(False)
    ax4.set_xticklabels([])
    ax4.set_yticklabels([])
    ax4.set_zticklabels([])
            
    # ax1.view_init(elev=40., azim=-45)
    # ax2.view_init(elev=40., azim=-45)
    ax3.view_init(elev=40., azim=-45)
    ax3.set_box_aspect([1,1,1], zoom=1)
    ax4.view_init(elev=40., azim=-45)
    ax4.set_box_aspect([1,1,1], zoom=1)
    
    # fig.view_init(elev=40., azim=-45,ax = [ax1,ax2,ax3,ax4])
        
    # ax1.set_title(r'$q$')
    # ax2.set_title(r'$\omega$')
    ax3.set_title(r'$\nabla \cdot \mathbf{u}$')
    ax4.set_title(r'$\zeta_G$')
    ax3.axis("off")
    ax4.axis("off")
    cbar = fig.colorbar(C, ax = [ax3,ax4],shrink  = 1,orientation = 'vertical',fraction = 0.01)
    cbar.formatter.set_powerlimits((0,0))
    cbar.ax.tick_params(labelsize=7)
    
    fig.subplotpars.update(wspace=0.1,hspace=0.1)
    # plt.tight_layout()
    plt.savefig(savePlot/f"All3D_{t:.1f}.png")
    plt.close(fig)
    t2 = time()
    # print(f"time taken to plot {(t2-t1)/60:.2f} mins")
    