"""
This code loads the calculated efficiencies from the CSV file and plots them as a function of k. 

Additionally, it plots $\epsilon \tau_e/ E$ for the different flows
"""

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


# --------------------------------------- # 

# %%
effsspectral = pathlib.Path("/mnt/pfs/rajarshi.chattopadhyay/codes/boussinesq/spectral_efficiencies.hdf5")
effsgaussian = pathlib.Path("/mnt/pfs/rajarshi.chattopadhyay/codes/boussinesq/gaussian_efficiencies.hdf5")
rotstrateffsspectral = pathlib.Path("/mnt/pfs/rajarshi.chattopadhyay/codes/boussinesq/rot_stratspectral_efficiencies.hdf5")
rotstrateffsgaussian = pathlib.Path("/mnt/pfs/rajarshi.chattopadhyay/codes/boussinesq/rot_stratgaussian_efficiencies.hdf5")
# %%
effs_gaussian = []
flux_gaussian = []
k_filts = [] 
with h5py.File(effsgaussian,'r') as file:
    for i,grp in enumerate(file.keys()):
        if "256" not in grp:
            effs_gaussian.append(file[f"{grp}/eff_mean"][:])
            flux_gaussian.append(file[f"{grp}/flux_mean"][:])
            k_filts.append(file[f"{grp}/k_filts"][:])
#%%
effs_spectral = []
flux_spectral = []
k_filts = [] 
with h5py.File(effsspectral,'r') as file:
    for i,grp in enumerate(file.keys()):
            effs_spectral.append(file[f"{grp}/eff_mean"][:])
            flux_spectral.append(file[f"{grp}/flux_mean"][:])
            k_filts.append(file[f"{grp}/k_filts"][:])
#%%
mpl.rc('text',usetex = True)
Ns = [384, 512, 1024]
cls = ["#FF6666","#5D2E8C",  "#2EC4B6" ]
fig,ax = plt.subplots(1,2,figsize = (12,6))


resolhandles = []
for i, N in enumerate(Ns):
    h1, = ax[0].plot(k_filts[i],effs_gaussian[i],'o-',label = f"Gaussian",color = cls[i])
    hr, = ax[0].plot(k_filts[i],effs_gaussian[i],'-',color = cls[i])
    ax[1].plot(k_filts[i],flux_gaussian[i],'o-',label = f"Gaussian",color = cls[i])
    ax[1].plot(k_filts[i],flux_gaussian[i],'-',color = cls[i])
    resolhandles.append(h1)
    
    h2, = ax[0].plot(k_filts[i],effs_spectral[i],'^',label = f"Spectral",color = cls[i])
    ax[0].plot(k_filts[i],effs_spectral[i],'-',color = cls[i])
    ax[1].plot(k_filts[i],flux_spectral[i],'^',label = f"Spectral",color = cls[i])
    ax[1].plot(k_filts[i],flux_spectral[i],'-',color = cls[i])
handles = [h1,h2]
labels = ["Gaussian","Spectral"]
for axis in ax:
    axis.set_xlabel(r"$k$")
    axis.set_xscale("log")
    leg = axis.legend(handles, labels,handlelength = 1, ncols = 1,labelcolor='black')
    for handle in leg.get_lines(): handle.set_color("black")
    
leg = fig.legend(resolhandles, Ns, ncols = len(Ns), handlelength = 1,loc = 'upper center', bbox_to_anchor = (0.5,1.1))
ax[0].set_ylabel(r"$\langle \Gamma(k) \rangle$")
ax[1].set_ylabel(r"$\langle \Pi(k) \rangle$")
fig.tight_layout()
# %%
