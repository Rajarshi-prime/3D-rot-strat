#%%
import numpy as np
import matplotlib.pyplot as plt
import pathlib,os,re
# %%
datapath1 = pathlib.Path("/mnt/pfs/rajarshi.chattopadhyay/codes/boussinesq/data/bsnq_phs_shifted/f_1.0_Nb_20/forced_True/N_256_Re_815674511287550568453837422592.0") # f10N100

datapath2 = pathlib.Path(f"/mnt/pfs/rajarshi.chattopadhyay/codes/boussinesq/data/bsnq/f_1.0_Nb_20/forced_True/N_384_Re_408867756835691895762583442948096.0") # f1N100

datapath3 = pathlib.Path("/mnt/pfs/rajarshi.chattopadhyay/codes/boussinesq/data/forced_True/N_384_Re_6189700196426916767658409984.0") # HIT

nu1 = 1/815674511287550568453837422592.0
nu2 = 1/408867756835691895762583442948096.0
nu3 = 1/6189700196426916767658409984.0
#%%
times1 = [float(i.split("_")[-1]) for i in os.listdir(datapath1) if "time" in i]
times2 = [float(i.split("_")[-1]) for i in os.listdir(datapath2) if "time" in i]
times3 = [float(i.split("_")[-1]) for i in os.listdir(datapath3) if "time" in i]

# %%
times1.sort()
times2.sort()
times3.sort()
#%%
times1, times2,times3
# %%
k = np.arange(384//2 + 1,dtype = np.float64)
etot1 = []
etot2 = []
etot3 = []
dissip1 =0.
Etot1 = 0.
dissip2 = 0.
Etot2 = 0.
dissip3 = 0.
Etot3 = 0.

count1 = 0.
count2 = 0.
count3 = 0.
for i,t in enumerate(times1):
    ek = np.load(datapath1/f"time_{t:.1f}/Energy_spectrum.npz")["ek"]
    # dissip1 += np.sum((nu1*k**16)*ek)
    # Etot1 += np.sum(ek)
    etot1.append(np.sum(ek))
    count1 += 1
    
for i,t in enumerate(times2):
    ek = np.load(datapath2/f"time_{t:.1f}/Energy_spectrum.npz")["ek"]
    # dissip2 += np.sum((nu2*k**16)*ek)
    # Etot2 += np.sum(ek)
    etot2.append(np.sum(ek))
    count2 += 1
    
for i,t in enumerate(times3):
    ek = np.load(datapath3/f"time_{t:.1f}/Energy_spectrum.npz")["ek"]
    # dissip2 += np.sum((nu2*k**16)*ek)
    # Etot2 += np.sum(ek)
    etot3.append(np.sum(ek))
    count3 += 1
    
# %%
# dissip1/Etot1, dissip2/Etot2
# %%
# Etot1/count1,Etot2/count2
# %%
# dissip1/count1,dissip2/count2
# %%
plt.plot(etot1,label = "f10N100")
plt.plot(etot2,label = "f1N100")
plt.plot(etot3,label = "HIT")
plt.legend()
plt.xlabel("t")
plt.ylabel("E",rotation = 0)
plt.show()
# %%
ek1 = 0.
ek2 = 0.
ek3 = 0.
pik1 = 0.
pik2 = 0.
pik3 = 0.
# for i in times1[-50:]:
#     ek1 += np.load(datapath1/f"time_{times1[-1]:.1f}/Energy_spectrum.npz")["ek"]/50
#     pik1 += np.load(datapath1/f"time_{times1[-1]:.1f}/Flux_spectrum.npz")["Pik"]/50
# for i in times2[-50:]:
#     ek2 += np.load(datapath2/f"time_{times2[-1]:.1f}/Energy_spectrum.npz")["ek"]/50
#     pik2 += np.load(datapath2/f"time_{times2[-1]:.1f}/Flux_spectrum.npz")["Pik"]/50
for i in times3[-50:]:
    ek3 += np.load(datapath3/f"time_{times3[-1]:.1f}/Energy_spectrum.npz")["ek"]/50
    pik3 += np.load(datapath3/f"time_{times3[-1]:.1f}/Flux_spectrum.npz")["Pik"]/50

# %%
plt.plot(ek1,label = "f10N100")
plt.plot(ek2,label = "f1N100")
plt.plot(ek3,label = "HIT")
plt.xscale("log")
plt.yscale("log")
plt.ylim(1e-6,None)
plt.legend()
nu1,nu2,nu3
# %%
plt.plot(pik1,label = "f10N100")
plt.plot(pik2,label = "f1N100")
plt.plot(pik3,label = "HIT")
plt.xscale("log")
# plt.yscale("log")
# plt.ylim(1e-6,None)
plt.legend()
nu1,nu2,nu3

# %%
# out_file_path = pathlib.Path("/mnt/pfs/rajarshi.chattopadhyay/codes/boussinesq/outfiles/f1N10_6927174.out")  # f0.1N10
out_file_path = pathlib.Path("/mnt/pfs/rajarshi.chattopadhyay/codes/boussinesq/outfiles/f1N10-Pwr_6927517.out")  # f0.1N10
e1 = []
t1 = []
with open(out_file_path, 'r') as file:
    for line in file:
        if "Energy at time" in line:
            e1.append(float(re.split(": | ,",line)[1].split(",")[0]))
            t1.append(float(re.split("time | is",line)[1]))


# %%
out_file_path = pathlib.Path("/mnt/pfs/rajarshi.chattopadhyay/codes/boussinesq/outfiles/f1N2.5-TWOPI_6927093.out")  # f1N20
e2 = []
t2 = []
with open(out_file_path, 'r') as file:
    for line in file:
        if "Energy at time" in line:
            e2.append(float(re.split(": | ,",line)[1].split(",")[0]))
            t2.append(float(re.split("time | is",line)[1]))
# %%
out_file_path = pathlib.Path("/mnt/pfs/rajarshi.chattopadhyay/codes/boussinesq/outfiles/f1N5-TWOPI_6927094.out")  # f1N20
e3 = []
t3 = []
with open(out_file_path, 'r') as file:
    for line in file:
        if "Energy at time" in line:
            e3.append(float(re.split(": | ,",line)[1].split(",")[0]))
            t3.append(float(re.split("time | is",line)[1]))
            
#%%
out_file_path = pathlib.Path("/mnt/pfs/rajarshi.chattopadhyay/codes/boussinesq/outfiles/f1N10_6927174.out")  # f1N20
e4 = []
t4 = []
with open(out_file_path, 'r') as file:
    for line in file:
        if "Energy at time" in line:
            e4.append(float(re.split(": | ,",line)[1].split(",")[0]))
            t4.append(float(re.split("time | is",line)[1]))
#%%

#%%

# plt.plot(t1,np.gradient(e1,t1)/np.array(e1),label = "f1N20")
plt.plot(t1,e1,label = "f1N10",ls = '--')
plt.plot(t2,e2,label = "f1N2.5",ls = '--')
plt.plot(t3,e3,label = "f1N5",ls = '--')
plt.plot(t4,e4,label = "f1N20",ls = '--')
# plt.ylim(0,0.01)
plt.legend()
plt.xlim()
plt.xlabel("t")
plt.ylabel("E(t)")
# plt.ylabel(r"$\dot{E}/E(t)$")
# plt.plot(t1,np.gradient(e1,t1))
# plt.plot(t2,np.gradient(e2,t2))
# %%
ek1= np.load(f"./data/bsnq/f_1.0_Nb_2.5/forced_True/N_256_Re_815674511287550568453837422592.0/last/Energy_spectrum.npz")["ek"]
# ek2= np.load(datapath2/f"last/Energy_spectrum.npz")["ek"]

# %%
plt.plot(np.arange(ek1.size),ek1,label = "f1N5")
# plt.plot(np.arange(ek2.size),ek2,label = "f1N20")
# plt.plot(np.arange(ek3.size),ek3)
plt.xscale("log")
plt.yscale("log")
plt.ylim(1e-6,None)
plt.legend()
plt.xlabel("k")
plt.ylabel(r"$e_k$")
  # %%
Pik1= np.load(f"./data/bsnq/f_1.0_Nb_2.5/forced_True/N_256_Re_815674511287550568453837422592.0/last/Flux_spectrum.npz")["Pik"]
# Pik2= np.load(datapath2/f"last/Flux_spectrum.npz")["Pik"]
# %%
plt.plot(np.arange(Pik1.size),Pik1,label = "f1N2.5")
# plt.plot(np.arange(Pik2.size),Pik2,label = "f1N20")
# plt.plot(np.arange(ek3.size),ek3)
plt.xscale("log")
plt.legend()
plt.xlabel("k")
plt.ylabel(r"$\Pi_k$")
# plt.yscale("log")
# plt.ylim(1e-6,None)
# %%
