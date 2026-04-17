
import numpy as np
import os
from scipy.fft import rfftn,dct,dst,idct,idst,irfftn

data  = np.load(f"/home/rudra_data/hd21/ictsuser1/Ro_0.5/forcedTide_ring_LW/time_1055.0/Fields_0.npz")

data.files

N = 384

def fft_cos_yz(x):
    return rfftn(dct(x,type=2, axis= -1),axes = -2)
def fft_sin_yz(x):
    return rfftn(dst(x,type=2, axis= -1),axes = -2)
def ifft_cos_yz(x):
    return idct(irfftn(x,N,axes = -2),type=2, axis= -1)
def ifft_sin_yz(x):
    return idst(irfftn(x,N,axes = -2),type=2, axis= -1)



def original_truncate(data,N = 384):
    u = data['u']
    v = data['v']
    w = data['w']
    b = data['b']

    uk = fft_cos_yz(u)
    vk = fft_cos_yz(v)
    wk = fft_sin_yz(w)
    bk = fft_sin_yz(b)

    uk_trunc = uk[...,:N//3+1,:(2*N)//3+1].copy()
    vk_trunc = vk[...,:N//3+1,:(2*N)//3+1].copy()
    wk_trunc = wk[...,:N//3+1,:(2*N)//3+1].copy()
    bk_trunc = bk[...,:N//3+1,:(2*N)//3+1].copy()

    return uk_trunc, vk_trunc, wk_trunc,bk_trunc

def load_trunc(data,N = 384):
    
    uk_trunc = data['uk_trunc']
    vk_trunc = data['vk_trunc']
    wk_trunc = data['wk_trunc']
    bk_trunc = data['bk_trunc']

    uk = np.zeros((*uk_trunc.shape[:-2], N//2+ 1,N),dtype = np.complex128)
    vk = np.zeros((*vk_trunc.shape[:-2], N//2+ 1,N),dtype = np.complex128)
    wk = np.zeros((*wk_trunc.shape[:-2], N//2+ 1,N),dtype = np.complex128)
    bk = np.zeros((*bk_trunc.shape[:-2], N//2+ 1,N),dtype = np.complex128)
    
    uk[...,:uk_trunc.shape[-2], :uk_trunc.shape[-1]] = uk_trunc
    vk[...,:vk_trunc.shape[-2], :uk_trunc.shape[-1]] = vk_trunc
    wk[...,:wk_trunc.shape[-2], :uk_trunc.shape[-1]] = wk_trunc
    bk[...,:bk_trunc.shape[-2], :uk_trunc.shape[-1]] = bk_trunc

    u = ifft_cos_yz(uk)
    v = ifft_cos_yz(vk)
    w = ifft_sin_yz(wk)
    b = ifft_sin_yz(bk)
    
    return u,v,w,b

def check_old_new(data_old,data_new):
    u = data['u']
    v = data['v']
    w = data['w']
    b = data['b']

    u_new,v_new,w_new,b_new = load_trunc(data_new)
    
    u_err = np.max(np.abs(u - u_new))
    v_err = np.max(np.abs(v - v_new))
    w_err = np.max(np.abs(w - w_new))
    b_err = np.max(np.abs(b - b_new))

    return u_err,v_err,w_err,b_err
    

uk_trunc, vk_trunc, wk_trunc,bk_trunc = original_truncate(data)

np.savez_compressed(f"/home/rudra_data/hd21/ictsuser1/Ro_0.5/forcedTide_ring_LW/time_1055.0/Fields_cmp_0.npz",uk_trunc = uk_trunc,vk_trunc = vk_trunc, wk_trunc = wk_trunc ,  bk_trunc = bk_trunc)

data_new = np.load(f"/home/rudra_data/hd21/ictsuser1/Ro_0.5/forcedTide_ring_LW/time_1055.0/Fields_cmp_0.npz")

check_old_new(data,data_new)


for time in np.arange(1000,1100,0.1):
    for slab in range(N):
        try: 
            data  = np.load(f"/home/rudra_data/hd21/ictsuser1/Ro_0.5/forcedTide_ring_LW/time_{time:.1f}/Fields_{slab}.npz")
            print(f"{time}, slab {slab}",end = "\r")
        except FileNotFoundError: 
            # print("File not Found")
            continue
        uk_trunc, vk_trunc, wk_trunc,bk_trunc = original_truncate(data)
        np.savez_compressed(f"/home/rudra_data/hd21/ictsuser1/Ro_0.5/forcedTide_ring_LW/time_{time:.1f}/Fields_cmp_{slab}.npz",uk_trunc = uk_trunc,vk_trunc = vk_trunc, wk_trunc = wk_trunc ,  bk_trunc = bk_trunc)
        data_new = np.load(f"/home/rudra_data/hd21/ictsuser1/Ro_0.5/forcedTide_ring_LW/time_{time:.1f}/Fields_cmp_{slab}.npz")
        u_err,v_err,w_err,b_err = check_old_new(data,data_new)
        if u_err + v_err + w_err + b_err > 1e-14: 
            print(f"Too much error: {u_err + v_err + w_err + b_err}. Deleting truncated file")
            os.remove(f"/home/rudra_data/hd21/ictsuser1/Ro_0.5/forcedTide_ring_LW/time_{time:.1f}/Fields_cmp_{slab}.npz")
        else: 
            os.remove(f"/home/rudra_data/hd21/ictsuser1/Ro_0.5/forcedTide_ring_LW/time_{time:.1f}/Fields_{slab}.npz")
        # print(f"{slab,u_err + v_err + w_err + b_err}",end = "\r")

# u,v,w,b = load_trunc(np.load(f"/home/rudra_data/hd21/ictsuser1/Ro_0.5/forcedTide_ring_LW/time_1099.9/Fields_cmp_0.npz"))




