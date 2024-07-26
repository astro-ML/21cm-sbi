import h5py as h5
import torch
import os, fnmatch
from py21cmfast import compute_tau
import numpy as np
from alive_progress import alive_bar
#from ps2d_for_sbi import *
from torchsummary import summary
from matplotlib import pyplot as plt
from py21cmfast.outputs import LightCone as lc
from py21cmfast.plotting import lightcone_sliceplot as lcplt
import py21cmfast as p21c

def cutoff_to_z(redshift_cutoff: float, path: str, prefix: str = "") -> None:
    files = fnmatch.filter(os.listdir(path), prefix + "*")
    z = np.asarray(h5.File(path + files[0], 'r')["lightcone_redshifts"])
    zidx = cut_z_idx(z, redshift_cutoff)
    print(z[zidx])


def convert_to_torch_batch(path: str, prefix: str = "run_", check_for_nan: bool = True, debug: bool = False,
                batch_size:int = 16, redshift_cutoff: int = 1175, xy_dim: int = 70, z_cut: float = 12) -> None:
    '''Given a path and an optinal prefix 
    (e.g. only convert all files named as run_, set prefix = "run_")
    this function converts .h5 files from 21cmfastwrapper to the common .npz format'''
    # image, label, tau, gxH
    # search for all files given in a path given a prefix an loop over those
    files = fnmatch.filter(os.listdir(path), prefix + "*")
    if debug: print(f"{files}")
    nan_counter = []
    zix = 88 #cut_z_idx(np.asarray(h5.File(path + files[0], 'r')["node_redshifts"]), z_cut)
    chunks = chunking(files, batch_size)
    with alive_bar(int(len(files)), disable=debug, force_tty=True) as fbar:
        for j, file_batch in enumerate(chunks):
            images = torch.empty(len(file_batch), xy_dim, xy_dim, redshift_cutoff)
            labels = torch.empty(len(file_batch), 6)
            taus = torch.empty(len(file_batch), 1)
            zs = torch.empty(len(file_batch), zix)
            gxHs = torch.empty(len(file_batch), zix)
            for i,file in enumerate(file_batch):
                if debug: print(f"load {path + file}")
                f = h5.File(path + file, 'r')
                img = torch.as_tensor(f['lightcones']['brightness_temp'])
                # check if there are NaNs in the brightness map
                if check_for_nan:
                    if torch.isnan(img).any():
                        nan_counter.append(file)
                        continue
                # load image
                images[i] = img[:,:,:redshift_cutoff]
                #load labels, WDM,OMm,LX,E0,Tvir,Zeta
                f_glob, f_cosmo, f_astro = dict(f['_globals'].attrs), dict(f['cosmo_params'].attrs), dict(f['astro_params'].attrs), 
                labels[i] = torch.as_tensor([
                    f_glob["M_WDM"],
                    f_cosmo["OMm"],
                    f_astro["L_X"],
                    f_astro["NU_X_THRESH"],
                    f_astro["ION_Tvir_MIN"],
                    f_astro["HII_EFF_FACTOR"]
                ])
                # load redshift
                redshifts = torch.as_tensor(f["node_redshifts"])
                # cut_idx = cut_z_idx(redshifts, z_cut)
                # compute tau
                gxH=torch.flip(torch.as_tensor(f["global_quantities"]["xH_box"]), dims=[0])#[:cut_idx]
                redshifts= torch.flip(redshifts, dims=[0])#[:cut_idx]
                #print(redshifts)
                tau=compute_tau(redshifts=redshifts,global_xHI=gxH)
                taus[i] = tau
                zs[i] = redshifts
                gxHs[i] = gxH

                new_format = {
                    "images": images,
                    "labels": labels,
                    "taus": taus,
                    "zs": zs,
                    "gxHs": gxHs
                }
                #save to new format
            np.savez(path + f"batch_{j}" + ".npz", **new_format)
            fbar()

    print(f"Done, {len(nan_counter)} NaNs encountered in \n{nan_counter}")   

def convert_to_torch(path: str, prefix: str = "run_", check_for_nan: bool = True, debug: bool = False, remove_zeros: bool = True,
                     redshift_cutoff: int = 1175, statistics: bool = False) -> None:
    '''Given a path and an optinal prefix 
    (e.g. only convert all files named as run_, set prefix = "run_")
    this function converts .h5 files from 21cmfastwrapper to the common .npz format'''
    # image, label, tau, gxH
    # search for all files given in a path given a prefix an loop over those
    if statistics:
        max_bt, min_bt, avg_bt, taus, zlc = ([] for i in range(5))
    if remove_zeros or statistics:
        zeros = []
    files = fnmatch.filter(os.listdir(path), prefix + "*")
    if debug: print(f"{files}")
    nan_counter = []
    # zix = 88 #cut_z_idx(np.asarray(h5.File(path + files[0], 'r')["node_redshifts"]), z_cut)
    with alive_bar(len(files), force_tty=True) as fbar:
        for i, file in enumerate(files):
            if debug: print(f"load {path + file}")
            f = h5.File(path + file, 'r')
            img = torch.as_tensor(f['lightcones']['brightness_temp'], dtype=torch.float32)
            # stuff good to know
            if statistics and i % 10 == 0:
                temp_lc = p21c.outputs.LightCone.read(path + file).lightcone_redshifts
                zlc.append(temp_lc[redshift_cutoff])
                zlc_min = temp_lc[0]
            # load image
            img = img[:,:,:redshift_cutoff]
            # check if there are NaNs in the brightness map
            if check_for_nan:
                if torch.isnan(img).any():
                    nan_counter.append(file)
                    continue

            # check for zero brightness_temp
            if statistics or remove_zeros:
                if not torch.any(img):
                    zeros.append(file)
                    if remove_zeros:
                        continue

            img = torch.unsqueeze(img, 0)
            #load labels, WDM,OMm,LX,E0,Tvir,Zeta
            f_glob, f_cosmo, f_astro = dict(f['_globals'].attrs), dict(f['cosmo_params'].attrs), dict(f['astro_params'].attrs), 
            label = torch.as_tensor([
                f_glob["M_WDM"],
                f_cosmo["OMm"],
                f_astro["L_X"],
                f_astro["NU_X_THRESH"],
                f_astro["ION_Tvir_MIN"],
                f_astro["HII_EFF_FACTOR"]
            ], dtype=torch.float32)

            if debug: print(f'{label=}')
            
            # load redshift
            redshifts = torch.as_tensor(f["node_redshifts"], dtype=torch.float32)
            # cut_idx = cut_z_idx(redshifts, z_cut)
            # compute tau
            gxH=torch.flip(torch.as_tensor(f["global_quantities"]["xH_box"], dtype=torch.float32), dims=[0])#[:cut_idx]
            redshifts= torch.flip(redshifts, dims=[0])#[:cut_idx]
            #print(redshifts)
            tau=torch.as_tensor(compute_tau(redshifts=redshifts,global_xHI=gxH), dtype=torch.float32)

            if statistics:
                max_bt.append(float(img.max()))
                min_bt.append(float(img.min()))
                avg_bt.append(float(img.mean()))
                taus.append(float(tau))

            new_format = {
                "images": img,
                "labels": label,
                "taus": tau,
                "zs": redshifts,
                "gxHs": gxH
            }
            #save to new format
            torch.save(new_format, path + f"batch_{i}" + ".pt")
            fbar()
    if statistics:
        plt.rcParams['text.usetex'] = True
        fig, ax = plt.subplots(2, 3, figsize=(12, 8))
        ax[0,0].hist(x = min_bt, bins = 10)
        ax[0,0].set_xlabel(r"$\max \delta T$")
        ax[0,1].hist(x = max_bt, bins = 10)
        ax[0,1].set_xlabel(r"$\min \delta T$")
        ax[1,0].hist(x = avg_bt, bins = 10)
        ax[1,0].set_xlabel("avg" + r"$ \delta T$")
        ax[1,1].hist(x = taus, bins = 10)
        ax[1,1].set_xlabel(r"$\tau$")
        ax[0,0].set_ylabel("count")
        ax[1,0].set_ylabel("count")
        ax[0,2].hist(x = zlc, bins = 10)
        ax[0,2].set_xlabel(r"$\max z$")
        ax[0,2].set_ylabel(f"count / mean={round(np.mean(zlc),2)} and zlc_min={round(zlc_min,2)}")
        ax[1,2].axis("off")
        
        fig.tight_layout()
        fig.savefig("./convert_results.png", dpi=200)
        fig.show()
        print(len(zeros), " brightness_temps are zero at pos:\n", zeros)


    print(f"Done, {len(nan_counter)} NaNs encountered in \n{nan_counter}")   



def convert_to_2dps(path: str, prefix: str = "", check_for_nan: bool = True, debug: bool = False) -> None:
    files = fnmatch.filter(os.listdir(path), prefix + "*")
    if debug: print(f"{files}")
    nan_counter = []
    #zix = cut_z_idx(np.asarray(h5.File(path + files[0], 'r')["node_redshifts"]), z_cut)
    # execute ps 1 time to find dims
    with alive_bar(len(files), force_tty=True) as fbar:
        for i, file in enumerate(files):
            if i == 0:
                
                ps2ds = torch.empty(len(file_batch), 13, xy_dim, z_dim)
                labels = torch.empty(len(file_batch), 6)
            #taus = torch.empty(len(file_batch), 1)
            #zs = torch.empty(len(file_batch), zix)
            #gxHs = torch.empty(len(file_batch), zix)
            for file in file_batch:
                if debug: print(f"load {path + file}")
                f = h5.File(path + file, 'r')
                # load image
                img = torch.as_tensor(f['lightcones']['brightness_temp'])
                # check if there are NaNs in the brightness map
                if check_for_nan:
                    if torch.isnan(img).any():
                        nan_counter.append(file)
                        continue

                # compute 2dps
                for j,z in enumerate(np.linspace(5.75,11.75,13)):
                    _,_, ps = run_2dps(path, file, z)
                    ps2ds[i,j] = torch.as_tensor(ps)
                
                #load labels, WDM,OMm,LX,E0,Tvir,Zeta
                f_glob, f_cosmo, f_astro = dict(f['_globals'].attrs), dict(f['cosmo_params'].attrs), dict(f['astro_params'].attrs), 
                labels[i] = torch.as_tensor([
                    f_glob["M_WDM"],
                    f_cosmo["OMm"],
                    f_astro["L_X"],
                    f_astro["NU_X_THRESH"],
                    f_astro["ION_Tvir_MIN"],
                    f_astro["HII_EFF_FACTOR"]
                ])
                
                # load redshift
                #redshifts = torch.as_tensor(f["node_redshifts"])
                #cut_idx = cut_z_idx(redshifts, z_cut)
                # compute tau
                #gxH=torch.flip(torch.as_tensor(f["global_quantities"]["xH_box"]), dims=[0])[:cut_idx]
                #redshifts= torch.flip(redshifts, dims=[0])[:cut_idx]
                #print(redshifts)
                #tau=compute_tau(redshifts=redshifts,global_xHI=gxH)
                #taus[i] = tau
                #zs[i] = redshifts
                #gxHs[i] = gxH
                

                new_format = {
                    "ps": ps2ds,
                    "labels": labels,
                    #"taus": taus,
                    #"zs": zs,
                    #"gxHs": gxHs
                }
                #save to new format
            torch.save(new_format, path + f"batch_{i}" + ".pt")
            fbar()

    print(f"Done, {len(nan_counter)} NaNs encountered in \n{nan_counter}")

def plot_random(num: int = 5, path: str = "./", prefix: str = "run_") -> None:
    files = fnmatch.filter(os.listdir(path), prefix + "*")
    idxs = np.random.randint(0, len(files), num)
    for idx in idxs:
        f = lc.read(path + files[idx])
        lcplt(f)
        plt.title(files[idx])
        plt.tight_layout()
        plt.show()


def convert_to_npz(path: str, prefix: str = "run_", check_for_nan: bool = True, debug: bool = False, remove_zeros: bool = True,
                     redshift_cutoff: int = 1175, statistics: bool = False) -> None:
    '''Given a path and an optinal prefix 
    (e.g. only convert all files named as run_, set prefix = "run_")
    this function converts .h5 files from 21cmfastwrapper to the common .npz format'''
    # image, label, tau, gxH
    # search for all files given in a path given a prefix an loop over those
    if statistics:
        max_bt, min_bt, avg_bt, taus, zlc = ([] for i in range(5))
    if remove_zeros or statistics:
        zeros = []
    files = fnmatch.filter(os.listdir(path), prefix + "*")
    if debug: print(f"{files}")
    nan_counter = []
    # zix = 88 #cut_z_idx(np.asarray(h5.File(path + files[0], 'r')["node_redshifts"]), z_cut)
    with alive_bar(len(files), force_tty=True) as fbar:
        for i, file in enumerate(files):
            if debug: print(f"load {path + file}")
            f = h5.File(path + file, 'r')
            img = torch.as_tensor(f['lightcones']['brightness_temp'], dtype=torch.float32)
            # stuff good to know
            if statistics and i % 10 == 0:
                temp_lc = p21c.outputs.LightCone.read(path + file).lightcone_redshifts
                zlc.append(temp_lc[redshift_cutoff])
                zlc_min = temp_lc[0]
            # load image
            img = img[:,:,:redshift_cutoff]
            # check if there are NaNs in the brightness map
            if check_for_nan:
                if torch.isnan(img).any():
                    nan_counter.append(file)
                    continue

            # check for zero brightness_temp
            if statistics or remove_zeros:
                if not torch.any(img):
                    zeros.append(file)
                    if remove_zeros:
                        continue

            img = torch.unsqueeze(img, 0)
            #load labels, WDM,OMm,LX,E0,Tvir,Zeta
            f_glob, f_cosmo, f_astro = dict(f['_globals'].attrs), dict(f['cosmo_params'].attrs), dict(f['astro_params'].attrs), 
            label = torch.as_tensor([
                f_glob["M_WDM"],
                f_cosmo["OMm"],
                f_astro["L_X"],
                f_astro["NU_X_THRESH"],
                f_astro["ION_Tvir_MIN"],
                f_astro["HII_EFF_FACTOR"]
            ], dtype=torch.float32)

            if debug: print(f'{label=}')
            
            # load redshift
            redshifts = torch.as_tensor(f["node_redshifts"], dtype=torch.float32)
            # cut_idx = cut_z_idx(redshifts, z_cut)
            # compute tau
            gxH=torch.flip(torch.as_tensor(f["global_quantities"]["xH_box"], dtype=torch.float32), dims=[0])#[:cut_idx]
            redshifts= torch.flip(redshifts, dims=[0])#[:cut_idx]
            #print(redshifts)
            tau=torch.as_tensor(compute_tau(redshifts=redshifts,global_xHI=gxH), dtype=torch.float32)

            if statistics:
                max_bt.append(float(img.max()))
                min_bt.append(float(img.min()))
                avg_bt.append(float(img.mean()))
                taus.append(float(tau))

            new_format = {
                "images": np.asarray(img, dtype=float),
                "labels": np.asarray(label, dtype=float),
                "taus": np.asarray(tau, dtype=float),
                "zs": np.asarray(redshifts, dtype=float),
                "gxHs": np.asarray(gxH, dtype=float)
            }
            #save to new format
            np.savez(new_format, path + f"batch_{i}" + ".npz")
            fbar()
    if statistics:
        plt.rcParams['text.usetex'] = True
        fig, ax = plt.subplots(2, 3, figsize=(12, 8))
        ax[0,0].hist(x = min_bt, bins = 10)
        ax[0,0].set_xlabel(r"$\max \delta T$")
        ax[0,1].hist(x = max_bt, bins = 10)
        ax[0,1].set_xlabel(r"$\min \delta T$")
        ax[1,0].hist(x = avg_bt, bins = 10)
        ax[1,0].set_xlabel("avg" + r"$ \delta T$")
        ax[1,1].hist(x = taus, bins = 10)
        ax[1,1].set_xlabel(r"$\tau$")
        ax[0,0].set_ylabel("count")
        ax[1,0].set_ylabel("count")
        ax[0,2].hist(x = zlc, bins = 10)
        ax[0,2].set_xlabel(r"$\max z$")
        ax[0,2].set_ylabel(f"count / mean={round(np.mean(zlc),2)} and zlc_min={round(zlc_min,2)}")
        ax[1,2].axis("off")
        
        fig.tight_layout()
        fig.savefig("./convert_results.png", dpi=200)
        fig.show()
        print(len(zeros), " brightness_temps are zero at pos:\n", zeros)


    print(f"Done, {len(nan_counter)} NaNs encountered in \n{nan_counter}")   


def chunking(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def cut_z_idx(z, z_cut):
    amidx = np.abs(z - z_cut).argmin()
    return int(len(z) - amidx)