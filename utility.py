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
from typing import Any, Dict, Optional
from torch.distributions import Distribution
from sbi.inference.posteriors.mcmc_posterior import MCMCPosterior
from sbi.inference.posteriors.rejection_posterior import RejectionPosterior
from sbi.inference.potentials.ratio_based_potential import RatioBasedPotential
from sbi.inference.potentials.likelihood_based_potential import LikelihoodBasedPotential
from sbi.inference.posteriors.vi_posterior import VIPosterior
from sbi.utils import mcmc_transform
from py21cmfast_tools import calculate_ps
from powerbox.tools import ignore_zero_absk

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
            img = torch.as_tensor(np.array(f['lightcones']['brightness_temp']), dtype=torch.float32)
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
                #"taus": tau,
                #"zs": redshifts,
                #"gxHs": gxH
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


def convert_pt_to_2dps(path: str, prefix: str = "", debug: bool = False,
                    summary_statistics_parameters = {
                    "BOX_LEN": 200,
                    "HII_DIM": 28,
                    "z-eval": np.linspace(7, 24, 10),
                    "bins": 8,
                    },) -> None:

    files = fnmatch.filter(os.listdir(path), prefix + "*")
    if debug: print(f"{files}")
    nan_counter = []
    #zix = cut_z_idx(np.asarray(h5.File(path + files[0], 'r')["node_redshifts"]), z_cut)
    # execute ps 1 time to find dims
    with alive_bar(len(files), force_tty=True) as fbar:
        for i, file in enumerate(files):
            file = torch.load(path + file)
            lab, img = file['labels'], file['images']
            WDM,OMm,LX,E0,Tvir,Zeta = lab
            with p21c.global_params.use(**{"M_WDM":WDM}):
                cosmo_params = p21c.inputs.CosmoParams({"OMm": OMm})
                astro_params = p21c.inputs.AstroParams({
                    "L_X": LX,
                    "NU_X_THRESH": E0,
                    "ION_Tvir_MIN": Tvir,
                    "HII_EFF_FACTOR": Zeta,
                })
                
                user_params = p21c.UserParams(
                    HII_DIM=summary_statistics_parameters['HII_DIM'], 
                    BOX_LEN=summary_statistics_parameters['BOX_LEN'], KEEP_3D_VELOCITIES=True
                )

                lcn_distances = p21c.RectilinearLightconer.with_equal_cdist_slices(
                    min_redshift=5.5,
                    max_redshift=35.05,
                    quantities=('brightness_temp', 'density', 'velocity_z'),
                    resolution=user_params.cell_size,
                    # index_offset=0,
                ).lc_distances

                flag_options, user_params, random_seed = p21c.inputs.FlagOptions(), p21c.inputs.UserParams(), 42

                lc = p21c.LightCone(redshift=5.5, 
                cosmo_params=cosmo_params,
                flag_options=flag_options,
                user_params=user_params,
                random_seed=random_seed,
                distances=lcn_distances,
                astro_params=astro_params,
                lightcones={"brightness_temp":img},
                current_redshift=35.05)

                res = calculate_ps(lc=img, lc_redshifts=lc.lightcone_redshifts[:img.shape[-1]], 
                    box_length=summary_statistics_parameters['BOX_LEN'], box_side_shape=summary_statistics_parameters["HII_DIM"],
                    log_bins=False, zs=summary_statistics_parameters["z-eval"], calc_1d=False, calc_2d=True, 
                    kpar_bins=summary_statistics_parameters["bins"], nbins=summary_statistics_parameters["bins"], bin_ave=True, 
                    k_weights=ignore_zero_absk, postprocess=True, get_variance=True)

            ps2d = res['final_ps_2D']
            std = torch.sqrt(torch.as_tensor(res["var_1D"]))

            new_format = {
                "images": torch.as_tensor(ps2d, dtype=torch.float32),
                "labels": torch.as_tensor(lab, dtype=torch.float32),
                "std": std.to(torch.float32),
                #"taus": taus,
                #"zs": zs,
                #"gxHs": gxHs
            }
                #save to new format
            torch.save(new_format, path + f"ps2d_{i}" + ".pt")
            fbar()


def convert_pt_to_1dps(path: str, prefix: str = "", debug: bool = False,
                    summary_statistics_parameters = {
                    "BOX_LEN": 200,
                    "HII_DIM": 28,
                    "z-eval": np.linspace(7, 24, 10),
                    "bins": 8,
                    },) -> None:

    files = fnmatch.filter(os.listdir(path), prefix + "*")
    if debug: print(f"{files}")
    nan_counter = []
    #zix = cut_z_idx(np.asarray(h5.File(path + files[0], 'r')["node_redshifts"]), z_cut)
    # execute ps 1 time to find dims
    with alive_bar(len(files), force_tty=True) as fbar:
        for i, file in enumerate(files):
            file = torch.load(path + file)
            lab, img = file['labels'], file['images']
            WDM,OMm,LX,E0,Tvir,Zeta = lab
            with p21c.global_params.use(**{"M_WDM":WDM}):
                cosmo_params = p21c.inputs.CosmoParams({"OMm": OMm})
                astro_params = p21c.inputs.AstroParams({
                    "L_X": LX,
                    "NU_X_THRESH": E0,
                    "ION_Tvir_MIN": Tvir,
                    "HII_EFF_FACTOR": Zeta,
                })
                
                user_params = p21c.UserParams(
                    HII_DIM=summary_statistics_parameters['HII_DIM'], 
                    BOX_LEN=summary_statistics_parameters['BOX_LEN'], KEEP_3D_VELOCITIES=True
                )

                lcn_distances = p21c.RectilinearLightconer.with_equal_cdist_slices(
                    min_redshift=5.5,
                    max_redshift=35.05,
                    quantities=('brightness_temp', 'density', 'velocity_z'),
                    resolution=user_params.cell_size,
                    # index_offset=0,
                ).lc_distances

                flag_options, user_params, random_seed = p21c.inputs.FlagOptions(), p21c.inputs.UserParams(), 42

                lc = p21c.LightCone(redshift=5.5, 
                cosmo_params=cosmo_params,
                flag_options=flag_options,
                user_params=user_params,
                random_seed=random_seed,
                distances=lcn_distances,
                astro_params=astro_params,
                lightcones={"brightness_temp":img},
                current_redshift=35.05)

                res = calculate_ps(lc=img, lc_redshifts=lc.lightcone_redshifts[:img.shape[-1]], 
                    box_length=summary_statistics_parameters['BOX_LEN'], box_side_shape=summary_statistics_parameters["HII_DIM"],
                    log_bins=False, zs=summary_statistics_parameters["z-eval"], calc_1d=True, calc_2d=False, 
                    nbins_1d=summary_statistics_parameters["bins"], bin_ave=True, 
                    k_weights=ignore_zero_absk, postprocess=True, get_variance=True)

            ps1d = res['ps_1D']
            std = torch.sqrt(torch.as_tensor(res["var_1D"]))

            new_format = {
                "images": torch.as_tensor(ps1d, dtype=torch.float32),
                "labels": torch.as_tensor(lab, dtype=torch.float32),
                "std": std.to(torch.float32),
                #"taus": taus,
                #"zs": zs,
                #"gxHs": gxHs
            }
                #save to new format
            torch.save(new_format, path + f"ps1d_{i}" + ".pt")
            fbar()



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


def convert_npz_to_pt(path: str, prefix: str = "run_", check_for_nan: bool = True, 
                      debug: bool = False, remove_zeros: bool = True,
                     redshift_cutoff: int = 0, statistics: bool = False) -> None:
    '''Given a path and an optinal prefix 
    (e.g. only convert all files named as run_, set prefix = "run_")
    this function converts .h5 files from 21cmfastwrapper to the common .npz format'''
    # image, label, tau, gxH
    # search for all files given in a path given a prefix an loop over those
    if statistics:
        max_bt, min_bt, avg_bt = ([] for i in range(3))
    if remove_zeros or statistics:
        zeros = []
    files = fnmatch.filter(os.listdir(path), prefix + "*")
    if debug: print(f"{files}")
    nan_counter = []
    # zix = 88 #cut_z_idx(np.asarray(h5.File(path + files[0], 'r')["node_redshifts"]), z_cut)
    with alive_bar(len(files), force_tty=True) as fbar:
        for i, file in enumerate(files):
            if debug: print(f"load {path + file}")
            f = np.load(path + file)
            img = torch.as_tensor(f['image'], dtype=torch.float32)
            # stuff good to know

            if redshift_cutoff > 0:
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
                    
            #load labels, WDM,OMm,LX,E0,Tvir,Zeta
            label = torch.as_tensor(f['label'], dtype=torch.float32)

            if debug: print(f'{label=}')

            if statistics:
                max_bt.append(float(img.max()))
                min_bt.append(float(img.min()))
                avg_bt.append(float(img.mean()))

            new_format = {
                "images": img,
                "labels": label,
            }
            #save to new format
            torch.save(new_format, path + f"batch_{i}" + ".pt")
            fbar()

        if statistics:
            plt.rcParams['text.usetex'] = True
            fig, ax = plt.subplots(1, 3, figsize=(12, 8))
            ax[0].hist(x = min_bt, bins = 10)
            ax[0].set_xlabel(r"$\max \delta T$")
            ax[1].hist(x = max_bt, bins = 10)
            ax[1].set_xlabel(r"$\min \delta T$")
            ax[2].hist(x = avg_bt, bins = 10)
            ax[2].set_xlabel("avg" + r"$ \delta T$")
            
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




def repeat_rows(x, num_reps):
    """Each row of tensor `x` is repeated `num_reps` times along leading dimension."""
    if not (type(num_reps) == int and num_reps > 0):
        raise TypeError("Number of repetitions must be a positive integer.")
    shape = x.shape
    x = x.unsqueeze(1)
    x = x.expand(shape[0], num_reps, *shape[1:])
    return merge_leading_dims(x, num_dims=2)

def merge_leading_dims(x, num_dims):
    """Reshapes the tensor `x` such that the first `num_dims` dimensions are merged to
    one."""
    if not (type(num_dims) == int and num_dims > 0):
        raise TypeError("Number of leading dims must be a positive integer.")
    if num_dims > x.dim():
        raise ValueError(
            "Number of leading dims can't be greater than total number of dims."
        )
    new_shape = torch.Size([-1]) + x.shape[num_dims:]
    return torch.reshape(x, new_shape)

def get_nre_posterior(
    ratio_estimator: torch.nn.Module,
    prior: Optional[Distribution] = None,
    sample_kwargs: dict = {"sample_with": "rejection"}
):
    """Try it.

    Args:
        density_estimator: The density estimator that the posterior is based on.
            If `None`, use the latest neural density estimator that was trained.
        prior: Prior distribution.
        sample_with: Method to use for sampling from the posterior. Must be one of
            [`mcmc` | `rejection` | `vi`].
        mcmc_method: Method used for MCMC sampling, one of `slice_np`, `slice`,
            `hmc`, `nuts`. Currently defaults to `slice_np` for a custom numpy
            implementation of slice sampling; select `hmc`, `nuts` or `slice` for
            Pyro-based sampling.
        vi_method: Method used for VI, one of [`rKL`, `fKL`, `IW`, `alpha`]. Note
            that some of the methods admit a `mode seeking` property (e.g. rKL)
            whereas some admit a `mass covering` one (e.g fKL).
        mcmc_parameters: Additional kwargs passed to `MCMCPosterior`.
        vi_parameters: Additional kwargs passed to `VIPosterior`.
        rejection_sampling_parameters: Additional kwargs passed to
            `RejectionPosterior`.
    """
    device = next(ratio_estimator.parameters()).device.type
    potential_fn = RatioBasedPotential(ratio_estimator, prior, x_o=None, device=device)
    theta_transform = mcmc_transform(
        prior, device=device, enable_transform=False
    )
    sample_with = sample_kwargs.pop('sample_with')
    if sample_with == "mcmc":
        posterior = MCMCPosterior(
            potential_fn=potential_fn,
            theta_transform=theta_transform,
            proposal=prior,
            device=device,
            **sample_kwargs,
        )
    elif sample_with == "rejection":
        posterior = RejectionPosterior(
            potential_fn=potential_fn,
            proposal=prior,
            device=device,
            **sample_kwargs,
        )
    else:
        raise NotImplementedError

    return posterior


def get_nle_posterior(
    likelihood_estimator: torch.nn.Module,
    prior: Optional[Distribution] = None,
    sample_kwargs: dict = {"sample_with": "rejection"}
):
    """Try it.

    Args:
        density_estimator: The density estimator that the posterior is based on.
            If `None`, use the latest neural density estimator that was trained.
        prior: Prior distribution.
        sample_with: Method to use for sampling from the posterior. Must be one of
            [`mcmc` | `rejection` | `vi`].
        mcmc_method: Method used for MCMC sampling, one of `slice_np`, `slice`,
            `hmc`, `nuts`. Currently defaults to `slice_np` for a custom numpy
            implementation of slice sampling; select `hmc`, `nuts` or `slice` for
            Pyro-based sampling.
        vi_method: Method used for VI, one of [`rKL`, `fKL`, `IW`, `alpha`]. Note
            that some of the methods admit a `mode seeking` property (e.g. rKL)
            whereas some admit a `mass covering` one (e.g fKL).
        mcmc_parameters: Additional kwargs passed to `MCMCPosterior`.
        vi_parameters: Additional kwargs passed to `VIPosterior`.
        rejection_sampling_parameters: Additional kwargs passed to
            `RejectionPosterior`.
    """
    device = next(likelihood_estimator.parameters()).device.type
    potential_fn = LikelihoodBasedPotential(likelihood_estimator, prior, x_o=None, device=device)
    theta_transform = mcmc_transform(
        prior, device=device, enable_transform=False
    )
    sample_with = sample_kwargs.pop('sample_with')
    if sample_with == "mcmc":
        posterior = MCMCPosterior(
            potential_fn=potential_fn,
            theta_transform=theta_transform,
            proposal=prior,
            device=device,
            **sample_kwargs,
        )
    elif sample_with == "rejection":
        posterior = RejectionPosterior(
            potential_fn=potential_fn,
            proposal=prior,
            device=device,
            **sample_kwargs,
        )
    #add VI
    else:
        raise NotImplementedError

    return posterior