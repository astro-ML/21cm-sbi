import py21cmfast as p21c
from matplotlib import pyplot as plt
import os 
import logging, sys, os
logger = logging.getLogger('21cmFAST')
logger.setLevel(logging.DEBUG)
from py21cmfast import plotting
from py21cmfast import cache_tools
from multiprocessing import Pool
#import timeit
import numpy as np
import random
import yaml
import itertools
from powerbox.tools import get_power
from schwimmbad import MPIPool
import h5py
import fnmatch
from collections.abc import Callable

# set your cache path here
cache_path = "./_cache"

if not os.path.exists(cache_path):
    os.mkdir(cache_path)

p21c.config['direc'] = cache_path
# uncomment this if you want to clear the cache everytime you execute the program
# cache_tools.clear_cache(direc=cache_path)

class Parameters():    
    '''Auxillary class to initialize and update parameters given a config file or on the fly.'''
    def __init__(self, parameter_path, file_name, data_path, override, debug):
        # set bool for box-lightcone-simulation switch
        self.box = False
        # load parameter file
        with open(parameter_path + "parameter.yaml", 'r') as file:
            parameter = yaml.safe_load(file)
        # set default configuration given by parameter file
        self.random_seed = not parameter["user_params"]["NO_RNG"]
        self.max_z = parameter["input_params"]["max_redshift"]
        use_default = []
        for key in parameter.keys():
            if key == "input_params":
                self.input_params = parameter["input_params"]
            else:
                print(f"""use {key} default config: {parameter[key]["use_default"]}""")
                use_default.append(parameter[key]["use_default"])
                parameter[key].pop("use_default")
        
        parameter.pop("input_params")
        
        self.astro_params = parameter["astro_params"] if not use_default[0] else p21c.AstroParams()
        self.cosmo_params = parameter["cosmo_params"] if not use_default[1] else p21c.CosmoParams()
        self.user_params = parameter["user_params"] if not use_default[2] else p21c.UserParams()
        self.flag_params = parameter["flag_options"] if not use_default[3] else p21c.FlagOptions()
        self.global_params = parameter["global_params"] if not use_default[4] else {}
        parameter.pop("global_params")
        
        self.input_params.update(parameter)
        
        # initialize saving procedure
        self.data_path = data_path
        self.data_name = file_name
        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)
        # check if some files may already exist
        if override: os.system(f'rm -f {self.data_path + self.data_name + "*"}')
        if debug: print(self.data_path,self.data_name + "*",len(fnmatch.filter(os.listdir(self.data_path), self.data_name + "*")), fnmatch.filter(os.listdir(self.data_path), self.data_name + "*"))
        self.run_counter = 0 if override else len(fnmatch.filter(os.listdir(self.data_path), self.data_name + "*"))
        self.override = override
        
        # save the initial configuration
        self.init_params = self.input_params.copy() 

    def kwargs_update(self, kargs):
        '''Update the parameter config given kargs'''
        self.input_params.update(kargs)
        
    def kwargs_revert(self):
        '''Revert changes in the parameters'''
        self.input_params = self.init_params
    
    @staticmethod
    def wrap_params(params):
        '''Change the parameter file to run a single box and revert the changes afterwards.
        This is necessary or else 21cmfast returns an error.'''
        params.pop("lightcone_quantities")
        params.pop("max_redshift")
        return p21c.run_coeval(**params)
            
    def randomize(self):
        '''Shuffle random_seed'''
        if self.random_seed: self.input_params["random_seed"] = random.randint(0,99999)
       
    
class Simulation(Parameters):
    '''Dynamically execute and plot simulations.'''
    def __init__(self, parameter_path="./", save_inclass = False, save_ondisk = True, 
                write_cache=False, clean_cache=False, data_path = "./data/", file_name = "run_", override = False, 
                 debug=False):
        '''parameter_file (str): path to parameter.yaml
        save_inclass (bool): If set true, results are saved as a list in the class, very useful for testing and quick analysis. If False, results are saved as a file
        save_ondisk (bool): If set True, save results on disk
        write_cache (bool): If true, use the included 21cmfast cache, usually brings performance improvements on lightcone simulations
        clean_cache (bool): If true, clean the cache after each simulation )
        data_path (str): path for saving the results of save_ondisk is True
        file_name (str): filename for the runs, final name will be: filename + run_id + .h5
        override (bool): If True, old runs will be overridden
        debug (bool): Print many things along the way if something is screwed again.'''
        super().__init__(parameter_path=parameter_path, data_path=data_path, 
                        file_name=file_name, override=override, debug=debug)
        print(f"Using 21cmFAST version {p21c.__version__}")
        self.sic = save_inclass
        self.sod = save_ondisk
        self.ccache = clean_cache
        self.debug = debug
        if save_inclass: self.data = []
        self.input_params['write'] = write_cache
        
    def __len__(self):
        '''Returns length of the data-array'''
        return len(self.data)
        
    def run_box(self, kargs={}, run_id=0, commit=False):
        '''Run a simple box simulation
        kargs (dict): Change parameters on-the-fly for this run
        commit (bool): If true, results are returned'''
        # depricated, but plotting still depends on it
        self.simtype = 0
        with p21c.global_params.use(**self.global_params):
            #self.randomize()
            self.kwargs_update(kargs)
            run = self.wrap_params(self.input_params)
            if self.ccache: cache_tools.clear_cache()
            if commit: return run
            if self.sic: self.data.append(run)
            if self.sod: self.save(run, self.data_name, self.data_path, run_id)
                
    
    def run_cone(self, kargs={}, run_id=0, commit = False):
        '''Run a simple lightcone simulation'''
        # depricated, but plotting still depends on it
        self.simtype = 1
        with p21c.global_params.use(**self.global_params):
            #self.randomize()
            self.kwargs_update(kargs)
            run = p21c.run_lightcone(**self.input_params)
            run = self.cut_lightcone(run, self.max_z)
            if self.ccache: cache_tools.clear_cache()
            if commit: return run
            if self.sic: self.data.append(run)
            if self.sod: self.save(run, self.data_name, self.data_path, run_id)
            
                
    def run_multi_lightcone(self, mkargs):
        '''Compute multiple lightcones given a list of parameters as a dict with list entries
        e.g. rargs = {"random_seed": [1,2], "astro_params": {"HII_EFF_FACTOR": [29,31]}, NU_X_THRESH": [1,2,3]}, ...}'''
        for run_params in self.generate_combinations(mkargs):
            print("Parameter run: ",run_params)
            self.run_cone(kargs=run_params)
            if self.ccache: cache_tools.clear_cache()
    
    def mpi_lcone_wrapper(self,args):
        return self.run_cone(*args)
    
    def mpi_box_wrapper(self, args):
        return self.run_box(*args)
   
    def run_samplef(self, nruns, args, samplef = (lambda a,b: np.random.uniform(a,b)), box = False, threads = 1, mpi=False):
        '''Sample parameters according to some function and run simulations
        nruns (int): specify the number of runs
        samplef (func): callable function which takes an array of parameter given in args (standard choice is uniform)
        args (dict): a dict of parameter (must obey the parameter.yaml structure) with argument arrays as value (standard choice is array [start, end])
        box (bool): If True, run a box simulation instead instead of the full lightcone
        threads (int): number of parralel running threads
        mpi (bool): uses MPI instead of pythons Pool -> use False if single CPU, True if multiple hardware CPU'''
        # multithread loop via Pool? -> Pool seems to work :) (schwimmbad mpi? (GLI a problem??? probably not because it wraps C code))
        m_params = []
        for run in range(nruns):
            m_params.append((self.generate_range(args, samplef), self.run_counter + run))
            if self.debug: print(m_params[-1])
        schwimmhalle = MPIPool() if mpi else Pool(threads)
        with schwimmhalle as p:
            p.map(self.mpi_box_wrapper if box else self.mpi_lcone_wrapper, m_params)

    
    # print functions need to be rewritten in a modular way ... may not working atm
    def plot_global_properties(self, run=[-1], observational_axis = False, print_params = ['']):
        '''Make a plot of the global quantities of the lightcone
        run: array of run_ids which should be printed
        observational_axis: plot the x-axis in MHz instead of redshift
        print_params: Additional array of parameters which are printed as plain text for debug reasons'''
        x_map = (lambda z: 1420.4/(z + 1)) if observational_axis else (lambda z: z)
        if run[0] == -1:
            run = range(len(self.data))
        if self.simtype != 1:
            print("Non-lightcone simulation not supported.")
            return
        w,h = len(run), len(self.data[0].global_quantities)
        fig, ax = plt.subplots(h,w, figsize=(4*w,4*h))
        ax = ax.reshape(h,w)
        for r in run:
            add_string = ''
            for p in print_params:
                if p != '': add_string += p + '=' + str(self.data[r].astro_params.defining_dict[p]) + ', '
            title = f"run={r}"
            if add_string != '': title += '\n' + add_string
            ax[0,r].set_title(f"{title}")
            for q, quantity in enumerate(self.data[r].global_quantities.keys()):
                ax[q,r].plot(x_map(np.array(self.data[r].node_redshifts)), self.data[r].global_quantities[quantity])
                ax[q,r].set_xlabel("MHz" if observational_axis else "z")
                ax[q,r].set_ylabel(quantity)
        plt.tight_layout()
        plt.savefig("./glob_prop_test.jpg")
        plt.show()
            
    def plot_imshow(self, run=-1, fields=["brightness_temp"], print_params = [''], extended=True):
        '''
        run: plot given run, -1 to print them all
        fiels: list of fields that should be printed
        print_params: also show given parameters on the plots (only astro parameter!)
        extended: make additional graphs and statistics'''
        if run == -1:
            data = self.data
        else: 
            data = [self.data[run]]
        match self.simtype:
            case 0:
                h = len(data)
                w = len(data[0])
                for field in fields:
                    fig, ax = plt.subplots(h,w, figsize=(4*w,4*h))
                    ax = ax.reshape(h,w)
                    for (j, axis) in zip(range(len(data)),  ax):
                        add_string = ''
                        for p in print_params:
                            if p != '': add_string += p + '=' + str(data[j].astro_params.defining_dict[p]) + ', '
                        for i, (dat, redshift, a) in enumerate(zip(data[j], [z.redshift for z in self.data[j]], axis)):
                            if i<1:
                                plotting.coeval_sliceplot(dat, ax=a, fig=fig, kind=field)
                            else:
                                plotting.coeval_sliceplot(dat, ax=a, fig=fig, kind=field, printlabel=False)
                            title = f"{field}, z = {round(redshift,2)}"
                            if add_string != '': title += '\n' + add_string
                            plt.title(title)
                    plt.tight_layout()
                    plt.savefig("./imshow_test.jpg")
                    plt.show()
            case 1:
                h = len(data)
                for field in fields:
                    fig, ax = plt.subplots(h,1, figsize=(12,4*h))
                    ax = ax.flatten() if h > 1 else [ax]
                    for i, dat in enumerate(data):
                        add_string = ''
                        for p in print_params:
                            if p != '': add_string += p + " = " + str(data[i].astro_params.defining_dict[p]) + ', '
                        z = dat.node_redshifts
                        plotting.lightcone_sliceplot(dat, ax=ax[i], fig=fig, kind=field)
                        title = f"{field}, z = [{round(max(z),2)} -> {round(min(z),2)}] in {len(z)} steps, run = {i+1} "
                        if add_string != '': title += '\n' + add_string
                        plt.title(title)
                    ax = np.reshape(ax, (h,1))
                    plt.tight_layout()
                    plt.savefig("./imshow_test.jpg")
                    plt.show()     
                    
    def plot_ps(self, bins=10, run=[-1], compare=False, observational_axis = False, logp = True):
        '''Plot the power spectra given a fixed bin number. (Future: Do the same with a dz range but p21cm saves lightcone with constant physical distances not redshift ->)
        run (array): plot ps of given run, -1 to print them all
        compare (bool): Activating compare puts all simulations with equal redshift range in a single plot (useful to compare the ps of different simualtions)
        in_f (bool): changes the x axis from k in Mpc^-3 to lambda in real space (Mpc^3)
        logp (bool): Activating make the plots in log-log axis.
        '''
        binspace = np.linspace(0,len(self.data[0].lightcone_redshifts)-1,bins).astype(int)
        
        x_map = (lambda k: 2*np.pi/k) if observational_axis else (lambda k: k)
        
        if run[0] == -1:
            run = range(len(self.data))
            
            
        if not compare: 
            h,w = len(run),bins-1
            fig, ax = plt.subplots(h,w, figsize=(4*w,4*h))
            ax = ax.reshape(h,w)
            for r in run:
                for bin in range(w):
                    physical_size = self.data[r].lightcone_distances[binspace[bin+1]] - self.data[r].lightcone_distances[binspace[bin]]
                    #print(f"binspace: {binspace[bin]}, brightness map: {self.data[r].brightness_temp[:,:,binspace[bin]:binspace[bin+1]].shape}, boxlength= {(*self.data[r].brightness_temp.shape[:2], physical_size)}")
                    ps = get_power(deltax= self.data[r].brightness_temp[:,:,binspace[bin]:binspace[bin+1]], boxlength=(*self.data[r].brightness_temp.shape[:2], physical_size), bin_ave=True, ignore_zero_mode=True, get_variance=False, bins=50)
                    ax[r,bin].plot(x_map(ps[1]), ps[0]*ps[1]**3)
                    ax[-1,bin].set_xlabel('m in Mpc^3' if observational_axis else "k in Mpc^-3")
                    ax[0,bin].set_title(f'{round(self.data[r].lightcone_redshifts[binspace[bin]],1)} < z < {round(self.data[r].lightcone_redshifts[binspace[bin+1]],1)}')
                    ax[r,0].set_ylabel(f"P(k) * k ^ 3 ; run: {r}")
                    if logp:
                        ax[r,bin].set_xscale("log")
                        ax[r,bin].set_yscale("log")
        else:
            h,w = len(run), bins-1
            fig, ax = plt.subplots(w,1, figsize=(4*h,4*w))
            for r in run:
                for bin in range(w):
                    physical_size = self.data[r].lightcone_distances[binspace[bin+1]] - self.data[r].lightcone_distances[binspace[bin]]
                    ps = get_power(deltax= self.data[r].brightness_temp[:,:,binspace[bin]:binspace[bin+1]], boxlength=(*self.data[r].brightness_temp.shape[:2], physical_size), bin_ave=True, ignore_zero_mode=True, get_variance=False,bins=50)
                    ax[bin].plot(x_map(ps[1]), ps[0]*ps[1]**3, label=f"run {r}")
                    ax[bin].set_title(f'{round(self.data[r].lightcone_redshifts[binspace[bin]],1)} < z < {round(self.data[r].lightcone_redshifts[binspace[bin+1]],1)}')
                    ax[bin].set_xlabel('m in Mpc^3' if observational_axis else "k in Mpc^-3")
                    ax[bin].set_ylabel(f"P(k) * k ^ 3 ; run: {r}")
                    ax[bin].legend()
                    if logp:
                        ax[bin].set_xscale("log")
                        ax[bin].set_yscale("log")
        
        plt.tight_layout()
        plt.savefig("./ps_test.jpg")
        plt.show()
        
    def generate_combinations(self, d):
        '''Helper function to recursively generate combinations'''
        keys, values = zip(*d.items())
        values = [self.generate_combinations(v) if isinstance(v, dict) else v for v in values]
        for combination in itertools.product(*values):
            yield dict(zip(keys, combination))
        return self.generate_combinations(d)
    
    def generate_range(self, nested_dict, func):
        '''Helper function which updates every values in a nested dict such that the values [a,b] -> func(*[a,b])'''
        res = {} 
        for key, value in nested_dict.items(): 
            if isinstance(value, dict): 
                res[key] = self.generate_range(value, func) 
            else: 
                res[key] = func(*value)
        return res
    
    def fill_dict(self, nested_dict, array, index=0):
        '''Helper function to recursively fill a dict given an array'''
        for key in nested_dict:
            if isinstance(nested_dict[key], dict):
                index = self.fill_dict(nested_dict[key], array, index)
            else:
                if index < len(array):
                    nested_dict[key] = array[index]
                    index += 1
                else:
                    break
        return nested_dict
    
    def num_elements(self, x):
        '''Helper function to recursively count the elements in a nested dict'''
        if isinstance(x, dict):
            return sum([self.num_elements(_x) for _x in x.values()])
        else: return 1
        
    def extract_values(self, nested_dict):
        '''Helper function to recursively extract all values from a nested dict'''
        values = []
        for key in nested_dict:
            if isinstance(nested_dict[key], dict):
                values.extend(self.extract_values(nested_dict[key]))
            else:
                values.append(nested_dict[key])
        return values
    
    def extract_keys(self, nested_dict):
        '''Helper function to recursively extract all keys from a nested dict'''
        keys = []
        for key in nested_dict:
            keys.append(key)
            if isinstance(nested_dict[key], dict):
                keys.extend(self.extract_keys(nested_dict[key]))
        return keys
            
    @staticmethod
    def cut_lightcone(cone, z_cut):
        amidx = np.abs(cone.lightcone_redshifts - z_cut).argmin()
        cone.brightness_temp = cone.brightness_temp[:,:,:amidx]
        return cone
            
    @staticmethod
    def save(obj, fname, direc, run_id):
        obj.save(fname=fname+str(run_id), direc=direc)
    
    @staticmethod
    def load(path_to_obj):
        return h5py.File(path_to_obj)
    
    
    @staticmethod
    def convert_to_npz(path: str, prefix: str = "", check_for_nan: bool = True, debug: bool = False) -> None:
        '''Given a path and an optinal prefix 
        (e.g. only convert all files named as run_, set prefix = "run_")
        this function converts .h5 files from 21cmfastwrapper to the common .npz format'''
        # image, label, tau, gxH
        # search for all files given in a path given a prefix an loop over those
        files = fnmatch.filter(os.listdir(path), prefix + "*")
        if debug: print(f"{files}")
        len_files = len(files)
        nan_counter = []
        # initalize the progress bar
        for i,file in enumerate(files):
            lcone = p21c.outputs.LightCone.read(path + file)
            # load image
            image = lcone.brightness_temp
            # check if there are NaNs in the brightness map
            if check_for_nan:
                if np.isnan(image).any():
                    nan_counter.append(file)
                    continue
            #load labels, WDM,OMm,LX,E0,Tvir,Zeta
            labels = [
                lcone.global_params["M_WDM"],
                lcone.cosmo_params.OMm,
                lcone.astro_params.L_X,
                lcone.astro_params.NU_X_THRESH,
                lcone.astro_params.ION_Tvir_MIN,
                lcone.astro_params.HII_EFF_FACTOR
            ]
            # load redshift
            redshifts = lcone.node_redshifts
            # compute tau
            gxH=lcone.global_xH
            gxH=gxH[::-1]
            redshifts=redshifts[::-1]
            tau=p21c.compute_tau(redshifts=redshifts,global_xHI=gxH)

            new_format = {
                "image": image,
                "label": labels,
                "tau": tau,
                "z": redshifts,
                "gxH": gxH
            }
            #save to new format
            np.savez(path + file + ".npz", **new_format)
            
            # progress counter
            if len_files >= 10:
                if i % (int(len_files/10)) == 0:
                    idx = int(i / int(len_files) * 10)
                    progress = "-"*(idx) + ">" + (10-idx)*" "
                    print(f"|{progress}|", end='', flush=True)
                    print("\r", end='')
        print(f"Done, {len(nan_counter)} NaNs encountered in \n{nan_counter}")   
    

"""
    def load(self, name):
        with open(name, 'rb') as f:
            self.data = pickle.load(f)
    def pop(self,idx):
        '''Delete a run with idx'''
        self.data.pop(idx)
        
    def clear(self):
        '''Clear the data/runs cache'''
        self.data.clear()
"""     
