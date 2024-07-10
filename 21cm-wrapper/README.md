p21cmfast wrapper v0.41 

<b>Quickstart</b> <br/>
(* - required for cluster deployment):<br/> <br/>

1*) Create an new virtual environment <br />
```python -m venv ./21cm-wrapper-penv```<br /><br />
2*) Activate the environment <br />
```source ./21cm-wrapper-penv/bin/activate``` <br /><br />
3) Install the requirements <br />
```pip install -r requirements.txt``` <br /><br />
4) (optional) change the cache path in p21cmfastwrapper.py, line 21 <br /><br />
5) Get familiar with the wrapper by open 21cmfast.ipynb in your jupyter notebook <br />
  You can import the classes and load all dependencies by running <br />
```from p21cmfastwrapper import *``` <br /><br />
6) Edit the parameter.yaml file (!) to your liking. It initializes parameters and allows for detailed control of fixed parameters. <br /> <br /> <br />

<b>Data generation on cluster</b> <br/>
For running the code on a cluster using PBS one needs to create two files:<br /> <br />

1) A python script which will be executed. For running simulations with uniform sampled parameters<br />
  the script may look like this: <br />
```python
#import the wrapper
from p21cmfastwrapper import *

# initialize the wrapper and define saving behavoir
sim = Simulation(save_ondisk=True, write_cache=False, save_inclass=False)

# define a sample function, in this case we sample uniformly from a given parameter range
samplef = (lambda a,b: np.random.uniform(a,b))

# To set the parameter ranges, the parameter must be given as a dict, following the dict
# structure defined in parameter.yaml. The values must be an array which is handed over
# to the samplef defined above. In this case it is [start, stop].
args = {"astro_params": 
        {"NU_X_THRESH": [400,700], 
        "HII_EFF_FACTOR": [10,100],
        "L_X": [1,2],
        "ION_Tvir_MIN": [4,5]},
    "cosmo_params":
        {"OMm": [0,1]},
    "global_params": 
        {"M_WDM": [1,2]}}

# Set the number of simulations with nruns and the number of threads (= #cores on the cluster)
# Note that if the cluster has multiple nodes, you may use mpi=True as an argument
# If not, it is considerable faster to leave it False (default)
sim.run_samplef(nruns=12, args=args, samplef=samplef, threads = 6)
```
And we save it as main.py. <br />  <br />

2) A bash script which is queued using the  <br />
```qsub``` <br />
command. It may look like this: <br />
```bash
#!/bin/bash
#PBS -l nodes=1:ppn=32:bigmemlong
#PBS -q bigmemlong
#PBS -l mem=80gb
#PBS -M YOUR_MAIL -m ae
#PBS -l walltime=RUNTIME
cd PATH_TO_YOUR_FOLDER
source ./21cm-wrapper-penv/bin/activate
python ./main.py
``` 
We save it as run_main.sh. <br />  <br />

Now can run it via <br />
```qsub run_main.sh```
. <br /> <br /> <br />

<b>MCMC</b> <br/><br/>
For affine-invariant ensemble sampling using emcee, <br/>
a starting script may look like this:<br/>
```python
# load class
from mcmc import *

# set first sampling range
init_params_ranges = {"astro_params": {"ION_Tvir_MIN": [4,5.3], "HII_EFF_FACTOR": [5, 100], 
                    "L_X": [30,50], "NU_X_THRESH": [100, 1500]}}

# define the log-prior
def log_prior(theta):
        T_vir, H_eff, LX, nu_x = theta
        if 5 < H_eff < 100 and 100 < nu_x < 1500 and 4 < T_vir < 5.3 and 30 < LX < 50:
            return 0
        return - np.inf

#define the log-likelihood
def log_likelihood(t,f):
    return - np.sum((t - f)**2/f)

# initialize sampler, choose nwalker twice the number of available cpu-cores
# for best performance
mcrun = mcmc(nwalker=8, z_bins = 11, debug = True,
	 log_likelihood=log_likelihood, prior=log_prior)
	 
# generate or load fiducial model
mcrun.make_fiducial(load = False)

# run the sampler
mcrun.run_aies(init_params_ranges)
```
The program can be queued as above. <br/> <br/>



For doing nested sampling via pymultinest, <br/>
a starting script may look like this:

```python
# load class
from mcmc import *

# set first sampling range
init_params_ranges = {"astro_params": {"ION_Tvir_MIN": [4,5.3], "HII_EFF_FACTOR": [5, 100], 
                    "L_X": [30,50], "NU_X_THRESH": [100, 1500]}}

# input: init cube for parameter [0,1)
# pymultinest samples parameter from the prior, not probabilities
def prior_sampling(theta):
        T_vir, H_eff, LX, nu_x = theta
        T_vir = 4 + 1.3*T_vir
        H_eff = 5 + 95*H_eff
        LX = 30 + 20*LX
        nu_x = 100 + 1400*nu_x
        return np.array([T_vir, H_eff, LX, nu_x])

# define the log-likelihood
def log_likelihood(t,f):
    return - np.sum((t - f)**2/f)

# initialize sampler
mcrun = mcmc(z_bins = 11, debug = True, log_likelihood=log_likelihood, prior=prior_sampling)

# generate or load fiducial model
mcrun.make_fiducial(load = True)

# run the sampler
mcrun.run_ns(init_params_ranges)
```
pymultinest requires multiprocessing via MPI <br/>
To run the script multithreaded, change the line in your run.sh to
```bash
mpiexec -n NUMBER-OF-CPU-CORES python main.py
```
<br/><br/>



Philosphy:
- Optionality: All functions have set default arguments to simplify execution but also allow for
lower level calls
- Flexibility: The classes are written to work with them in a .ipynb notebook

Addendum:

There are two classes: Parameters (which handles everything related to the parameters) and Simulation (which handles p21cmfast and the visualization).
Simulation inherits Parameters, so there is no need to work with the Parameters class at all.

To-do (Priority):
- [x] implement (uniform / distribution function) sampling of parameters, run sims, and save them

To-do:
- [x] true multiprocessing of py21cmfast (GIL a problem??)
- [x] add saving method (!)
- [ ] make ps using custom z-bins
- [ ] show progress bars
- [ ] more cleanup, especially the plotting routines
- [ ] make a compare method (like ps) for global_props plot 


