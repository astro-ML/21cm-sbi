from ps2d_for_sbi import *
from utility import *
from models import * 
from dataloader import *
my_module_path = os.path.join("./", '..', '21cm-wrapper')
sys.path.append(my_module_path)
from Leaf import *

from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
from sbi import utils as utils

simparams = p21c.outputs.LightCone.read("./data/run_36690")
simulator = Leaf(user_params = simparams.user_params.defining_dict,
                 flag_options = simparams.flag_options.defining_dict,
                 debug=True)

prior_range = torch.tensor([
            [0.3,10.0], # M_WDM
            [0.2,0.4], # OMm
            [38, 42], # L_X
            [100, 1500], # NU_X_THRESH
            [4, 5.3], # ION_Tvir_MIN
            [10.0, 250.0], # HII_EFF_FACTOR
], dtype = torch.float32)

prior = utils.BoxUniform(low=prior_range[:,0], high=prior_range[:,1])

def simulation(theta):
    M_WDM, OMm, L_X, NU_X_THRESH, ION_Tvir_MIN, HII_EFF_FACTOR = theta
    cosmo_params = {
        "OMm": OMm
        }
    astro_params = {
        "L_X": L_X,
        "NU_X_THRESH": NU_X_THRESH,
        "ION_Tvir_MIN": ION_Tvir_MIN,
        "HII_EFF_FACTOR": HII_EFF_FACTOR
    }
    global_params = {
        "M_WDM": M_WDM
    }
    return simulator.run_lightcone(
        redshift = 5.5, save = False, sanity_check = True, filter_peculiar = False,
        astro_params = astro_params, global_params = global_params, cosmo_params = cosmo_params
    )

print(simulation(prior.sample()))
