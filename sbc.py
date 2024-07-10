from ps2d_for_sbi import *
from utility import *
from models import * 
from dataloader import *
my_module_path = os.path.join("./", "..", '21cm-wrapper')
sys.path.append(my_module_path)
from Leaf import *

# hyperparams
data_path = "./data/"
batch_size = 8
epochs = 120
train_test_data_ration = 0.95

optimizer = torch.optim.Adam
optimizer_params = {
    "lr": 6e-4,
}

loss = torch.nn.MSELoss
loss_params = {}

norm_range = torch.tensor([
            [0.3,10.0], # M_WDM
            [0.2,0.4], # OMm
            [38, 42], # L_X
            [100, 1500], # NU_X_THRESH
            [4, 5.3], # ION_Tvir_MIN
            [10.0, 250.0], # HII_EFF_FACTOR
], dtype = torch.float32)


# transform trainingsdata
# perhaps add check if file is there: continue + override option in the future
#convert_to_torch(path = data_path, prefix="run", redshift_cutoff=600, debug=False, statistics=True)

# load data
train_data = DataHandler(path=data_path, prefix="batch", load_to_ram=False,
                            split = train_test_data_ration, training_data = True,
                            apply_norm=True, norm_range=norm_range, augmentation_probability=0)
test_data = DataHandler(path=data_path, prefix="batch", load_to_ram=False,
                            split = train_test_data_ration, training_data = False,
                            apply_norm=True, norm_range=norm_range, augmentation_probability=0)
# import data to torch dataloader
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                num_workers = 2, pin_memory = True, prefetch_factor=2)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                                num_workers = 2, pin_memory = True, prefetch_factor=2)

# init model
model = ModelHandler(Model = Summary_net_lc_smol,
                        Training_data=train_dataloader, Test_data=test_dataloader, device='cpu')

from sbi.inference import SNPE, SNLE
from sbi import utils, analysis

# define model hyperparemeter
user_params = {
"HII_DIM": 40,
"BOX_LEN": 160,
"N_THREADS": 2,
"USE_INTERPOLATION_TABLES": True,
"PERTURB_ON_HIGH_RES": True
}

flag_options = {
"INHOMO_RECO": True,
"USE_TS_FLUCT": True
}

#simparams = p21c.outputs.LightCone.read("./data/run_36690")
# load the simulator class
Leaf_simulator = Leaf(debug=True, user_params=user_params, flag_options=flag_options, redshift = 5.5)

# define the prior ranges (only need for denormalization!)
prior_range = torch.tensor([
            [0.3,10.0], # M_WDM
            [0.2,0.4], # OMm
            [38, 42], # L_X
            [100, 1500], # NU_X_THRESH
            [4, 5.3], # ION_Tvir_MIN
            [10.0, 250.0], # HII_EFF_FACTOR
], dtype = torch.float32)

# define the prior (uniform prior in this case)
prior = utils.BoxUniform(low=torch.zeros((6)), high=torch.ones((6)))

# define the simulator
def simulation(theta):
    M_WDM, OMm, L_X, NU_X_THRESH, ION_Tvir_MIN, HII_EFF_FACTOR = theta
    cosmo_params = {
        "OMm": OMm
        }
    astro_params = {
        "L_X": L_X,
        "NU_X_THRESH": NU_X_THRESH,
        "ION_Tvir_MIN": ION_Tvir_MIN,
        "HII_EFF_FACTOR": HII_EFF_FACTOR,
        "INHOMO_RECO": True
    }
    global_params = {
        "M_WDM": M_WDM
    }

    res = torch.as_tensor(Leaf_simulator.run_lightcone(
        save = False, sanity_check = True, filter_peculiar = False,
        astro_params = astro_params, global_params = global_params, cosmo_params = cosmo_params).brightness_temp, dtype=torch.float32)
    
    return res


def simulator(theta: torch.FloatTensor, Model: object, threads: int = 1):
    tshape = theta.shape
    schwimmhalle = Pool(max_workers=threads, max_tasks_per_child=1, mp_context=get_context('spawn'))
    runner = [params.tolist() for params in theta]
    result = torch.empty((0,6), dtype=torch.float32)
    with alive_bar(len(runner), force_tty=True) as bar: 
        with schwimmhalle as p:
            data = p.map(simulation, runner)
            for dat in as_completed(data):
                pred = model.fast_forward(dat.result())
                result = torch.cat((result, pred))
                bar()
    return result

if __name__ == '__main__':
    ### SNPE ###

    from sbi.utils.get_nn_models import (
        posterior_nn,
    )  # For SNPE: posterior_nn(), SNLE: likelihood_nn(). For SNRE: classifier_nn()

    # load the summary model
    model = ModelHandler(Model = Summary_net_lc_smol, device='cpu')
    model.load_model("./summary_net.pt")

    # define the maf
    density_estimator_build_fun = posterior_nn(
        model="maf", hidden_features=60, num_transforms=6
    )

    # do inference using a freezed summary model
    x,y = model.full_inference(train_dataloader)

    # train the maf
    inference = SNPE(prior=prior, density_estimator=density_estimator_build_fun)
    inference.append_simulations(y,x)
    density_estimator = inference.train()
    posterior = inference.build_posterior(density_estimator, prior=prior)
    # define point of interest
    # [M_WDM, OMm, L_X, NU_X_THRESH, ION_Tvir_MIN, HII_EFF_FACTOR]
    _, x_o = train_data.normalize(labels=torch.tensor([2, 0.30964144154550644, 40.0, 500.0, 4.69897, 30.0], dtype=torch.float32))

    # A PPC is performed after we trained or neural posterior
    posterior.set_default_x(x_o)

    # We draw theta samples from the posterior. This part is not in the scope of SBI

    posterior_samples = train_data.denormalize(labels=posterior.sample((50,)))

    # We use posterior theta samples to generate x data

    x_pp = torch.as_tensor(simulator(theta = posterior_samples, Model = model, threads=6))

    # We verify if the observed data falls within the support of the generated data
    _ = analysis.pairplot(
        samples=x_pp,
        points=x_o
    )
    


