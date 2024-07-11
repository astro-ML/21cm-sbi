from ps2d_for_sbi import *
from utility import *
from models import * 
from dataloader import *
my_module_path = os.path.join("./", "..", '21cm-wrapper')
sys.path.append(my_module_path)
from Leaf import *
from sbi.utils.get_nn_models import (
        posterior_nn,
    )  # For SNPE: posterior_nn(), SNLE: likelihood_nn(). For SNRE: classifier_nn()

# define the simulator
def simulation(theta):

    user_params = {
    "HII_DIM": 40,
    "BOX_LEN": 160,
    "N_THREADS": 1,
    "USE_INTERPOLATION_TABLES": True,
    "PERTURB_ON_HIGH_RES": True
    }

    flag_options = {
    "INHOMO_RECO": True,
    "USE_TS_FLUCT": True
    }

    #simparams = p21c.outputs.LightCone.read("./data/run_36690")
    # load the simulator class
    Leaf_simulator = Leaf(debug=False, user_params=user_params, flag_options=flag_options, redshift = 5.5)

    M_WDM, OMm, L_X, NU_X_THRESH, ION_Tvir_MIN, HII_EFF_FACTOR = theta
    cosmo_params = {
        "OMm": OMm
        }
    astro_params = {
        "L_X": L_X,
        "NU_X_THRESH": NU_X_THRESH,
        "ION_Tvir_MIN": ION_Tvir_MIN,
        "HII_EFF_FACTOR": HII_EFF_FACTOR,
    }
    global_params = {
        "M_WDM": M_WDM
    }

    res = Leaf_simulator.run_lightcone(
        save = False, sanity_check = True, filter_peculiar = False,
        astro_params = astro_params, global_params = global_params, cosmo_params = cosmo_params).brightness_temp[:,:,:600]
    print('done simulating')
    return res, theta


def simulator(theta: torch.FloatTensor, Model: object, data_loader: object, threads: int = 1):
    tshape = theta.shape
    schwimmhalle = Pool(max_workers=threads, max_tasks_per_child=1, mp_context=get_context('spawn'))
    first = True
    with alive_bar(tshape[0], force_tty=True, refresh_secs=5) as bar: 
        with schwimmhalle as p:
            futures = [p.submit(simulation, params.tolist()) for params in theta]
            for future in as_completed(futures):
                lc_bt, lab = future.result()
                lc_bt, lab = torch.as_tensor(lc_bt, dtype=torch.float32), torch.as_tensor(lab, dtype=torch.float32)
                bt, lab = data_loader.normalize(images=torch.unsqueeze(lc_bt, dim=0), labels=torch.unsqueeze(lab, dim=0))
                pred = Model.fast_forward(bt)
                if first:
                    x = torch.empty(pred.shape, dtype=torch.float32)
                    first=False
                # may be unsorted
                x = torch.cat((x, pred),0)    
                bar()
    return x



if __name__ == '__main__':
    # hyperparams
    data_path = "./data/"
    train_test_data_ration = 0.95

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
    convert_to_torch(path = data_path, prefix="run", redshift_cutoff=600, debug=False, statistics=False)

    # load data
    train_data = DataHandler(path=data_path, prefix="batch", load_to_ram=False,
                                split = train_test_data_ration, training_data = True,
                                apply_norm=True, norm_range=norm_range, augmentation_probability=0)

    # import data to torch dataloader
    train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True,
                                    num_workers = 2, pin_memory = True, prefetch_factor=2)


    # init model
    model = ModelHandler(Model = Summary_net_lc_smol,
                            Training_data=train_dataloader, device='cpu')

    from sbi.inference import SNPE, SNLE
    from sbi import utils, analysis

    # define model hyperparemeter


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

    ### SNPE ###

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

    posterior_samples = train_data.denormalize(labels=posterior.sample((5000,)))

    # We use posterior theta samples to generate x data

    x_pp = simulator(theta = posterior_samples, Model = model, threads=6, data_loader=train_data)

    # We verify if the observed data falls within the support of the generated data
    fig, _ = analysis.pairplot(
        samples=x_pp,
        points=x_o,
        limits=[[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],], figsize=(5, 5),
        labels = ["M_WDM", "OMm", "L_X", "NU_X_THRESH", "ION_Tvir_MIN", "HII_EFF_FACTOR"],
    )
    fig.savefig("sbc.png", dpi=300)




