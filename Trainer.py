import torch
import torch.nn as nn
from typing import Callable
import matplotlib.pyplot as plt
import numpy as np
from alive_progress import alive_bar
from logging import info, warning, error

import os
import tempfile

from ray import train, tune
from ray.tune.schedulers import ASHAScheduler

from ray.train import Checkpoint

from ray.tune.search.hyperopt import HyperOptSearch

import fff.loss as fffloss


class Trainer:
    def __init__(self, 
                 NetworkHandlerDE,
                 NetworkHandlerSN,
                 training_data, 
                 test_data, 
                 device = 'cuda'):
        self.sn_net = NetworkHandlerSN
        self.de_net = NetworkHandlerDE
        self.training_data = training_data
        self.test_data = test_data
        self.device = device
        self.opti_hype = False
        self.use_dec = self.sn_net.use_dec
    
    def train(self, config: dict,
              epochs: int,
              pretrain_epochs: int = 0,
              freezed_epochs: int = 0,):
        """_summary_

        Args:
            config (dict): Example config might look like:
                            {
                                "grad_clip": 1,
                                "optimizer": torch.optim.Adam,
                                "optimizer_kwargs": {"lr": 1e-3},
                            }
            epochs (int): _description_
            pretrain_epochs (int, optional): _description_. Defaults to 0.
            freezed_epochs (int, optional): _description_. Defaults to 0.

        Returns:
            _type_: _description_
        """        
        
        # 
        if self.opti_hype:
            # quck and dirty way to reinizialize the networks
            self.sn_net.__init__(**config["summary_network_kwargs"])
            self.de_net.__init__(**config["density_network_kwargs"])
        
        
        # set bool for gradient clipping
        if config["grad_clip"] > 0:
            grad_clipping = True
            info("Gradient clipping activated")
        else:
            grad_clipping = False
            info("No gradient clipping")
        
        
        self.sn_net.to(self.device)
        self.de_net.to(self.device)
        
        # pretrain summary_net
        if pretrain_epochs > 0:
            info("Pretraining summary net...")
            self.sn_net.train(pretrain_epochs, 
                              self.training_data, 
                              self.test_data, 
                              config["optimizer"], 
                              config["optimizer_kwargs"])
        
        # begin main trainingsloop
        info("Initialize optimizer for density estimator training with freezed summary...")
        self.optimizer = config["optimizer"](self.de_net.parameters(),  **config["optimizer_kwargs"])
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        
        info("Begin training...")
        with alive_bar(epochs, force_tty=True, refresh_secs=1) as bar:
                        
            train_loss_de, test_loss_de = [], []
            train_loss_sn, test_loss_sn = [], []
            
            # training loop
            for epoch in range(epochs):
                
                # initialize optimizer
                if epoch == freezed_epochs:
                    info("Initialize optimizer for joint training...")
                    self.optimizer = config["optimizer"](list(self.sn_net.parameters()) + list(self.de_net.parameters()),  **config["optimizer_kwargs"])
                    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)

                self.de_net.train()
                if epoch < freezed_epochs:
                    self.sn_net.eval()
                else:
                    self.sn_net.train()
                    
                train_loss_de_tmp = 0
                train_loss_sn_tmp = 0
                test_loss_de_tmp = 0
                test_loss_sn_tmp = 0
                    
                    
                for lab, img, rnge in self.training_data:
                    
                    img, lab, rnge = img.to(self.device), lab.to(self.device), rnge.to(self.device)

                    
                    img, losssn = self.sn_net.loss(img, lab, rnge)
                    loss_de = self.de_net.loss(img, lab, rnge)
                    if self.use_dec:
                        loss = loss_de.mean() + losssn.mean()
                    else:
                        loss = loss_de.mean()
                    train_loss_sn_tmp += losssn.mean().item()

                    train_loss_de_tmp += loss_de.mean().item()
                    
                    loss.backward()
                    
                    # grad clipping
                    if grad_clipping:
                        torch.nn.utils.clip_grad_norm_(self.de_net.parameters(), config["grad_clip"])
                        torch.nn.utils.clip_grad_norm_(self.sn_net.parameters(), config["grad_clip"])
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                info(f"Epoch {epoch} finished. Trainloss DE: {train_loss_de_tmp / len(self.training_data)}, Trainloss SN: {train_loss_sn_tmp / len(self.training_data)}")
                lr_scheduler.step()
                # testing loop
                
                for lab, img, rnge in self.test_data:
                    img, lab, rnge = img.to(self.device), lab.to(self.device), rnge.to(self.device)
                    
                    img, loss_sn = self.sn_net.loss(img, lab, rnge)
                        
                    loss_de = self.de_net.loss(img, lab, rnge)
                    test_loss_de_tmp += loss_de.mean().item()
                    test_loss_sn_tmp += loss_sn.mean().item()
                
                info(f"Epoch {epoch} finished. Testloss DE: {test_loss_de_tmp}, Testloss SN: {test_loss_sn_tmp}")
                
                train_loss_de.append(train_loss_de_tmp / len(self.training_data))
                test_loss_de.append(test_loss_de_tmp / len(self.test_data))
                train_loss_sn.append(train_loss_sn_tmp / len(self.training_data))
                test_loss_sn.append(test_loss_sn_tmp / len(self.test_data))
                
                if self.opti_hype:
                    with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                        checkpoint = None
                        if (epoch + 1) % 5 == 0:
                            # This saves the model to the trial directory
                            torch.save(
                                self.de_net.state_dict(),
                                os.path.join(temp_checkpoint_dir, "model.pth")
                            )
                            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)

                        # Send the current training result back to Tune
                        train.report({"loss": test_loss_de_tmp}, checkpoint=checkpoint)
                    
                bar()
                
        self.train_loss_de = np.array(train_loss_de)
        self.test_loss_de = np.array(test_loss_de)
        self.train_loss_sn = np.array(train_loss_sn)
        self.test_loss_sn = np.array(test_loss_sn)
                
        self.sn_net.eval()
        self.de_net.eval()
    
    def opti_params(self, search_space: dict, train_kwargs: dict = {}):
        """Perform a grid search using Baysian optimization.

        Args:
            search_space (dict): Dict containing the search space. An example might look like
            from hyperopt import hp
            search_space = {
                "lr": hp.loguniform("lr", -10, -1),
                "momentum": hp.uniform("momentum", 0.1, 0.9),
            }
            train_kwargs (dict, optional): Addition parameter passed to the training function. Defaults to {}.
        """
        
        self.opti_hype  = True
        hyperopt_search = HyperOptSearch(search_space, metric="loss", mode="min")
        results = tune.run(
            tune.with_parameters(self.train, **train_kwargs),
            num_samples=10,
            scheduler=ASHAScheduler(metric="loss", mode="min"),
            search_alg=hyperopt_search,
            resources_per_trial={"GPU": 1},
        )
        #results = tuner.fit()

        # Obtain a trial dataframe from all run trials of this `tune.run` call.
        torch.save(results.get_all_configs, "./opt_configs.pt")
        torch.save(results.results, "./opt_results.pt")
        
    def save_model(self, path: str = "./"):
        self.de_net.save(path + "density_model.pt")
        self.sn_net.save(path + "summary_model.pt")

    def load_model(self, path: str = "./"):
        self.de_net.load(path + "density_model.pt")
        self.de_net.to(self.device)
        self.de_net.eval()
        self.sn_net.load(path + "summary_model.pt")
        self.sn_net.to(self.device)
        self.sn_net.eval()

    def plot(self, filename: str):
        plt.plot(np.arange(len(self.train_loss_de))+1, self.train_loss_de, label='Trainingsloss DE', alpha=0.5)
        plt.plot(np.arange(len(self.test_loss_de)), self.test_loss_de, label='Testloss DE')
        plt.plot(np.arange(len(self.train_loss_sn)), self.train_loss_sn, label='Trainingsloss SN', alpha=0.5)
        plt.plot(np.arange(len(self.test_loss_sn)), self.test_loss_sn, label='Testloss SN')
        plt.xlabel("epochs")
        plt.ylabel("log loss")
        plt.title("Log loss during training")
        plt.legend()
        plt.savefig(f"{filename}.png", dpi=300)
        plt.clf()
        
        
        

class SNHandler:
    def __init__(self, encoder,
                 encoder_kwargs: dict = {},
                 decoder = None,
                 decoder_kwargs: dict = {},
                 device = 'cuda',
                 no_progress_bar = False,
                 beta: float = 1.0):
        self.encoder = encoder(**encoder_kwargs).to(device)
        if decoder is not None:
            self.decoder = decoder(**decoder_kwargs).to(device)
        self.use_dec = True if decoder is not None else False
        self.device = device
        self.opti_hype = False
        self.no_progress_bar = no_progress_bar
        self.beta = beta
        self.latent_dist = torch.distributions.Normal(0,1)
        
        info("Succesfully initialized SNHandler")

    def __call__(self, img, cond=None):
        return self.encoder(img, cond)
    
    def to(self, device):
        self.encoder.to(device)
        if self.use_dec:
            self.decoder.to(device)
        
        
    def opti_params(self, search_space: dict, train_kwargs: dict = {}):
        """Perform a grid search using Baysian optimization.

        Args:
            search_space (dict): Dict containing the search space. An example might look like
            from hyperopt import hp
            search_space = {
                "lr": hp.loguniform("lr", -10, -1),
                "momentum": hp.uniform("momentum", 0.1, 0.9),
            }
            train_kwargs (dict, optional): Addition parameter passed to the training function. Defaults to {}.
        """
        import multiprocessing as mp
        mp.set_start_method('fork', force=True)

        self.opti_hype  = True
        hyperopt_search = HyperOptSearch(search_space, metric="loss", mode="min")
        results = tune.run(
            tune.with_parameters(self.training, **train_kwargs),
            num_samples=50,
            scheduler=ASHAScheduler(metric="loss", mode="min"),
            search_alg=hyperopt_search,
            resources_per_trial={"gpu": 1, "cpu": 1},
        )
        #results = tuner.fit()


        torch.save(results.get_all_configs, "./opt_configs.pt")
        torch.save(results.results, "./opt_results.pt")
        #ax = None  # This plots everything on the same plot
        #for d in dfs.values():
        #    ax = d.mean_accuracy.plot(ax=ax, legend=False)
        
    def training(self,
                 config: dict,
              epochs: int,
              training_data: object,
              test_data: object,
              lossf: Callable = nn.MSELoss(reduction = 'none')):
        
        if self.opti_hype:
            # quick and dirty way to reinizialize the networks
            self.encoder.__init__(**config["summary_network_kwargs"])
            if self.use_dec:
                self.decoder.__init__(**config["decoder_kwargs"])
        
        self.encoder.to(self.device)
        if self.use_dec:
            self.decoder.to(self.device)

        if self.use_dec:
            self.optimizer = config["optimizer"](list(self.encoder.parameters()) + list(self.decoder.parameters()), **config["optimizer_kwargs"])
        else:
            self.optimizer = config["optimizer"](self.encoder.parameters(), **config["optimizer_kwargs"])
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        losstrain = []
        losstest = []
        with alive_bar(epochs, force_tty=True, refresh_secs=1, disable=self.no_progress_bar) as bbar:
            for epoch in range(epochs):
                self.encoder.train()
                losstrain_tmp = 0
                losstest_tmp = 0
                self.train()
                for lab, img, rnge in training_data:
                    img = img.to(self.device)
                    lab = lab.to(self.device)
                    rnge = rnge.to(self.device)
                    x, loss_sn = self.loss(img, lab, rnge)
                    loss_sn = loss_sn.mean()
                    loss_sn.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    losstrain_tmp += loss_sn.item()
                losstrain.append(losstrain_tmp / len(training_data))
                
                lr_scheduler.step()
                
                self.eval()
                for lab, img, rnge in test_data:
                    
                    img = img.to(self.device)
                    lab = lab.to(self.device)
                    rnge = rnge.to(self.device)
                    x,loss_sn = self.loss(img,lab, rnge)
                    loss_sn = loss_sn.mean()
                    losstest_tmp += loss_sn.item()
                losstest.append(losstest_tmp / len(test_data))
                
                bbar()
                
                if self.opti_hype:
                    with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                        checkpoint = None
                        if (epoch + 1) % 20 == 0:
                            # This saves the model to the trial directory
                            torch.save(
                                self.encoder.state_dict(),
                                os.path.join(temp_checkpoint_dir, "model.pth")
                            )
                            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)

                        # Send the current training result back to Tune
                        train.report({"loss": losstest[-1]}, checkpoint=checkpoint)
                        
                        
        self.losstest = np.array(losstest)
        self.losstrain = np.array(losstrain)
        
    def train(self):
        self.encoder.train()
        if self.use_dec:
            self.decoder.train()
        
    def eval(self):
        self.encoder.eval()
        if self.use_dec:
            self.decoder.eval()
        
    def parameters(self):
        if self.use_dec:
            return list(self.encoder.parameters()) + list(self.decoder.parameters())
        else:
            return self.encoder.parameters()
        
    def plot(self, filename: str):
        plt.plot(np.arange(len(self.losstrain)), self.losstrain, label='Trainingsloss', alpha=0.5)
        plt.plot(np.arange(len(self.losstest)), self.losstest, label='Testloss')
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.legend()
        plt.savefig(f"{filename}.png", dpi=300)
        plt.show()
        plt.clf()
    
    def forward(self, img, cond=None):
        if self.use_dec:
            z = self.encoder(img, cond)
            x = self.decoder(z, cond)
            return z,x
        else:
            return self.encoder(img, cond)
    
    def inverse(self, lab, cond=None):
        try:
            result = self.encoder.inverse(lab, cond)
            return result
        except Exception as e:
            e.add_note("Inverse not implemented for this model")
        
    def save(self, path: str = "./"):
        torch.save(self.encoder.state_dict(), path + "encoder.pt")
        if self.use_dec:
            torch.save(self.decoder.state_dict(), path + "decoder.pt")
        
    def load(self, path: str = "./"):
        self.encoder.load_state_dict(torch.load(path + "density_model.pt"))
        self.encoder.to(self.device)
        self.encoder.eval()
        if self.use_dec:
            self.decoder.load_state_dict(torch.load(path + "decoder.pt"))
            self.decoder.to(self.device)
            self.decoder.eval()

    def loss(self, img, lab, rnge):
        if self.use_dec:
            loss = fffloss.fff_loss(img, self.encoder, self.decoder, self.latent_dist, self.beta)
            #z = self.encoder(img)
        else:
            z = self.encoder(img)
            loss = nn.MSELoss(reduction='none')(z, lab).mean(0)
        return z, loss
        
        
### Depricated, will removed in the future ###

"""class RecNetHandler(SNHandler):
    def __init__(self, Model: object, summary_net: nn.Module = None, device = "cuda"):
        super().__init__(Model=Model, device= device)
        if summary_net is None:
            self.sum_net = False
        else:
            self.sum_net = True
            self.Sum_Net = summary_net.to(device)
        
    def train(self, epochs: int, training_data: object, test_data: object,
              optimizer: object, optimizer_kwargs: dict = {}, lossf: Callable = nn.MSELoss(reduction = 'none'),
              comp_kwargs: dict = {"kernel_size": 1}, plot: bool = True):
        
        # currently no ps because metadata like redshift is lost in preprocessing step
        # post-computation possible but expensive in trainingsloop
        '''        
        res = calculate_ps(lc = lightcone.lightcones['brightness_temp'] , 
                        lc_redshifts=lightcone.lightcone_redshifts, 
                        box_length=lightcone.user_params.BOX_LEN, 
                        box_side_shape=lightcone.user_params.HII_DIM,
                        log_bins=False, zs = self.z_eval, 
                        calc_1d=True, calc_2d=False,
                        nbins_1d=self.bins, bin_ave=True, 
                        k_weights=ignore_zero_absk,postprocess=True)'''
        
        def _lossf(x, y):
            x = self.compress(x, **comp_kwargs)
            return lossf(x, y)
        
        self.lossf = _lossf
        if self.sum_net:
            self.optimizer = optimizer(list(self.Model.parameters()) + list(self.Sum_Net.parameters()), **optimizer_kwargs)
        else:
            self.optimizer = optimizer(self.Model.parameters(), **optimizer_kwargs)
    

        losstrain = []
        losstest = []
        with alive_bar(epochs, force_tty=True, refresh_secs=5) as bbar:
            for epoch in range(epochs):
                self.Model.train()
                if self.sum_net:
                    self.Sum_Net.train()
                for lab, img, _ in training_data:
                    img = img.to(self.device)
                    # add prior summnet
                    if self.sum_net:
                        lab = self.Sum_Net(img)
                    else:
                        lab = lab.to(self.device)
                    x = self.Model(lab)
                    loss = self.lossf(img, x).sum(-1).mean()
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    losstrain.append(loss.item())
                losstest.append(self.test_self(test_data))
                bbar()
        if plot:
            plt.plot(np.linspace(0, epochs, len(losstrain)), losstrain, label='Trainingsloss', alpha=0.5)
            plt.plot(np.linspace(0, epochs, len(losstest)), losstest, label='Testloss')
            plt.xlabel("epochs")
            plt.ylabel("loss")
            #plt.yscale('log')
            #plt.xticks(np.linspace(0,epochs*lentrain,10, dtype=int), np.linspace(0,epochs,10, dtype=int))
            plt.legend()
            plt.savefig("./run.png", dpi=400)
            plt.show()
            plt.clf()
        return {"trainloss": losstrain, 
        "testloss": losstest}

    @staticmethod
    def compress(x, kernel_size: int = 10 ):
        x = torch.squeeze(torch.mean(x, (-3,-2)),-2)
        x = nn.AvgPool1d(kernel_size=kernel_size, stride = kernel_size, padding = 0)(x)
        return x

    def test_self(self, testdata):
        self.Model.eval()
        if self.sum_net:
            self.Sum_Net.eval()
        test_loss = []
        with torch.no_grad():
            for lab, img, _  in testdata:
                img, lab,  = img.to(self.device), lab.to(self.device)
                if self.sum_net:
                    lab = self.Sum_Net(img)
                pred = self.Model(lab)
                test_loss.append(self.lossf(img, pred).sum(-1).mean().to('cpu').item())
        return np.mean(test_loss)

    def test_specific(self, Validation_data: object = None, lossf: Callable = nn.MSELoss(reduction='none'), 
                      num_samples: int = 3, device = 'cuda',
                      denormalize: Callable = (lambda x: x)):
        self.Model.to(device)
        self.Model.eval()
        if Validation_data is None:
            Validation_data = self.TestD
        # plot num_samples plots with avg, min/best, max/worst (evaluated via loss?) for each batch
        fig, ax = plt.subplots(1,num_samples, figsize=(5*num_samples,5), sharey=True)
        #test_idx = np.random.randint(0, len(Validation_data), num_samples)
        #test_loss = []
        with torch.no_grad():
            for i, (lab, img, _) in enumerate(Validation_data):
                if num_samples > 0:
                    img, lab = img.to(device), lab.to(self.device)
                    self.Model.eval()
                    if self.sum_net:
                        self.Sum_Net.eval()
                        lab = self.Sum_Net(img)
                    pred = self.Model(lab).cpu()
                    truth = self.compress(img, kernel_size=1).cpu() # (batch_size, event_size)
                    loss = lossf(truth,pred)
                    loss_batch, loss_features = loss.mean(-1), loss.mean(0)
                    sorted_l_idx = torch.argsort(loss_batch) # ascending
                    bestl, worstl = loss[sorted_l_idx[0]], loss[sorted_l_idx[-1]]
                    avgl = loss_features
                    best, worst, avg = pred[sorted_l_idx[0]], pred[sorted_l_idx[-1]], pred.mean(0)
                    ax[i].plot(best, label=f'Log Best, {round(torch.log10(bestl.mean()).item(),2)}', c='green')
                    ax[i].plot(truth[sorted_l_idx[0]], label=f"Best Truth", ls="dotted", c="darkgreen")
                    ax[i].plot(worst, label=f'Log Worst, {round(torch.log10(worstl.mean()).item(),2)}', c='red')
                    ax[i].plot(truth[sorted_l_idx[-1]], label=f"Worst Truth", ls="dotted", c="darkred")
                    ax[i].plot(avg, label=f'Log Avg, {round(torch.log10(avgl.mean()).item(), 2)}', c='blue')
                    ax[i].plot(truth.mean(0), label=f"Avg Truth", ls="dotted", c="darkblue")
                    '''print(f'loss: ' + str(lossf(denormalize(pred), denormalize(lab)).item()) + '\n',
                        f'pred: ' + str(denormalize(pred.to('cpu'))) + '\n',
                        f'truth: ' + str(denormalize(lab.to('cpu'))))'''
                    ax[i].legend()
                    ax[i].set_xlabel(r'$z$ bin')
                    ax[i].set_ylabel(r'$\delta T_b$')
                    num_samples -= 1
                else:
                    break
"""