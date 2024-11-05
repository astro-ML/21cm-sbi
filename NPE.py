import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
from alive_progress import alive_bar
from logging import info, warning, error
from Trainer import SumnetHandler

import os
import tempfile

from ray import train, tune
from ray.tune.schedulers import ASHAScheduler

from ray.train import Checkpoint

from ray.tune.search.hyperopt import HyperOptSearch


class NPEHandler():
    def __init__(self, density_estimator, summary_net = None,
                 device = 'cuda'):
        self.density_estimator = density_estimator
        if summary_net is None:
            self.sum_net = False
        else:
            self.sum_net = True
            self.summary_net = summary_net
        self.summary_net = summary_net
        self.device = device
        self.opti_hype = False
        
        info("Succesfully initialized NPEHandler")

    def train(self, training_data: object, test_data: object, epochs: int = 20, freezed_epochs: int = 0, pretrain_epochs: int = 0, optimizer = torch.optim.Adam,
              optimizer_kwargs: dict = {"lr": 1e-4}, loss_function: Callable = torch.nn.MSELoss, loss_params: dict = {}, device: str = None, plot: bool = True,
              grad_clip: float = 0, lossfile: str = ""):
        
        # set bool for gradient clipping
        if grad_clip > 0:
            grad_clipping = True
        else:
            grad_clipping = False
        
        # watch for local device changes
        if device is None: 
            device = self.device
            info(f"Using device {device}")
        info("Begin training...")
        
        # initialize loss function
        loss_function = loss_function(**loss_params)
        
        if self.sum_net:
            self.summary_net.to(device)
        self.density_estimator.to(device)
        
        # pretrain summary_net
        if pretrain_epochs > 0:
            summary_net = SumnetHandler(self.summary_net, device)
            summary_net.train(epochs=pretrain_epochs, training_data=training_data, test_data= test_data, lossf=loss_function, optimizer=optimizer(self.summary_net.parameters(),**optimizer_kwargs))
            summary_net = summary_net.Model
        
        # begin main trainingsloop
        
        with alive_bar(epochs, force_tty=True, refresh_secs=5) as bar:
                        
            train_loss_de, test_loss_de = [], []
            if self.sum_net:
                train_loss_sn, test_loss_sn = [], []
            
            # training loop
            for epoch in range(epochs):
                
                # initialize optimizer
                if self.sum_net and epoch == freezed_epochs:
                    info("Initialize optimizer for joint training...")
                    self.optimizer = optimizer(list(self.density_estimator.parameters()) + list(self.summary_net.parameters()), **optimizer_kwargs)
                if not self.sum_net or ( epoch == 0 and epoch < freezed_epochs):
                    info("Initialize optimizer for density estimator training with freezed summary...")
                    self.optimizer = optimizer(self.density_estimator.parameters(), **optimizer_kwargs)
                
                self.density_estimator.train()
                if self.sum_net and epoch < freezed_epochs:
                    self.summary_net.eval()
                else:
                    self.summary_net.train()
                    
                train_loss_de_tmp = 0
                if self.sum_net:
                    train_loss_sn_tmp = 0
                    
                    
                for lab, img, rnge in training_data:
                    
                    img, lab, rnge = img.to(device), lab.to(device), rnge.to(device)

                    loss, _train_loss_sn = self._loss(img, lab, rnge, epoch, freezed_epochs, loss_function)

                    train_loss_de_tmp += _train_loss_sn
                    
                    loss.backward()
                    
                    # grad clipping
                    if grad_clipping:
                        torch.nn.utils.clip_grad_norm_(self.density_estimator.parameters(), grad_clip)
                        if self.sum_net:
                            torch.nn.utils.clip_grad_norm_(self.summary_net.parameters(), grad_clip)
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    train_loss_de_tmp += loss.item()
                    
                # testing loop
                test_loss_de_tmp, test_loss_sn_tmp = self.test_self(test_data, loss_function)
                    
                train_loss_de.append(train_loss_de_tmp / len(training_data))
                test_loss_de.append(test_loss_de_tmp)
                if self.sum_net:
                    train_loss_sn.append(train_loss_sn_tmp / len(training_data))
                    test_loss_sn.append(test_loss_sn_tmp)
                
                if self.opti_hype:
                    with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                        checkpoint = None
                        if (i + 1) % 5 == 0:
                            # This saves the model to the trial directory
                            torch.save(
                                self.density_estimator.state_dict(),
                                os.path.join(temp_checkpoint_dir, "model.pth")
                            )
                            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)

                        # Send the current training result back to Tune
                        train.report({"mean_loss": test_loss_de_tmp}, checkpoint=checkpoint)
                    
                bar()
                
        train_loss_de = np.array(train_loss_de)
        test_loss_de = np.array(test_loss_de)
        if self.sum_net:
            train_loss_sn = np.array(train_loss_sn)
            test_loss_sn = np.array(test_loss_sn)
                
        self.summary_net.eval()
        self.density_estimator.eval()
        
        if plot:     
            plt.plot(np.linspace(0, epochs, len(train_loss_de)), train_loss_de, label='Trainingsloss DE', alpha=0.5)
            plt.plot(np.linspace(0, epochs, len(test_loss_de)), test_loss_de, label='Testloss DE')
            if self.sum_net:
                plt.plot(np.linspace(0, epochs, len(train_loss_sn)), np.log(train_loss_sn), label='Trainingsloss SN', alpha=0.5)
                plt.plot(np.linspace(0, epochs, len(test_loss_sn)), np.log(test_loss_sn), label='Testloss SN')
            plt.xlabel("epochs")
            plt.ylabel("norm loss")
            plt.title("Log loss during training")
            plt.legend()
            if lossfile == "": plt.show()
            else: plt.savefig(f"./{lossfile}.png", dpi=400)
            plt.clf()
        return {"trainloss": train_loss_de, 
        "testloss": test_loss_de}
    
    def grid_search(self, search_space: dict, training_data: object, test_data: object, train_kwargs: dict = {}):
        """Perform a grid search using Baysian optimization.

        Args:
            search_space (dict): Dict containing the search space. An example might look like
            from hyperopt import hp
            search_space = {
                "lr": hp.loguniform("lr", -10, -1),
                "momentum": hp.uniform("momentum", 0.1, 0.9),
            }
            training_data (object): Torch dataloader containing the trainingsdata
            test_data (object): Torch dataloader containing the testdata
            train_kwargs (dict, optional): Addition parameter passed to the training function. Defaults to {}.
        """
        hyperopt_search = HyperOptSearch(search_space, metric="mean_loss", mode="min")
        tuner = tune.Tuner(
            tune.with_parameters(self.train, training_data=training_data, test_data=test_data, **train_kwargs),
            tune_config=tune.TuneConfig(
                num_samples=10,
                scheduler=ASHAScheduler(metric="mean_loss", mode="min",
                search_algo=hyperopt_search),
            ),
            param_space=search_space,
        )
        results = tuner.fit()

        # Obtain a trial dataframe from all run trials of this `tune.run` call.
        dfs = {result.path: result.metrics_dataframe for result in results}
        torch.save(dfs, "./dfs_results.pt") 
        ax = None  # This plots everything on the same plot
        for d in dfs.values():
            ax = d.mean_accuracy.plot(ax=ax, legend=False)
        
    
    @torch.no_grad()   
    def test_self(self, validation_data: object, loss_function: Callable = torch.nn.MSELoss()):
        if self.sum_net:
            self.summary_net.eval()
        self.density_estimator.eval()
        test_loss_de_tmp = 0
        if self.sum_net:
            test_loss_sn_tmp = 0
        for lab, img, rnge  in validation_data:
            lab, img, rnge = lab.to(self.device), img.to(self.device), rnge.to(self.device)
            loss, _test_loss_sn = self._loss(img, lab, rnge, 0, 1, loss_function)
            test_loss_sn_tmp += _test_loss_sn
            test_loss_de_tmp += loss.mean().item() 
        return test_loss_de_tmp / len(validation_data), test_loss_sn_tmp / len(validation_data)


        
    def save_model(self, path: str = "./"):
        torch.save(self.density_estimator.state_dict(), path + "density_model.pt")
        torch.save(self.summary_net.state_dict(), path + "summary_model.pt")
        
    def load_model(self, path: str = "./"):
        self.density_estimator.load_state_dict(torch.load(path + "density_model.pt"))
        self.density_estimator.to(self.device)
        self.density_estimator.eval()
        self.summary_net.load_state_dict(torch.load(path + "summary_model.pt"))
        self.summary_net.to(self.device)
        self.summary_net.eval()

    @torch.no_grad()
    def test(self, val_data: object = None, plotname: str = "", nsamples: int = 10000):
        self.density_estimator.eval()
        if self.sum_net:
            self.summary_net.eval()
            
        for i in range(3):
            lab, img, _ = val_data.dataset[i]

            lab, img = lab.to(self.device), img.to(self.device)
            img = img.unsqueeze(0)

            if self.sum_net:
                summary = self.summary_net(img)
            else:
                summary = img

            samples, _ = self.density_estimator.sample(nsamples, summary)
            # plot posterior samples
            figure, axis = pairplot(samples = samples.detach().to('cpu').numpy(), points=lab.detach().to('cpu').numpy(),
                limits=[[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],], figsize=(10, 10),
                labels = [r"$M_{WDM}$", r"$\Omega_m$", r"$L_X$", r"$E_0$", r"$T_{vir, ion}$", r"$\zeta$"],
                #quantiles=((0.16, 0.84, 0.0015, 0.99815)), levels=(1 - np.exp(-0.5),1 - np.exp(-9/2)),
                upper = 'hist', lower = 'contour', diag = 'kde')
            if plotname != "": figure.savefig(f"{plotname}_{i}.png", dpi=300)
            figure.show()

    def _loss(self, img, lab, rnge, epoch, freezed_epochs, loss_function):
        if self.sum_net:
            if epoch < freezed_epochs:
                summary = self.summary_net(img, rnge).detach()
                train_loss_sn_tmp = loss_function(summary, lab).mean().item()
            else:
                summary = self.summary_net(img, rnge)
                train_loss_sn_tmp = loss_function(summary, lab).mean().item()
            
            
        else:
            summary = img

        if summary.shape != lab.shape:
            raise error(f"Summary {summary.shape} and label {lab.shape} shape do not match")                    
        # computing loss
        loss = self.density_estimator.loss(lab, cond=summary)
        loss = loss.mean(0)

        return loss, train_loss_sn_tmp