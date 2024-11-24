import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest, uniform, gaussian_kde
from alive_progress import alive_bar
from typing import Callable
from logging import info, warning, error
from NPE import NPEHandler




class NLEHandler(NPEHandler):
    def __init__(self, density_estimator: nn.Module, summary_net: nn.Module = None,  
                 device = 'cuda'):
        super().__init__(density_estimator, summary_net, device)

        
        info("Succesfully initialized SBIHandler")
        
    def train(self, training_data: object, test_data: object, epochs: int = 20, freezed_epochs: int = 0, pretrain_epochs: int = 0, optimizer = torch.optim.Adam,
              optimizer_kwargs: dict = {"lr": 1e-4}, loss_function: Callable = torch.nn.MSELoss, loss_params: dict = {'reduction': 'none', 'kernel_size': 10}, device: str = None, plot: bool = True,
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
        if self.sum_net:
            loss_function = loss_function(**loss_params)
        
        # set summary net and push to device
        if self.sum_net:
            self.summary_net.to(device)
        loss_sn = (lambda x,y: loss_function(x,y).sum(-1))
        
        # push density net to device
        self.density_estimator.to(device)
        
        # pretrain summary_net
        if pretrain_epochs > 0:
            summary_net = SumnetHandler(self.summary_net, training_data, test_data, device)
            summary_net.train(pretrain_epochs, loss_function, optimizer(self.summary_net.parameters(), **optimizer_kwargs))
            summary_net = summary_net.Model
        
        # hack number of freezed epochs for test loss
        self.stpe = freezed_epochs
        
        with alive_bar(epochs, force_tty=True, refresh_secs=5) as bar:
                        
            train_loss_de, test_loss_de = [], []
            train_loss_sn, test_loss_sn = [], []
            
            
            # training loop
            for epoch in range(epochs):
                
                # initialize optimizer
                if self.sum_net and epoch == freezed_epochs:
                    self.optimizer = optimizer(list(self.density_estimator.parameters()) + list(self.summary_net.parameters()), **optimizer_kwargs)
                    info("Optimizer initialized for joint training of summary and density network.")
                elif epoch == 0 and epoch < freezed_epochs:
                    info("Initialize optimizer for density estimator training with freezed summary...")
                    self.optimizer = optimizer(self.density_estimator.parameters(), **optimizer_kwargs)
                
                # set state of neural nets
                self.density_estimator.train()
                if self.sum_net and epoch < freezed_epochs:
                    self.summary_net.eval()
                elif self.sum_net:
                    self.summary_net.train()
                
                
                # reset tmp loss counter
                train_loss_de_tmp = 0
                train_loss_sn_tmp = 0
                    
                # start main trainingsloop
                for lab, img, rnge in training_data:
                    
                    img, lab, rnge = img.to(device), lab.to(device), rnge.to(device)
                    # navigate through right propagation and compute primary and auxillary losses
                    loss, _train_loss_sn = self._loss(img, lab, rnge, epoch, freezed_epochs, loss_function)
                    
                    # backprop 
                    loss.mean().backward()
                    
                    train_loss_sn_tmp += _train_loss_sn
                    train_loss_de_tmp += loss.mean().item()
                    
                    # grad clipping
                    if grad_clipping:
                        torch.nn.utils.clip_grad_norm_(self.density_estimator.parameters(), grad_clip)
                        if epoch >= freezed_epochs:
                            if self.sum_net:
                                torch.nn.utils.clip_grad_norm_(self.summary_net.parameters(), grad_clip)
                            
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                # testing loop
                test_loss_de_tmp, test_loss_sn_tmp = self.test_self(test_data, epoch, freezed_epochs, loss_sn)
                # save losses for plot
                train_loss_de.append(train_loss_de_tmp / len(training_data))
                test_loss_de.append(test_loss_de_tmp)
                if self.sum_net:
                    train_loss_sn.append(train_loss_sn_tmp / len(training_data))
                    test_loss_sn.append(test_loss_sn_tmp)
                    
                bar()

        if self.sum_net: 
            self.summary_net.eval()
            self.summary_net.zero_grad(set_to_none=True)
        self.density_estimator.eval()
        self.density_estimator.zero_grad(set_to_none=True)
        
        if plot:      
            plt.plot(np.linspace(0, epochs, len(train_loss_de)), train_loss_de, label='Trainingsloss DE', alpha=0.5)
            plt.plot(np.linspace(0, epochs, len(test_loss_de)), test_loss_de, label='Testloss DE')
            if self.sum_net:
                plt.plot(np.linspace(0, epochs, len(train_loss_sn)), np.log(train_loss_sn), label='Trainingsloss SN', alpha=0.5)
                plt.plot(np.linspace(0, epochs, len(test_loss_sn)), np.log(test_loss_sn), label='Testloss SN')
            plt.xlabel("epochs")
            plt.ylabel("loss")
            plt.title("Log loss during training")
            plt.legend()
            if lossfile == "": plt.show()
            else: plt.savefig(f"./{lossfile}.png", dpi=400)
            plt.clf()
    
    @torch.no_grad()   
    def test_self(self, validation_data: object, epoch: int = 0, freezed_epochs: int = 0, loss_sn: Callable = torch.nn.MSELoss()):
        if self.sum_net:
            self.summary_net.eval()
        self.density_estimator.eval()
        
        tmp_loss_de = 0
        tmp_loss_sn = 0
        
        for lab, img, rnge  in validation_data:
            img, lab, rnge = img.to(self.device), lab.to(self.device), rnge.to(self.device)
            
            loss, _test_loss_sn = self._loss(img, lab, rnge, epoch, freezed_epochs, loss_sn)
            tmp_loss_sn += _test_loss_sn
            tmp_loss_de += loss.mean().item()
        return tmp_loss_de / len(validation_data), tmp_loss_sn / len(validation_data)

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
        loss = self.density_estimator.loss(summary, cond=lab)
        loss = loss.mean(0)

        return loss, train_loss_sn_tmp