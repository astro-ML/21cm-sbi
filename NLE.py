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
                 reconstruction_net: nn.Module = None, 
                 device = 'cuda'):
        super().__init__(density_estimator, summary_net, device)
        if reconstruction_net is None:
            self.rec_net = False
        else:
            self.rec_net = True
            self.reconstruction_net = reconstruction_net

        
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
        if self.rec_net:
            kernel_size = loss_params.pop('kernel_size')
        if self.sum_net:
            loss_function = loss_function(**loss_params)
        
        # set summary net and push to device
        if self.sum_net:
            self.summary_net.to(device)
        loss_sn = (lambda x,y: loss_function(x,y).sum(-1))
        
        # set reconstruction net and push to device
        if self.rec_net:
            self.reconstruction_net.to(device)
        loss_rec = (lambda x,y: loss_function(RecNetHandler.compress(x, kernel_size=kernel_size),y).sum(-1))
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
            train_loss_rec, test_loss_rec = [], []
            
            
            # training loop
            for epoch in range(epochs):
                
                # initialize optimizer
                if self.sum_net and epoch == freezed_epochs and not self.rec_net:
                    self.optimizer = optimizer(list(self.density_estimator.parameters()) + list(self.summary_net.parameters()), **optimizer_kwargs)
                if self.sum_net and epoch == freezed_epochs and self.rec_net:
                    info("Initialize optimizer for joint training...")
                    self.optimizer = optimizer(list(self.density_estimator.parameters()) + list(self.summary_net.parameters()) + list(self.reconstruction_net.parameters()), **optimizer_kwargs)
                if not self.sum_net or ( epoch == 0 and epoch < freezed_epochs):
                    info("Initialize optimizer for density estimator training with freezed summary...")
                    self.optimizer = optimizer(self.density_estimator.parameters(), **optimizer_kwargs)
                
                # set state of neural nets
                self.density_estimator.train()
                if self.sum_net and epoch < freezed_epochs:
                    self.summary_net.eval()
                elif self.sum_net:
                    self.summary_net.train()
                if self.rec_net and epoch < freezed_epochs:
                    self.reconstruction_net.eval()
                elif self.rec_net:
                    self.reconstruction_net.train()
                
                
                # reset tmp loss counter
                train_loss_de_tmp = 0
                train_loss_sn_tmp = 0
                train_loss_rec_tmp = 0
                    
                # start main trainingsloop
                for lab, img, _ in training_data:
                    
                    img, lab = img.to(device), lab.to(device)
                    # navigate through right propagation and compute primary and auxillary losses
                    loss, train_loss_de_tmp, train_loss_sn_tmp, train_loss_rec_tmp = self.compute_loss(epoch=epoch,
                    freezed_epochs=freezed_epochs,
                    img = img,
                    lab=lab,
                    train_loss_de_tmp=train_loss_de_tmp,
                    train_loss_sn_tmp=train_loss_sn_tmp,
                    train_loss_rec_tmp=train_loss_rec_tmp,
                    loss_sn=loss_sn,
                    loss_rec=loss_rec)
                    
                    # backprop 
                    loss.backward()
                    
                    # grad clipping
                    if grad_clipping:
                        torch.nn.utils.clip_grad_norm_(self.density_estimator.parameters(), grad_clip)
                        if epoch >= freezed_epochs:
                            if self.sum_net:
                                torch.nn.utils.clip_grad_norm_(self.summary_net.parameters(), grad_clip)
                            if self.rec_net:
                                torch.nn.utils.clip_grad_norm_(self.reconstruction_net.parameters(), grad_clip)
                            
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                # testing loop
                test_loss_de_tmp, test_loss_sn_tmp, test_loss_rec_tmp = self.test_self(test_data, epoch, freezed_epochs, loss_sn, loss_rec)
                # save losses for plot
                train_loss_de.append(train_loss_de_tmp / len(training_data))
                test_loss_de.append(test_loss_de_tmp)
                if self.sum_net:
                    train_loss_sn.append(train_loss_sn_tmp / len(training_data))
                    test_loss_sn.append(test_loss_sn_tmp)
                if self.rec_net:
                    train_loss_rec.append(train_loss_rec_tmp / len(training_data))
                    test_loss_rec.append(test_loss_rec_tmp)
                    
                bar()

        if self.sum_net: 
            self.summary_net.eval()
            self.summary_net.zero_grad(set_to_none=True)
        self.density_estimator.eval()
        self.density_estimator.zero_grad(set_to_none=True)
        if self.rec_net: 
            self.reconstruction_net.eval()
            self.reconstruction_net.zero_grad(set_to_none=True)
        
        if plot:      
            plt.plot(np.linspace(0, epochs, len(train_loss_de)), train_loss_de, label='Trainingsloss DE', alpha=0.5)
            plt.plot(np.linspace(0, epochs, len(test_loss_de)), test_loss_de, label='Testloss DE')
            if self.sum_net:
                plt.plot(np.linspace(0, epochs, len(train_loss_sn)), np.log(train_loss_sn), label='Trainingsloss SN', alpha=0.5)
                plt.plot(np.linspace(0, epochs, len(test_loss_sn)), np.log(test_loss_sn), label='Testloss SN')
            if self.rec_net:
                plt.plot(np.linspace(0, epochs, len(train_loss_rec)), np.log(train_loss_rec), label='Trainingsloss REC', alpha=0.5)
                plt.plot(np.linspace(0, epochs, len(test_loss_rec)), np.log(test_loss_rec), label='Testloss REC')
            plt.xlabel("epochs")
            plt.ylabel("loss")
            plt.title("Log loss during training")
            plt.legend()
            if lossfile == "": plt.show()
            else: plt.savefig(f"./{lossfile}.png", dpi=400)
            plt.clf()
    
    @torch.no_grad()   
    def test_self(self, validation_data: object, epoch: int = 0, freezed_epochs: int = 0, loss_sn: Callable = torch.nn.MSELoss(), loss_rec: Callable = torch.nn.MSELoss()):
        if self.sum_net:
            self.summary_net.eval()
        if self.rec_net:
            self.reconstruction_net.eval()
        self.density_estimator.eval()
        test_loss_de_tmp = 0
        test_loss_sn_tmp = 0
        test_loss_rec_tmp = 0
            
        for lab, img, _  in validation_data:
            img, lab,  = img.to(self.device), lab.to(self.device)
            
            _, test_loss_de_tmp, test_loss_sn_tmp, test_loss_rec_tmp = self.compute_loss(epoch=epoch,
                    freezed_epochs=freezed_epochs,
                    img = img,
                    lab=lab,
                    train_loss_de_tmp=test_loss_de_tmp,
                    train_loss_sn_tmp=test_loss_sn_tmp,
                    train_loss_rec_tmp=test_loss_rec_tmp,
                    loss_sn=loss_sn,
                    loss_rec=loss_rec)
        return test_loss_de_tmp / len(validation_data), test_loss_sn_tmp / len(validation_data), test_loss_rec_tmp / len(validation_data)

    def compute_loss(self, epoch, freezed_epochs, img, lab, train_loss_de_tmp, 
        train_loss_sn_tmp = None, train_loss_rec_tmp= None, 
        loss_sn: Callable = None, loss_rec: Callable = None):
        loss = 0
        if epoch < freezed_epochs:
            if self.sum_net:
                summary = self.summary_net(img).detach()
                _sn_loss = loss_sn(summary, lab).mean()
                train_loss_sn_tmp += _sn_loss.item()
            else:
                summary = img
            if self.rec_net:
                reconstruction = self.reconstruction_net(summary).detach()
                _rec_loss = loss_rec(img, reconstruction).mean()
                train_loss_rec_tmp += _rec_loss.item()
            _de_loss = self.density_estimator.loss(summary, cond=lab).mean()
            train_loss_de_tmp += _de_loss.item()
            loss += _de_loss
        else:
            if self.sum_net:
                if self.rec_net:
                    # loss for sum_net
                    summary = self.summary_net(img)
                    
                    # _latent_loss = 1/torch.var(summary, dim=1).mean()
                    # _latent_loss = - torch.log(torch.var(summary, dim=1))
                    # loss += _latent_loss
                    
                    _sn_loss = loss_sn(summary, lab).mean()
                    train_loss_sn_tmp += _sn_loss.item()
                    # loss for de_net
                    _de_loss, u = self.density_estimator.loss(summary, cond=lab)
                    _de_loss= _de_loss.mean()
                    # loss for rec_net
                    reconstruction = self.reconstruction_net(u)
                    _rec_loss = loss_rec(img, reconstruction).mean()
                    loss += _rec_loss
                    train_loss_rec_tmp += _rec_loss.item()
                else:
                    # loss for sum_net
                    summary = self.summary_net(img)
                    _sn_loss = loss_sn(summary, lab).mean()
                    train_loss_sn_tmp += _sn_loss.item()
                    # loss for de_net
                    _de_loss = self.density_estimator.loss(summary, cond=lab).mean()
            else:
                # loss for de_net
                summary = img
                _de_loss = self.density_estimator.loss(summary, cond=lab).mean()
            # add loss de_net
            train_loss_de_tmp += _de_loss.item()
            loss += _de_loss
        return loss, train_loss_de_tmp, train_loss_sn_tmp, train_loss_rec_tmp

    def run_sbc(self, Validation_Dataset = None, num_samples: int = 1000,
                plotname: str = "", 
                sampling_parameter: dict = {}):
        
        if sampling_parameter == {}:
            sample_attr = {
                        "sample_with": "mcmc", 
                        "method": "slice_np_vectorized",
                        "warmup_steps": 200,
                        "num_chains": 20, # change 
                        "init_strategy": "proposal", # try 'sir' here
                        "num_workers": 1,
            }
        else:
            sample_attr = sampling_parameter
        self.density_estimator.build_posterior(sample_attr)
        save = False if plotname == "" else True 
        self.density_estimator.eval()
        self.density_estimator.to(self.device)
        if self.sum_net:
            self.summary_net.eval()
            self.summary_net.to(self.device)
        
        #mp = True if num_workers > 1 else False
        # run sbc on full Validation Dataset
        lengthd = len(Validation_Dataset.dataset)
        info("Run SBC...")
        with alive_bar(len(Validation_Dataset), force_tty=True, refresh_secs=1) as bar:
            for k, (lab, img,_) in enumerate(Validation_Dataset):
                img, lab = img.to(self.device), lab.to(self.device)

                pred = self.summary_net(img).detach()
                if k == 0:
                    ranks = torch.empty((lengthd, *pred.shape[1:]), device = self.device)
                    dap_samples = torch.empty(ranks.shape, device=self.device)
                # sbc rank stat
                for i in range(pred.shape[0]):
                    samples = self.density_estimator.sample(x = pred[i].unsqueeze(0), 
                    num_samples=num_samples).detach()
                    dap_samples[k*pred.shape[0] + i] = samples[0]
                    for j in range(pred.shape[1]):
                        ranks[k*pred.shape[0] + i,j] = (samples[:,j]<lab[i,j]).sum().item()
                bar()
                
        # plot rank statistics
        ranks, dap_samples = ranks.cpu().numpy(), dap_samples.cpu()
        labels_txt = [r"$M_\text{WDM}$", r"$\Omega_m$", r"$L_X$", r"$E_0$", r"$T_\text{vir, ion}$", r"$\zeta$"]
        fig, ax = plt.subplots(1,lab.shape[1], figsize=(5*lab.shape[1],5))
        for i in range(lab.shape[1]):
            ax[i].hist(ranks[:,i], bins='auto', range=(0, num_samples), density=True)
            ax[i].set_title(f"{labels_txt[i]}")
            ax[i].set_xlabel("Rank")
            kde = gaussian_kde(ranks[:,i])
            xx = np.linspace(0, num_samples, num_samples)
            ax[i].plot(xx, kde(xx), c='orange')
        if save: fig.savefig(f"{plotname}_rank_stat.png", dpi=400)
        fig.show()
        fig.clf()
        


        # ks_pvals: check how uniform the ranks are (frequentist approach)
        kstest_pvals = torch.tensor(
        [
            kstest(rks, uniform(loc=0, scale=num_samples).cdf)[1]
            for rks in ranks.T
        ],
        dtype=torch.float32,
        )
        print(kstest_pvals)
        
        # c2st, train a small classifier to distinguish between samples from rank and uniform distribution
        # if 0.5, both a equal = classifier is not able to distinguish the two distributions
        
        # compute tarp
        labels = torch.empty((0, lab.shape[1]))
        for  lab, _,_ in Validation_Dataset:
            labels = torch.cat((labels, lab))
        bins = int(np.sqrt(num_samples))
        sorted_labels, idx = torch.sort(labels, dim=0)
        sorted_samples = torch.gather(dap_samples, dim=0, index=idx).numpy()
        dap_samples = dap_samples.numpy()
        fig, ax = plt.subplots(1,lab.shape[1], figsize=(5*lab.shape[1],5), sharey=True)
        h = []
        for i in range(lab.shape[1]):
            h.append(ax[i].hist2d(sorted_labels[:,i], sorted_samples[:,i], 
                             bins=bins, range=[[0,1],[0,1]], density=True)[0])
        hmax = np.max(h, axis=(1,2))
        vmax = np.max(hmax)
        arg_vmax = np.argmax(hmax)
        for i in range(lab.shape[1]):
            h = ax[i].hist2d(sorted_labels[:,i], sorted_samples[:,i], 
                             bins=bins, range=[[0,1],[0,1]], density=True, vmin=0, vmax=vmax)
            ax[i].plot([0,1],[0,1], c='black', linestyle='--', lw=2)
            ax[i].set_title(rf"{labels_txt[i]}")
            ax[i].set_aspect('equal', 'box')
            ax[i].set_xlabel("Truth")
            ax[i].set_ylabel("Predicted")
        fig.tight_layout()
        fig.subplots_adjust(right=0.96)
        cbar_ax = fig.add_axes([0.966, 0.15, 0.01, 0.7])
        fig.colorbar(h[3], cax=cbar_ax, label="Count")
        if save: fig.savefig(f"{plotname}_tarp.png", dpi=400)
        fig.show()
        fig.clf()
        