import torch
import torch.nn as nn
from typing import Callable
import matplotlib.pyplot as plt
import numpy as np
from alive_progress import alive_bar
from logging import info, warning, error
from torch.optim import AdamW
from cl_models import RNVP
from classifier import ResNet
import sbi

import os
import tempfile

from ray import train, tune
from ray.tune.schedulers import ASHAScheduler

from ray.train import Checkpoint

from ray.tune.search.hyperopt import HyperOptSearch


class Trainer:
    def __init__(self,
                 summary_network,
                 sn_kwargs,
                 de_kwargs,
                 training_data, 
                 test_data, 
                 path,
                 name,
                 NLE = False,
                 device = 'cuda',):
        if sn_kwargs == None or sn_kwargs == {}:
            self.sn_net = summary_network
        else:
            self.sn_net = summary_network(**sn_kwargs)
        self.training_data = training_data
        self.test_data = test_data
        self.device = device
        self.opti_hype = False
        de_net_type = de_kwargs.pop('net_type')
        self.path = path
        self.name = name
        self.NLE = NLE
        self.de_net_type = de_net_type
        if de_net_type == 'inn':
            self.de_net = RNVP(**de_kwargs)
        elif de_net_type == 'maf':
            self.de_net = sbi.neural_nets.net_builders.build_maf(
                **de_kwargs
            )
        elif de_net_type == 'nsf':
            self.de_net = sbi.neural_nets.net_builders.build_nsf(
                **de_kwargs
            )
        elif de_net_type == 'resnet':
            self.de_net = ResNet(**de_kwargs)
        else:
            raise NameError("Network type not found!")

    
    def train(self, config: dict,
            epochs: int,
            pretrain_epochs: int = 0,
            freezed_epochs: int = 0):
        """Config should contain
            {
                'pretrain_kwargs': {'lr', 'weight_decay'},
                'optimizer_kwargs': {'lr', 'weight_decay'},
            }"""
        # Reinitialize networks if needed
        if self.opti_hype:
            self.sn_net.__init__(**config["summary_network_kwargs"])
            self.de_net.__init__(**config["density_network_kwargs"])

        # Device move
        self.sn_net.to(self.device)
        self.de_net.to(self.device)
        self.freezed_epochs = freezed_epochs

        if self.NLE and (freezed_epochs < epochs):
            raise AssertionError("NLE won't converge! Mode collapse immement.")

        # Pretrain summary net
        if pretrain_epochs > 0:
            info("Pretraining summary net...")
            self.sn_net.train()
            handler = SNHandler(encoder=self.sn_net, device=self.device)
            handler.training(epochs=pretrain_epochs, config={
                "optimizer_kwargs": config['pretrain_kwargs']
            }, training_data=self.training_data, test_data=self.test_data)
            self.sn_pretrain_trainloss = handler.losstrain
            self.sn_pretrain_testloss = handler.losstest
            
        mse = torch.nn.MSELoss(reduction = 'none')
        # Prepare main optimizer for density estimator only
        optimizer = AdamW(list(self.de_net.parameters()), **config["optimizer_kwargs"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1)

        # Training loop
        info("Begin training...")
        with alive_bar(epochs, force_tty=True, refresh_secs=10) as bar:
            history = { 'train_de': [], 'test_de': [], 'train_sn': [], 'test_sn': [] }

            for epoch in range(epochs):
                if freezed_epochs == epoch:
                    optimizer = AdamW(list(self.de_net.parameters()) + list(self.sn_net.parameters()), **config["optimizer_kwargs"])
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1)
                # Switch optimizer when joint training starts
                self.de_net.train()
                fixed_flag = epoch < freezed_epochs
                if fixed_flag:
                    self.sn_net.eval()
                else:
                    self.sn_net.train()

                # Accumulate losses
                train_loss_de = 0.0
                train_loss_sn = 0.0
                # Train
                for lab, img in self.training_data:
                    img = img.to(self.device)
                    lab = lab.to(self.device)

                    # Forward
                    z = self.sn_net(img)
                    if self.NLE:
                        loss = self.de_net.loss(z, condition=lab).mean(0).mean()
                    else:
                        loss = self.de_net.loss(lab, condition=z).mean(0).mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    train_loss_de += loss.item()
                
                # Evaluate
                self.de_net.eval()
                self.sn_net.eval()
                val_loss_de = 0.0
                val_loss_sn = 0.0
                with torch.no_grad():
                    for lab, img in self.test_data:
                        img = img.to(self.device)
                        lab = lab.to(self.device)
                        z = self.sn_net(img)
                        val_loss_sn += torch.log(mse(z,lab).mean(0).mean()).item()
                        if self.NLE:
                            val_loss_de = self.de_net.loss(z, condition=lab).mean(0).mean().item()
                        else:
                            val_loss_de = self.de_net.loss(lab, condition=z).mean(0).mean().item()

                # Logging
                history['train_de'].append(train_loss_de / len(self.training_data))
                history['test_de'].append(val_loss_de / len(self.test_data))
                history['train_sn'].append(train_loss_sn / len(self.training_data))
                history['test_sn'].append(val_loss_sn / len(self.test_data))

                #print("LR loss: ", history['train_de'][-1])
                #print("Eval loss: ", history['test_de'][-1])

                scheduler.step(history['test_de'][-1])
                bar()

                # Optional reporting
                if self.opti_hype and (epoch + 1) % 5 == 0:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        path = os.path.join(tmpdir, "model.pth")
                        torch.save(self.de_net.state_dict(), path)
                        ckpt = Checkpoint.from_directory(tmpdir)
                        train.report({"loss": history['test_de'][-1]}, checkpoint=ckpt)

        # Store history
        self.train_loss_de = history['train_de']
        self.test_loss_de = history['test_de']
        self.train_loss_sn = history['train_sn']
        self.test_loss_sn = history['test_sn']

        np.save(self.path + f'loss_{self.de_net_type}_{freezed_epochs}.npy', np.vstack([self.train_loss_de, self.test_loss_de, self.train_loss_sn, self.test_loss_sn]))


        
        self.sn_net.eval()
        self.de_net.eval()
        
    def save_model(self, path: str = "./"):
        torch.save(self.de_net.state_dict(), self.path + self.name + "_de.pt")
        torch.save(self.sn_net.state_dict(), self.path + self.name + "_sn.pt")

    def load_model(self, path: str = "./"):
        self.de_net.load(path)
        self.de_net.to(self.device)
        self.de_net.eval()
        self.sn_net.load(path)
        self.sn_net.to(self.device)
        self.sn_net.eval()
        
    def plot(self):
        plt.clf()
        try:
            len_pre = len(self.sn_pretrain_trainloss)
            len_main = len(self.test_loss_de)
            pre = True
        except:
            len_main = len(self.test_loss_de)
            pre = False
            len_pre = 0

        if pre:
            epochs = list(range(len_pre+len_main))
            plt.plot(epochs[:len_pre], self.sn_pretrain_trainloss, label='Pre-SN Train', c='limegreen')
            plt.plot(epochs[:len_pre], self.sn_pretrain_testloss, label='Pre-SN Test', c='forestgreen')
            all = self.train_loss_de + self.train_loss_sn + self.test_loss_de + self.test_loss_sn + self.sn_pretrain_trainloss + self.sn_pretrain_testloss
            glob_min, glob_max = np.min(all), np.max(all)
            plt.vlines(len_pre-0.5, glob_min, glob_max, linestyles='dashdot', colors='black', lw=1.5)
        else:
            epochs = list(range(len_main))
        if self.freezed_epochs > 0:
            plt.plot(epochs[len_pre:(len_pre + self.freezed_epochs)], self.train_loss_sn[:self.freezed_epochs], label='Fixed-SN Train', c='gold')
            plt.plot(epochs[len_pre:(len_pre + self.freezed_epochs)], self.test_loss_sn[:self.freezed_epochs], label='Fixed-SN Test', c='darkgoldenrod')
            plt.plot(epochs[len_pre:(len_pre + self.freezed_epochs)], self.train_loss_de[:self.freezed_epochs], label='Fixed-DE Train', c='lightcoral')
            plt.plot(epochs[len_pre:(len_pre + self.freezed_epochs)], self.test_loss_de[:self.freezed_epochs], label='Fixed-DE Test', c='red')
            if pre:
                plt.vlines(len_pre + self.freezed_epochs - 0.5, glob_min, glob_max, linestyles='dashdot', colors='black', lw=1.5)
            else:
                all = self.train_loss_de + self.train_loss_sn + self.test_loss_de + self.test_loss_sn
                glob_min, glob_max = np.min(all), np.max(all)
                plt.vlines(len_pre + self.freezed_epochs - 0.5, glob_min, glob_max, linestyles='dashdot', colors='black', lw=1.5)
        
        plt.plot(epochs[(len_pre + self.freezed_epochs):], self.train_loss_sn[self.freezed_epochs:], label='Joint-SN Train', c='cornflowerblue')
        plt.plot(epochs[(len_pre + self.freezed_epochs):], self.test_loss_sn[self.freezed_epochs:], label='Joint-SN Test', c='mediumvioletred')
        plt.plot(epochs[(len_pre + self.freezed_epochs):], self.train_loss_de[self.freezed_epochs:], label='Joint-DE Train', c='magenta')
        plt.plot(epochs[(len_pre + self.freezed_epochs):], self.test_loss_de[self.freezed_epochs:], label='Joint-DE Test', c='indigo')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Log Loss')
        plt.tight_layout()
        plt.savefig(self.path + self.name + '_loss.pdf', dpi=300)
        plt.show()
        plt.clf()



class SNHandler:
    def __init__(self, encoder,
                 device,
                 path: str = ''):
        self.encoder = encoder.to(device)
        self.device = device
        self.opti_hype = False
        self.path = path
        
        info("Succesfully initialized SNHandler")

    def __call__(self, img, cond=None):
        return self.encoder(img, cond)
    
    def to(self, device):
        self.encoder.to(device)
        if self.use_dec:
            self.decoder.to(device)

    
        
    def training(self,
                 config: dict,
              epochs: int,
              training_data: object,
              test_data: object,
              lossf: Callable = nn.MSELoss(reduction = 'none',)):
        
        if self.opti_hype:
            # quick and dirty way to reinizialize the networks
            self.encoder.__init__(**config["summary_network_kwargs"])
        
        self.encoder.to(self.device)

        self.optimizer = AdamW(self.encoder.parameters(), **config["optimizer_kwargs"])
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=.5)
        losstrain = []
        losstest = []
        with alive_bar(epochs, force_tty=True, refresh_secs=10) as bbar:
            for epoch in range(epochs):
                self.encoder.train()
                losstrain_tmp = 0
                losstest_tmp = 0
                for lab, img in training_data:
                    img = img.to(self.device)
                    lab = lab.to(self.device)
                    loss_sn = self.loss(img, lab)
                    loss_sn = torch.log(loss_sn.mean())
                    loss_sn.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    losstrain_tmp += loss_sn.item()

                losstrain.append(losstrain_tmp / len(training_data))
                
                
                self.encoder.eval()
                for lab, img in test_data:
                    
                    img = img.to(self.device)
                    lab = lab.to(self.device)
                    loss_sn = self.loss(img,lab)
                    loss_sn = loss_sn.mean()
                    losstest_tmp += torch.log(loss_sn).item()
                losstest.append(losstest_tmp / len(test_data))

                lr_scheduler.step(losstrain[-1])

                
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
                    
                    
        self.losstest = losstest
        self.losstrain = losstrain

        np.save(self.path + 'sn_loss.npy', np.vstack([self.losstest,self.losstrain]))


    def loss(self, img, lab):
        z = self.encoder(img)
        loss = nn.MSELoss(reduction='none')(z, lab).mean(0)
        return loss
