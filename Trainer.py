import torch
import torch.nn as nn
from typing import Callable
import matplotlib.pyplot as plt
import numpy as np
from alive_progress import alive_bar

# DEPRICATED, WILL BE REMOVED IN THE FUTURE
'''class DensnetHandler():
    def __init__(self, Model: object, Training_data: object = None, Test_data: object = None, device = "cuda"):
        self.TrainingD = Training_data
        self.TestD = Test_data
        self.device = device
        self.xshape = Training_data.dataset[0][1].shape
        self.yshape = Training_data.dataset[0][0].shape
        self.in_dim = self.yshape[-1]
        self.out_dim = self.yshape[-1]
        self.Model = Model(in_dim = self.in_dim, n_blocks=6, n_nodes=60, cond_dims=self.in_dim).to(device)
        
    def train(self, epochs: int,
                optimizer: object, lossf: Callable = nn.MSELoss(), plot: bool = True):
        self.lossf = lossf
        self.optimizer = optimizer

        losstrain = []
        losstest = []
        with alive_bar(epochs, force_tty=True, refresh_secs=5) as bbar:
            for epoch in range(epochs):
                self.Model.train()
                for lab, img, _ in self.TrainingD:

                    img, lab = img.to(self.device), lab.to(self.device)


                    loss = self.Model.loss(x=lab, cond=img)
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    losstrain.append(loss.item())
                losstest.append(self.test_self())
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
                
    def sample(self, num_samples: int, c: torch.FloatTensor, z: torch.FloatTensor = None) -> torch.FloatTensor:
        if z is None:
            z = torch.randn(num_samples, self.out_dim).to(self.device)
        return self.Model(z, c = [c.repeat((num_samples,1)).to(self.device)], rev=True)
    
    def save(self, path: str = "de_model.pt"):
        torch.save(self.Model.state_dict(), path)
    
    def load(self, path: str = "de_model.pt"):
        self.Model.load_state_dict(torch.load(path, map_location=torch.device(self.device)))
        self.Model.eval()'''

class SumnetHandler():
    def __init__(self, Model: object, device = "cuda",
                 model_init_kwargs: dict = {}):
        self.Model = Model(**model_init_kwargs).to(device)
        self.device = device
    
    def train(self, epochs: int,  training_data: object, test_data: object,
              optimizer: object, optimizer_kwargs: dict = {},
            lossf: Callable = nn.MSELoss(), plot: bool = True):
        self.lossf = lossf
        self.optimizer = optimizer

        losstrain = []
        losstest = []
        with alive_bar(epochs, force_tty=True, refresh_secs=5) as bbar:
            for epoch in range(epochs):
                self.Model.train()
                for lab, img, rnge in training_data:
                    img, lab, rnge = img.to(self.device), lab.to(self.device), rnge.to(self.device)
                    x = self.Model(img,rnge)
                    loss = self.lossf(lab, x)
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

    def test_self(self, TestD):
        return SumnetHandler.test(TestD, self.Model, self.lossf, plot = False, device=self.device)

    @staticmethod
    def test(Validation_data: object, Model: object, lossf: Callable, plot: bool = True, device = 'cuda',
            denormalize: Callable = (lambda x: x)):
        Model.eval()
        test_loss = []
        with torch.no_grad():
            for lab, img, rnge  in Validation_data:
                img, lab, rnge = img.to(device), lab.to(device), rnge.to(device)
                pred = Model(img, rnge)
                test_loss.append(lossf(denormalize(pred), denormalize(lab)).to('cpu').item())
        if plot:
            if len(test_loss) > 1:
                plt.hist(test_loss)
                plt.title(f'test loss: {np.mean(test_loss)}')
                plt.xlabel("loss")
                plt.ylabel("count")
                plt.savefig("./eval.png", dpi=400)
                plt.show()
        else: 
            return np.mean(test_loss)
        
    
    @staticmethod
    def test_specific(Validation_data: object, Model: object, lossf: Callable, num_samples: int, device = 'cuda',
                    denormalize: Callable = (lambda x: x)):
        Model.to(device)
        Model.eval()
        test_idx = np.random.randint(0, len(Validation_data), num_samples)
        test_loss = []
        with torch.no_grad():
            for lab, img, rnge in Validation_data:
                if num_samples > 0:
                    img, lab, rnge = img.to(device), lab.to('cpu'), rnge.to('cpu')
                    pred = Model(img, rnge).to('cpu')
                    print(f'loss: ' + str(lossf(denormalize(pred), denormalize(lab)).item()) + '\n',
                        f'pred: ' + str(denormalize(pred.to('cpu'))) + '\n',
                        f'truth: ' + str(denormalize(lab.to('cpu'))))
                    num_samples -= 1
                else:
                    break

    def fast_forward(self, data: torch.FloatTensor, data_dim: float = 5) -> torch.FloatTensor:
        '''data: The input which should be passed to the neural network
        data_dim: The expected dimension of the neural network (useful to unsqueeze first dimension [batch dim])'''
        while len(data.shape) < data_dim:
            data = data.unsqueeze(0)
        self.Model.eval()
        with torch.no_grad():
            res = self.Model(data)
        return res
    
    def full_inference(self, dataloader: object) -> torch.FloatTensor:
        with alive_bar(len(dataloader), force_tty=True) as bar:
            with torch.no_grad():
                for i, (lab, img, _) in enumerate(dataloader):
                    img, lab = img.to(self.device), lab.to(self.device)
                    
                    if not i:
                        summary_vec = torch.empty(0,lab.shape[1])
                        labels = torch.empty(0,lab.shape[1])

                    pred = self.Model(img)
                    summary_vec = torch.cat((summary_vec, pred), dim=0)
                    labels = torch.cat((labels, lab), dim=0)
                    bar()
        return (summary_vec, labels)
            

    def save(self, name: str = "model.pt"):
        torch.save(self.Model.state_dict(), name)

    def load(self, name: str = "model.pt"):
        self.Model.load_state_dict(torch.load(name, map_location=torch.device(self.device), weights_only=True))
        self.Model.eval()
        
        
        
class RecNetHandler(SumnetHandler):
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
