
from logging import info, warning, error
from matplotlib import pyplot as plt
import os, fnmatch, sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
from typing import Callable
for i in range(torch.cuda.device_count()):
   print(torch.cuda.get_device_properties(i).name)

from matplotlib.pyplot import imshow
from alive_progress import alive_bar
from utility import *
from torchvision.transforms import v2
from plot import pairplot
from scipy.stats import kstest, uniform, gaussian_kde
from py21cmfast_tools import calculate_ps
from powerbox.tools import ignore_zero_absk
from py21cmfast_tools import calculate_ps


# write additional class for model itself
        
                
class DensnetHandler():
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
                
    def sample(self, num_samples: int, c: torch.FloatTensor, z: torch.FloatTensor = None) -> torch.FloatTensor:
        if z is None:
            z = torch.randn(num_samples, self.out_dim).to(self.device)
        return self.Model(z, c = [c.repeat((num_samples,1)).to(self.device)], rev=True)
    
    def save(self, path: str = "de_model.pt"):
        torch.save(self.Model.state_dict(), path)
    
    def load(self, path: str = "de_model.pt"):
        self.Model.load_state_dict(torch.load(path, map_location=torch.device(self.device)))
        self.Model.eval()

class SumnetHandler():
    def __init__(self, Model: object, device = "cuda"):
        self.Model = Model().to(device)
        self.device = device
    
    def train(self, epochs: int,  training_data: object, test_data: object,
              optimizer: object, optimizer_kwargs: dict = {},
            lossf: Callable = nn.MSELoss(), plot: bool = True):
        self.lossf = lossf
        self.optimizer = optimizer(self.Model.parameters(), **optimizer_kwargs)

        losstrain = []
        losstest = []
        with alive_bar(epochs, force_tty=True, refresh_secs=5) as bbar:
            for epoch in range(epochs):
                self.Model.train()
                for lab, img, _ in training_data:
                    img, lab = img.to(self.device), lab.to(self.device)

                    x = self.Model(img)
                    loss = self.lossf(lab, x)
                    self.lossf.backward()
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

    def test_self(self, TestD):
        return SumnetHandler.test(TestD, self.Model, self.lossf, plot = False, device=self.device)

    @staticmethod
    def test(Validation_data: object, Model: object, lossf: Callable, plot: bool = True, device = 'cuda',
            denormalize: Callable = (lambda x: x)):
        Model.eval()
        test_loss = []
        with torch.no_grad():
            for lab, img, _  in Validation_data:
                img, lab,  = img.to(device), lab.to(device)
                pred = Model(img)
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
            for lab, img, _ in Validation_data:
                if num_samples > 0:
                    img, lab = img.to(device), lab.to('cpu')
                    pred = Model(img).to('cpu')
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

        
        

class DataHandler():
    def __init__(self, path: str = "./", prefix: str = "batch_",
                 split: float = 1, training_data: bool = True, noise_model: object = None,
                 norm_range: torch.FloatTensor = None, apply_norm: bool = False, 
                 augmentation_probability: float = 0.5) -> None:
        #super().__init__()
        self.path = path
        self.prefix = prefix
        self.files = fnmatch.filter(os.listdir(path), prefix + "*" + ".pt")
        if training_data: self.files = self.files[:int(len(self.files)*split)]
        else: self.files = self.files[int(len(self.files)*split):]
        self.norm_range = norm_range
        self.apply_norm = apply_norm

        if 1 > augmentation_probability > 0:
            # augmentation probability
            self.transforms = v2.Compose([
            v2.RandomHorizontalFlip(p=augmentation_probability),
            v2.RandomVerticalFlip(p=augmentation_probability),
            Transpose(p=augmentation_probability)])
            self.augment_data = True
        else:
            self.augment_data = False

        if noise_model is not None:
            self.noise_model = noise_model
            self.noise = True
        else:
            self.noise = False
                
    def __call__(self) -> tuple:
        return self.load_file(0)

    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, idx) -> tuple:
        return self.load_file(idx)
    
    def give_all(self):
        for i in range(len(self.files)):
            img[i], lab[i] = self.__getitem__(i)
            if i == 0:
                img, lab = torch.empty((len(self), *img.shape)),  torch.empty((len(self), 6))
        return lab, img
        
    def load_file(self, idx: int) -> tuple:
        '''files should contain [batch_size, ...]'''
        file = self.files[idx]
        data = torch.load(self.path + file, weights_only=False)
        images, labels = data['images'], data['labels']
        # TO-DO: Find better way to handle batch loading, right now only bathch loading with size of
        # 1 is possible and dimension is just thrown away
        # Idea: Load sub-batch of data for better io-performance
        # keep in mind the order! Here, noise will also be augmented
        if self.noise: images = self.noise_model(images)
        if self.apply_norm: images, labels = self.normalize(images=images, labels=labels)
        if self.augment_data: images = self.transforms(images)
        images = images.unsqueeze(0)
        return (labels, images, idx)
    
    def save_file(self, file: str, data: dict) -> None:
            np.savez(file, **data)
    
    def plot_data(self, idx: int = -1):
        if idx == -1:
            idx = np.random.randint(self.__len__())
        imshow(self.images[idx])
        plt.title = str(self.labels[idx])
        plt.show()

    def normalize(self, labels: torch.FloatTensor, images: torch.FloatTensor = None, 
                  epsilon: float = 1e-2) -> tuple[torch.FloatTensor, ...]:
        if images is not None:
            #print(f'{images.shape}')
            diff = images.max() - images.min()
            # normalize to [0,1]
            if diff != 0: images = (images - images.min()) / diff
            # normalize to [0 + epsilon, 1 - epsilon]
        labels = (labels - self.norm_range[:,0] + epsilon) / (self.norm_range[:,1] - self.norm_range[:,0] + 2*epsilon)
        return (images, labels)
    
    def denormalize(self, labels: torch.FloatTensor,
                  epsilon: float = 1e-2) -> torch.FloatTensor:
        labels = labels * (self.norm_range[:,1] - self.norm_range[:,0] + 2*epsilon) + self.norm_range[:,0] - epsilon
        return labels

class mock_noise:
    def __init__(self, path: str = "./"):
        print()

    def noise_mock(self, brightness_temp: np.ndarray, parameters: np.array) -> np.ndarray:
        with open("21cm_pie/generate_data/redshifts5.npy", "rb") as data:
            box_redshifts = list(np.load(data, allow_pickle=True))
            box_redshifts.sort()
        cosmo_params = p21c.CosmoParams(OMm=parameters[1])
        astro_params = p21c.AstroParams(INHOMO_RECO=True)
        user_params = p21c.UserParams(HII_DIM=140, BOX_LEN=200)
        flag_options = p21c.FlagOptions()
        sim_lightcone = p21c.LightCone(5., user_params, cosmo_params, astro_params, flag_options, 0,
                                       {"brightness_temp": brightness_temp}, 35.05)
        redshifts = sim_lightcone.lightcone_redshifts
        box_len = np.array([])
        y = 0
        z = 0
        for x in range(len(brightness_temp[0][0])):
            if redshifts[x] > (box_redshifts[y + 1] + box_redshifts[y]) / 2:
                box_len = np.append(box_len, x - z)
                y += 1
                z = x
        box_len = np.append(box_len, x - z + 1)
        y = 0
        delta_T_split = []
        for x in box_len:
            delta_T_split.append(brightness_temp[:,:,int(y):int(x+y)])
            y+=x
            
        mock_lc = np.zeros(brightness_temp.shape)
        cell_size = 200 / 140
        hii_dim = 140
        k140 = np.fft.fftfreq( 140, d=cell_size / 2. / np.pi)
        index1 = 0
        index2 = 0
        files = self.read_noise_files()
        for x in range(len(box_len)):
            with np.load(files[x]) as data:
                ks = data["ks"]
                T_errs = data["T_errs"]
            kbox = np.fft.rfftfreq(int(box_len[x]), d=cell_size / 2. / np.pi)
            volume = hii_dim * hii_dim * box_len[x] * cell_size ** 3
            err21a = np.random.normal(loc=0.0, scale=1.0, size=(hii_dim, hii_dim, int(box_len[x])))
            err21b = np.random.normal(loc=0.0, scale=1.0, size=(hii_dim, hii_dim, int(box_len[x])))
            deldel_T = np.fft.rfftn(delta_T_split[x], s=(hii_dim, hii_dim, int(box_len[x])))
            deldel_T_noise = np.zeros((hii_dim, hii_dim, int(box_len[x])), dtype=np.complex_)
            deldel_T_mock = np.zeros((hii_dim, hii_dim, int(box_len[x])), dtype=np.complex_)
            
            for n_x in range(hii_dim):
                for n_y in range(hii_dim):
                    for n_z in range(int(box_len[x] / 2 + 1)):
                        k_mag = np.sqrt(k140[n_x] ** 2 + k140[n_y] ** 2 + kbox[n_z] ** 2)
                        err21 = np.interp(k_mag, ks, T_errs)
                        
                        if k_mag:
                            deldel_T_noise[n_x, n_y, n_z] = np.sqrt(np.pi * np.pi * volume / k_mag ** 3 * err21) * (err21a[n_x, n_y, n_z] + err21b[n_x, n_y, n_z] * 1j)
                        else:
                            deldel_T_noise[n_x, n_y, n_z] = 0
                        
                        if err21 >= 1000:
                            deldel_T_mock[n_x, n_y, n_z] = 0
                        else:
                            deldel_T_mock[n_x, n_y, n_z] = deldel_T[n_x, n_y, n_z] + deldel_T_noise[n_x, n_y, n_z] / cell_size ** 3
            
            delta_T_mock = np.fft.irfftn(deldel_T_mock, s=(hii_dim, hii_dim, box_len[x]))
            index1 = index2
            index2 += delta_T_mock.shape[2]
            mock_lc[:, :, index1:index2] = delta_T_mock
            if x % 5 == 0:
                self.debug(f'mock created to {int(100 * index2 / 2350)}%')
        return mock_lc
    


class gaussian_noise:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
    def __call__(self, data: torch.FloatTensor) -> torch.FloatTensor:
        return data + torch.normal(torch.full(data.shape,self.mu, dtype=torch.float32),self.sigma)


class Transpose(torch.nn.Module):
    def __init__(self,p: float):
        super().__init__()
        self.p = p
    def forward(self, img):
        if torch.rand(1).item() < self.p:
            return torch.transpose(img, -3,-2)
        else:
            return img
        
        
        
        
class SBIHandler():
    def __init__(self, density_estimator: DensnetHandler, summary_net: SumnetHandler = None,
                 summary_statistics = 'none', 
                 summary_statistics_parameters = {
                    "BOX_LEN": 200,
                    "HII_DIM": 40,
                    "z-eval": np.linspace(7, 24, 10),
                    "bins": 8,
                    },
                 device = 'cuda'):
        self.density_estimator = density_estimator
        if summary_net is None:
            self.sum_net = False
        else:
            self.sum_net = True
            self.summary_net = summary_net
        self.summary_net = summary_net
        self.device = device
        if summary_statistics != 'none':
            if summary_statistics == '1dps':
                def sum_stat(self, brightness_temp, labels):
                    WDM,OMm,LX,E0,Tvir,Zeta = labels
                    global_params = {"M_WDM": WDM}
                    cosmo_params = {"OMm": OMm}
                    astro_params = {
                        "L_X": LX,
                        "NU_X_THRESH": E0,
                        "ION_Tvir_MIN": Tvir,
                        "HII_EFF_FACTOR": Zeta,
                    }
                    
                    lc = p21c.LightCone(redshift=5.5, 
                    cosmo_params=cosmo_params,
                    astro_params=astro_params,
                    _globals=global_params,
                    lightcones={"brightness_temp":brightness_temp},
                    current_redshift=35.05)


                    res = calculate_ps(lc=lc.lightcones['brightness_temp'], lc_redshifts=lc.lightcone_redshifts, 
                       box_length=summary_statistics_parameters['BOX_LEN'], box_side_shape=summary_statistics_parameters["HII_DIM"],
                       log_bins=False, zs=summary_statistics_parameters["z-eval"], calc_1d=True, calc_2d=False, 
                       nbins_1d=summary_statistics_parameters["bins"], bin_ave=True, 
                       k_weights=ignore_zero_absk, postprocess=True)
                    return res['ps_1D']
            elif summary_statistics == '2dps':
                def sum_stat(self, brightness_temp, labels):
                    WDM,OMm,LX,E0,Tvir,Zeta = labels
                    global_params = {"M_WDM": WDM}
                    cosmo_params = {"OMm": OMm}
                    astro_params = {
                        "L_X": LX,
                        "NU_X_THRESH": E0,
                        "ION_Tvir_MIN": Tvir,
                        "HII_EFF_FACTOR": Zeta,
                    }
                    
                    lc = p21c.LightCone(redshift=5.5, 
                    cosmo_params=cosmo_params,
                    astro_params=astro_params,
                    _globals=global_params,
                    lightcones={"brightness_temp":brightness_temp},
                    current_redshift=35.05)


                    res = calculate_ps(lc=lc.lightcones['brightness_temp'], lc_redshifts=lc.lightcone_redshifts, 
                       box_length=summary_statistics_parameters['BOX_LEN'], box_side_shape=summary_statistics_parameters["HII_DIM"],
                       log_bins=False, zs=summary_statistics_parameters["z-eval"], calc_1d=False, calc_2d=True, 
                       kpar_bins=summary_statistics_parameters["bins"], nbins=summary_statistics_parameters["bins"], bin_ave=True, 
                       k_weights=ignore_zero_absk, postprocess=True)
                    return res['final_ps_2D']
            else: 
                error("Summary statistics ", summary_statistics, " not defined.")
        else:
            self.sum_stat = (lambda x,y: x)



            
        
        info("Succesfully initialized SBIHandler")
        
    def train(self, training_data: object, test_data: object, epochs: int = 20, freezed_epochs: int = 0, pretrain_epochs: int = 0, optimizer = torch.optim.Adam,
              optimizer_kwargs: dict = {"lr": 1e-4}, loss_function: Callable = torch.nn.MSELoss, loss_params: dict = {}, device: str = None, plot: bool = True,
              grad_clip: float = 0):
        
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
            summary_net = SumnetHandler(self.summary_net, training_data, test_data, device)
            summary_net.train(pretrain_epochs, loss_function, optimizer(self.summary_net.parameters(), **optimizer_kwargs))
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
                    
                    
                for lab, img, _ in training_data:
                    
                    img, lab = img.to(device), lab.to(device)

                    img = self.sum_stat(img, lab)

                    loss, _train_loss_sn = _loss(img, lab, epoch, freezed_epochs)

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
            plt.savefig("./training_loss.png", dpi=400)
            plt.show()
            plt.clf()
    
    @torch.no_grad()   
    def test_self(self, validation_data: object, loss_function: Callable = torch.nn.MSELoss()):
        if self.sum_net:
            self.summary_net.eval()
        self.density_estimator.eval()
        test_loss_de_tmp = 0
        if self.sum_net:
            test_loss_sn_tmp = 0
        for lab, img, _  in validation_data:
            img = self.sum_stat(img, lab)
            loss, _test_loss_sn = _loss(img, lab, 0, 0 if self.sum_net else 1)
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

    def _loss(self, img, lab, epoch, freezed_epochs):
        if self.sum_net:
            if epoch < freezed_epochs:
                summary = self.summary_net(img).detach()
                train_loss_sn_tmp = loss_function(summary, lab).mean().item()
            else:
                summary = self.summary_net(img)
                train_loss_sn_tmp = 0
            
            
        else:
            summary = img

        if summary.shape != lab.shape:
            raise error(f"Summary {summary.shape} and label {lab.shape} shape do not match")                    
        # computing loss
        loss = self.density_estimator.loss(lab, cond=summary).mean(0)

        return loss, train_loss_sn_tmp
    
    @torch.no_grad()
    def run_sbc(self, Validation_Dataset = None, num_samples: int = 1000, num_workers: int = 1,
                plotname: str = ""):
        
        
        save = False if plotname == "" else True 
        
        self.density_estimator.eval()
        if self.sum_net:
            self.summary_net.eval()
        #mp = True if num_workers > 1 else False
        # run sbc on full Validation Dataset
        info("Run SBC...")
        for i, (lab, img,_) in enumerate(Validation_Dataset):
            img, lab = img.to(self.device), lab.to(self.device)
            
            if not i:
                summary_vec = torch.empty(0,lab.shape[1], device=self.device)
                labels = torch.empty(0,lab.shape[1], device=self.device)

            pred = self.summary_net(img)
            summary_vec = torch.cat((summary_vec, pred), dim=0)
            labels = torch.cat((labels, lab), dim=0)

        ranks = torch.empty(summary_vec.shape)
        dap_samples = torch.empty((summary_vec.shape[0], summary_vec.shape[1]))
        # sbc rank stat
        with alive_bar(summary_vec.shape[0], force_tty=True, refresh_secs=1) as bar:
            for i in range(summary_vec.shape[0]):
                samples, _ = self.density_estimator.sample(x = summary_vec[i].unsqueeze(0), num_samples=num_samples)
                dap_samples[i] = samples[0]
                for j in range(summary_vec.shape[1]):
                    ranks[i,j] = (samples[:,j]<labels[i,j]).sum().item()
                bar()
                
        # plot rank statistics
        
        labels_txt = [r"$M_\text{WDM}$", r"$\Omega_m$", r"$L_X$", r"$E_0$", r"$T_\text{vir, ion}$", r"$\zeta$"]
        fig, ax = plt.subplots(1,summary_vec.shape[1], figsize=(5*summary_vec.shape[1],5))
        for i in range(summary_vec.shape[1]):
            ax[i].hist(ranks[:,i].cpu().numpy(), bins=50, range=(0, num_samples), density=True)
            ax[i].set_title(f"{labels_txt[i]}")
            ax[i].set_xlabel("Rank")
            kde = gaussian_kde(ranks[:,i].cpu().numpy())
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
        bins = int(np.sqrt(num_samples))
        sorted_labels, idx = torch.sort(labels, dim=0)
        sorted_samples = torch.gather(dap_samples.cpu(), dim=0, index=idx.cpu())
        fig, ax = plt.subplots(1,summary_vec.shape[1], figsize=(5*summary_vec.shape[1],5), sharey=True)
        h = []
        for i in range(summary_vec.shape[1]):
            h.append(ax[i].hist2d(sorted_labels[:,i].cpu().numpy(), sorted_samples[:,i].cpu().numpy(), 
                             bins=bins, range=[[0,1],[0,1]], density=True)[0].max())
        vmax = np.max(h)
        for i in range(summary_vec.shape[1]):
            h = ax[i].hist2d(sorted_labels[:,i].cpu().numpy(), sorted_samples[:,i].cpu().numpy(), 
                             bins=bins, range=[[0,1],[0,1]], density=True, vmin=0, vmax=vmax)
            ax[i].plot([0,1],[0,1], c='black', linestyle='--', lw=2)
            ax[i].set_title(rf"{labels_txt[i]}")
            ax[i].set_aspect('equal', 'box')
            ax[i].set_xlabel("Truth")
            ax[i].set_ylabel("Predicted")
        #fig.subplots_adjust(right=0.8)
        #cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(h[3], cax=ax[-1])
        if save: fig.savefig(f"{plotname}_tarp.png", dpi=400)
        fig.show()
        fig.clf()
        
        # Local Classifier Two-Sample Tests
        
        # sensitivity analysis
        
        
        
        
class NLEHandler(SBIHandler):
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
              grad_clip: float = 0):
        
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
        kernel_size = loss_params.pop('kernel_size')
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

        if self.sum_net: self.summary_net.eval()
        self.density_estimator.eval()
        if self.rec_net: self.reconstruction_net.eval()
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
            plt.savefig("./training_loss.png", dpi=400)
            plt.show()
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
            print("freezed epoch")
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
            print("rec:",_rec_loss.item())
        return loss, train_loss_de_tmp, train_loss_sn_tmp, train_loss_rec_tmp