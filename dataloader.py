import torch
from matplotlib import pyplot as plt
import os, fnmatch, sys
import numpy as np
from torch.utils.data import DataLoader
from typing import Callable
print("CUDA is available: ", torch.cuda.is_available())
from matplotlib.pyplot import imshow
from alive_progress import alive_bar
from utility import *
from torchvision.transforms import v2


# write additional class for model itself


'''
class SBIHandler(ModelHandler):
    def __init__(self, )
'''



class ModelHandler():
    def __init__(self, Model: object, Training_data: object = None, Test_data: object = None, device = "cuda"):
        self.Model = Model().to(device)
        self.TrainingD = Training_data
        self.TestD = Test_data
        self.device = device
    
    def train(self, epochs: int, 
              lossf: Callable, optimizer: object):
        self.lossf = lossf
        self.optimizer = optimizer

        losstrain = []
        losstest = []
        with alive_bar(epochs, force_tty=True, refresh_secs=5) as bbar:
            for epoch in range(epochs):
                self.Model.train()
                for lab, img, _ in self.TrainingD:

                    img, lab = img.to(self.device), lab.to(self.device)


                    pred = self.Model(img)
                    loss = self.lossf(pred, lab)
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    losstrain.append(loss.item())
                losstest.append(self.test_self())
                bbar()
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

    def test_self(self):
        return ModelHandler.test(self.TestD, self.Model, self.lossf, plot = False, device=self.device)

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
            

    def save_model(self, name: str = "model.pt"):
        torch.save(self.Model.state_dict(), name)

    def load_model(self, name: str = "model.pt"):
        self.Model.load_state_dict(torch.load(name, map_location=torch.device(self.device)))
        self.Model.eval()




        


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
        data = torch.load(self.path + file)
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
        """
        Add noise to the simulation.
        
        Args:
            brightness_temp (np.ndarray): The brightness temperature data.
            parameters (np.array): The simulation parameters.
        
        Returns:
            np.ndarray: The brightness temperature data with added noise.
        """
        self.debug('Create mock')
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
        k140 = np.fft.fftfreq(140, d=cell_size / 2. / np.pi)
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
        
        