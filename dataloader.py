import torch
from torchvision.transforms import v2  
import os
import fnmatch
import numpy as np
import matplotlib.pyplot as plt
import py21cmfast as p21c 
from torchrl.modules import TruncatedNormal   
import sys, glob, logging


class DataHandler():
    def __init__(self, path: str = "./", prefix: str = "batch_",
                 split: float = 1, training_data: bool = True, noise_model: object = None,
                 norm_range: torch.FloatTensor = None, apply_norm: bool = False, 
                 augmentation_probability: float = 0.5, expand_dim: bool = True,
                 psvar: bool = False) -> None:
        #super().__init__()
        self.path = path
        self.prefix = prefix
        self.files = fnmatch.filter(os.listdir(path), prefix + "*" + ".pt")
        if training_data: self.files = self.files[:int(len(self.files)*split)]
        else: self.files = self.files[int(len(self.files)*split):]
        self.norm_range = norm_range
        self.apply_norm = apply_norm
        self.psvar = psvar
        if 1 > augmentation_probability > 0 and not psvar:
            # augmentation probability
            self.transforms = v2.Compose([
            v2.RandomHorizontalFlip(p=augmentation_probability),
            v2.RandomVerticalFlip(p=augmentation_probability),
            Transpose(p=augmentation_probability)])
            self.augment_data = True
        elif 1 > augmentation_probability > 0 and psvar:
            self.transforms = PSCosmicVariance(p=augmentation_probability)
            self.augment_data = True
        else:
            self.augment_data = False

        if noise_model is not None:
            self.noise_model = noise_model
            self.noise = True
        else:
            self.noise = False
        self.expand_dim = expand_dim
                
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
        # Changed ordering from range_calc -> noise -> norm -> augment
        # to: noise -> augment -> range_calc -> norm
        if self.noise: images = self.noise_model(images, labels)
        images = torch.tensor(images)
        if self.augment_data: images = self.transforms(images) if not self.psvar else self.transforms(images, data['std'])
        ranges = torch.stack((torch.amax(images),
        torch.amin(images)), dim=-1)
        if self.apply_norm:
            images, labels = self.normalize(images=images, labels=labels)
        if self.expand_dim: images = images.unsqueeze(0)
        return (labels, images, ranges)
    
    def save_file(self, file: str, data: dict) -> None:
            np.savez(file, **data)
    
    def plot_data(self, idx: int = -1):
        if idx == -1:
            idx = np.random.randint(self.__len__())
        imshow(self.images[idx])
        plt.title = str(self.labels[idx])
        plt.show()

    def normalize(self, labels: torch.FloatTensor, images: torch.FloatTensor = None, 
                  epsilon: float = 1e-4) -> tuple[torch.FloatTensor, ...]:
        if images is not None:
            #print(f'{images.shape}')
            diff = images.max() - images.min()
            # normalize to [0,1]
            if diff != 0: images = (images - images.min()) / diff
            # normalize to [0 + epsilon, 1 - epsilon]
        labels = (labels - self.norm_range[:,0] + epsilon) / (self.norm_range[:,1] - self.norm_range[:,0] + 2*epsilon)
        return (images, labels)
    
    def denormalize(self, labels: torch.FloatTensor,
                  epsilon: float = 1e-4) -> torch.FloatTensor:
        labels = labels * (self.norm_range[:,1] - self.norm_range[:,0] + 2*epsilon) + self.norm_range[:,0] - epsilon
        return labels

class mock_noise:
    def __init__(self, path: str, noise_level: str = "opt", z_cut: int = 470):
        self.path = path
        self.noise_level = noise_level
        from astropy.cosmology import Planck18
        
    def read_noise_files(self) -> list:
        """
        Read noise files based on the noise level specified in the parameters.
        
        Returns:
            list: List of filenames for the noise data.
        """
        if self.noise_level == "opt":
            files = glob.glob(f"{self.path}/twentyone_cm_pie/generate_data/calcfiles/opt_mocks/SKA1_Lowtrack_6.0hr_opt_0.*_LargeHII_Pk_Ts1_Tb9_nf0.52_v2.npz")
        elif self.noise_level == "mod":
            files = glob.glob(f"{self.path}/twentyone_cm_pie/generate_data/calcfiles/mod_mocks/SKA1_Lowtrack_6.0hr_mod_0.*_LargeHII_Pk_Ts1_Tb9_nf0.52_v2.npz")
        else:
            logging.info("Please choose a valid foreground model")
            sys.exit()
        files.sort(reverse=True)
        return files
            
    def __call__(self, brightness_temp, labels) -> np.ndarray:
        """
        Add noise to the simulation.
        
        Args:
            brightness_temp (np.ndarray): The brightness temperature data.
            parameters (np.array): The simulation parameters.
        
        Returns:
            np.ndarray: The brightness temperature data with added noise.
        """
        if self.path[-1] == "/":
            self.path = self.path[:-1]
            
        
        logging.info('Create mock')
        with open(f"{self.path}/twentyone_cm_pie/generate_data/redshifts5.npy", "rb") as data:
            box_redshifts = list(np.load(data, allow_pickle=True))
            box_redshifts.sort()
        from astropy.cosmology import Planck18
        user_params = p21c.UserParams(BOX_LEN=200, HII_DIM=28)
        with p21c.global_params.use(**{"M_WDM": labels[0]}):
            Planck18 = Planck18.clone(Om0=labels[1])
            lightcone = p21c.RectilinearLightconer.with_equal_cdist_slices(min_redshift=5, max_redshift=35,
                    resolution=user_params.cell_size, cosmo=Planck18)
        
            redshifts = lightcone.lc_redshifts[:470]
        box_len = np.array([])
        y = 0
        z = 0
        for x in range(len(brightness_temp[0][0])):
            if (redshifts[x] > (box_redshifts[y + 1] + box_redshifts[y]) / 2) and x - z > 0:
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
        cell_size = user_params.cell_size.value
        hii_dim = user_params.HII_DIM
        k140 = np.fft.fftfreq(hii_dim, d=cell_size / 2. / np.pi)
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
                logging.info(f'mock created to {int(100 * index2 / 2350)}%')
        print("done")
        return mock_lc.astype(np.float32)


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

class PSCosmicVariance(torch.nn.Module):
    def __init__(self,p: float):
        super().__init__()
        self.p = p
    def forward(self, img, std):
        if torch.rand(1).item() < self.p:
            return img + TruncatedNormal(torch.zeros(std.shape), std, low=-2*std,high=2*std+1e-4).sample((1,)).squeeze(0).to(torch.float32)
        else:
            return img
        
        

        
        
        
