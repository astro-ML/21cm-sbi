import torch
from torchvision.transforms import v2  
import os
import fnmatch
import numpy as np
import matplotlib.pyplot as plt
import py21cmfast as p21c 
from torchrl.modules import TruncatedNormal   
import sys, glob, logging
from astropy.cosmology import Planck18


class DataHandler:
    """
    Handles data loading, preprocessing, augmentation, and normalization for machine learning tasks.
    """

    def __init__(self, path="./", prefix="batch_", split=1, training_data=True, noise_model=None,
                 norm_range=None, apply_norm=False, augmentation_probability=0.5, datatype="box_npy"):
        """
        Initializes the data handler with specified parameters.

        Args:
            path (str): Path to the data directory.
            prefix (str): Prefix for the data files.
            split (float): Fraction of the dataset to use for training or validation.
            training_data (bool): Whether the data is for training or validation/testing.
            noise_model (object): Noise model to apply to the data.
            norm_range (torch.FloatTensor): Range for normalization.
            apply_norm (bool): Whether to apply normalization.
            augmentation_probability (float): Probability of applying data augmentation.
            datatype (str): Type of data ("box", "ps1d", or "ps2d").
        """
        self.path = path
        self.prefix = prefix
        self.files = fnmatch.filter(os.listdir(path), f"{prefix}*.pt") #.pt
        self.files = self.files[:int(len(self.files) * split)] if training_data else self.files[int(len(self.files) * split):]
        self.norm_range = norm_range
        self.apply_norm = apply_norm
        self.datatype = {"box": 0, "ps1d": 1, "ps2d": 2, "box_npy": 3}.get(datatype, -1)
        self.train = training_data

        if self.datatype == -1:
            logging.error("Invalid datatype. Choose 'box', 'ps1d', or 'ps2d'.")
            sys.exit()
        if self.datatype == 3:
            self.files = fnmatch.filter(os.listdir(path), f"{prefix}*.npz")
            self.files = self.files[:int(len(self.files) * split)] if training_data else self.files[int(len(self.files) * split):]
        
        self.augment_data = 0 < augmentation_probability < 1
        if self.augment_data:
            if self.datatype == 0 or self.datatype == 3:
                self.transforms = v2.Compose([
                    v2.RandomHorizontalFlip(p=augmentation_probability),
                    v2.RandomVerticalFlip(p=augmentation_probability),
                    Transpose(p=augmentation_probability)
                ])
            else:
                self.transforms = PSCosmicVariance(p=augmentation_probability)

        self.noise_model = noise_model
        self.noise = noise_model is not None
        self.expand_dim = self.datatype == 0 or self.datatype == 3

    def __call__(self):
        return self.load_file(0)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return self.load_file(idx)

    def give_all(self):
        """
        Loads and returns all data and labels from the files.
        """
        img, lab = None, None
        for i in range(len(self.files)):
            temp_img, temp_lab = self.__getitem__(i)
            if i == 0:
                img = torch.empty((len(self), *temp_img.shape))
                lab = torch.empty((len(self), 6))
            img[i], lab[i] = temp_img, temp_lab
        return lab, img

    def load_file(self, idx):
        """
        Loads and processes the data from the file at the specified index.

        Args:
            idx (int): Index of the file to load.

        Returns:
            tuple: Processed labels, images, and range.
        """
        file = self.files[idx]
        #data = torch.load(os.path.join(self.path, file), weights_only=False)
        if self.datatype != 3:
            data = torch.load(os.path.join(self.path, file), weights_only=False)
            image_key = "brightness_temp" if self.datatype == 0 else f"ps{self.datatype}d"
            #image_key = "image"
            image = data[image_key]
            label = data["parameter"]
            if self.noise:
                image = self.noise_model(image, label)
            if self.augment_data:
                if self.datatype == 0:
                    image = self.transforms(image)
                else:
                    variance_key = f"ps{self.datatype}d_var"
                    image = self.transforms(image, data[variance_key])

            #range_ = torch.tensor([torch.amax(image), torch.amin(image)])
            if self.apply_norm:
                image, label = self.normalize(images=image, labels=label)
            if self.train:
                image += torch.normal(mean=0, std=torch.full(image.shape,1e-2)).to(torch.float32)
            if self.expand_dim:
                image = image.unsqueeze(0)
            return label, image#, range_
        elif self.datatype == 3:
            data = np.load(os.path.join(self.path, file))
            image = torch.tensor(data['image']).to(torch.float32)
            label = torch.tensor(data['label']).to(torch.float32)
            if self.augment_data:
                image = self.transforms(image)
            if self.apply_norm:
                image, label = self.normalize(images=image, labels=label)
            if self.train:
                image += torch.normal(mean=0, std=torch.full(image.shape,1e-2)).to(torch.float32)
            if self.expand_dim:
                image = image.unsqueeze(0)
            return label, image#, range_

    def save_file(self, file, data):
        """
        Saves the given data to a file in `.npz` format.

        Args:
            file (str): File path to save the data.
            data (dict): Data to save.
        """
        np.savez(file, **data)

    def plot_data(self, idx=-1):
        """
        Plots the data at the specified index or a random index if none is provided.

        Args:
            idx (int): Index of the data to plot. Defaults to -1 (random index).
        """
        if idx == -1:
            idx = np.random.randint(len(self))
        plt.imshow(self.images[idx])
        plt.title(str(self.labels[idx]))
        plt.show()

    def normalize(self, labels, images=None, epsilon=1e-4):
        """
        Normalizes the labels and optionally the images using Z-score normalization.

        Args:
            labels (torch.FloatTensor): Labels to normalize.
            images (torch.FloatTensor, optional): Images to normalize.
            epsilon (float): Small value to avoid division by zero (used for labels).

        Returns:
            tuple: Normalized images and labels.
        """
        if images is not None:
            imgmin, imgmax = images.min(), images.max()
            if imgmin != imgmax:
                images = (images - imgmin) / (imgmax - imgmin) * 2 - 1

        labels = (labels - self.norm_range[:, 0]) / (self.norm_range[:, 1] - self.norm_range[:, 0])
        labels = (epsilon/2 + (1 - 2 * epsilon/2) * labels)*2 - 1
        return images, labels

    def denormalize(self, norm_labels, epsilon=1e-4):
        """
        Denormalizes the normalized labels and optionally the images.

        Args:
            norm_labels (torch.FloatTensor): Normalized labels to denormalize.
            norm_images (torch.FloatTensor, optional): Normalized images to denormalize.
            epsilon (float): Small value used during normalization.

        Returns:
            tuple: Denormalized images and labels.
        """
        norm_labels = (norm_labels + 1) / 2
        norm_labels = (norm_labels - epsilon / 2) / (1 - epsilon)
        labels = norm_labels * (self.norm_range[:, 1] - self.norm_range[:, 0]) + self.norm_range[:, 0]

        return labels

class MockNoise:
    """
    Class to add mock noise to brightness temperature simulations.
    """

    def __init__(self, path: str, noise_level: str = "opt"):
        self.path = path.rstrip("/") + "/"
        self.noise_level = noise_level

    def read_noise_files(self) -> list:
        """
        Reads noise files based on the specified noise level.

        Returns:
            list: Sorted list of filenames for the noise data.
        """
        if self.noise_level == "opt":
            files = glob.glob(f"{self.path}/twentyone_cm_pie/generate_data/calcfiles/opt_mocks/SKA1_Lowtrack_6.0hr_opt_0.*_LargeHII_Pk_Ts1_Tb9_nf0.52_v2.npz")
        elif self.noise_level == "mod":
            files = glob.glob(f"{self.path}/twentyone_cm_pie/generate_data/calcfiles/mod_mocks/SKA1_Lowtrack_6.0hr_mod_0.*_LargeHII_Pk_Ts1_Tb9_nf0.52_v2.npz")
        else:
            logging.error("Invalid foreground model. Choose 'opt' or 'mod'.")
            sys.exit()
        return sorted(files, reverse=True)

    def __call__(self, brightness_temp, labels) -> np.ndarray:
        """
        Adds noise to the brightness temperature simulation.

        Args:
            brightness_temp (np.ndarray): Brightness temperature data.
            labels (np.ndarray): Simulation parameters.

        Returns:
            np.ndarray: Brightness temperature data with added noise.
        """
        logging.info("Creating mock noise...")
        with open(f"{self.path}/twentyone_cm_pie/generate_data/redshifts5.npy", "rb") as data:
            box_redshifts = sorted(np.load(data, allow_pickle=True))

        user_params = p21c.UserParams(BOX_LEN=200, HII_DIM=28)
        lightcone = p21c.RectilinearLightconer.with_equal_cdist_slices(
            min_redshift=6.2, max_redshift=8.5, resolution=user_params.cell_size, cosmo=Planck18
        )
        redshifts = lightcone.lc_redshifts

        box_lengths = self._calculate_box_lengths(redshifts, box_redshifts)
        delta_T_split = self._split_brightness_temp(brightness_temp, box_lengths)

        mock_lc = np.zeros_like(brightness_temp)
        files = self.read_noise_files()

        for idx, box_len in enumerate(box_lengths):
            mock_lc = self._add_noise_to_box(mock_lc, delta_T_split[idx], files[idx], idx, box_len, user_params)

        logging.info("Mock noise creation complete.")
        return mock_lc.astype(np.float32)

    def _calculate_box_lengths(self, redshifts, box_redshifts):
        box_lengths = []
        y, z = 0, 0
        for x in range(len(redshifts)):
            if redshifts[x] > (box_redshifts[y + 1] + box_redshifts[y]) / 2 and x - z > 0:
                box_lengths.append(x - z)
                y += 1
                z = x
        box_lengths.append(len(redshifts) - z)
        return box_lengths

    def _split_brightness_temp(self, brightness_temp, box_lengths):
        delta_T_split = []
        start = 0
        for length in box_lengths:
            delta_T_split.append(brightness_temp[:, :, start:start + length])
            start += length
        return delta_T_split

    def _add_noise_to_box(self, mock_lc, delta_T_box, noise_file, idx, box_len, user_params):
        with np.load(noise_file) as data:
            ks, T_errs = data["ks"], data["T_errs"]

        cell_size = user_params.cell_size.value
        hii_dim = user_params.HII_DIM
        k140 = np.fft.fftfreq(hii_dim, d=cell_size / (2 * np.pi))
        kbox = np.fft.rfftfreq(box_len, d=cell_size / (2 * np.pi))
        volume = hii_dim ** 2 * box_len * cell_size ** 3

        err21a = np.random.normal(size=(hii_dim, hii_dim, box_len))
        err21b = np.random.normal(size=(hii_dim, hii_dim, box_len))
        deldel_T = np.fft.rfftn(delta_T_box, s=(hii_dim, hii_dim, box_len))

        deldel_T_mock = np.zeros_like(deldel_T, dtype=np.complex_)
        for n_x in range(hii_dim):
            for n_y in range(hii_dim):
                for n_z in range(kbox.size):
                    k_mag = np.sqrt(k140[n_x] ** 2 + k140[n_y] ** 2 + kbox[n_z] ** 2)
                    err21 = np.interp(k_mag, ks, T_errs)

                    if k_mag:
                        noise = np.sqrt(np.pi ** 2 * volume / k_mag ** 3 * err21)
                        deldel_T_mock[n_x, n_y, n_z] = deldel_T[n_x, n_y, n_z] + noise * (err21a[n_x, n_y, n_z] + 1j * err21b[n_x, n_y, n_z])
                    else:
                        deldel_T_mock[n_x, n_y, n_z] = deldel_T[n_x, n_y, n_z]

        delta_T_mock = np.fft.irfftn(deldel_T_mock, s=(hii_dim, hii_dim, box_len))
        mock_lc[:, :, idx:idx + box_len] = delta_T_mock
        return mock_lc


class GaussianNoise:
    """
    Class to add Gaussian noise to data.
    """

    def __init__(self, mu: float, sigma: float):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, data: torch.FloatTensor) -> torch.FloatTensor:
        noise = torch.normal(mean=self.mu, std=self.sigma, size=data.shape, dtype=torch.float32)
        return data + noise


class Transpose(torch.nn.Module):
    """
    Randomly transposes the last two dimensions of an image with a given probability.
    """

    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def forward(self, img: torch.FloatTensor) -> torch.FloatTensor:
        if torch.rand(1).item() < self.p:
            return img.transpose(-3, -2)
        return img


class PSCosmicVariance(torch.nn.Module):
    """
    Adds cosmic variance to power spectrum data with a given probability.
    """

    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def forward(self, img: torch.FloatTensor, std: torch.FloatTensor) -> torch.FloatTensor:
        if torch.rand(1).item() < self.p:
            noise = TruncatedNormal(torch.zeros_like(std), std, low=-2 * std, high=2 * std + 1e-4).sample().to(torch.float32)
            return img + noise
        return img
        
        

        
        
        
