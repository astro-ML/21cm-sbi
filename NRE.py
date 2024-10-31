import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, kstest, uniform
from alive_progress import alive_bar
from NPE import NPEHandler
from logging import info, warning, error


class NREHandler(NPEHandler):
    def __init__(self, density_estimator, summary_net = None,
                 device = 'cuda'):
        super().__init__(density_estimator, summary_net, device)
    
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