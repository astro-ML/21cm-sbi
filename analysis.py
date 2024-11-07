import torch  
import numpy as np 
import matplotlib.pyplot as plt  
from scipy.stats import gaussian_kde, kstest, uniform 
from sbi.utils import BoxUniform
from sbi.diagnostics import check_tarp
from plot import pairplot 
from alive_progress import alive_bar  
from sbi.inference.potentials import posterior_estimator_based_potential
from sbi.inference.potentials import likelihood_estimator_based_potential
from sbi.inference.potentials import ratio_estimator_based_potential
import json
import seaborn as sns
from sbi.analysis import conditional_pairplot, conditional_corrcoeff
from logging import info, warning, error

from typing import Callable, Optional, Tuple
from torch import Tensor


class Analysis:
    def __init__(self, NPEHandler, 
                Validation_Dataset,
                filename: str = "",
                path: str = "",
                prior = None,
                epsilon: float = 1e-4,
                labels: list = [r"$M_{WDM}$", r"$\Omega_m$", r"$L_X$", r"$E_0$", r"$T_{vir, ion}$", r"$\zeta$"],
                transform: bool = False,
                posterior_kwargs: dict = {}):
        """Class to analyse a neural posterior (NPE) estimator on several metrics.

        Args:
            NPEHandler (_type_): _description_
            Validation_Dataset (_type_): _description_
            filename (str, optional): _description_. Defaults to "".
            path (str, optional): _description_. Defaults to "".
            prior (_type_, optional): _description_. Defaults to None.
            epsilon (float, optional): _description_. Defaults to 1e-4.
            labels (list, optional): _description_. Defaults to [r"{WDM}$", r"$\Omega_m$", r"$", r"$", r"{vir, ion}$", r"$\zeta$"].
            transform (bool, optional): _description_. Defaults to False.
        """
        self.NPE = NPEHandler
        self.valdat = Validation_Dataset
        self.device = self.NPE.device
        # workaround for now
        self.NPE.density_estimator.zero_grad(set_to_none=True)
        self.NPE.summary_net.zero_grad(set_to_none=True)
        self.features = len(labels)
        if path == "":
            self.save = False
        else:
            self.save = True
            self.filename = filename
            self.path = path if path[-1] == "/" else path + "/"
        if prior is None:
            self.prior = BoxUniform(torch.zeros(self.features) + epsilon, torch.ones(self.features) - epsilon, device=self.device)
        else: self.prior = prior
        self.potential, _ = posterior_estimator_based_potential(posterior_estimator=self.NPE.density_estimator,
                        prior = self.prior, enable_transform=transform,x_o = None)
        self.transform = transform
        if self.save:
            with open(path + filename + "_results.json", 'w') as f:
                json.dump({}, f, indent=4)
        self.labels = labels
        self.posterior_kwargs = posterior_kwargs

    def marginals(self, num_points: int = 3,
                num_samples_stat: int = 10000,):
        """Plot the marginals with [num_samples_stat] samples.

        Args:
            num_points (int, optional): _description_. Defaults to 3.
            num_samples_stat (int, optional): _description_. Defaults to 10000.
            sample_kwargs (dict, optional): _description_. Defaults to { 'sample_with' :'rejection',}.
        """
        labs, imgs, _ = self.sampler(num_samples=num_points, sumnet=True)
        for i, (lab, img) in enumerate(zip(labs, imgs)):
            with torch.no_grad():
                samples = self.NPE.density_estimator.sample(num_samples_stat, img.unsqueeze(0), self.posterior_kwargs)
            # plot posterior samples
            figure, axis = pairplot(samples = samples.detach().cpu().numpy(), points=lab.detach().cpu().numpy(),
                limits=[[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],], figsize=(10, 10),
                labels = self.labels,
                #quantiles=((0.16, 0.84, 0.0015, 0.99815)), levels=(1 - np.exp(-0.5),1 - np.exp(-9/2)),
                upper = 'hist', lower = 'contour', diag = 'kde')
            if self.save: 
                figure.savefig(self.path + f"{self.filename}_marginal_{i}.png", dpi=300)
            else: figure.show()
        torch.cuda.empty_cache()
    
    def conditionals(self, num_points: int = 3,
                num_samples: int = 100,):
        """Plot the 2D conditionals given some fiducial value.

        Args:
            num_points (int, optional): _description_. Defaults to 3.
            num_samples_state (int, optional): _description_. Defaults to 10000.
            sample_kwargs (dict, optional): _description_. Defaults to { 'sample_with' :'rejection',}.
        """
        self.NPE.density_estimator._device = 'cpu'
        self.NPE.density_estimator.to('cpu')
        labs, imgs, _ = self.sampler(num_samples=num_points, sumnet=True)
        for (i, lab, img) in zip(range(num_points), labs, imgs):
            with torch.no_grad():
                figure, axis = conditional_pairplot(
                    density=self.NPE.density_estimator,
                    condition=img.unsqueeze(0),
                    limits=[[0, 1]]*6, figsize=(10, 10),
                    labels = self.labels,
                    points=lab.detach().cpu().numpy(),
                    resolution=num_samples
                )
            if self.save:
                figure.savefig(self.path + f"{self.filename}_marginal_{i}.png", dpi=300)
            else: figure.show()
        torch.cuda.empty_cache()



    def run_sbc(self, num_samples: int = 1000,):
        self.NPE.density_estimator.eval()
        self.NPE.density_estimator.to(self.device)
        if self.NPE.sum_net:
            self.NPE.summary_net.eval()
            self.NPE.summary_net.to(self.NPE.device)
        
        #mp = True if num_workers > 1 else False
        # run sbc on full Validation Dataset
        lengthd = len(self.valdat.dataset)
        info("Run SBC...")
        batch_size = self.valdat.batch_size
        with alive_bar(len(self.valdat), force_tty=True, refresh_secs=1) as bar:
            with torch.no_grad():
                for k, (lab, img, rnge) in enumerate(self.valdat):
                    img, lab, rnge = img.to(self.device), lab.to(self.device), rnge.to(self.device)

                    pred = self.NPE.summary_net(img, rnge)
                    if k == 0:
                        ranks = torch.empty((lengthd, *pred.shape[1:]))
                        dap_samples = torch.empty(ranks.shape)
                    # sbc rank stat
                    for i in range(pred.shape[0]):
                        with torch.no_grad():
                            samples = self.NPE.density_estimator.sample(x = pred[i].unsqueeze(0), 
                        num_samples=num_samples,
                        sample_kwargs=self.posterior_kwargs)
                        dap_samples[k*batch_size + i] = samples[0].cpu()
                        for j in range(pred.shape[1]):
                            ranks[k*batch_size + i,j] = (samples[:,j]<lab[i,j]).sum().cpu().item()
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
        if self.save: 
            fig.savefig(self.path + f"{self.name}_rank.png", dpi=400)
        fig.show()
        fig.clf()
        torch.cuda.empty_cache()
        


        # ks_pvals: check how uniform the ranks are (frequentist approach)
        kstest_pvals = torch.tensor(
        [
            kstest(rks, uniform(loc=0, scale=num_samples).cdf)[1]
            for rks in ranks.T
        ],
        dtype=torch.float32,
        )

        self.append_result({"kstest_pvals": kstest_pvals})

        # c2st, train a small classifier to distinguish between samples from rank and uniform distribution
        # if 0.5, both a equal = classifier is not able to distinguish the two distributions
        
        # compute tarp
        labels = torch.empty((0, lab.shape[1]))
        for  lab, _,_ in self.valdat:
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
        #arg_vmax = np.argmax(hmax)
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
        if self.save: 
            fig.savefig(self.path + f"{self.filename}_coverage.png", dpi=400)
        fig.show()
        fig.clf()
        torch.cuda.empty_cache()
        
    def run_ppc(self, simulator, num_points: int = 5, num_samples: int = 100):
        """_summary_

        Args:
            simulator (class):  Function or object with __call__ method. For a tensor of size (batch_dim, posterior_samples) it should return a
                                tensor of size (batch_dim, likelihood_sample). It can be a simulator, an emulator etc.
            num_points (int, optional): Number of posterior predictive checks. Defaults to 5.
        """
        
        _,thetas,_ = self.sampler(num_samples=num_points, sumnet=True)
        with alive_bar(num_points, force_tty=True, refresh_secs=1) as bar:
            for i in range(num_points):
                with torch.no_grad():
                    posterior_samples = self.NPE.density_estimator.sample(num_samples, x=thetas[i],sample_kwargs=self.posterior_kwargs)
                likelihood_samples = simulator(posterior_samples)
                figure, axis = pairplot(samples = likelihood_samples, points=thetas[i],
                limits=[[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],], figsize=(10, 10),
                labels = self.labels,
                #quantiles=((0.16, 0.84, 0.0015, 0.99815)), levels=(1 - np.exp(-0.5),1 - np.exp(-9/2)),
                upper = 'hist', lower = 'contour', diag = 'kde')
                if self.save: 
                    figure.savefig(self.path + f"{self.filename}_ppc_{i}.png", 
                                             dpi=300)
                else: figure.show()
        torch.cuda.empty_cache()
        
    def run_sensitivity_analysis(self,
        num_points: int = 10,
        num_samples: int = 1000,
        summary_net: bool = True,
        axis: Tuple[int,...] = [-1]):
        
        final = {}
        
        # first check avg. sensitivity for sum_net
        for i, (_, imgs, rnges) in enumerate(self.valdat):
            if i >= num_points:
                break
            imgs, rnges = imgs.to(self.device), rnges.to(self.device)
            if summary_net:
                # we are interested in how the imgs and rnges change when we change the input
                
                imgs.requires_grad = True
                rnges.requires_grad = True
                out = self.NPE.summary_net(imgs, rnges)
                out.mean(0).mean().backward()
            if i == 0:
                if axis == [-1]:
                    axis_imgs = [imgs.dim() - 1]
                    axis_rnges = [rnges.dim() -1]
                else:
                    axis_imgs = axis
                    axis_rnges = axis
                axes_to_reduce_imgs = [i for i in range(imgs.dim()) if i not in axis_imgs]
                axes_to_reduce_rnges = [i for i in range(rnges.dim()) if i not in axis_rnges]

                sens_sumnet_temp_img = torch.mean(torch.abs(imgs.grad), dim=axes_to_reduce_imgs).detach().cpu().unsqueeze(0)
                sens_sumnet_temp_rnge = torch.mean(torch.abs(rnges.grad), dim=axes_to_reduce_rnges).detach().cpu().unsqueeze(0)
            else:
                sens_sumnet_temp_img = torch.cat([sens_sumnet_temp_img,
                                                  torch.mean(torch.abs(imgs.grad), dim=axes_to_reduce_imgs).detach().cpu().unsqueeze(0)],
                                                 dim=0)
                sens_sumnet_temp_rnge = torch.cat([sens_sumnet_temp_rnge,
                                                   torch.mean(torch.abs(rnges.grad), dim=axes_to_reduce_rnges).detach().cpu().unsqueeze(0)],
                                                  dim=0)
            
            imgs = out.detach()
            for j, img in enumerate(imgs):
                img = img.unsqueeze(0)
                with torch.no_grad():
                    thetas = self.NPE.density_estimator.sample(num_samples = num_samples,
                                                            x = img, sample_kwargs=self.posterior_kwargs)
                img.requires_grad = True
                res = self.NPE.density_estimator.forward(thetas, img)[1]
                res.mean(0).mean().backward()
                res = res.cpu()
                if i == 0 and j == 0:
                    if axis == [-1]:
                        axis = [img.dim() - 1]
                    axes_to_reduce = [i for i in range(img.dim()) if i not in axis]
                    grad_tmp =  torch.mean(torch.abs(img.grad), dim=axes_to_reduce).detach().cpu().unsqueeze(0)
                else:
                    grad_tmp = torch.cat([grad_tmp, torch.mean(torch.abs(img.grad), dim=axes_to_reduce).detach().cpu().unsqueeze(0)], dim=0)

                
                
        final.update({
            "sensitivity_sum_net_image": torch.mean(sens_sumnet_temp_img, dim=0).cpu(),
            "sensitivity_sum_net_range": torch.mean(sens_sumnet_temp_rnge, dim=0).cpu(),
            "sensitivity_density_net_logprob": torch.mean(grad_tmp, dim=0).cpu()
        })
        self.append_result(final)
        torch.cuda.empty_cache()
        return final
        
        
            
    def run_sensitivity_analysis_eigenspace(
        self,
        num_points: int = 10,
        norm_gradients_to_prior: bool = True,
        num_samples: int = 1000,
        ):

        r"""Code heavily inspired by https://github.com/sbi-dev/sbi/blob/main/sbi/analysis/sensitivity_analysis.py

        Return eigenvectors and values corresponding to directions of sensitivity.

        The directions of sensitivity are the directions along which a specific
        property changes in the fastest way. They will have the largest eigenvalues.

        This computes the matrix:
        $\mathbf{M} = \mathbb{E}_{p(\theta|x_o)}[\nabla_{\theta} f(\theta)^T
        \nabla_{\theta}
        f(\theta)]$
        where $f(\cdot)$ is the trained regression network. The expected value is
        approximated with a Monte-Carlo mean. Next, do an eigenvalue
        decomposition of the matrix $\mathbf{M}$:

        $\mathbf{M} = \mathbf{Q} \mathbf{\Lambda} \mathbf{Q}^{-1}$

        We then return the eigenvectors and eigenvalues found by this decomposition.
        Eigenvectors with large eigenvalues are directions along which the property is
        sensitive to changes in the parameters $\theta$ (`active` directions).
        Increases along these directions will increase the value of the property.

        Args:
            x: Fiducial point at which the sensitivity analysis will be performed.
            norm_gradients_to_prior: Whether to normalize each entry of the gradient
                by the standard deviation of the prior in each dimension. If set to
                `False`, the directions with the strongest eigenvalues might correspond
                to directions in which the prior is broad.
            num_monte_carlo_samples: Number of Monte Carlo samples that the average is
                based on. A larger value will make the results more accurate while
                requiring more compute time.

        Returns:
            Eigenvectors and corresponding eigenvalues. They are sorted in ascending
            order. The column `eigenvectors[:, j]` is the eigenvector corresponding to
            the `j`-th eigenvalue.
        """

        self._gradients_are_normed = norm_gradients_to_prior
        
        def comp_eig(x, num_samples):
            with torch.no_grad():
                thetas = self.NPE.density_estimator.sample(num_samples = num_samples,
                                                        x = x, sample_kwargs=self.posterior_kwargs).detach()
            thetas.requires_grad = True

            #thetas.requires_grad = True
            self.potential.set_x(x)
            predictions = self.potential(thetas, track_gradients=True)

            loss = predictions.mean()
            loss.backward()
            gradients = torch.squeeze(thetas.grad)
            if norm_gradients_to_prior:
                prior_samples = self.prior.sample((10000,))
                prior_scale = torch.std(prior_samples, dim=0)
                gradients *= prior_scale
            outer_products = torch.einsum("bi,bj->bij", (gradients, gradients))
            average_outer_product = outer_products.mean(dim=0)

            eigen_values, eigen_vectors = torch.linalg.eigh(average_outer_product, UPLO="U")

            # Identify the direction of the eigenvectors. Above, we have computed an outer
            # product m*mT=A. Note that the same matrix A can be constructed with the
            # negative vector (-m)(-mT)=A. Thus, when performing an eigen-decomposition of
            # A, we can not determine if the eigenvector was -m or m. We solve this issue
            # below. We use that the average gradient m should be obtained by a mean over
            # the eigenvectors (weighted by the eigenvalues).
            av_gradient = torch.mean(gradients, dim=0)
            av_gradient = av_gradient / torch.norm(av_gradient)
            av_eigenvec = torch.mean(eigen_vectors * eigen_values, dim=1)
            av_eigenvec = av_eigenvec / torch.norm(av_eigenvec)

            # Invert if the negative eigenvectors are closer to the average gradient.
            if (torch.mean((av_eigenvec - av_gradient) ** 2)) > (
                torch.mean((-av_eigenvec - av_gradient) ** 2)
            ):
                eigen_vectors = -eigen_vectors
            return eigen_values, eigen_vectors

        _, img ,_ = self.sampler(num_points, True)
        for i,x in enumerate(img):
            x = x.to(self.device).unsqueeze(0)
            eigval, eigvec = comp_eig(x, num_samples)
            eigval, eigvec = eigval.unsqueeze(0), eigvec.unsqueeze(0)
            if i == 0:
                eigen_values = eigval
                eigen_vectors = eigvec
            else:
                eigen_values = torch.cat([eigen_values, eigval], dim=0)
                eigen_vectors = torch.cat([eigen_vectors, eigvec], dim=0)

        self.append_result({
            "sensitivity_eigen_values": eigen_values.mean(0),
            "sensitivity_eigen_vectors": eigen_vectors.mean(0)
        })
        torch.cuda.empty_cache()
    
    @staticmethod
    def get_tarp_refpoint(thetas: Tensor) -> Tensor:
        """Returns reference points for the TARP diagnostic, sampled from a uniform."""

        # obtain min/max per dimension of theta
        lo = thetas.min(dim=0).values  # min for each theta dimension
        hi = thetas.max(dim=0).values  # max for each theta dimension

        refpdf = torch.distributions.Uniform(low=lo, high=hi)

        # sample one reference point for each entry in theta
        return refpdf.sample(torch.Size([thetas.shape[0]]))

    def run_tarp(self, num_points: int = 32, 
                 num_samples: int = 500,
                 num_bins: int = 30):

        lab, img, _ = self.sampler(num_samples=num_points, sumnet=True)
            
        for i in range(num_points):
            with torch.no_grad():
                sample = self.NPE.density_estimator.sample(num_samples=num_samples, x=img[i], sample_kwargs=self.posterior_kwargs).unsqueeze(1).detach().cpu()
            if i == 0:
                samples = sample
            else:
                samples = torch.cat([samples, sample], dim=1)
        del img, _

        if num_bins is None:
            num_bins = num_samples // 10

        distance = torch.nn.MSELoss(reduction='none')

        reference = Analysis.get_tarp_refpoint(lab)

        # distances between references and samples
        sample_dists = distance(reference, samples)

        # distances between references and true values
        theta_dists = distance(reference, lab)

        # compute coverage, f in algorithm 2
        coverage_values = (
            torch.sum(sample_dists < theta_dists, dim=0) / num_samples
        )
        hist, alpha = torch.histogram(coverage_values, density=True, bins=num_bins)
        # calculate empirical CDF via cumsum and normalize
        ecp = torch.cumsum(hist, dim=0) / hist.sum()
        # add 0 to the beginning of the ecp curve to match the alpha grid
        ecp = torch.cat([Tensor([0]), ecp])

        # Similar to SBC, we can check then check whether the distribution of ecp is close to
        # that of alpha.
        atc, ks_pval = check_tarp(ecp, alpha)
        self.append_result({"atc": atc,
                            "ks_val": ks_pval})

        fig = plt.figure(figsize=(6, 6))
        ax = plt.gca()

        ax.plot(alpha, ecp, color="blue", label="TARP")
        ax.plot(alpha, alpha, color="black", linestyle="--", label="ideal")
        ax.set_xlabel(r"Credibility Level $\alpha$")
        ax.set_ylabel(r"Expected Coverage Probability")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_title("TARP ")
        ax.legend()
        fig.tight_layout()
        if self.save:
            fig.savefig(self.path + self.filename + "_tarp.png", dpi=300)
        else:
            fig.show()
        torch.cuda.empty_cache()


    def person_coeff(self, num_points: int = 32, num_samples_stat = 200):
        _,prior_samples,_ = self.sampler(num_samples=num_points, sumnet=True)
        with alive_bar(num_points, force_tty=True,refresh_secs=1) as bar:
            for i, psamp in enumerate(prior_samples):
                psamp.to(self.device)
                with torch.no_grad():
                    samples = self.NPE.density_estimator.sample(num_samples_stat, psamp.unsqueeze(0), sample_kwargs = self.posterior_kwargs).cpu()
                if i == 0:
                    corrcoef = torch.corrcoef(samples.T).unsqueeze(0)
                else:
                    corrcoef = torch.cat([corrcoef, torch.corrcoef(samples.T).unsqueeze(0)], dim=0)
                bar()
            print(corrcoef.shape)
            corrcoef = corrcoef.mean(dim=0)
            print(corrcoef.shape)
            self.append_result({"person_coeff_marginal": corrcoef})

            plt.figure(figsize=(8, 6))
            sns.heatmap(corrcoef, annot=True, cmap='coolwarm', vmin=-1, vmax=1, xticklabels=self.labels, yticklabels=self.labels)
            plt.title("Marginal Pearson correlation coefficients")
            if self.save:
                plt.savefig(self.path + self.filename + "_marginal_correlcoeff.png", dpi=320)
            else:
                plt.show()
        
        torch.cuda.empty_cache()
                
    def person_coeff_conditionals(self, num_points: int = 32, num_samples_stat = 200):
        _,prior_samples,_ = self.sampler(num_samples=num_points, sumnet=True)
        self.NPE.density_estimator._device = self.device
        with torch.no_grad():
            corrcoef = conditional_corrcoeff(
                density=self.NPE.density_estimator,
                limits=torch.tensor([[1e-4,1-1e-4]]*6,device=self.device),
                condition = prior_samples,resolution=num_samples_stat
                
            ).cpu()
        self.append_result({"person_coeff_conditional": corrcoef})

        plt.figure(figsize=(8, 6))
        sns.heatmap(corrcoef, annot=True, cmap='coolwarm', vmin=-1, vmax=1, xticklabels=self.labels, yticklabels=self.labels)
        plt.title("Conditional Pearson correlation coefficients")
        if self.save:
            plt.savefig(self.path + self.filename + "_conditional_correlcoeff.png", dpi=320)
        else:
            plt.show()
            
        torch.cuda.empty_cache()
            


    def append_result(self, in_dict: dict):
        """Add key-value key to statistics on disk.

        Args:
            in_dict (dict): Dictionary which is saved on disk.
        """
        if self.save:
            with open(self.path + self.filename + "_results.json", 'r') as f:
                data = json.load(f)
            data.update(in_dict)
            with open(self.path + self.filename + "_results.json", 'w') as f:
                json.dump(data, f, intend=4)
        else:
            print(in_dict)
            
        torch.cuda.empty_cache()

    def sampler(self, num_samples: int, sumnet: bool = False):
        with alive_bar(int(num_samples/self.valdat.batch_size), force_tty=True,refresh_secs=1) as bar:
            if num_samples > self.valdat.batch_size * len(self.valdat):
                raise ValueError("Number of samples larger than available data.")
            for i, (lab, img, rnge) in enumerate(self.valdat):
                if sumnet:
                    with torch.no_grad():
                        img = self.NPE.summary_net(images.to(self.device), ranges.to(self.device)).cpu()
                if i == 0:
                    labels, images, ranges = lab, img, rnge
                else:
                    labels = torch.cat([labels,lab], dim=0)
                    images = torch.cat([images, img], dim=0)
                    ranges = torch.cat([ranges,rnge],dim=0)
                if labels.shape[0] >= num_samples:
                    return labels[:num_samples], images[:num_samples], ranges[:num_samples]