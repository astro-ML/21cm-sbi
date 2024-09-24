import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional
from torch.distributions import Distribution
from utility import repeat_rows, merge_leading_dims, get_sbi_posterior
from sbi.utils import BoxUniform


class FlowSequential(nn.Sequential):
    """ Container for layers of a normalizing flow """
    def forward(self, x, y):
        for module in self:
            x  = module(x, y)
        return x

class BasicBlock(nn.Module):
    def __init__(self, in_dim, out_dim, n_nodes, hidden_layer, activation_fn, batch_norm, conditional_dim, idx):
        super(BasicBlock, self).__init__()
        self.out_dim = out_dim
        self.first = not bool(idx)
        self.net = self.build_base_net(in_dim, out_dim, hidden_layer, n_nodes, activation_fn, batch_norm, conditional_dim)

    def forward(self, x, cond=None):
        
        if self.first:
            x = torch.cat([x, cond], dim=-1)
        
        out = self.net(x)
        if self.cond:
            out = F.glu(torch.cat((out, self.cond_net(cond)), dim=1), dim=1)
        return x[:,:self.out_dim] + out
    
    def build_base_net(self, in_dim, out_dim, hidden_layer, n_nodes, activation_fn, batch_norm, conditional_dim):
        # alternatively just concat input and conditional
        if conditional_dim > 0:
            self.cond_net = nn.Linear(conditional_dim, out_dim)
            self.cond = True
        else:
            self.cond = False
            
        net = []
        net += [nn.Linear(in_dim, n_nodes)]
        net += [activation_fn]
        for _ in range(1, hidden_layer - 1):
            net += [nn.Linear(n_nodes, n_nodes)]
            net += [activation_fn]
            if batch_norm:
                net += [nn.BatchNorm1d(n_nodes)]
        net += [nn.Linear(n_nodes, out_dim)]
        return nn.Sequential(*net)
    
class LLinear(nn.Linear):
    def __init__(self, in_dim, out_dim):
        super().__init__(in_dim, out_dim)

    def forward(self, x, cond=None):
        return super().forward(x)


class ResNet(nn.Module):
    def __init__(self, in_dim, n_nodes, hidden_layer, batch_norm, n_blocks, K=4, gamma=1.0, activation_fn = nn.ReLU(),
                 device = 'cuda'):
        super(ResNet, self).__init__()
        self.in_dim = in_dim
        self.classifier = [BasicBlock(self.in_dim *2 if _ == 0 else self.in_dim,
                                        in_dim,
                                      n_nodes, hidden_layer, activation_fn, batch_norm, in_dim, _) for _ in range(n_blocks)]
        self.classifier += [LLinear(self.in_dim, 1)]
        self.classifier = FlowSequential(*self.classifier).to(device)
        self.K = K
        self.gamma = gamma
        self.device = device
        
    def forward(self, x, cond):
        # Initial convolution
        out = self.classifier(x, cond)
        return out
    
    def prior(self, batch_size, epsilon=1e-2):
        return torch.rand((batch_size, self.in_dim), device=self.device)*(1-2*epsilon) + epsilon
    
    # add theta prior sampled from prior for better performance
    def logits(self, x, theta, theta_prior = True):
        
        batch_size = batch_size_prior = theta.shape[0]
        
        if theta_prior:
            theta_prior = self.prior(batch_size_prior)
        else:
            theta_prior = torch.empty((0, self.in_dim), device=theta.device)
         
        repeated_x = repeat_rows(x, self.K+1)
        
        # one could add a new batch_size sampled from the prior
        probs = torch.cat(
        [(1 - torch.eye(batch_size)), torch.ones(batch_size, batch_size_prior)], dim=-1
        ) / (batch_size + batch_size_prior - 1)
        #probs = torch.ones(batch_size, batch_size) * (1 - torch.eye(batch_size)) / (batch_size - 1)
        choices = torch.multinomial(probs, num_samples=self.K, replacement=False)
        contrasting_theta = torch.cat([theta, theta_prior], dim=0)[choices]
        atomic_theta = torch.cat((theta[:, None, :], contrasting_theta), dim=1).reshape(
            batch_size * (self.K+1), -1
        )
        return self.forward(atomic_theta, repeated_x)
    
    def loss_on_logits(self, logits_marginal, logits_joint, batch_size):
        
        logits_marginal = logits_marginal[:, 1:]
        logits_joint = logits_joint[:, :-1]
        
        loggamma = torch.tensor(self.gamma, dtype=logits_marginal.dtype, device=logits_marginal.device).log()
        logK = torch.tensor(self.K, dtype=logits_marginal.dtype, device=logits_marginal.device).log()
        denominator_marginal = torch.concat(
            [loggamma + logits_marginal, logK.expand((batch_size, 1))],
            dim=-1,
        )
        denominator_joint = torch.concat(
            [loggamma + logits_joint, logK.expand((batch_size, 1))],
            dim=-1,
        )
        
        log_prob_marginal = logK - torch.logsumexp(denominator_marginal, dim=-1)
        log_prob_joint = (
            loggamma + logits_joint[:, 0] - torch.logsumexp(denominator_joint, dim=-1)
        )
        # relative weights. pm := p_0, and pj := p_K * K from the notation.
        pm, pj = ResNet.get_pmarginal_pjoint(self.gamma)
        return -(pm * log_prob_marginal + pj * log_prob_joint)
    
    def loss(self, x, cond):
        #theta_prior = torch.rand(theta.shape, device=theta.device)
        assert cond.shape[0] == x.shape[0], "Batch sizes for theta and x must match."
        batch_size = cond.shape[0] 
        
        logits_marginal = self.logits(x, cond, theta_prior=True).reshape(
            batch_size, self.K + 1
        )
        logits_joint = self.logits(x, cond, theta_prior=True).reshape(
            batch_size, self.K + 1
        )
        
        return self.loss_on_logits(logits_marginal, logits_joint, batch_size)
    
    def decision(self, x, cond):
        """Given a batch of data, return the decision of the classifier.

        Args:
            x (_type_): Parameter proposal of shape (batch_size, in_dim * (K+1)).
            cond (_type_): Output of summary network of shape (batch_size, cond_dim).

        Returns:
            _type_: _description_
        """
        return self.forward(x, cond)
    
    def test_decision(self, x, cond):
        """Given a batch of data, return the decision of the classifier.
        (Meant for performance evaluation)

        Args:
            x (_type_): True parameter of shape (batch_size, in_dim).
            cond (_type_): Output of summary network of shape (batch_size, cond_dim).

        Returns:
            _type_: _description_
        """
        batch_size = x.shape[0]
        probs = torch.ones(batch_size, batch_size) * (1 - torch.eye(batch_size)) / (batch_size - 1)
        choices = torch.multinomial(probs, num_samples=self.K, replacement=False)
        contrasting_theta = cond[choices]
        atomic_theta = torch.cat((cond[:, None, :], contrasting_theta), dim=1).reshape(
            batch_size, self.in_dim
        )
        return self.forward(atomic_theta, x)
    
    def sample(self,
    n_samples: int,
    x: torch.Tensor,
    prior: Optional[Distribution] = None,
    sample_with: str = "rejection",
    mcmc_method: str = "slice_np",
    mcmc_parameters: Dict[str, Any] = {},
    rejection_sampling_parameters: Dict[str, Any] = {},
    enable_transform: bool = False,
    device = 'cuda',):
        """Sample from the posterior distribution of the classifier.

        Args:
            ratio_estimator (torch.nn.Module): A neural network that estimates the ratio of the likelihoods.
            prior (Optional[Distribution], optional): Prior distribution of the parameter. Defaults to None.
            sample_with (str, optional): Sampling method. Defaults to "rejection".
            mcmc_method (str, optional): MCMC method to use. Defaults to "slice_np".
            mcmc_parameters (Dict[str, Any], optional): Parameters for the MCMC method. Defaults to {}.
            rejection_sampling_parameters (Dict[str, Any], optional): Parameters for rejection sampling. Defaults to {}.
            enable_transform (bool, optional): Enable transformation of the samples. Defaults to False.

        Returns:
            torch.Tensor: Samples from the posterior distribution.
        """
        prior = BoxUniform(low=torch.zeros((6)), high=torch.ones((6)), device=device)
        posterior = get_sbi_posterior(
            ratio_estimator=self.classifier.to(device),
            prior=prior,
            sample_with=sample_with,
            mcmc_method=mcmc_method,
            mcmc_parameters=mcmc_parameters,
            rejection_sampling_parameters=rejection_sampling_parameters,
            enable_transform=enable_transform,
        )
        samples = posterior.sample((n_samples,), x=x.to(device))
        return samples
        

        
    @staticmethod
    def get_pmarginal_pjoint(gamma: float) -> float:
        r"""Return a tuple (p_marginal, p_joint) where `p_marginal := `$p_0$,
        `p_joint := `$p_K \cdot K$.

        We let the joint (dependently drawn) class to be equally likely across K
        options. The marginal class is therefore restricted to get the remaining
        probability.
        """
        p_joint = gamma / (1 + gamma)
        p_marginal = 1 / (1 + gamma)
        return p_marginal, p_joint
    
    
    
