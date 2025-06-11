import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional
from torch.distributions import Distribution
from torch import distributions as D
from utility import repeat_rows, get_nre_posterior
from torch.nn import init
from sbi.utils import BoxUniform

class CLinear(nn.Linear):
    def __init__(self, in_dim, out_dim, cond_dim, activation_fn, batch_norm):
        super().__init__(in_dim, out_dim)
        if cond_dim > 0:
            self.cond_net = nn.Linear(cond_dim, out_dim)
            self.cond = True
        else:
            self.cond = False
            
        self.activation_fn = activation_fn
        self.batch_norm = nn.BatchNorm1d(out_dim) if batch_norm else (lambda x: x)
        
    def forward(self, x, y=None):
        out = F.linear(x, self.weight, self.bias)
        if y is not None and self.cond:
            out = out + F.glu(torch.cat((out, self.cond_net(y)), dim=1), dim=1)
        out = self.activation_fn(out)
        out = self.batch_norm(out)
        return out

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

        
        out = self.net(x, cond)
        return x + out
    
    def build_base_net(self, in_dim, out_dim, hidden_layer, n_nodes, activation_fn, batch_norm, conditional_dim):
            
        net = []
        #if batch_norm:
        #    net += [nn.BatchNorm1d(in_dim)]
        net += [CLinear(in_dim, n_nodes, cond_dim=0, activation_fn=activation_fn, batch_norm=batch_norm)]
        for _ in range(1, hidden_layer - 1):
            net += [CLinear(n_nodes, n_nodes, cond_dim=0, activation_fn=activation_fn, batch_norm=batch_norm)]
        net += [CLinear(n_nodes, out_dim, conditional_dim, activation_fn, batch_norm=batch_norm)]
        return FlowSequential(*net)

class ResNet(nn.Module):
    def __init__(self, in_dim, n_nodes, hidden_layer, batch_norm, n_blocks, K=4, gamma=1.0, activation_fn = nn.ReLU(),
                 device = 'cuda', prior = BoxUniform, epsilon: float = 1e-4):
        super(ResNet, self).__init__()
        self.in_dim = in_dim
        self.classifier = []
        self.classifier += [CLinear(in_dim + in_dim, n_nodes, cond_dim=0, activation_fn=activation_fn, batch_norm=batch_norm)]
        for _ in range(n_blocks):
            self.classifier += [BasicBlock(n_nodes ,
                                        n_nodes,
                                      n_nodes, hidden_layer, activation_fn, batch_norm, in_dim, _)]
        self.classifier += [CLinear(n_nodes, 1, cond_dim=0, activation_fn=(lambda x: x), batch_norm=False)]
        self.classifier = nn.Sequential(*self.classifier).to(device)
        
        # shallow init for last layer
        init.uniform_(self.classifier[-1].weight, -1e-3, 1e-3)
        init.uniform_(self.classifier[-1].bias, -1e-3, 1e-3)
        
        self.K = K
        self.gamma = gamma
        self.device = device
        
        self.prior = prior(low=-torch.ones((in_dim)) + epsilon, high=torch.ones((in_dim)) - epsilon, device = device)
        
    def forward(self, x, condition):
        # Initial convolution
        out = torch.cat([x, condition], dim=-1)
        out = self.classifier(out)
        return out
    
    @property
    def base_dist(self):
        return D.Uniform(0,1)
    
    def loss(
        self, theta, condition):
        batch_size = theta.shape[0]

        logits_marginal = self.logits(theta, condition, self.K + 1).reshape(
            batch_size, self.K + 1
        )
        logits_joint = self.logits(theta, condition, self.K).reshape(
            batch_size, self.K
        )

        dtype = logits_marginal.dtype
        device = logits_marginal.device

        logits_marginal = logits_marginal[:, 1:]

        loggamma = torch.tensor(self.gamma, dtype=dtype, device=device).log()
        logK = torch.tensor(self.K, dtype=dtype, device=device).log()
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
        p_marginal, p_joint = self.get_pmarginal_pjoint(self.gamma)
        return -(p_marginal * log_prob_marginal + p_joint * log_prob_joint)

    
    def logits(self, theta, cond, num_classes):

        batch_size = theta.shape[0]
        repeated_x = repeat_rows(cond, num_classes)

        # TODO: Implement sampling from prior to explore full marginal
        probs = torch.ones(batch_size, batch_size) * (1 - torch.eye(batch_size)) / (batch_size - 1)
        choices = torch.multinomial(probs, num_samples=num_classes -1, replacement=False)

        contrasting_theta = theta[choices]

        atomic_theta = torch.cat((theta[:, None, :], contrasting_theta), dim=1).reshape(
            batch_size * num_classes, -1
        )

        return self.forward(atomic_theta, repeated_x)

    def build_posterior(self, sample_kwargs):
        self.classifier.eval()
        posterior = get_nre_posterior(
            ratio_estimator=self,
            prior=self.prior,
            sample_kwargs=sample_kwargs,)
        self.posterior = posterior
            

    def sample(self, num_samples, x):
        return self.posterior.sample((num_samples,), x, show_progress_bars=False)    
        

        
    @staticmethod
    def get_pmarginal_pjoint(gamma: float) -> float:
        p_joint = gamma / (1 + gamma)
        p_marginal = 1 / (1 + gamma)
        return p_marginal, p_joint
    
    


