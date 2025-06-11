import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from made_backbone import MADE, BatchNorm, FlowSequential
import torch.nn.init as init
from cl_models import unconstrained_rational_quadratic_spline
from math import sqrt
from typing import Dict, Any
from utility import get_nle_posterior
from sbi.utils import BoxUniform

class cond_FCL(nn.Linear):
    def __init__(self, in_dim, out_dim, cond_dim, activation_fn):
        super().__init__(in_dim, out_dim)
        if cond_dim > 0:
            self.cond_weight = nn.Parameter(torch.rand(out_dim, cond_dim) / sqrt(cond_dim))

        self.activation_fn = activation_fn
        
    def forward(self, x, y=None):
        out = F.linear(x, self.weight, self.bias)
        if y is not None:
            out = out + F.linear(y, self.cond_weight)
        out = self.activation_fn(out)
        return out

# alternative conditional linear layer, which should be more expressive
class CLinear(nn.Module):
    def __init__(self, in_dim, out_dim, cond_dim, activation_fn, batch_norm):
        super().__init__()
        # Main linear layer (outputs twice the number of channels for GLU)
        self.linear = nn.Linear(in_dim, out_dim * 2)

        # Conditioning projection (maps cond_dim -> out_dim * 2)
        self.cond_proj = nn.Linear(cond_dim, out_dim * 2)

        self.activation_fn = activation_fn
        self.batch_norm = nn.BatchNorm1d(out_dim) if batch_norm else nn.Identity()

    def forward(self, x, cond):
        """
        x:     (batch, in_dim)
        cond:  (batch, cond_dim)
        """
        linear_out = self.linear(x)  # (batch, out_dim * 2)

        # Project conditioning vector
        cond_out = self.cond_proj(cond)  # (batch, out_dim * 2)

        # Add conditioning
        out = linear_out + cond_out

        # Split for GLU
        a, b = out.chunk(2, dim=1)  # Split along channel dimension
        out = a * torch.sigmoid(b)

        out = self.activation_fn(out)
        out = self.batch_norm(out)
        return out
    

class CondSequential(nn.Sequential):
    def forward(self, x, y):
        for module in self:
            x = module(x, y)
        return x
    
        


# modified version of https://github.com/tonyduan/normalizing-flows/blob/master/src/flows.py
class NSF_AR_Block(nn.Module):
    """
    Neural spline flow, auto-regressive.

    [Durkan et al. 2019]
    """
    def __init__(self, in_dim, hidden_layer, n_nodes, activation = 'tanh', 
                 cond_dim=6, K = 5, B = 1, device='cuda'):
        super().__init__()
        
        self.register_buffer('base_dist_mean', torch.zeros(in_dim))
        self.register_buffer('base_dist_var', torch.ones(in_dim))
        self.device = device
        
        if activation == 'relu':
            activation_fn = nn.ReLU()
        elif activation == 'tanh':
            activation_fn = nn.Tanh()
        elif activation == 'gelu':
            activation_fn = nn.GELU()
        else:
            raise ValueError('Check activation function.')
        
        self.device = device
        
        self.dim = in_dim
        self.K = K
        self.B = B
        self.init_param = nn.Parameter(torch.Tensor(3 * K - 1))
        
        self.net = nn.ModuleList()
        for i in range(1, in_dim):
            self.net += [self.build_base_net(i, K, hidden_layer, 
                        n_nodes, activation_fn, cond_dim)]
            
        self.net.to(device)
        self.reset_parameters()
        
    def build_base_net(self, in_dim, K, hidden_layer, n_nodes, activation_fn, cond_dim):
        net = []
        net += [cond_FCL(in_dim, n_nodes, cond_dim, activation_fn)]
        for _ in range(1, hidden_layer - 1):
            net += [cond_FCL(n_nodes, n_nodes, cond_dim,activation_fn)]
        net += [cond_FCL(n_nodes, 3 * K - 1, cond_dim, activation_fn=(lambda x: x))]
        return CondSequential(*net)
    
    def reset_parameters(self):
        init.uniform_(self.init_param, - 1 / 2, 1 / 2).to(self.device)

    def forward(self, x, cond=None):
        
        z = torch.zeros_like(x, device=self.device)
        log_det = torch.zeros(z.shape[0],z.shape[1], device=self.device)
        for i in range(self.dim):
            if i == 0:
                init_param = self.init_param.expand(x.shape[0], 3 * self.K - 1)
                W, H, D = torch.split(init_param, self.K, dim = 1)
            else:
                out = self.net[i - 1](x[:, :i], cond)
                W, H, D = torch.split(out, self.K, dim = 1)
            W, H = torch.softmax(W, dim = 1), torch.softmax(H, dim = 1)
            W, H = 2 * self.B * W, 2 * self.B * H
            D = F.softplus(D)
            z[:, i], ld = unconstrained_rational_quadratic_spline(
                x[:, i], W, H, D, inverse=False, tail_bound=self.B, device=self.device)
            log_det[:,i] = ld
        
        return z, log_det

    def inverse(self, z, cond=None):
   
        x = torch.zeros_like(z, device=self.device)
        log_det = torch.zeros(x.shape[0],x.shape[1], device=self.device)
        for i in range(self.dim):
            if i == 0:
                init_param = self.init_param.expand(x.shape[0], 3 * self.K - 1)
                W, H, D = torch.split(init_param, self.K, dim = 1)
            else:
                out = self.net[i - 1](x[:, :i],cond)
                W, H, D = torch.split(out, self.K, dim = 1)
            W, H = torch.softmax(W, dim = 1), torch.softmax(H, dim = 1)
            W, H = 2 * self.B * W, 2 * self.B * H
            D = F.softplus(D)
            x[:, i], ld = unconstrained_rational_quadratic_spline(
                z[:, i], W, H, D, inverse = True, tail_bound = self.B, device = self.device)
            log_det[:,i] = ld
            
        return x, log_det
    
    
class NSF_AR(nn.Module):
    def __init__(self, n_blocks, in_dim, hidden_layer, n_nodes, 
                 cond_dim=6, K = 10, B=3., activation='tanh', 
                 batch_norm=True, prior = BoxUniform,
                 device='cuda', epsilon = 1e-4):
        super().__init__()
        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(in_dim))
        self.register_buffer('base_dist_var', torch.ones(in_dim))
        self.device = device
        
        self.condition_shape = torch.tensor([1,cond_dim])
        
        self.B = B
        self.reversed = False
        
        # hack in prior, bettor solution TBA
        self.prior = BoxUniform(low=-torch.ones(cond_dim)+epsilon, high=torch.ones(cond_dim)-epsilon, device=device)
        
        # construct model
        modules = []
        for i in range(n_blocks):
            modules += [NSF_AR_Block(in_dim=in_dim, hidden_layer=hidden_layer, 
                                     n_nodes=n_nodes, activation=activation, 
                                     cond_dim=cond_dim, device=device,
                                     B=B, K=K)]
            modules += batch_norm * [BatchNorm(in_dim)]
        
        self.condition_shape = torch.Size((cond_dim,))
        self.net = FlowSequential(*modules).to(device)
        
    @property
    def base_dist(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var)

    def forward(self, x, cond=None):
        x, logp = self.net(x, cond)
        return x, logp
    
    def inverse(self, x, cond=None):
        x, logp = self.net.inverse(x, cond)
        return x, logp
    
    def log_prob(self, x, condition=None):
        xshape = x.shape
        if len(xshape) > 2:
            x = x.reshape(xshape[0] * xshape[1], xshape[2])
        elif len(xshape) > 3:
            raise ValueError(f"Shape of x is {x.shape} but is expected to be (sample_shape, batch_shape, event_shape) or (batch_shape, event_shape)") 
        # only there to handle weird sbi package stuff
        
        s, p = self.forward(x, condition)
        
        if len(xshape) > 2:
            p = p.reshape(xshape[0], xshape[1], xshape[2])
            s = s.reshape(xshape[0], xshape[1], xshape[2])
            
        p = p.sum(-1) + self.base_dist.log_prob(s).sum(-1)

        return p
    
    def build_posterior(self, sample_kwargs = None):
        self.net.eval()
        self.posterior = get_nle_posterior(
                likelihood_estimator=self,
                prior=self.prior,
                sample_kwargs = sample_kwargs
                )
        self.reversed = True
            

    def sample(self, num_samples, x):
        if self.reversed:
            return self.posterior.sample((num_samples,), x, show_progress_bars=False)    
        else:
            u = self.base_dist.sample((num_samples,)).to(self.device)
            labels = x.repeat(num_samples,1).to(self.device)
            return self.net.inverse(u, labels)[0]

    def loss(self, x, cond=None):
        u, sum_log_abs_det_jacobians = self.forward(x, cond)
        return - torch.sum(self.base_dist.log_prob(u) + sum_log_abs_det_jacobians, dim=1)

    
    
    
class MAF(nn.Module):
    def __init__(self, n_blocks, in_dim, hidden_layer, n_nodes, 
                 cond_dim=6, activation='relu', input_order='sequential', 
                 batch_norm=True, device='cuda', epsilon = 1e-4):
        super().__init__()
        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(in_dim))
        self.register_buffer('base_dist_var', torch.ones(in_dim))
        self.device = device
        self.condition_shape = torch.Size((cond_dim,))

        # Hack in prior, better solution TBA
        self.prior = BoxUniform(low=-torch.ones(cond_dim)+epsilon, high=torch.ones(cond_dim)-epsilon, device=device)
        
        # construct model
        modules = []
        self.input_degrees = None
        for i in range(n_blocks):
            modules += [MADE(in_dim, n_nodes, hidden_layer, cond_dim, activation, input_order, self.input_degrees)]
            self.input_degrees = modules[-1].input_degrees.flip(0)
            modules += batch_norm * [BatchNorm(in_dim)]

        
        self.net = FlowSequential(*modules).to(device)
        self.reversed = False

    @property
    def base_dist(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var)

    def forward(self, x, cond=None):
        return self.net(x, cond)
    
    def inverse(self, x, cond=None):
        return self.net.inverse(x, cond)
    
    def log_prob(self, x, condition=None):
        xshape = x.shape
        if len(xshape) > 2:
            x = x.reshape(xshape[0] * xshape[1], xshape[2])
        elif len(xshape) > 3:
            raise ValueError(f"Shape of x is {x.shape} but is expected to be (sample_shape, batch_shape, event_shape) or (batch_shape, event_shape)") 
        s, p = self.forward(x, condition)
        
        if len(xshape) > 2:
            p = p.reshape(xshape[0], xshape[1], xshape[2])
            s = s.reshape(xshape[0], xshape[1], xshape[2])
            
        p = p.sum(-1) + self.base_dist.log_prob(s).sum(-1)

        return p
    
    def build_posterior(self, sample_kwargs = None):
        self.net.eval()
        self.posterior = get_nle_posterior(
                likelihood_estimator=self,
                prior=self.prior,
                sample_kwargs = sample_kwargs
                )
        self.reversed = True
            

    def sample(self, num_samples, x, sample_kwargs = None):
        if self.reversed:
            return self.posterior.sample((num_samples,), x, show_progress_bars=False)    
        else:
            u = self.base_dist.sample((num_samples,)).to(self.device)
            labels = x.repeat(num_samples,1).to(self.device)
            return self.net.inverse(u, labels)[0]

    def loss(self, x, cond=None):
        u, sum_log_abs_det_jacobians = self.forward(x, cond)
        return - torch.sum(self.base_dist.log_prob(u) + sum_log_abs_det_jacobians, dim=1)
