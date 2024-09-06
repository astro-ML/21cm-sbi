import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from made_backbone import MADE, BatchNorm, FlowSequential
import torch.nn.init as init
from cl_models import unconstrained_rational_quadratic_spline
from summary_models import FCNN
from math import sqrt


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
                 cond_dim=6, K = 5, B = 1., device='cuda'):
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
                 batch_norm=True, device='cuda'):
        super().__init__()
        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(in_dim))
        self.register_buffer('base_dist_var', torch.ones(in_dim))
        self.device = device
        self.cond_dim = cond_dim
        self.B = B
        
        # construct model
        modules = []
        for i in range(n_blocks):
            modules += [NSF_AR_Block(in_dim=in_dim, hidden_layer=hidden_layer, 
                                     n_nodes=n_nodes, activation=activation, 
                                     cond_dim=cond_dim, device=device,
                                     B=B, K=K)]
            modules += batch_norm * [BatchNorm(in_dim)]
        self.net = FlowSequential(*modules).to(device)
        
    @property
    def base_dist(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var)

    def forward(self, x, cond=None):
        # rescale x to (-B,B)
        #x -= self.B
        x, logp = self.net(x, cond)
        return x, logp
    
    def inverse(self, x, cond=None):
        # rescale x to (-B,B)
        #x -= self.B
        x, logp = self.net.inverse(x, cond)
        return x, logp

    @torch.no_grad()
    def sample(self, num_samples, x):
        self.net.eval()
        u = self.base_dist.sample((num_samples,)).to(self.device)
        labels = x.repeat(num_samples,1).to(self.device)
        return self.net.inverse(u, labels)

    def loss(self, x, cond=None):
        u, sum_log_abs_det_jacobians = self.forward(x, cond)
        return - torch.sum(self.base_dist.log_prob(u) + sum_log_abs_det_jacobians, dim=1)

        
    
    
    
class MAF(nn.Module):
    def __init__(self, n_blocks, in_dim, hidden_layer, n_nodes, 
                 cond_dim=6, activation='relu', input_order='sequential', 
                 batch_norm=True, device='cuda'):
        super().__init__()
        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(in_dim))
        self.register_buffer('base_dist_var', torch.ones(in_dim))
        self.device = device
        self.cond_dim = cond_dim

        # construct model
        modules = []
        self.input_degrees = None
        for i in range(n_blocks):
            modules += [MADE(in_dim, n_nodes, hidden_layer, cond_dim, activation, input_order, self.input_degrees)]
            self.input_degrees = modules[-1].input_degrees.flip(0)
            modules += batch_norm * [BatchNorm(in_dim)]

        
        self.net = FlowSequential(*modules).to(device)

    @property
    def base_dist(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var)

    def forward(self, x, cond=None):
        return self.net(x, cond)
    
    def inverse(self, x, cond=None):
        return self.net.inverse(x, cond)

    @torch.no_grad()
    def sample(self, num_samples, x):
        self.net.eval()
        u = self.base_dist.sample((num_samples,)).to(self.device)
        labels = x.repeat(num_samples,1).to(self.device)
        return self.net.inverse(u, labels)

    def loss(self, x, cond=None):
        u, sum_log_abs_det_jacobians = self.forward(x, cond)
        return - torch.sum(self.base_dist.log_prob(u) + sum_log_abs_det_jacobians, dim=1)
