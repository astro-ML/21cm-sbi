import torch
from torch import nn
import FrEIA.framework as Ff
import FrEIA.modules as Fm
from typing import Callable, List, Tuple
import torch.nn.functional as F
from summary_models import FCNN
import numpy as np
from typing import Dict, Any
from utility import get_nle_posterior
from sbi.utils import BoxUniform

DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3

# modified version of https://github.com/astro-ML/21cm_pie

# best performacne with hiddlen+layer = 1, n_nodes = 128
class RNVP(nn.Module):
    def __init__(self, in_dim: int, n_blocks: int, n_nodes: int, cond_dim: int = 6,
                 hidden_layer = 1, batch_norm: bool = False, 
                 activation = 'relu', device = 'cuda'):
        super().__init__()
        self.in_dim = in_dim
        self.n_blocks = n_blocks
        self.n_nodes = n_nodes
        self.cond_dim = cond_dim
        self.batch_norm = batch_norm
        self.reversed = False
        self.device = device
        
        # setup activation
        if activation == 'relu':
            activation_fn = nn.ReLU()
        elif activation == 'tanh':
            activation_fn = nn.Tanh()
        elif activation == 'gelu':
            activation_fn = nn.GELU()
        else:
            raise ValueError('Check activation function.')
        
        self.net = self.model(in_dim, n_blocks, n_nodes, cond_dim, hidden_layer, activation_fn).to(device)
        
        # Hack in prior, bettert oslution TBA
        self.prior = BoxUniform(low=torch.zeros(in_dim), high=torch.ones(in_dim), device=device)
        
    def model(self, n_dim: int, n_blocks: int, n_nodes: int, cond_dims: int, hidden_layer: int,
              activation_fn: Callable) -> Ff.SequenceINN:
        
        """
            Constructs the flow model.
            Batchnorm seeems to worsen the performance by a huge margin, keep it deactivated.
            (Probably broken because of how conditioning is done in the high-level network builder)

        Args:
            n_dim (int): The dimensionality of the input.
            n_blocks (int): The number of blocks in the model.
            n_nodes (int): The number of nodes in the subnet.

        """

        def subnet_fc(dims_in: int, dims_out: int) -> nn.Sequential:
            layers = []
            if self.batch_norm:
                layers += [nn.Linear(dims_in, n_nodes), activation_fn, nn.BatchNorm1d(n_nodes)]
                for _ in range(hidden_layer-1):
                    layers += [nn.Linear(n_nodes, n_nodes), activation_fn, nn.BatchNorm1d(n_nodes)]
            else:
                layers += [nn.Linear(dims_in, n_nodes), activation_fn]
                for _ in range(hidden_layer-1):
                    layers += [nn.Linear(n_nodes, n_nodes), activation_fn]
            layers += [nn.Linear(n_nodes, dims_out)]
            return nn.Sequential(*layers)
        
        flow = Ff.SequenceINN(n_dim)
        permute_soft = True if n_dim != 1 else False
        for k in range(n_blocks):
            flow.append(Fm.AllInOneBlock, cond=0, cond_shape=([cond_dims]),
                        subnet_constructor=subnet_fc, permute_soft=permute_soft)
        return flow
    
    def forward(self, x, cond):
        return self.net(x, c=[cond])
    
    def log_prob(self, x, condition=None):
        xshape = x.shape
        if len(xshape) > 2:
            x = x.reshape(xshape[0] * xshape[1], xshape[2])
        elif len(xshape) > 3:
            raise ValueError(f"Shape of x is {x.shape} but is expected to be (sample_shape, batch_shape, event_shape) or (batch_shape, event_shape)") 
        # only there to handle weird sbi package stuff
        
        z, jac = self.net(x, c=[condition], rev=False)
        p = 0.5*torch.sum(z**2,1) - jac
        
        if len(xshape) > 2:
            p = p.reshape(xshape[0], xshape[1])
        
        p = p / self.in_dim

        return - p

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
            z = torch.randn(num_samples, self.in_dim).to(self.device)
            # add rejection sampling TODO
            samples,_ = self.net(z, c = [x.repeat((num_samples,1)).to(self.device)], rev=True)
            return samples
    
    def loss(self, x: torch.FloatTensor, cond: torch.FloatTensor) -> torch.FloatTensor:
        z, jac = self.net(x, c=[cond], rev=False)
        loss = 0.5*torch.sum(z**2,1) - jac
        loss = loss.mean() / self.in_dim
        return loss

# modified version of https://github.com/tonyduan/normalizing-flows/blob/master/src/flows.py

class NSF_CL(nn.Module):
    """
    Neural spline flow, coupling layer.

    [Durkan et al. 2019]
    """
    def __init__(self, dim, K = 5, B = 3, hidden_dim = 8, base_network = FCNN):
        super().__init__()
        self.dim = dim
        self.K = K
        self.B = B
        self.f1 = base_network(dim // 2, (3 * K - 1) * dim // 2, hidden_dim)
        self.f2 = base_network(dim // 2, (3 * K - 1) * dim // 2, hidden_dim)

    def forward(self, x):
        log_det = torch.zeros(x.shape[0])
        lower, upper = x[:, :self.dim // 2], x[:, self.dim // 2:]
        out = self.f1(lower).reshape(-1, self.dim // 2, 3 * self.K - 1)
        W, H, D = torch.split(out, self.K, dim = 2)
        W, H = torch.softmax(W, dim = 2), torch.softmax(H, dim = 2)
        W, H = 2 * self.B * W, 2 * self.B * H
        D = F.softplus(D)
        upper, ld = unconstrained_rational_quadratic_spline(
            upper, W, H, D, inverse=False, tail_bound=self.B)
        log_det += torch.sum(ld, dim = 1)
        out = self.f2(upper).reshape(-1, self.dim // 2, 3 * self.K - 1)
        W, H, D = torch.split(out, self.K, dim = 2)
        W, H = torch.softmax(W, dim = 2), torch.softmax(H, dim = 2)
        W, H = 2 * self.B * W, 2 * self.B * H
        D = F.softplus(D)
        lower, ld = unconstrained_rational_quadratic_spline(
            lower, W, H, D, inverse=False, tail_bound=self.B)
        log_det += torch.sum(ld, dim = 1)
        return torch.cat([lower, upper], dim = 1), log_det

    def inverse(self, z):
        log_det = torch.zeros(z.shape[0])
        lower, upper = z[:, :self.dim // 2], z[:, self.dim // 2:]
        out = self.f2(upper).reshape(-1, self.dim // 2, 3 * self.K - 1)
        W, H, D = torch.split(out, self.K, dim = 2)
        W, H = torch.softmax(W, dim = 2), torch.softmax(H, dim = 2)
        W, H = 2 * self.B * W, 2 * self.B * H
        D = F.softplus(D)
        lower, ld = unconstrained_rational_quadratic_spline(
            lower, W, H, D, inverse=True, tail_bound=self.B)
        log_det += torch.sum(ld, dim = 1)
        out = self.f1(lower).reshape(-1, self.dim // 2, 3 * self.K - 1)
        W, H, D = torch.split(out, self.K, dim = 2)
        W, H = torch.softmax(W, dim = 2), torch.softmax(H, dim = 2)
        W, H = 2 * self.B * W, 2 * self.B * H
        D = F.softplus(D)
        upper, ld = unconstrained_rational_quadratic_spline(
            upper, W, H, D, inverse = True, tail_bound = self.B)
        log_det += torch.sum(ld, dim = 1)
        return torch.cat([lower, upper], dim = 1), log_det
    
    
    
    
def unconstrained_rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    tails="linear",
    tail_bound=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
    enable_identity_init=False,
    device='cpu'
):
    offset = 0.5
    inside_interval_mask = (inputs >= -tail_bound+offset) & (inputs <= tail_bound+offset)
    outside_interval_mask = ~inside_interval_mask

    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    if tails == "linear":
        unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1))
        constant = np.log(np.exp(1 - min_derivative) - 1)
        unnormalized_derivatives[..., 0] = constant
        unnormalized_derivatives[..., -1] = constant

        outputs[outside_interval_mask] = inputs[outside_interval_mask]
        logabsdet[outside_interval_mask] = 0
    else:
        raise RuntimeError("{} tails are not implemented.".format(tails))

    if torch.any(inside_interval_mask):
        (
            outputs[inside_interval_mask],
            logabsdet[inside_interval_mask],
        ) = rational_quadratic_spline(
            inputs=inputs[inside_interval_mask],
            unnormalized_widths=unnormalized_widths[inside_interval_mask, :],
            unnormalized_heights=unnormalized_heights[inside_interval_mask, :],
            unnormalized_derivatives=unnormalized_derivatives[inside_interval_mask, :],
            inverse=inverse,
            left=-tail_bound+ offset,
            right=tail_bound+offset,
            bottom=-tail_bound+offset,
            top=tail_bound+offset,
            min_bin_width=min_bin_width,
            min_bin_height=min_bin_height,
            min_derivative=min_derivative,
            enable_identity_init=enable_identity_init,
            device=device
        )

    return outputs, logabsdet


def rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    left=0.0,
    right=1.0,
    bottom=0.0,
    top=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
    enable_identity_init=False,
    device='cpu',
):
    if torch.min(inputs) < left or torch.max(inputs) > right:
        raise ValueError("Input outside of interval")

    num_bins = unnormalized_widths.shape[-1]

    if min_bin_width * num_bins > 1.0:
        raise ValueError("Minimal bin width too large for the number of bins")
    if min_bin_height * num_bins > 1.0:
        raise ValueError("Minimal bin height too large for the number of bins")

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    if enable_identity_init: #flow is the identity if initialized with parameters equal to zero
        beta = np.log(2) / (1 - min_derivative)
    else: #backward compatibility
        beta = 1
    derivatives = min_derivative + F.softplus(unnormalized_derivatives, beta=beta)

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode="constant", value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    if inverse:
        bin_idx = searchsorted(cumheights, inputs)[..., None]
    else:
        bin_idx = searchsorted(cumwidths, inputs)[..., None]

    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]

    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]

    input_heights = heights.gather(-1, bin_idx)[..., 0]

    if inverse:
        a = (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        ) + input_heights * (input_delta - input_derivatives)
        b = input_heights * input_derivatives - (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        )
        c = -input_delta * (inputs - input_cumheights)

        discriminant = b.pow(2) - 4 * a * c
        assert (discriminant >= 0).all()

        root = (2 * c) / (-b - torch.sqrt(discriminant))
        # root = (- b + torch.sqrt(discriminant)) / (2 * a)
        outputs = root * input_bin_widths + input_cumwidths

        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
            * theta_one_minus_theta
        )
        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * root.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - root).pow(2)
        )
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

        return outputs, -logabsdet
    else:
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_heights * (
            input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta
        )
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
            * theta_one_minus_theta
        )
        outputs = input_cumheights + numerator / denominator

        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * theta.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - theta).pow(2)
        )
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

        return outputs, logabsdet
    
    
def searchsorted(bin_locations, inputs, eps=1e-6):
    bin_locations[..., -1] += eps
    return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1