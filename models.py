import torch
from torch import nn
import FrEIA.framework as Ff
import FrEIA.modules as Fm
from typing import Callable


class flow_net(nn.Module):
    def __init__(self, in_dim: int, n_blocks: int, n_nodes: int, cond_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.n_blocks = n_blocks
        self.n_nodes = n_nodes
        self.cond_dim = cond_dim
        self.Model = self.model(in_dim, n_blocks, n_nodes, cond_dim)
        
        
        
    def model(self, n_dim: int, n_blocks: int, n_nodes: int, cond_dims: int) -> Ff.SequenceINN:
        """
        Constructs the flow model.

        Args:
            n_dim (int): The dimensionality of the input.
            n_blocks (int): The number of blocks in the model.
            n_nodes (int): The number of nodes in the subnet.
            cond_dims (tuple): The dimensions of the conditional input.

        Returns:
            Ff.SequenceINN: The constructed flow model.
        """
        def subnet_fc(dims_in: int, dims_out: int) -> nn.Sequential:
            return nn.Sequential(nn.Linear(dims_in, n_nodes), nn.ReLU(),
                                 nn.Linear(n_nodes, dims_out))
        
        flow = Ff.SequenceINN(n_dim)
        permute_soft = True if n_dim != 1 else False
        for k in range(n_blocks):
            flow.append(Fm.AllInOneBlock, cond=0, cond_shape=([cond_dims]),
                        subnet_constructor=subnet_fc, permute_soft=permute_soft)
        return flow
    
    def forward(self, lab, cond):
        return self.Model(lab, c=[cond])
    
    def sample(self, num_sampels: int, c: torch.FloatTensor, z: torch.FloatTensor = None,
               device: str = 'cuda') -> torch.FloatTensor:
        if z is None:
            z = torch.randn(num_sampels, self.in_dim).to(device)
        return self.Model(z, c = [c.repeat((num_sampels,1)).to(device)], rev=True)
    
    def loss(self, lab: torch.FloatTensor, cond: torch.FloatTensor, loss: Callable) -> torch.FloatTensor:
        z, jac = self.Model(lab, c=[cond], rev=False)
        loss = 0.5*torch.sum(z**2,1) - jac
        loss = loss.mean() / self.in_dim
        return loss
        
        

class Summary_net_lc(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=32, kernel_size=(3, 3, 102), stride=(1, 1, 102)), 
            nn.ReLU(),
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(3, 3, 2)), 
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 1)),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 2)), 
            nn.ReLU(),
            nn.ZeroPad2d((1, 1, 0, 0)),  # Padding for width and height only, no depth padding
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 2)), 
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 1)),
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(3, 3, 2)), 
            nn.ReLU(),
            nn.ZeroPad2d((1, 1, 0, 0)),  # Padding for width and height only, no depth padding
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(3, 3, 2)), 
            nn.ReLU()
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 6),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = nn.functional.adaptive_avg_pool3d(x, (1, 1, 1)).view(x.size(0), -1)
        x = self.fc_layers(x)
        return x
    
class Summary_net_lc_lil(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=32, kernel_size=(3, 3, 51), stride=(1, 1, 51)), 
            nn.ReLU(),
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(3, 3, 2)), 
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 1)),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 2)), 
            nn.ReLU(),
            nn.ZeroPad2d((1, 1, 0, 0)),  # Padding for width and height only, no depth padding
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 2)), 
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 1)),
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(3, 3, 2)), 
            nn.ReLU(),
            nn.ZeroPad2d((1, 1, 0, 0)),  # Padding for width and height only, no depth padding
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(3, 3, 2)), 
            nn.ReLU()
        )
        self.pooling = nn.Sequential(
            nn.MaxPool3d(kernel_size=3, stride=3, padding=1),
            nn.AvgPool3d(kernel_size = 4, stride=1, padding=0)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 6),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.pooling(x)
        x = torch.flatten(x, 1, -1)
        #x = nn.functional.adaptive_avg_pool3d(x, (1, 1, 1)).view(x.size(0), -1)
        x = self.fc_layers(x)
        return x
    
class Summary_net_lc_smol(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=48, kernel_size=(3, 3, 20), stride=(1, 1, 20)), 
            nn.GELU(),
            nn.BatchNorm3d(48),
            nn.Conv3d(in_channels=48, out_channels=48, kernel_size=(3, 3, 3)), 
            nn.GELU(),
            nn.BatchNorm3d(48),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Conv3d(in_channels=48, out_channels=64, kernel_size=(3, 3, 3)), 
            nn.GELU(),
            nn.BatchNorm3d(64),
            nn.ZeroPad2d((1, 1, 0, 0)),  # Padding for width and height only, no depth padding
            nn.Conv3d(in_channels=64, out_channels=96, kernel_size=(3, 3, 3)), 
            nn.GELU(),
            nn.BatchNorm3d(96),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Conv3d(in_channels=96, out_channels=128, kernel_size=(3, 3, 2)), 
            nn.GELU(),
            nn.BatchNorm3d(128),
        )
        self.pooling = nn.Sequential(
            nn.MaxPool3d(kernel_size=3, stride=3, padding=1),
            nn.AvgPool3d(kernel_size=2, stride=2, padding=0)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(128, 96),  # Adjusted input dimension
            nn.Dropout(0.1),
            nn.GELU(),
            nn.Linear(96, 64),
            nn.Dropout(0.1),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.Dropout(0.1),
            nn.GELU(),
            nn.Linear(32, 6),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.pooling(x)
        x = torch.flatten(x, 1, -1)
        x = self.fc_layers(x)
        return x
    
class Summary_net_lc_super_smol(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=48, kernel_size=(3, 3, 10), stride=(1, 1, 10)), 
            nn.GELU(),
            nn.BatchNorm3d(48),
            nn.Conv3d(in_channels=48, out_channels=48, kernel_size=(3, 3, 3)), 
            nn.GELU(),
            nn.BatchNorm3d(48),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Conv3d(in_channels=48, out_channels=64, kernel_size=(3, 3, 3)), 
            nn.GELU(),
            nn.BatchNorm3d(64),
            nn.ZeroPad2d((1, 1, 0, 0)),  # Padding for width and height only, no depth padding
            nn.Conv3d(in_channels=64, out_channels=96, kernel_size=(3, 3, 3)), 
            nn.GELU(),
            nn.BatchNorm3d(96),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Conv3d(in_channels=96, out_channels=96, kernel_size=(3, 3, 3)), 
            nn.GELU(),
            nn.BatchNorm3d(96),
        )
        self.pooling = nn.Sequential(
            nn.MaxPool3d(kernel_size=(1,1,4), stride=(1,1,4), padding=0),
            nn.AvgPool3d(kernel_size=2, stride=2, padding=0)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(96, 96),  # Adjusted input dimension
            nn.Dropout(0.1),
            nn.GELU(),
            nn.Linear(96, 64),
            nn.Dropout(0.1),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.Dropout(0.1),
            nn.GELU(),
            nn.Linear(32, 6),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.pooling(x)
        x = torch.flatten(x, 1, -1)
        x = self.fc_layers(x)
        return x

class Summary_net_lc_super_smol_inv(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(6, 32),
            nn.GELU(),
            nn.Linear(32, 64),
            nn.GELU(),
            nn.Linear(64, 96),
            nn.GELU(),
            nn.Linear(96, 96),
            nn.GELU(),
        )
        self.unpooling = nn.Sequential(
            nn.Upsample(scale_factor=(2, 2, 2), mode='nearest'),
            nn.Conv3d(in_channels=96, out_channels=96, kernel_size=(3, 3, 3)),
            nn.GELU(),
            nn.BatchNorm3d(96),
            nn.Upsample(scale_factor=(1, 1, 4), mode='nearest'),
            nn.Conv3d(in_channels=96, out_channels=64, kernel_size=(3, 3, 3)),
            nn.GELU(),
            nn.BatchNorm3d(64),
            nn.Upsample(scale_factor=(2, 2, 2), mode='nearest'),
            nn.Conv3d(in_channels=64, out_channels=48, kernel_size=(3, 3, 3)),
            nn.GELU(),
            nn.BatchNorm3d(48),
            nn.Upsample(scale_factor=(2, 2, 2), mode='nearest'),
            nn.Conv3d(in_channels=48, out_channels=48, kernel_size=(3, 3, 3)),
            nn.GELU(),
            nn.BatchNorm3d(48),
        )
        self.conv_layers = nn.Sequential(
            nn.Upsample(scale_factor=(2, 2, 2), mode='nearest'),
            nn.Conv3d(in_channels=48, out_channels=1, kernel_size=(3, 3, 10), stride=(1, 1, 10)),
            nn.GELU()
        )

    def forward(self, x):
        x = self.fc_layers(x)
        x = x.view(x.size(0), -1, 1, 1, 1)
        x = self.unpooling(x)
        x = self.conv_layers(x)
        return x

class Summary_net_lc_benedikt(nn.Module):
    
    def __init__(self, in_ch=1, ch=32, N_parameter=6,sigmoid=False):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, ch, kernel_size=(3,3,51), bias=True, stride=(1,1,51))
        self.conv2 = nn.Conv3d(ch, ch, kernel_size=(3,3,2), bias=True)
        self.conv3 = nn.Conv3d(ch, 2*ch, kernel_size=(3,3,2), bias=True)
        self.conv3_zero = nn.Conv3d(2*ch, 2*ch, kernel_size=(3,3,2), bias=True, padding=(1,1,0))
        self.conv4 = nn.Conv3d(2*ch, 4*ch, kernel_size=(3,3,2), bias=True)
        self.conv4_zero = nn.Conv3d(4*ch, 4*ch, kernel_size=(3,3,2), bias=True, padding=(1,1,0))
        self.max = nn.MaxPool3d(kernel_size=(2,2,1), stride=(2,2,1))
        self.avg = nn.AvgPool3d(kernel_size = (13,13,18))
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(128,128, bias=True)
        self.linear2 = nn.Linear(128,96, bias=True)
        self.linear3 = nn.Linear(96,64, bias=True)
        self.out = nn.Linear(64, N_parameter, bias=True)
        self.sigmoid = sigmoid
    
    def forward(self,x):
        x = nn.ReLU()(self.conv1(x))
        x = nn.ReLU()(self.conv2(x))
        x = self.max(x)
        x = nn.ReLU()(self.conv3(x))
        x = nn.ReLU()(self.conv3_zero(x))
        x = self.max(x)
        x = nn.ReLU()(self.conv4(x))
        x = nn.ReLU()(self.conv4_zero(x))
        x = self.avg(x)
        x = self.flatten(x)
        x = nn.ReLU()(self.linear1(x))
        x = nn.ReLU()(self.linear2(x))
        x = nn.ReLU()(self.linear3(x))
        x = self.out(x)
        if self.sigmoid:
            x = nn.Sigmoid()(self.out(x))
        else:
            x = self.out(x)

        return x


class Summary_net_2dps(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_conv_stack = nn.Sequential(
            torch.nn.Conv2d(10, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(64, 128, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,32),
            nn.ReLU(),
            nn.Linear(32,6),
            nn.Sigmoid()
        )

    def forward(self, x):
        
        x = self.linear_conv_stack(x)
        x = self.flatten(x)
        x = self.linear_stack(x)
        return x

class cINN_net(nn.Module):
    def __init__(self):
        super().__init__()