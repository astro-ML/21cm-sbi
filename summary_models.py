import torch
import torch.nn as nn
import torch.nn.functional as F
import FrEIA.framework as Ff
import FrEIA.modules as Fm
from ar_models import CLinear, CondSequential
import math


class cConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, stride=stride)
        self.cond_scale = nn.Linear(cond_dim, out_channels)
        self.cond_shift = nn.Linear(cond_dim, out_channels)

    def forward(self, x, cond):
        conv_out = self.conv(x)
        scale = self.cond_scale(cond).unsqueeze(-1)
        shift = self.cond_shift(cond).unsqueeze(-1)
        out = conv_out * (1 + scale) + shift
        return out


class cConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride)
        self.cond_scale = nn.Linear(cond_dim, out_channels)
        self.cond_shift = nn.Linear(cond_dim, out_channels)

    def forward(self, x, cond):
        conv_out = self.conv(x)
        scale = self.cond_scale(cond).unsqueeze(-1).unsqueeze(-1)
        shift = self.cond_shift(cond).unsqueeze(-1).unsqueeze(-1)
        out = conv_out * (1 + scale) + shift
        return out

class cConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, stride=stride)
        self.cond_scale = nn.Linear(cond_dim, out_channels)
        self.cond_shift = nn.Linear(cond_dim, out_channels)

    def forward(self, x, cond):
        conv_out = self.conv(x)
        scale = self.cond_scale(cond).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        shift = self.cond_shift(cond).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        out = conv_out * (1 + scale) + shift
        return out

# 35x35x252 = ([7*5]^2, [7*3*3*2*2])
class Summary_net_lc_master_stride_smoll(nn.Module):
    def __init__(self, num_features: int = 6):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(3, 3, 14), stride=(3, 3, 14))
        self.post1 = nn.Sequential(
            nn.GELU(),
            nn.BatchNorm3d(8),
        )
        
        self.conv2 = nn.Conv3d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.post2 = nn.Sequential(
            nn.GELU(),
            nn.BatchNorm3d(8),
        )
        self.conv3 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(3, 3, 3), stride=(3, 3, 3))
        self.post3 = nn.Sequential(
            nn.GELU(),
            nn.BatchNorm3d(16),
        )

        self.conv4 = nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.post4 = nn.Sequential(
            nn.GELU(),
            nn.BatchNorm3d(16),
        )
        self.conv5 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(2,2,3), stride=(2,2,3), padding=0)
        self.post5 = nn.Sequential(
            nn.GELU(),
            nn.BatchNorm3d(32),
        )

        self.conv8 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.post8 = nn.Sequential(
            nn.GELU(),
            nn.MaxPool3d(kernel_size=(1,1,2))
        )
        
        
        self.fc_layers = nn.Sequential(
            nn.Linear(32, 32),
            nn.GELU(),
            nn.Linear(32, 24),
            nn.GELU(),
            nn.Linear(24, 16),
            nn.GELU(),
            nn.Linear(16, num_features),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.post1(x)
        x = self.conv2(x)
        x = self.post2(x)
        x = self.conv3(x)
        x = self.post3(x)
        x = self.conv4(x)
        x = self.post4(x)
        x = self.conv5(x)
        x = self.post5(x)
        x = self.conv8(x)
        x = self.post8(x)
        x = torch.squeeze(x, (-3,-2,-1))
        x = self.fc_layers(x)
        return x 
    
class Summary_net_lc_master_pool_smoll(nn.Module):
    def __init__(self, num_features: int = 6,
                 channels1: int = 32,
                 channels3: int = 48,
                 channels4: int = 64,
                 channels5: int = 96,
                 hidden_dim1: int = 128,
                 hidden_dim2: int = 64,):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=channels1, kernel_size=3, stride=1, padding=1)
        self.post1 = nn.Sequential(
            nn.GELU(),
            nn.BatchNorm3d(channels1),
            nn.MaxPool3d((3,3,14))
        )
        
        self.conv3 = nn.Conv3d(in_channels=channels1, out_channels=channels3, kernel_size=3, stride=1, padding=1)
        self.post3 = nn.Sequential(
            nn.GELU(),
            nn.BatchNorm3d(channels3),
            nn.MaxPool3d((2,2,3))
        )

        self.conv5 = nn.Conv3d(in_channels=channels3, out_channels=channels4, kernel_size=3, stride=1, padding=1)
        self.post5 = nn.Sequential(
            nn.GELU(),
            nn.BatchNorm3d(channels4),
            nn.MaxPool3d((2,2,3))
        )

        self.conv7 = nn.Conv3d(in_channels=channels4, out_channels=channels5, kernel_size=3, stride=1, padding=1)
        self.post7 = nn.Sequential(
            nn.GELU(),
            nn.BatchNorm3d(channels5),
            nn.MaxPool3d(2)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(channels5, hidden_dim1),
            nn.GELU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.GELU(),
            nn.Linear(hidden_dim2, 16),
            nn.GELU(),
            nn.Linear(16, num_features),
            nn.Tanh()
        )
    
    def forward(self, x):

        x = self.conv1(x)
        x = self.post1(x)

        x = self.conv3(x)
        x = self.post3(x)

        x = self.conv5(x)
        x = self.post5(x)

        x = self.conv7(x)
        x = self.post7(x)

        x = torch.squeeze(x, (-4,-3,-2,-1))
        x = self.fc_layers(x)
        return x 
    

class Summary_net_lc_master_pool_smoll_cond(nn.Module):
    def __init__(self, num_features: int = 6):
        super().__init__()
        self.conv0 = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.post0 = nn.Sequential(
            nn.GELU(),
            nn.BatchNorm3d(8),
        )
        self.conv1 = nn.Conv3d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.post1 = nn.Sequential(
            nn.GELU(),
            nn.BatchNorm3d(8),
            nn.MaxPool3d((3,3,14))
        )
        
        self.conv2 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.post2 = nn.Sequential(
            nn.GELU(),
            nn.BatchNorm3d(16),
        )
        self.conv3 = nn.Conv3d(in_channels=16, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.post3 = nn.Sequential(
            nn.GELU(),
            nn.BatchNorm3d(24),
            nn.MaxPool3d((2,2,3))
        )

        self.conv4 = nn.Conv3d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.post4 = nn.Sequential(
            nn.GELU(),
            nn.BatchNorm3d(24),
        )
        self.conv5 = nn.Conv3d(in_channels=24, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.post5 = nn.Sequential(
            nn.GELU(),
            nn.BatchNorm3d(32),
            nn.MaxPool3d((2,2,3))
        )

        self.conv6 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.post6 = nn.Sequential(
            nn.GELU(),
            nn.BatchNorm3d(32),
        )
        self.conv7 = nn.Conv3d(in_channels=32, out_channels=48, kernel_size=3, stride=1, padding=1)
        self.post7 = nn.Sequential(
            nn.GELU(),
            nn.BatchNorm3d(48),
            nn.MaxPool3d(2)
        )

        self.conv8 = nn.Conv3d(in_channels=48, out_channels=48, kernel_size=1, stride=1, padding=0)
        self.post8 = nn.GELU()
        
        self.fc_layers = CondSequential(
            CLinear(48, 64, cond_dim=2, activation_fn=nn.GELU(),batch_norm=False),
            CLinear(64, 32, cond_dim=2, activation_fn=nn.GELU(),batch_norm=False),
            CLinear(32, 16, cond_dim=2, activation_fn=nn.GELU(),batch_norm=False),
            CLinear(16, num_features, cond_dim=2, activation_fn=nn.Identity(),batch_norm=False),
        )
    
    def forward(self, x, cond):
        x = self.conv0(x)
        x = self.post0(x)

        x = self.conv1(x)
        x = self.post1(x)

        x = self.conv2(x)
        x = self.post2(x)

        x = self.conv3(x)
        x = self.post3(x)

        x = self.conv4(x)
        x = self.post4(x)

        x = self.conv5(x)
        x = self.post5(x)

        x = self.conv6(x)
        x = self.post6(x)

        x = self.conv7(x)
        x = self.post7(x)

        x = self.conv8(x)
        x = self.post8(x)

        x = torch.squeeze(x, (-4,-3,-2,-1))
        x = self.fc_layers(x,cond)
        return x 


# 140x140x1000 = ([2*2*5*7]^2, [2*2*10*25])
class Summary_net_lc_master_stride(nn.Module):
    def __init__(self, num_features: int, cond_dim: int):
        super().__init__()
        self.conv1 = cConv3d(in_channels=1, out_channels=32, kernel_size=(7, 7, 25), stride=(7, 7, 25), cond_dim=cond_dim)
        self.post1 = nn.Sequential(
            nn.GELU(),
            nn.BatchNorm3d(32),
        )
        
        self.conv2 = cConv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, cond_dim=cond_dim)
        self.post2 = nn.GELU() 
        self.conv3 = cConv3d(in_channels=32, out_channels=32, kernel_size=(5, 5, 10), stride=(5, 5, 10), cond_dim=cond_dim)
        self.post3 = nn.Sequential(
            nn.GELU(),
            nn.BatchNorm3d(32),
        )

        self.conv4 = cConv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, cond_dim=cond_dim)
        self.post4 = nn.GELU()
        self.conv5 = cConv3d(in_channels=64, out_channels=64, kernel_size=2, stride=2, cond_dim=cond_dim)
        self.post5 = nn.Sequential(
            nn.GELU(),
            nn.BatchNorm3d(64),
        )

        self.conv6 = cConv3d(in_channels=64, out_channels=96, kernel_size=3, stride=1, padding=1, cond_dim=cond_dim)
        self.post6 = nn.GELU()
        self.conv7 = cConv3d(in_channels=96, out_channels=128, kernel_size=2, stride=2, cond_dim=cond_dim)
        self.post7 = nn.Sequential(
            nn.GELU(),
            nn.BatchNorm3d(128),
        )

        self.conv8 = cConv3d(in_channels=128, out_channels=128, kernel_size=2, stride=1, padding=0, cond_dim=cond_dim)
        self.post8 = nn.GELU()
        
        self.fc_layers = CondSequential(
            CLinear(128, 96, cond_dim, nn.GELU(), False),
            CLinear(96, 64, cond_dim, nn.GELU(), False),
            CLinear(64, 32, cond_dim, nn.GELU(), False),
            CLinear(32, num_features, cond_dim, nn.Tanh(), False),
        )
    
    def forward(self, x, cond):
        x = self.conv1(x, cond)
        x = self.post1(x)
        x = self.conv2(x, cond)
        x = self.post2(x)
        x = self.conv3(x, cond)
        x = self.post3(x)
        x = self.conv4(x, cond)
        x = self.post4(x)
        x = self.conv5(x, cond)
        x = self.post5(x)
        x = self.conv6(x, cond)
        x = self.post6(x)
        x = self.conv7(x, cond)
        x = self.post7(x)
        x = self.conv8(x, cond)
        x = self.post8(x)
        x = torch.squeeze(x, (-4,-3,-2,-1))
        x = self.fc_layers(x, cond)
        return x 
    
# 140x140x536 = ([2*2*5*7]^2, [2*2*2*67])
class Summary_net_lc_master_pool(nn.Module):
    def __init__(self, num_features: int, cond_dim: int):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=32, kernel_size=(3, 3, 67), stride=(1, 1, 67), padding=(1,1,0)), 
            nn.ReLU(),
            #nn.Max

            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(5, 5, 2), stride=(5, 5, 2)), 
            nn.ReLU(),

            nn.Conv3d(in_channels=64, out_channels=96, kernel_size=2, stride=2), 
            nn.ReLU(),

            nn.Conv3d(in_channels=96, out_channels=128, kernel_size=2, stride=2), 
            nn.ReLU(),
        )
        
        self.fc_layers = nn.Sequential(
            CLinear(128, 96, cond_dim, nn.GELU(), False),
            CLinear(96, 64, cond_dim, nn.GELU(), False),
            CLinear(64, 32, cond_dim, nn.GELU(), False),
            CLinear(32, num_features, cond_dim, nn.Tanh(), False),
        )
    
    def forward(self, x, cond):
        x = self.conv_layers(x)
        x = torch.squeeze(x, (-4,-3,-2,-1))
        x = self.fc_layers(x)
        return x 
    
# def __init__(self, in_dim, out_dim, cond_dim, activation_fn, batch_norm):

class Summary_net_linear_2dps(nn.Module):
    def __init__(self, num_features, cond_dim):
        super().__init__()
        self.fc_layers = CondSequential(
            CLinear(300, 100, cond_dim, nn.GELU(), False),
            CLinear(100, 50, cond_dim, nn.GELU(), False),
            CLinear(50, 20, cond_dim, nn.GELU(), False),
            CLinear(20, num_features, cond_dim, nn.Tanh(), False),
        )
    
    def forward(self, x, cond):
        xshape = torch.tensor(x.shape)
        x = x.reshape((*xshape[:-3], torch.prod(xshape[-3:])))
        x = self.fc_layers(x,cond)
        return x

# (batch_dim, channels, ps_dim, k_dim)
# __init__(self, in_channels, out_channels, cond_dim, kernel_size=3, padding=1, stride=1):
class Summary_net_convolution_2dps(nn.Module):
    def __init__(self, num_features, cond_dim):
        super().__init__()
        self.conv_layers = cConv2d(1, 10, 2, kernel_size=(3,1), padding=0, stride=1)
        self.post = nn.Sequential(
            nn.GELU(),
            nn.BatchNorm2d(10),
            nn.MaxPool2d(kernel_size=(1,10), stride=None, padding=0)
        )
        # (batch_dim, channels, 1, k_dim)
        self.fc_layers = CondSequential(
            CLinear(100, 50, cond_dim, nn.GELU(), False),
            CLinear(50, 20, cond_dim, nn.GELU(), False),
            CLinear(20, num_features, cond_dim, nn.Tanh(), False),
        )
    
    def forward(self, x, cond):
        xshape = torch.tensor(x.shape)
        x = x.reshape((*xshape[:-2], torch.prod(xshape[-2:]))).unsqueeze(1)
        x = self.conv_layers(x, cond)
        x = self.post(x)
        xshape = torch.tensor(x.shape)
        x = x.reshape((*xshape[:-3], torch.prod(xshape[-3:])))
        x = self.fc_layers(x, cond)
        return x
    
class Summary_net_linear_1dps(nn.Module):
    def __init__(self, num_features, cond_dim):
        super().__init__()
        self.fc_layers = CondSequential(
            CLinear(30, 30, cond_dim, nn.GELU(), False),
            CLinear(30, 10, cond_dim, nn.GELU(), False),
            CLinear(10, num_features, cond_dim, nn.Tanh(),False),
            
        )
    
    def forward(self, x, cond):
        xshape = torch.tensor(x.shape)
        x = x.reshape((*xshape[:-2], torch.prod(xshape[-2:])))
        x = self.fc_layers(x,cond)
        return x

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
            nn.Tanh()
        )
    
    def forward(self, x, cond):
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
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.pooling(x)
        x = torch.flatten(x, 1, -1)
        #x = nn.functional.adaptive_avg_pool3d(x, (1, 1, 1)).view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

# wants (28,28,680) input
class Summary_net_lc_smol(nn.Module):
    def __init__(self, in_channels = 1,
                 init_layers = {
            "layer_size1": 1,
            "channel1": 48,
            
            "layer_size2": 1,
            "channel2": 48,
            
            "layer_size3": 1,
            "channel3": 64,
            
            "layer_size4": 1,
            "channel4": 96,
        }):
        super().__init__()
        
        for j in range(1,5):
            setattr(self, f"layercount{j}", init_layers[f"layer_size{j}"])
            
            out_channels = init_layers[f"channel{j}"]
            # first layer
            if j == 1:
                kernel_size = torch.tensor([3,3,17])
                stride = torch.tensor([1,1,17])
                padding = torch.tensor([1,1,0])
            else:
                kernel_size = torch.tensor([3,3,3])
                padding = (kernel_size/2).to(torch.int32)
                stride = 1
                in_channels = init_layers[f"channel{j-1}"]
            setattr(self, f'conv{j}0', nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding = padding))
            setattr(self, f'bn{j}0', nn.BatchNorm3d(out_channels))
            setattr(self, f'relu{j}0', nn.GELU())
            in_channels = out_channels
            
            for i in range(1,getattr(self, f"layercount{j}")):
                setattr(self, f'conv{j}{i}', nn.Conv3d(in_channels, out_channels, kernel_size=(3,3,3), stride = (1,1,1), padding = 1))
                setattr(self, f'bn{j}{i}', nn.BatchNorm3d(out_channels))
                setattr(self, f'relu{j}{i}', nn.GELU())
            
            
            setattr(self, f"pool{j}", nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2)))

        
        self.fc_layers = nn.Sequential(
            nn.Linear(init_layers["channel4"], 96),  # Adjusted input dimension
            nn.Dropout(0.1),
            nn.GELU(),
            nn.Linear(96, 64),
            nn.Dropout(0.1),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.Dropout(0.1),
            nn.GELU(),
            nn.Linear(32, 6),
            nn.Tanh()
        )
    
    def forward(self, x):
        
        for j in range(1, 5):
            #print(f"Before conv{j}0, x shape: {x.shape}, cond shape: {cond.shape}")
            x = getattr(self, f'conv{j}0')(x)
            x = getattr(self, f'bn{j}0')(x)
            x = getattr(self, f'relu{j}0')(x)
            for i in range(1, getattr(self, f"layercount{j}")):
                #print(f"Before conv{j}{i}, x shape: {x.shape}, cond shape: {cond.shape}")
                x = getattr(self, f'conv{j}{i}')(x)
                x = getattr(self, f'bn{j}{i}')(x)
                x = getattr(self, f'relu{j}{i}')(x)
                #print(f"Before cond_coup{j}, x shape: {x.shape}, cond shape: {cond.shape}")
            x = getattr(self, f'pool{j}')(x)
            #print(f"After pool{j}, x shape: {x.shape}")
        x = torch.mean(x, dim=(2,3,4))
        
        x = torch.flatten(x, 1, -1)
        x = self.fc_layers(x)

        return x
    
class Summary_net_lc_super_smol(nn.Module):
    def __init__(self, num_features: int = 6,
                 channels1: int = 48,
                 channels2: int = 64,
                 channels3: int = 96,
                 channels4: int = 96,
                 hidden_layers1 = 96,
                 hidden_layers2 = 64,
                 ):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=channels1, kernel_size=(3, 3, 14), stride=(1, 1, 7)),
            nn.BatchNorm3d(channels1),
            nn.GELU(),
            nn.Dropout3d(0.1),

            nn.Conv3d(in_channels=channels1, out_channels=channels2, kernel_size=(3, 3, 3)),
            nn.GELU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),

            nn.Conv3d(in_channels=channels2, out_channels=channels2, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(channels2),
            nn.GELU(),
            nn.Dropout3d(0.1),

            nn.Conv3d(in_channels=channels2, out_channels=channels3, kernel_size=(3, 3, 3)),
            nn.GELU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),

            nn.Conv3d(in_channels=channels3, out_channels=channels3, kernel_size=1, padding=1),
            nn.BatchNorm3d(channels3),
            nn.GELU(),
            nn.Dropout3d(0.1),

            nn.Conv3d(in_channels=channels3, out_channels=channels4, kernel_size=(3, 3, 3)), # Adjusted kernel for depth to fit
            nn.GELU(),
            nn.MaxPool3d(kernel_size=(3,3,4), stride=(3,3,4), padding=0),
        )

        self.flatten = nn.Flatten()

        self.fc_layers = nn.Sequential(
            nn.Linear(channels4, hidden_layers1),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.Linear(hidden_layers1, hidden_layers2),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.Linear(hidden_layers2, 16),
            nn.GELU(),
            nn.Linear(16, num_features),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x
    

# out_dim is a bit clunky, better option will be added soon
# in: (batch_dim, event_dim) ; out: (batch_dim, event_dim)
class global_temp_smol_inv_super_smol(nn.Module):
    def __init__(self, in_dim = 6, out_dim = 470):
        super().__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.GELU(),
            nn.Linear(32, 48),
            nn.GELU(),
            nn.Linear(48, 64),
            nn.GELU(),
            nn.Linear(64, 92),
            nn.GELU(),
        )
        self.unpooling = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(8),
            nn.Upsample(size = 100, mode='linear'),
            nn.Conv1d(in_channels=8, out_channels=8, kernel_size=11, stride=1, padding=5),
            nn.GELU(),
            nn.BatchNorm1d(8),
            nn.Upsample(size = 200, mode='linear'),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=21, stride=1, padding=10),
            nn.GELU(),
            nn.BatchNorm1d(16),
            nn.Upsample(size = 350, mode='linear'),
            nn.Conv1d(in_channels=16, out_channels=24, kernel_size=21, stride=1, padding=10),
            nn.GELU(),
            nn.BatchNorm1d(24),
            nn.Upsample(size = 470, mode='linear'),
            nn.Conv1d(in_channels=24, out_channels=16, kernel_size=11, stride=1, padding=5),
            nn.GELU(),
            nn.BatchNorm1d(16),
            nn.Conv1d(in_channels=16, out_channels=1, kernel_size=5, stride=1, padding=2),
            nn.GELU(),
        )
        self.out = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1, stride =1, padding=0),
        )

    def forward(self, x):
        x = self.fc_layers(x)
        x = x.unsqueeze(-2)
        x = self.unpooling(x)
        x = self.out(x)
        x = x.squeeze(-2)
        return x


class Summary_net_lc_benedikt(nn.Module):
    
    def __init__(self, in_ch=1, ch=32, N_parameter=6,sigmoid=True):
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
            x = nn.Tanh()(self.out(x))
        else:
            x = self.out(x)

        return x

'''class Summary_net_1dps(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_conv_stack = nn.Sequential(
            nn.Conv1d(10, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 48, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.flatten = nn.Flatten()
        """
        self.linear_conv_stack_z = nn.Sequential(
            nn.Conv1d(1, 4, 3, 1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(4, 8, 3, 1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(4, 8, 3, 1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        """
        self.linear_stack = nn.Sequential(
            nn.Linear(48,32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,6),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.linear_conv_stack(x)
        x = self.flatten(x)
        x = self.linear_stack(x)
        return x
'''
# old: .044
# old_charged: .039
# more_params: .037
# more_params with add linear: .036
# linear_conv_stack_z: .037
# linear_conv_stack_z with max first: .038

class Summary_net_1dps_cond(nn.Module):
    def __init__(self,
                 layer_per_block1 = 2,
                 layersize1 = 32,
                 filter_size1 = 3,
                 layer_per_block2 = 2,
                 layersize2 = 32,
                 filter_size2 = 3,
                 layer_per_block3 = 2,
                 layersize3 = 32,
                 filter_size3 = 3,):
        super().__init__()
        
        self.conv1 = cConv1d(10, layersize1, layer_per_block1, filter_size1, 1, int(filter_size1/2))
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = cConv1d(layersize1, layersize2, layer_per_block2, filter_size2,1, int(filter_size2/2))
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = cConv1d(layersize2, layersize3, layer_per_block3, filter_size3,1, int(filter_size3/2))
        self.pool3 = nn.MaxPool1d(2)

        self.flatten = nn.Flatten()

        self.linear_stack = nn.Sequential(
            nn.Linear(layersize3,32),
            nn.Dropout(0.1),
            nn.GELU(),
            nn.Linear(32,16),
            nn.Dropout(0.1),
            nn.GELU(),
            nn.Linear(16,8),
            nn.GELU(),
            nn.Linear(8,6),
            nn.Tanh()
        )

    def forward(self, x, cond=None): 
        x = self.conv1(x,cond)
        x = self.pool1(x)

        x = self.conv2(x, cond)
        x = self.pool2(x)

        x = self.conv3(x, cond)
        x = self.pool3(x)


        x = self.flatten(x)
        x = self.linear_stack(x)
        return x
    
    
class Summary_net_1dps_det(nn.Module):
    def __init__(self,
                    z_slices: int = 10,
                    cond_size: int = 2):
        super().__init__()
        
        self.z_slices = z_slices
        
        
        def subnet_fc(c_in, c_out):
            return nn.Sequential(nn.Linear(c_in, 256), nn.GELU(),
                                nn.Linear(256,  c_out), nn.BatchNorm1d(c_out))

        def subnet_conv(c_in, c_out):
            return nn.Sequential(nn.Conv1d(c_in, 256,   3, padding=1), nn.GELU(),
                                nn.Conv1d(256,  c_out, 3, padding=1))

        def subnet_conv_1x1(c_in, c_out):
            return nn.Sequential(nn.Conv1d(c_in, 256,   1), nn.GELU(),
                                nn.Conv1d(256,  c_out, 1))
            
        self.down = nn.AvgPool1d(2)
    
        self.mlp1 = Ff.SequenceINN(80)
        self.mlp1.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, permute_soft=True)
        
        self.mlp2 = Ff.SequenceINN(40)
        self.mlp2.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, permute_soft=True)
        
        self.mlp3 = Ff.SequenceINN(20)
        self.mlp3.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, permute_soft=True)
        
        self.mlp4 = Ff.SequenceINN(10)
        self.mlp4.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, permute_soft=True)
        
        self.mlp5 = Ff.SequenceINN(8)
        self.mlp5.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, permute_soft=True)
        
        self.mlp6 = Ff.SequenceINN(6)
        self.mlp6.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, permute_soft=True)

        self.flatten = nn.Flatten()
        
        self.out = nn.Tanh()


    def forward(self, x, cond=None):
        loss = 0
    
        x = x.flatten(1)
        x, det = self.mlp1(x)
        loss += 0.5*torch.sum(x**2, 1) - det
        
        x = self.down(x)
        x, det = self.mlp2(x)
        loss += 0.5*torch.sum(x**2, 1) - det
        
        x = self.down(x)
        x, det = self.mlp3(x)
        loss += 0.5*torch.sum(x**2, 1) - det
        
        x = self.down(x)
        x, det = self.mlp4(x)
        loss += 0.5*torch.sum(x**2, 1) - det
        
        x = x[:, :-2] * x[:, -1].unsqueeze(-1) + x[:, -2].unsqueeze(-1)
        
        x, det = self.mlp5(x)
        loss += 0.5*torch.sum(x**2, 1) - det
        
        x = x[:, :-2] * x[:, -1].unsqueeze(-1) + x[:, -2].unsqueeze(-1)
        
        x, det = self.mlp6(x)
        
        x = self.out(x)
        
        loss += 0.5*torch.sum(x**2, 1) - det
        
        return x, loss

class Summary_net_1dps(nn.Module):
    def __init__(self, num_features: int = 6, 
                 channels1: int = 16,
                 channels2: int = 32,
                 channels3: int = 42,
                 hidden_dim1: int = 42):
        super().__init__()

        
        self.conv1 = nn.Sequential(
            nn.Conv1d(7, channels1, 3, padding=1),
            nn.BatchNorm1d(channels1),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Dropout1d(0.1),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(channels1, channels2, 3, padding=1),
            nn.GELU(),
            nn.MaxPool1d(2),
        )


        self.conv3 = nn.Sequential(
            nn.Conv1d(channels2, channels3, 3, padding=1),
            nn.BatchNorm1d(channels3),
            nn.GELU(),
            nn.MaxPool1d(2),
        )
        
        self.flatten = nn.Flatten()

        self.linear_stack = nn.Sequential(
            nn.Linear(channels3,hidden_dim1),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.Linear(hidden_dim1,32),
            nn.GELU(),
            nn.Linear(32,num_features),
            nn.Tanh()
        )

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.flatten(x)

        x = self.linear_stack(x)

        return x


    
class Summary_net_2dps(nn.Module):
    def __init__(self, num_features: int = 6,
                channels1: int = 32,
                channels2: int = 48,
                channels3: int = 64,
                channels4: int = 42,
                hidden_dim1: int = 128):
        super().__init__()
    

        self.conv1 = nn.Sequential(
            nn.Conv2d(7, channels1, 3, padding=1),
            nn.BatchNorm2d(channels1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(channels1, channels2, 3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(channels2, channels3, 3, padding=1),
            nn.GELU(),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(channels3, channels4, 3, padding=1),
            nn.BatchNorm2d(channels4),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
        
        self.flatten = nn.Flatten()

        self.linear_stack = nn.Sequential(
            nn.Linear(channels4,hidden_dim1),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.Linear(hidden_dim1,32),
            nn.GELU(),
            nn.Linear(32,num_features),
            nn.Tanh()
        )

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.flatten(x)

        x = self.linear_stack(x)

        return x


class PositionalEncoding2D(nn.Module):
    """
    Injects some information about the relative or absolute position of the tokens in the sequence.
    The positional encodings have the same dimension as the embeddings, so they can be summed.
    Here, we use sine and cosine functions of different frequencies.
    This version is robust to odd d_model values.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000): # Increased max_len for safety
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        if d_model <= 0:
            raise ValueError("d_model must be a positive integer.")

        position = torch.arange(max_len).unsqueeze(1).float() # Ensure position is float
        pe = torch.zeros(max_len, d_model)

        # Calculate div_term. It has math.ceil(d_model / 2.0) elements.
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term) # Assign to columns 0, 2, 4, ...

        if d_model > 1: # Only apply cosine if there are odd indices to fill
            # For pe[:, 1::2] (cols 1, 3, 5, ...), there are d_model // 2 columns.
            # We need to use the first d_model // 2 elements of div_term for these.
            pe[:, 1::2] = torch.cos(position * div_term[:d_model//2])
        
        # Add a batch dimension: (max_len, d_model) -> (1, max_len, d_model)
        pe = pe.unsqueeze(0) 
        
        # register_buffer ensures 'pe' is part of the model's state_dict, but not a learnable parameter
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # Add positional encoding to the input tensor x
        # self.pe is (1, max_len, d_model). We slice it to (1, seq_len, d_model)
        # to match the input sequence length x.size(1).
        # The embedding dimension x.size(2) must match self.pe.size(2) (d_model).
        if x.size(2) != self.pe.size(2):
            raise RuntimeError(
                f"The embedding dimension of input x ({x.size(2)}) "
                f"does not match the d_model of positional encoding ({self.pe.size(2)})."
            )
        
        x = x + self.pe[:, :x.size(1), :] # Use explicit slicing for all dims for clarity
        return self.dropout(x)

class PatchEmbedding2D(nn.Module):
    """
    Image to Patch Embedding.
    Treats the input image as a collection of patches and projects each patch
    into a vector of `embed_dim`. This serves as the "tokenizer" for the image.
    """
    def __init__(self, img_size_h: int = 14, img_size_w: int = 14, 
                 patch_size_h: int = 2, patch_size_w: int = 2, 
                 in_chans: int = 7, embed_dim: int = 256):
        super().__init__()
        self.img_size = (img_size_h, img_size_w)
        self.patch_size = (patch_size_h, patch_size_w)
        
        if img_size_h % patch_size_h != 0 or img_size_w % patch_size_w != 0:
            raise ValueError("Image dimensions must be divisible by patch dimensions.")

        # Calculate the number of patches in height and width
        self.grid_size = (img_size_h // patch_size_h, img_size_w // patch_size_w)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        # Convolutional layer to project patches.
        # Kernel size and stride are set to patch_size to create non-overlapping patches.
        self.proj = nn.Conv2d(in_chans, embed_dim, 
                              kernel_size=(patch_size_h, patch_size_w), 
                              stride=(patch_size_h, patch_size_w))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: (batch_size, in_chans, img_size_h, img_size_w)
        # e.g., (B, 7, 14, 14)
        B, C, H, W = x.shape
        
        if H != self.img_size[0] or W != self.img_size[1]:
            raise ValueError(
                f"Input image size ({H}*{W}) doesn't match model's expected size ({self.img_size[0]}*{self.img_size[1]})."
            )

        # Project patches using convolution:
        # (B, C, H, W) -> (B, embed_dim, grid_size_h, grid_size_w)
        x = self.proj(x)
        
        # Flatten the spatial dimensions (grid_size_h, grid_size_w) into a single sequence dimension
        # (B, embed_dim, grid_size_h, grid_size_w) -> (B, embed_dim, num_patches)
        x = x.flatten(2)
        
        # Transpose to get the format (batch_size, num_patches, embed_dim)
        # This is the standard input format for batch_first=True Transformers
        x = x.transpose(1, 2)
        
        return x

class ImageSummarizerTransformer2D(nn.Module):
    """
    A small Transformer-based network to summarize image-like inputs.
    Input shape: (batch_size, in_chans, img_h, img_w) e.g. (B, 7, 14, 14)
    Output shape: (batch_size, d_model) - the summarized representation.
    """
    def __init__(self, img_size_h: int = 14, img_size_w: int = 14, 
                 patch_size_h: int = 2, patch_size_w: int = 2, 
                 in_chans: int = 7, d_model: int = 64, nhead: int = 4, 
                 num_encoder_layers: int = 4, dim_feedforward: int = 64, 
                 dropout: float = 0.1):
        super().__init__()
        
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead}).")

        self.d_model = d_model

        # 1. Patch Embedding ("Tokenizer")
        self.patch_embed = PatchEmbedding2D(
            img_size_h, img_size_w, patch_size_h, patch_size_w, in_chans, d_model
        )
        num_patches = self.patch_embed.num_patches

        # 2. Positional Encoding
        # max_len for positional encoding should be at least num_patches.
        # Increased default max_len in PositionalEncoding class itself.
        self.pos_encoder = PositionalEncoding2D(d_model, dropout, max_len=num_patches + 1) 

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True, norm_first=False # common practice
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_encoder_layers
        )

        self.summary = nn.Linear(d_model,6)
        self.final = nn.Tanh()

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # src: (batch_size, in_chans, img_h, img_w), e.g., (B, 7, 14, 14)
        
        # Apply patch embedding
        # Output: (batch_size, num_patches, d_model)
        x = self.patch_embed(src)
        
        # Add positional encoding
        # Output: (batch_size, num_patches, d_model)
        x = self.pos_encoder(x)
        
        # Pass through Transformer encoder
        # Output: (batch_size, num_patches, d_model)
        memory = self.transformer_encoder(x) # No mask needed for standard encoder self-attention
        
        # Summarize by averaging across the sequence (patch) dimension
        # Output: (batch_size, d_model)
        summary = torch.mean(memory, dim=1)
        summary = self.summary(summary)
        summary = self.final(summary)

        
        return summary
    


class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000): # Increased max_len for safety
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        if d_model <= 0:
            raise ValueError("d_model must be a positive integer.")

        position = torch.arange(max_len).unsqueeze(1).float() # Ensure position is float
        pe = torch.zeros(max_len, d_model)

        # Calculate div_term. It has math.ceil(d_model / 2.0) elements.
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term) # Assign to columns 0, 2, 4, ...

        if d_model > 1: # Only apply cosine if there are odd indices to fill
            # For pe[:, 1::2] (cols 1, 3, 5, ...), there are d_model // 2 columns.
            # We need to use the first d_model // 2 elements of div_term for these.
            # Ensure div_term has enough elements for this slicing.
            div_term_cos = div_term[:pe[:, 1::2].size(1)] # Match the number of columns to fill
            pe[:, 1::2] = torch.cos(position * div_term_cos)
        
        # Add a batch dimension: (max_len, d_model) -> (1, max_len, d_model)
        pe = pe.unsqueeze(0) 
        
        # register_buffer ensures 'pe' is part of the model's state_dict, but not a learnable parameter
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # Add positional encoding to the input tensor x
        # self.pe is (1, max_len, d_model). We slice it to (1, seq_len, d_model)
        # to match the input sequence length x.size(1).
        # The embedding dimension x.size(2) must match self.pe.size(2) (d_model).
        if x.size(2) != self.pe.size(2):
            raise RuntimeError(
                f"The embedding dimension of input x ({x.size(2)}) "
                f"does not match the d_model of positional encoding ({self.pe.size(2)})."
            )
        
        x = x + self.pe[:, :x.size(1), :] # Use explicit slicing for all dims for clarity
        return self.dropout(x)

class PatchEmbedding1D(nn.Module):
    """
    1D Sequence to Patch Embedding.
    Treats the input sequence as a collection of patches and projects each patch
    into a vector of `embed_dim`. This serves as the "tokenizer" for the 1D sequence.
    """
    def __init__(self, seq_len_in: int = 14, 
                 patch_len: int = 2, 
                 in_chans: int = 7, embed_dim: int = 256):
        super().__init__()
        self.seq_len_in = seq_len_in
        self.patch_len = patch_len
        
        if seq_len_in % patch_len != 0:
            raise ValueError("Input sequence length must be divisible by patch length.")

        self.num_patches = seq_len_in // patch_len

        # 1D Convolutional layer to project patches.
        # Kernel size and stride are set to patch_len to create non-overlapping patches.
        self.proj = nn.Conv1d(in_chans, embed_dim, 
                              kernel_size=patch_len, 
                              stride=patch_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: (batch_size, in_chans, seq_len_in)
        # e.g., (B, 7, 14)
        B, C, L = x.shape
        
        if L != self.seq_len_in:
            raise ValueError(
                f"Input sequence length ({L}) doesn't match model's expected length ({self.seq_len_in})."
            )

        # Project patches using 1D convolution:
        # (B, C, L) -> (B, embed_dim, num_patches)
        # e.g., (B, 7, 14) -> (B, 256, 7) if patch_len is 2
        x = self.proj(x)
        
        # Transpose to get the format (batch_size, num_patches, embed_dim)
        # This is the standard input format for batch_first=True Transformers
        # e.g., (B, 256, 7) -> (B, 7, 256)
        x = x.transpose(1, 2)
        
        return x

class SequenceSummarizerTransformer1D(nn.Module):
    """
    A small Transformer-based network to summarize 1D sequence inputs.
    Input shape: (batch_size, in_chans, seq_len) e.g. (B, 7, 14)
    Output shape: (batch_size, d_model) - the summarized representation.
    """
    def __init__(self, seq_len: int = 14, 
                 patch_len: int = 2, 
                 in_chans: int = 7, d_model: int = 32, nhead: int = 4, 
                 num_encoder_layers: int = 4, dim_feedforward: int = 32, 
                 dropout: float = 0.1):
        super().__init__()
        
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead}).")

        self.d_model = d_model

        # 1. Patch Embedding ("Tokenizer") for 1D sequences
        self.patch_embed = PatchEmbedding1D(
            seq_len, patch_len, in_chans, d_model
        )
        num_patches = self.patch_embed.num_patches

        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding1D(d_model, dropout, max_len=num_patches + 1) 

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True, norm_first=False 
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_encoder_layers
        )

        self.summary = nn.Linear(d_model,6)
        self.final = nn.Tanh()

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # src: (batch_size, in_chans, seq_len), e.g., (B, 7, 14)
        
        # Apply 1D patch embedding
        # Output: (batch_size, num_patches, d_model)
        x = self.patch_embed(src)
        
        # Add positional encoding
        # Output: (batch_size, num_patches, d_model)
        x = self.pos_encoder(x)
        
        # Pass through Transformer encoder
        # Output: (batch_size, num_patches, d_model)
        memory = self.transformer_encoder(x)
        
        # Summarize by averaging across the sequence (patch) dimension
        # Output: (batch_size, d_model)
        summary = torch.mean(memory, dim=1)

        summary = self.summary(summary)
        summary = self.final(summary)
        
        return summary