import torch
from torch import nn
import FrEIA.framework as Ff
import FrEIA.modules as Fm

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
            nn.Conv3d(in_channels=1, out_channels=32, kernel_size=(3, 3, 41), stride=(1, 1, 41)), 
            nn.ReLU(),
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(3, 3, 2)), 
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 1)),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 2)), 
            nn.ReLU(),
            nn.ZeroPad2d((1, 1, 0, 0)),  # Padding for width and height only, no depth padding
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 2)), 
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(3, 3, 2)), 
            nn.ReLU()
        )
        self.pooling = nn.Sequential(
            nn.MaxPool3d(kernel_size=3, stride=3, padding=1),
            nn.AvgPool3d(kernel_size=2, stride=2, padding=0)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(128, 64),  # Adjusted input dimension
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 6),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.pooling(x)
        x = torch.flatten(x, 1, -1)
        x = self.fc_layers(x)
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
        self.linear2 = nn.Linear(128,128, bias=True)
        self.linear3 = nn.Linear(128,128, bias=True)
        self.out = nn.Linear(128, N_parameter, bias=True)
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