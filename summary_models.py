import torch
import torch.nn as nn
import torch.nn.functional as F
        

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

# wants (28,28,680) input
class Summary_net_lc_smol(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=48, kernel_size=(3, 3, 15), stride=(1, 1, 15)), 
            nn.GELU(),
            nn.BatchNorm3d(48),)
        self.conv_2 = nn.Sequential(
            nn.Conv3d(in_channels=48, out_channels=48, kernel_size=(3, 3, 3)), 
            nn.GELU(),
            nn.BatchNorm3d(48),)
        self.conv_3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(2, 2, 3)),
            nn.Conv3d(in_channels=48, out_channels=64, kernel_size=(3, 3, 3)), 
            nn.GELU(),
            nn.BatchNorm3d(64),)
        self.conv_4 = nn.Sequential(
            nn.ZeroPad3d((0, 0, 1,1,1,1)),  # Padding for width and height only, no depth padding
            nn.Conv3d(in_channels=64, out_channels=96, kernel_size=(3, 3, 3)), 
            nn.GELU(),
            nn.BatchNorm3d(96),)
        self.conv_5 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            #nn.ZeroPad3d((0, 0, 1,1,1,1)),
            nn.Conv3d(in_channels=96, out_channels=128, kernel_size=(3, 3, 3)), 
            nn.GELU(),
            nn.BatchNorm3d(128),
        )
        self.pooling = nn.Sequential(
            nn.MaxPool3d(kernel_size=(2,2,1), stride=(2,2,1), padding=0),
            nn.AvgPool3d(kernel_size=(3,3,3), stride=1, padding=0)
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
        #print(x.shape)
        x = self.conv_1(x)
        #print(x.shape)
        x = self.conv_2(x)
        #print(x.shape)
        x = self.conv_3(x)
        #print(x.shape)
        x = self.conv_4(x)
        #print(x.shape)
        x = self.conv_5(x)
        #print(x.shape)
        x = self.pooling(x)
        x = torch.flatten(x, 1, -1)
        x = self.fc_layers(x)
        return x
    
# wants (28,28,470) input
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

# out_dim is a bit clunky, better option will be added soon
# in: (batch_dim, event_dim) ; out: (batch_dim, event_dim)
class global_temp_smol_inv_super_smol(nn.Module):
    def __init__(self, in_dim = 6, out_dim = 470):
        super().__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(in_dim, 12),
            nn.GELU(),
            nn.Linear(12, 24),
            nn.GELU(),
            nn.Linear(24, 48),
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
        

# input: (in_dim, out_dim, hidden_layer, n_nodes, activation, batch_norm)
class FCNN(nn.Module):
    """
    Simple fully connected neural network.
    """
    def __init__(self, in_dim, out_dim, hidden_layer, n_nodes, activation, batch_norm,
                 device):
        super().__init__()
        if batch_norm:
            layers = [nn.Linear(in_dim, n_nodes), activation,nn.BatchNorm1d(n_nodes)] + [layer for _ in range(hidden_layer-1) for layer in (nn.Linear(n_nodes, n_nodes), activation, nn.BatchNorm1d(n_nodes))] + [nn.Linear(n_nodes, out_dim)]
            self.network = nn.Sequential(*layers).to(device)
        else:
            layers = [nn.Linear(in_dim, n_nodes), activation] + [layer for _ in range(hidden_layer-1) for layer in (nn.Linear(n_nodes, n_nodes), activation)] + [nn.Linear(n_nodes, out_dim)]
            self.network = nn.Sequential(*layers).to(device)

    def forward(self, x):
        return self.network(x)