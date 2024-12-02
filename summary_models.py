import torch
import torch.nn as nn
import torch.nn.functional as F
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

# wants (28,28,680) input
class Summary_net_lc_smol(nn.Module):
    def __init__(self, in_channels = 1,
                 init_layers = {
            "layer_size1": 1,
            "channel1": 48,
            "kernel_size1_xy": 3,
            "kernel_size1_z": 15,
            "stride1": 1,
            
            "layer_size2": 1,
            "channel2": 48,
            "kernel_size2": 3,
            
            "layer_size3": 1,
            "channel3": 64,
            "kernel_size3": 3,
            
            "layer_size4": 1,
            "channel4": 96,
            "kernel_size4": 3,
            
            "layer_size5": 1,
            "channel5": 128,
            "kernel_size5": 3,   
        }):
        super().__init__()
        
        for j in range(1,6):
            setattr(self, f"layercount{j}", init_layers[f"layer_size{j}"])
            
            out_channels = init_layers[f"channel{j}"]
            if j == 1:
                kernel_size = torch.tensor([init_layers[f"kernel_size{j}_xy"],init_layers[f"kernel_size{j}_xy"],init_layers[f"kernel_size{j}_z"]])
                stride = torch.tensor([init_layers[f"stride{j}"],init_layers[f"stride{j}"],init_layers[f"kernel_size{j}_z"]])
            else:
                kernel_size = torch.tensor([init_layers[f"kernel_size{j}"],init_layers[f"kernel_size{j}"],init_layers[f"kernel_size{j}"]])
                stride = torch.tensor([int(init_layers[f"kernel_size{j}"]/2),int(init_layers[f"kernel_size{j}"]/2), 1])
                in_channels = init_layers[f"channel{j-1}"]
            setattr(self, f'conv{j}0', nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding = kernel_size-stride))
            setattr(self, f'bn{j}0', nn.BatchNorm3d(out_channels))
            setattr(self, f'relu{j}0', nn.GELU())
            setattr(self, f'cond{j}0', nn.Sequential(nn.Linear(2, out_channels), nn.Tanh()))
            in_channels = out_channels
            
            for i in range(1,getattr(self, f"layercount{j}")):
                setattr(self, f'cond{j}{i}', nn.Sequential(nn.Linear(2, in_channels), nn.Tanh()))
                setattr(self, f'conv{j}{i}', nn.Conv3d(in_channels, out_channels, kernel_size=(3,3,3), stride = (1,1,1), padding = (1,1,1)))
                setattr(self, f'bn{j}{i}', nn.BatchNorm3d(out_channels))
                setattr(self, f'relu{j}{i}', nn.GELU())
            
            setattr(self, f'cond_coup{j}', nn.Conv3d(out_channels, out_channels, kernel_size=1, stride=1, padding = 0))
            
            setattr(self, f"pool{j}", nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2)))
            
        self.lpool = nn.AvgPool3d(kernel_size=(2,2,2), stride=1, padding=0)
        
        self.fc_layers = nn.Sequential(
            nn.Linear(init_layers["channel5"], 96),  # Adjusted input dimension
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
    
    def forward(self, x, cond):
        
        for j in range(1, 6):
            #print(f"Before conv{j}0, x shape: {x.shape}, cond shape: {cond.shape}")
            x = getattr(self, f'conv{j}0')(x)
            x = getattr(self, f'bn{j}0')(x)
            x = getattr(self, f'relu{j}0')(x)
            for i in range(1, getattr(self, f"layercount{j}")):
                #print(f"Before conv{j}{i}, x shape: {x.shape}, cond shape: {cond.shape}")
                x = x * getattr(self, f'cond{j}{i}')(cond).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                x = getattr(self, f'conv{j}{i}')(x)
                x = getattr(self, f'bn{j}{i}')(x)
                x = getattr(self, f'relu{j}{i}')(x)
                #print(f"Before cond_coup{j}, x shape: {x.shape}, cond shape: {cond.shape}")
            x = x * getattr(self, f'cond_coup{j}')(x)
            x = getattr(self, f'pool{j}')(x)
            #print(f"After pool{j}, x shape: {x.shape}")
            
        x = self.lpool(x)
        
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
    
    def forward(self, x,c):
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

    def forward(self, x,c):
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

    def forward(self, x,c):
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
            nn.Sigmoid()
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

class cConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, layers, kernel_size, stride, padding, bias=True):
        super().__init__()
        
        self.layers = layers
        
        for i in range(layers-1):
            setattr(self, f'conv{i}', nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=bias))
            setattr(self, f'bn{i}', nn.BatchNorm1d(out_channels))
            setattr(self, f'relu{i}', nn.GELU())
            setattr(self, f'cond{i}', nn.Sequential(nn.Linear(2, in_channels), nn.Tanh()))
            in_channels = out_channels
        setattr(self, f'cond{layers-1}', nn.Sequential(nn.Linear(2, in_channels), nn.Tanh()))
        setattr(self, f'conv{layers-1}', nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=bias))
        setattr(self, f'bn{layers-1}', nn.BatchNorm1d(out_channels))
        setattr(self, f'relu{layers-1}', nn.GELU())

    
    def forward(self, x, cond):
        
        
        for i in range(self.layers):
            x = x*getattr(self, f'cond{i}')(cond).unsqueeze(-1)
            x = getattr(self, f'conv{i}')(x)
            x = getattr(self, f'bn{i}')(x)
            x = getattr(self, f'relu{i}')(x)
        
        return x

class Summary_net_1dps(nn.Module):
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
            nn.Sigmoid()
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
        
        self.out = nn.Sigmoid()


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

class Summary_net_2dps(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_conv_stack = nn.Sequential(
            nn.Conv2d(10, 32, 3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 96, 3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(96),
        )
        self.flatten = nn.Flatten()

        self.linear_conv_stack_z = nn.Sequential(
            nn.Conv1d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.BatchNorm1d(16),
            nn.Conv1d(16, 24, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.BatchNorm1d(24),
            nn.Conv1d(24, 32, 3, padding=1),
            nn.ReLU(),
            nn.AvgPool1d(4),
            nn.BatchNorm1d(32),
        )

        self.linear_stack = nn.Sequential(
            nn.Linear(96,64),
            nn.Dropout(0.1),
            nn.GELU(),
            nn.Linear(64,32),
            nn.Dropout(0.1),
            nn.GELU(),
            nn.Linear(32,16),
            nn.GELU(),
            nn.Linear(16,6),
            nn.Sigmoid()
        )

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