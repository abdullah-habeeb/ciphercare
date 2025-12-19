import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .S4Model import S4Layer

def swish(x):
    return x * torch.sigmoid(x)

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(Conv, self).__init__()
        self.padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=self.padding)
        self.conv = nn.utils.weight_norm(self.conv)
        nn.init.kaiming_normal_(self.conv.weight)
    def forward(self, x):
        return self.conv(x)
    
class Residual_block(nn.Module):
    def __init__(self, res_channels, skip_channels, in_channels,
                 s4_lmax, s4_d_state, s4_dropout, s4_bidirectional, s4_layernorm):
        super(Residual_block, self).__init__()
        self.res_channels = res_channels
        
        # Removed diffusion step embedding layers
        
        self.S41 = S4Layer(features=2*self.res_channels, 
                           lmax=s4_lmax,
                           N=s4_d_state,
                           dropout=s4_dropout,
                           bidirectional=s4_bidirectional,
                           layer_norm=s4_layernorm)
        self.conv_layer = Conv(self.res_channels, 2 * self.res_channels, kernel_size=3)
        self.S42 = S4Layer(features=2*self.res_channels, 
                           lmax=s4_lmax,
                           N=s4_d_state,
                           dropout=s4_dropout,
                           bidirectional=s4_bidirectional,
                           layer_norm=s4_layernorm)
        
        self.res_conv = nn.Conv1d(res_channels, res_channels, kernel_size=1)
        self.res_conv = nn.utils.weight_norm(self.res_conv)
        nn.init.kaiming_normal_(self.res_conv.weight)
        
        self.skip_conv = nn.Conv1d(res_channels, skip_channels, kernel_size=1)
        self.skip_conv = nn.utils.weight_norm(self.skip_conv)
        nn.init.kaiming_normal_(self.skip_conv.weight)

    def forward(self, x):
        # x shape: [B, C, L]
        h = x
        B, C, L = x.shape
        assert C == self.res_channels

        # Removed diffusion step addition
        
        h = self.conv_layer(h)
        h = self.S41(h.permute(2,0,1)).permute(1,2,0)

        # Removed label embedding addition
        
        h = self.S42(h.permute(2,0,1)).permute(1,2,0)

        out = torch.tanh(h[:,:self.res_channels,:]) * torch.sigmoid(h[:,self.res_channels:,:])
        res = self.res_conv(out)
        skip = self.skip_conv(out)
        return (x + res) * math.sqrt(0.5), skip

class Residual_group(nn.Module):
    def __init__(self, res_channels, skip_channels, num_res_layers, 
                 in_channels, s4_lmax, s4_d_state, s4_dropout, s4_bidirectional, s4_layernorm):
        super(Residual_group, self).__init__()
        self.num_res_layers = num_res_layers
        
        self.residual_blocks = nn.ModuleList()
        for _ in range(self.num_res_layers):
            self.residual_blocks.append(Residual_block(res_channels, skip_channels, 
                                                       in_channels=in_channels,
                                                       s4_lmax=s4_lmax,
                                                       s4_d_state=s4_d_state,
                                                       s4_dropout=s4_dropout,
                                                       s4_bidirectional=s4_bidirectional,
                                                       s4_layernorm=s4_layernorm))
            
    def forward(self, x):
        h = x
        skip = 0
        for block in self.residual_blocks:
            h, skip_n = block(h)  
            skip += skip_n  
        return skip * math.sqrt(1.0 / self.num_res_layers)

class ECGClassifier(nn.Module):
    def __init__(self, in_channels, res_channels, skip_channels, num_classes, 
                 num_res_layers, s4_lmax, s4_d_state, s4_dropout, s4_bidirectional, s4_layernorm,
                 input_length=1000):
        super(ECGClassifier, self).__init__()
        self.init_conv = nn.Sequential(Conv(in_channels, res_channels, kernel_size=1), nn.ReLU())
        
        self.residual_layer = Residual_group(res_channels=res_channels, 
                                             skip_channels=skip_channels, 
                                             num_res_layers=num_res_layers, 
                                             in_channels=in_channels,
                                             s4_lmax=s4_lmax,
                                             s4_d_state=s4_d_state,
                                             s4_dropout=s4_dropout,
                                             s4_bidirectional=s4_bidirectional,
                                             s4_layernorm=s4_layernorm)
        
        # Classification Head
        # Global Average Pooling produces (B, skip_channels)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(skip_channels, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x: [B, in_channels, L]
        x = self.init_conv(x)
        x = self.residual_layer(x) # Returns skip connection sum [B, skip_channels, L]
        
        # Pooling
        x = self.pool(x).squeeze(-1) # [B, skip_channels]
        
        # Classifier
        logits = self.classifier(x)
        return logits
