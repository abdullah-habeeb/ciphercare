import torch
import torch.nn as nn

class VitalsEncoder(nn.Module):
    """
    MLP Encoder for Vitals data (tabular).
    Input: [B, 15]
    Output: [B, 64]
    """
    def __init__(self, input_dim=15, output_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, x):
        return self.net(x)
