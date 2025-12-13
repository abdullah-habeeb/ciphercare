import torch
import torch.nn as nn

class UnifiedFLModel(nn.Module):
    """
    Unified Model for simulated FL across heterogeneous hospitals.
    Input: (B, Input_Dim)
    Output: (B, Classes)
    
    We standardized on Input_Dim=1200 (flattened 12*100 ECG or padded vitals)
    Classes=5
    """
    def __init__(self, input_dim=1200, classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, classes)
        )
        
    def forward(self, x):
        # Flatten input if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
            
        # Handle size mismatch by padding or cropping
        # This is a hack for the simulation to work with heterogeneous dummy data
        if x.size(1) != 1200:
            if x.size(1) < 1200:
                padding = torch.zeros(x.size(0), 1200 - x.size(1), device=x.device)
                x = torch.cat([x, padding], dim=1)
            else:
                x = x[:, :1200]
                
        return self.net(x)
