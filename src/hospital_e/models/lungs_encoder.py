import torch
import torch.nn as nn

class LungsEncoder(nn.Module):
    """
    Simple projection for Lung Embeddings.
    Input: [B, 128] (already embeddings)
    Output: [B, 64]
    """
    def __init__(self, input_dim=128, output_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, x):
        return self.net(x)
