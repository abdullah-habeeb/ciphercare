import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.getcwd())
from src.hospital_a.models.encoder import ECGClassifier

class ECGEncoder(ECGClassifier):
    """
    S4-based ECG Encoder.
    Inherits from Hospital A's ECGClassifier but returns embeddings.
    """
    def __init__(
        self, 
        in_channels=8, 
        d_model=64,  # Reduced from 128
        n_layers=4,  # Reduced from 12 - MUCH faster!
        pool_output=True # Ignored, always pools in this architecture
    ):
        # Initialize parent with lightweight S4 params
        super().__init__(
            in_channels=in_channels,
            res_channels=d_model,
            skip_channels=d_model,
            num_classes=5, # Dummy, we remove the head
            num_res_layers=n_layers,
            s4_lmax=1000,
            s4_d_state=32,  # Reduced from 64
            s4_dropout=0.0,
            s4_bidirectional=1,
            s4_layernorm=1
        )
        
        # Remove classification head
        del self.classifier
        
    def forward(self, x):
        """
        Input: [B, 8, L]
        Output: [B, d_model]
        """
        x = self.init_conv(x)
        x = self.residual_layer(x) 
        # x is [B, skip_channels, L]
        
        # Global Average Pooling
        x = self.pool(x).squeeze(-1) # [B, d_model]
        
        return x
