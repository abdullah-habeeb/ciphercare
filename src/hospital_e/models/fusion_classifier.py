import torch
import torch.nn as nn
from .ecg_encoder import ECGEncoder
from .vitals_encoder import VitalsEncoder
from .lungs_encoder import LungsEncoder

class FusionClassifier(nn.Module):
    """
    Multimodal Fusion Classifier for Hospital E.
    Combines ECG, Vitals, and Lungs with masking for missing modalities.
    """
    def __init__(self, num_classes=5):
        super().__init__()
        
        # Encoders (Lightweight)
        self.ecg_encoder = ECGEncoder() # Output: 64
        self.vitals_encoder = VitalsEncoder() # Output: 64
        self.lungs_encoder = LungsEncoder() # Output: 64
        
        # Fusion Head
        # Concatenated dim = 64 + 64 + 64 = 192
        self.fusion = nn.Sequential(
            nn.Linear(192, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, ecg, vitals, lungs, mask):
        """
        ecg: [B, 8, 1000]
        vitals: [B, 15]
        lungs: [B, 128]
        mask: [B, 3] (1=present, 0=absent for ECG, Vitals, Lungs)
        """
        # Encode (if mask says present? Or encode all and zero out? 
        # Simpler to encode all if input is available, but input might be dummy/zeros if missing.
        # We assume input is always valid tensor shape, potentially zeros if missing.)
        
        e_ecg = self.ecg_encoder(ecg)       # [B, 64]
        e_vitals = self.vitals_encoder(vitals) # [B, 64]
        e_lungs = self.lungs_encoder(lungs)   # [B, 64]
        
        # Apply masks
        # mask[:, 0] is ECG, mask[:, 1] is Vitals, mask[:, 2] is Lungs
        # Expand masks to match embedding dim
        m_ecg = mask[:, 0].unsqueeze(1)    # [B, 1]
        m_vitals = mask[:, 1].unsqueeze(1) # [B, 1]
        m_lungs = mask[:, 2].unsqueeze(1)  # [B, 1]
        
        e_ecg = e_ecg * m_ecg
        e_vitals = e_vitals * m_vitals
        e_lungs = e_lungs * m_lungs
        
        # Concatenate
        fused = torch.cat([e_ecg, e_vitals, e_lungs], dim=1) # [B, 192]
        
        # Classify
        logits = self.fusion(fused)
        return logits
