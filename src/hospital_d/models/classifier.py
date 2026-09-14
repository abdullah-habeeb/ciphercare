import torch.nn as nn
from src.hospital_d.models.encoder_s4 import HospitalDEncoder

class HospitalDClassifier(HospitalDEncoder):
    """
    Classifier for Hospital D (Geriatric Cohort).
    Identical architecture to Hospital A for compatibility in FL.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # The BaseEncoder (ECGClassifier) already includes the classification head.
        # If we wanted to customize the head, we would override it here.
        # For now, we use the standard head:
        # self.final_conv = nn.Sequential(...) 
