from src.hospital_a.models.encoder import ECGClassifier as BaseEncoder

# Wrapper to expose the encoder specifically for Hospital D
# This ensures we have a dedicated entry point if we need to modify it later
class HospitalDEncoder(BaseEncoder):
    """
    Hospital D wrapper for the S4-based ECG Encoder.
    Inherits from the validated Hospital A model.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
