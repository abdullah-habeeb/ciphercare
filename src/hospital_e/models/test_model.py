import torch
import sys
import os

sys.path.append(os.getcwd())
from src.hospital_e.models.fusion_classifier import FusionClassifier

def test_fusion_model():
    print("Initializing FusionClassifier...")
    try:
        model = FusionClassifier(num_classes=5)
        print("✓ Model initialized")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return

    print("Testing Forward Pass...")
    batch_size = 4
    
    # Random inputs
    ecg = torch.randn(batch_size, 8, 1000)
    vitals = torch.randn(batch_size, 15)
    lungs = torch.randn(batch_size, 128)
    
    # Test masks
    # Batch 0: All present
    # Batch 1: ECG only
    # Batch 2: Vitals only
    # Batch 3: Lungs only
    masks = torch.tensor([
        [1, 1, 1],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=torch.float32)
    
    try:
        logits = model(ecg, vitals, lungs, masks)
        print(f"✓ Forward pass successful. Output shape: {logits.shape}")
        
        assert logits.shape == (batch_size, 5)
        print("✓ Shape assertion passed")
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")

if __name__ == "__main__":
    test_fusion_model()
