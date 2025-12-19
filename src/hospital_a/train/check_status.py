import torch
import os

checkpoint_dir = "src/hospital_a/train/checkpoints"

# Check for checkpoints
if os.path.exists(os.path.join(checkpoint_dir, "best_model.pth")):
    print("✓ Training completed!")
    print(f"\nBest model saved at: {checkpoint_dir}/best_model.pth")
    print(f"File size: {os.path.getsize(os.path.join(checkpoint_dir, 'best_model.pth')) / 1024 / 1024:.2f} MB")
    
    # Try to load and check
    try:
        state_dict = torch.load(os.path.join(checkpoint_dir, "best_model.pth"), map_location='cpu')
        print(f"Model parameters loaded: {len(state_dict)} layers")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print("No checkpoint found")
