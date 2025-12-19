import torch
import os

checkpoint_dir = "src/hospital_a/train/checkpoints"

print("="*70)
print("SEARCHING FOR TRAINING METRICS (LOSS & AUROC)")
print("="*70)

# Check for epoch checkpoints
checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
print(f"\nFound {len(checkpoint_files)} checkpoint file(s):")
for f in checkpoint_files:
    print(f"  - {f}")

# Try to load latest checkpoint if it exists
latest_checkpoint = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
if os.path.exists(latest_checkpoint):
    print(f"\n✓ Found latest_checkpoint.pth")
    ckpt = torch.load(latest_checkpoint, map_location='cpu')
    
    print("\n📊 TRAINING METRICS FROM CHECKPOINT:")
    print(f"  Epoch: {ckpt.get('epoch', 'N/A')}")
    print(f"  Train Loss: {ckpt.get('train_loss', 'N/A'):.4f}" if 'train_loss' in ckpt else "  Train Loss: N/A")
    print(f"  Val Loss: {ckpt.get('val_loss', 'N/A'):.4f}" if 'val_loss' in ckpt else "  Val Loss: N/A")
    print(f"  Val AUROC: {ckpt.get('auroc', 'N/A'):.4f}" if 'auroc' in ckpt else "  Val AUROC: N/A")
    print(f"  Best Val Loss: {ckpt.get('best_val_loss', 'N/A'):.4f}" if 'best_val_loss' in ckpt else "  Best Val Loss: N/A")
else:
    print("\n⚠ No latest_checkpoint.pth found")
    
# Check for individual epoch checkpoints
epoch_checkpoints = sorted([f for f in checkpoint_files if f.startswith('checkpoint_epoch_')])
if epoch_checkpoints:
    print(f"\n📈 EPOCH-BY-EPOCH METRICS:")
    print("-" * 70)
    print(f"{'Epoch':<8} {'Train Loss':<12} {'Val Loss':<12} {'Val AUROC':<12}")
    print("-" * 70)
    
    for ckpt_file in epoch_checkpoints:
        ckpt_path = os.path.join(checkpoint_dir, ckpt_file)
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu')
            epoch = ckpt.get('epoch', '?')
            train_loss = ckpt.get('train_loss', 0.0)
            val_loss = ckpt.get('val_loss', 0.0)
            auroc = ckpt.get('auroc', 0.0)
            print(f"{epoch:<8} {train_loss:<12.4f} {val_loss:<12.4f} {auroc:<12.4f}")
        except Exception as e:
            print(f"  Error loading {ckpt_file}: {e}")
    print("-" * 70)
else:
    print("\n⚠ No epoch checkpoints found (training may not have saved them)")

print("\n" + "="*70)
