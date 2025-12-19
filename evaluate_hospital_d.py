"""
Evaluation script for Hospital D geriatric model.
Run this after training completes to get final metrics.
"""
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, classification_report
import sys

sys.path.append('.')
from src.hospital_d.models.classifier import HospitalDClassifier

def evaluate():
    print("="*60)
    print("Hospital D - Final Evaluation")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Load test data
    print("Loading test data...")
    X_test = np.load('src/hospital_d/data/X_real_test_fixed.npy')
    Y_test = np.load('src/hospital_d/data/Y_real_test.npy')
    
    print(f"Test samples: {len(X_test)}")
    print(f"Positive labels: {Y_test.sum(axis=0)}\n")
    
    test_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(Y_test, dtype=torch.float32)
        ),
        batch_size=32, shuffle=False
    )
    
    # Load model with safe parameter loading
    print("Loading trained model...")
    model = HospitalDClassifier(
        in_channels=8,
        res_channels=128,
        skip_channels=128,
        num_classes=5,
        num_res_layers=12,
        s4_lmax=1000,
        s4_d_state=64,
        s4_dropout=0.0,
        s4_bidirectional=1,
        s4_layernorm=1
    ).to(device)
    
    # Safe loading to avoid S4 memory error
    checkpoint = torch.load('src/hospital_d/train/checkpoints/best_model_geriatric.pth', map_location=device)
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in checkpoint:
                try:
                    param.copy_(checkpoint[name])
                except RuntimeError as e:
                    if "single memory location" in str(e):
                        param.data = checkpoint[name].clone().to(device)
                    else:
                        print(f"Error loading {name}: {e}")
    
    model.eval()
    
    print("✓ Model loaded\n")
    
    # Evaluate
    print("Evaluating...")
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(y.numpy())
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    # Calculate metrics
    print("="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    try:
        macro_auroc = roc_auc_score(all_targets, all_preds, average='macro')
        print(f"\n📊 Macro AUROC: {macro_auroc:.4f}")
        
        # Per-class AUROC
        classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
        print("\n📈 Per-Class AUROC:")
        for i, cls in enumerate(classes):
            try:
                auroc = roc_auc_score(all_targets[:, i], all_preds[:, i])
                print(f"   {cls}: {auroc:.4f}")
            except:
                print(f"   {cls}: N/A (insufficient samples)")
                
    except Exception as e:
        print(f"Error calculating AUROC: {e}")
    
    # Binary predictions (threshold 0.5)
    binary_preds = (all_preds > 0.5).astype(int)
    
    print("\n📋 Classification Report:")
    print(classification_report(all_targets, binary_preds, target_names=classes, zero_division=0))
    
    print("\n" + "="*60)
    print("Evaluation complete!")
    print("="*60)

if __name__ == "__main__":
    evaluate()
