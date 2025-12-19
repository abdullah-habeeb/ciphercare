import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import numpy as np

sys.path.append(os.getcwd())

from src.hospital_a.models.encoder import ECGClassifier
from src.hospital_a.utils.ecg_preprocessing import PTBXL_DiseaseDataset

# Load config
with open('src/hospital_a/config/disease_config.json', 'r') as f:
    config = json.load(f)

device = torch.device('cpu')  # Force CPU to avoid issues
print(f"Using device: {device}\n")

# Load validation dataset only
print("Loading validation dataset...")
val_dataset = PTBXL_DiseaseDataset(
    signals_path=config['val_signals'],
    labels_csv_path=config['labels_csv'],
    ids_path=config.get('val_ids'),
    leads_idx=config.get('leads_idx', [0,1,6,7,8,9,10,11])
)

val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)  # Smaller batch

# Build model
print("Building model...")
mc = config['model_config']
model = ECGClassifier(
    in_channels=mc['in_channels'],
    res_channels=mc['res_channels'],
    skip_channels=mc['skip_channels'],
    num_classes=5,
    num_res_layers=mc['num_res_layers'],
    s4_lmax=mc['s4_lmax'],
    s4_d_state=mc['s4_d_state'],
    s4_dropout=mc['s4_dropout'],
    s4_bidirectional=mc['s4_bidirectional'],
    s4_layernorm=mc['s4_layernorm']
)

# Load trained weights
checkpoint_path = 'src/hospital_a/train/checkpoints/best_model.pth'
print(f"Loading trained model from {checkpoint_path}...\n")
state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
model.load_state_dict(state_dict)
model.eval()

criterion = nn.BCEWithLogitsLoss()

print("="*70)
print("EVALUATING TRAINED MODEL ON VALIDATION SET")
print("="*70)

val_loss = 0
val_preds = []
val_targets = []
batch_count = 0

print("\nProcessing batches...")
with torch.no_grad():
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        val_loss += loss.item()
        val_preds.append(torch.sigmoid(logits).cpu().numpy())
        val_targets.append(y.cpu().numpy())
        batch_count += 1
        if batch_count % 10 == 0:
            print(f"  Processed {batch_count} batches...")

val_preds = np.concatenate(val_preds)
val_targets = np.concatenate(val_targets)
val_loss = val_loss / len(val_loader)

try:
    val_auroc = roc_auc_score(val_targets, val_preds, average='macro')
except Exception as e:
    print(f"Error calculating AUROC: {e}")
    val_auroc = 0.0

print(f"\n📊 VALIDATION METRICS:")
print(f"  Val Loss: {val_loss:.4f}")
print(f"  Val AUROC (Macro): {val_auroc:.4f}")

# Per-class AUROC
classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
print(f"\n📈 PER-CLASS AUROC:")
print("-" * 40)
per_class_auroc = {}
for i, cls in enumerate(classes):
    try:
        if len(np.unique(val_targets[:, i])) > 1:
            auroc = roc_auc_score(val_targets[:, i], val_preds[:, i])
            per_class_auroc[cls] = float(auroc)
            print(f"  {cls:8s}: {auroc:.4f}")
        else:
            print(f"  {cls:8s}: N/A (only one class present)")
    except Exception as e:
        print(f"  {cls:8s}: Error - {e}")

print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation AUROC: {val_auroc:.4f}")
print("="*70)

# Save metrics
metrics = {
    "val_loss": float(val_loss),
    "val_auroc": float(val_auroc),
    "per_class_auroc": per_class_auroc,
    "num_samples": len(val_dataset)
}

with open('src/hospital_a/train/final_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"\n✓ Metrics saved to: src/hospital_a/train/final_metrics.json")
