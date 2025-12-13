import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
import sys
sys.path.append('.')
from src.hospital_d.models.classifier import HospitalDClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Load test data
X_test = np.load('src/hospital_d/data/X_real_test_fixed.npy')
Y_test = np.load('src/hospital_d/data/Y_real_test.npy')
print(f"Test samples: {len(X_test)}")

test_loader = DataLoader(
    TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(Y_test, dtype=torch.float32)),
    batch_size=32, shuffle=False
)

# Load model
model = HospitalDClassifier(
    in_channels=8, res_channels=128, skip_channels=128, num_classes=5,
    num_res_layers=12, s4_lmax=1000, s4_d_state=64, s4_dropout=0.0,
    s4_bidirectional=1, s4_layernorm=1
).to(device)

checkpoint = torch.load('src/hospital_d/train/checkpoints/best_model_geriatric.pth', map_location=device)

with torch.no_grad():
    for name, param in model.named_parameters():
        if name in checkpoint:
            try:
                param.copy_(checkpoint[name])
            except RuntimeError:
                param.data = checkpoint[name].clone().to(device)

model.eval()
print("Model loaded")

# Evaluate
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

# Calculate AUROC
macro_auroc = roc_auc_score(all_targets, all_preds, average='macro')
print(f"\nMacro AUROC: {macro_auroc:.4f}")

classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
print("\nPer-Class AUROC:")
for i, cls in enumerate(classes):
    try:
        auroc = roc_auc_score(all_targets[:, i], all_preds[:, i])
        print(f"  {cls}: {auroc:.4f}")
    except:
        print(f"  {cls}: N/A")

print("\nDone!")
