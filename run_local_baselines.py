"""
Computes genuine "before FL" (local-only, no federation) baseline AUROC for
each hospital, training solely on that hospital's own real data with no
communication with any other hospital or the FL server.

This replaces the hardcoded baseline numbers that used to live in
run_full_performance_simulation.py's stage_1_baselines() (previously just
returned {"A": 0.65, "B": 0.82, ...} without training anything). Data loading
here intentionally mirrors each run_hospital_*_client_enhanced.py script
exactly, rather than importing from them, so this script carries no risk of
disturbing the already-verified federated training path.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent
import sys
sys.path.append(str(PROJECT_ROOT))
from fl_utils.unified_model import UnifiedFLModel

EPOCHS = 3
DEVICE = torch.device("cpu")


def _auroc(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            out = torch.sigmoid(model(x.to(DEVICE)))
            preds.append(out.cpu().numpy())
            targets.append(y.cpu().numpy())
    y_true, y_pred = np.vstack(targets), np.vstack(preds)
    valid_cols = [c for c in range(y_true.shape[1]) if len(np.unique(y_true[:, c])) > 1]
    if not valid_cols:
        return 0.5
    return float(np.mean([roc_auc_score(y_true[:, c], y_pred[:, c]) for c in valid_cols]))


def _train_local(train_loader, val_loader, lr=1e-3):
    model = UnifiedFLModel().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(EPOCHS):
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
    return _auroc(model, val_loader)


def baseline_a():
    d = PROJECT_ROOT / "src" / "hospital_a" / "data"
    X_train = torch.from_numpy(np.load(d / "X_train.npy")).float()
    y_train = torch.from_numpy(np.load(d / "Y_train.npy")).float()
    X_val = torch.from_numpy(np.load(d / "X_val.npy")).float()
    y_val = torch.from_numpy(np.load(d / "Y_val.npy")).float()
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32)
    return _train_local(train_loader, val_loader)


def baseline_b():
    df = pd.read_csv(PROJECT_ROOT / "data" / "hospital_b" / "processed_vitals.csv")
    vitals_cols = [c for c in df.columns if c not in ("patient_id", "deterioration_risk")]
    X = torch.from_numpy(df[vitals_cols].values).float()
    y_raw = torch.from_numpy(df["deterioration_risk"].values).long()
    y_onehot = torch.zeros(len(y_raw), 5)
    y_onehot.scatter_(1, y_raw.unsqueeze(1), 1)

    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(sss.split(X.numpy(), y_raw.numpy()))
    train_loader = DataLoader(TensorDataset(X[train_idx], y_onehot[train_idx]), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X[val_idx], y_onehot[val_idx]), batch_size=32)
    return _train_local(train_loader, val_loader, lr=1e-3)


def baseline_c():
    from PIL import Image
    import torchvision.transforms as transforms

    data_dir = PROJECT_ROOT / "data" / "hospital_c"
    labels_df = pd.read_csv(data_dir / "labels.csv")
    img_transform = transforms.Compose([transforms.Resize((20, 20)), transforms.ToTensor()])
    classes = ["No Finding", "Cardiomegaly", "Effusion", "Infiltration", "Atelectasis"]

    class XrayDataset(torch.utils.data.Dataset):
        def __init__(self, paths, targets, transform):
            self.paths, self.targets, self.transform = paths, targets, transform

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            try:
                img = Image.open(self.paths[idx]).convert("RGB")
                return self.transform(img).view(-1), self.targets[idx]
            except Exception:
                return torch.zeros(1200), self.targets[idx]

    X_list, y_list = [], []
    for _, row in labels_df.iterrows():
        img_path = data_dir / "images" / row["Image Index"]
        if not img_path.exists():
            continue
        X_list.append(str(img_path))
        y_vec = torch.zeros(5)
        lbl_str = row["Finding Labels"]
        if "No Finding" in lbl_str:
            y_vec[0] = 1.0
        else:
            found = False
            for i, c in enumerate(classes[1:], 1):
                if c in lbl_str:
                    y_vec[i] = 1.0
                    found = True
            if not found:
                y_vec[4] = 1.0
        y_list.append(y_vec)

    full_dataset = XrayDataset(X_list, y_list, img_transform)
    train_size = int(0.8 * len(full_dataset))
    train_ds, val_ds = torch.utils.data.random_split(
        full_dataset, [train_size, len(full_dataset) - train_size],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)
    return _train_local(train_loader, val_loader, lr=1e-4)


def baseline_d():
    d = PROJECT_ROOT / "src" / "hospital_d" / "data"
    X_train = torch.from_numpy(np.load(d / "X_real_train.npy")).float()
    y_train = torch.from_numpy(np.load(d / "Y_real_train.npy")).float()
    X_val = torch.from_numpy(np.load(d / "X_real_test.npy")).float()
    y_val = torch.from_numpy(np.load(d / "Y_real_test.npy")).float()
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32)
    return _train_local(train_loader, val_loader)


def baseline_e():
    d = PROJECT_ROOT / "src" / "hospital_e" / "data"
    X_ecg = np.load(d / "X_ecg.npy")
    X_vitals = np.load(d / "X_vitals.npy")
    X_lungs = np.load(d / "X_lungs.npy")
    Y = np.load(d / "Y_labels.npy")
    train_idx = np.load(d / "train_indices.npy")
    test_idx = np.load(d / "test_indices.npy")

    X_combined = np.concatenate([X_ecg, X_vitals, X_lungs], axis=1)
    X_train = torch.from_numpy(X_combined[train_idx]).float()
    y_train = torch.from_numpy(Y[train_idx]).float()
    X_val = torch.from_numpy(X_combined[test_idx]).float()
    y_val = torch.from_numpy(Y[test_idx]).float()
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=16)
    return _train_local(train_loader, val_loader, lr=1e-4)


def main():
    baselines = {}
    for name, fn in [("A", baseline_a), ("B", baseline_b), ("C", baseline_c),
                      ("D", baseline_d), ("E", baseline_e)]:
        print(f"\n{'='*50}\nTraining local-only baseline: Hospital {name}\n{'='*50}")
        auroc = fn()
        baselines[name] = round(auroc, 4)
        print(f"Hospital {name} local-only AUROC: {auroc:.4f}")

    from fl_utils.simulation_utils import save_simulation_metrics
    save_simulation_metrics("before_fl", baselines)
    print(f"\nSaved real local-only baselines: {baselines}")


if __name__ == "__main__":
    main()
