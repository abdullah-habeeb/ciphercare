"""
Real personalization step: loads the actual aggregated global model saved by
fl_server_enhanced.py (fl_results/checkpoints/global_model_latest.pth), then
for each hospital freezes the shared encoder and fine-tunes only the
classifier head on that hospital's own real local data.

Replaces the old stage_3_personalization() in run_full_performance_simulation.py,
which never loaded any model and instead fabricated a result with
`improved = min(0.99, base + 0.035)`.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from fl_utils.unified_model import UnifiedFLModel
from fl_utils.personalization import freeze_encoder, unfreeze_head
from run_local_baselines import _auroc, DEVICE

CHECKPOINT_PATH = PROJECT_ROOT / "fl_results" / "checkpoints" / "global_model_latest.pth"


def _get_loaders(name):
    """Re-run just the data-loading half of run_local_baselines' baseline_* functions."""
    # baseline_* trains AND evaluates; here we need the loaders themselves, so
    # duplicate the minimal loading logic rather than force baseline_* to
    # return loaders (keeping run_local_baselines.py's public behavior as-is).
    import numpy as np
    import pandas as pd
    from torch.utils.data import DataLoader, TensorDataset

    if name == "A":
        d = PROJECT_ROOT / "src" / "hospital_a" / "data"
        X_train = torch.from_numpy(np.load(d / "X_train.npy")).float()
        y_train = torch.from_numpy(np.load(d / "Y_train.npy")).float()
        X_val = torch.from_numpy(np.load(d / "X_val.npy")).float()
        y_val = torch.from_numpy(np.load(d / "Y_val.npy")).float()
        return (DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True),
                DataLoader(TensorDataset(X_val, y_val), batch_size=32))

    if name == "D":
        d = PROJECT_ROOT / "src" / "hospital_d" / "data"
        X_train = torch.from_numpy(np.load(d / "X_real_train.npy")).float()
        y_train = torch.from_numpy(np.load(d / "Y_real_train.npy")).float()
        X_val = torch.from_numpy(np.load(d / "X_real_test.npy")).float()
        y_val = torch.from_numpy(np.load(d / "Y_real_test.npy")).float()
        return (DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True),
                DataLoader(TensorDataset(X_val, y_val), batch_size=32))

    if name == "B":
        df = pd.read_csv(PROJECT_ROOT / "data" / "hospital_b" / "processed_vitals.csv")
        vitals_cols = [c for c in df.columns if c not in ("patient_id", "deterioration_risk")]
        X = torch.from_numpy(df[vitals_cols].values).float()
        y_raw = torch.from_numpy(df["deterioration_risk"].values).long()
        y_onehot = torch.zeros(len(y_raw), 5)
        y_onehot.scatter_(1, y_raw.unsqueeze(1), 1)
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, val_idx = next(sss.split(X.numpy(), y_raw.numpy()))
        return (DataLoader(TensorDataset(X[train_idx], y_onehot[train_idx]), batch_size=32, shuffle=True),
                DataLoader(TensorDataset(X[val_idx], y_onehot[val_idx]), batch_size=32))

    if name == "E":
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
        return (DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True),
                DataLoader(TensorDataset(X_val, y_val), batch_size=16))

    raise ValueError(f"Hospital C uses image data loaded via run_local_baselines.baseline_c()")


def personalize(name, train_loader, val_loader, global_state_dict, epochs=2, lr=1e-3):
    model = UnifiedFLModel().to(DEVICE)
    model.load_state_dict(global_state_dict)
    pre_auroc = _auroc(model, val_loader)

    freeze_encoder(model)
    unfreeze_head(model)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(epochs):
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

    post_auroc = _auroc(model, val_loader)
    return pre_auroc, post_auroc


def main():
    if not CHECKPOINT_PATH.exists():
        print(f"No global model checkpoint found at {CHECKPOINT_PATH}.")
        print("Run fl_server_enhanced.py + the 5 hospital clients first to produce one.")
        return

    global_state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    results = {}

    for name in ["A", "B", "D", "E"]:
        print(f"\n{'='*50}\nPersonalizing Hospital {name}\n{'='*50}")
        train_loader, val_loader = _get_loaders(name)
        pre, post = personalize(name, train_loader, val_loader, global_state_dict)
        results[name] = round(post, 4)
        print(f"Hospital {name}: global AUROC {pre:.4f} -> personalized AUROC {post:.4f}")

    print(f"\n{'='*50}\nPersonalizing Hospital C\n{'='*50}")
    # Hospital C's loader construction lives in run_local_baselines.baseline_c(),
    # which trains+evaluates in one call; reuse its dataset-building logic here.
    import pandas as pd
    from PIL import Image
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader, TensorDataset

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
    train_loader_c = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader_c = DataLoader(val_ds, batch_size=16)
    pre, post = personalize("C", train_loader_c, val_loader_c, global_state_dict, lr=1e-4)
    results["C"] = round(post, 4)
    print(f"Hospital C: global AUROC {pre:.4f} -> personalized AUROC {post:.4f}")

    from fl_utils.simulation_utils import save_simulation_metrics
    save_simulation_metrics("after_personalization", results)
    print(f"\nSaved real personalization results: {results}")


if __name__ == "__main__":
    main()
