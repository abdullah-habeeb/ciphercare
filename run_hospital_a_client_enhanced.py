"""
Enhanced Hospital A FL Client - ECG/S4
Self-contained for robust FL testing
"""

import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import roc_auc_score
import sys

sys.path.append('.')
from fl_utils.dp_utils import apply_dp_to_gradients, DPConfig

# Simplified S4-like blocks for robustness
class SimpleS4(nn.Module):
    def __init__(self, channels=12, classes=5):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        self.head = nn.Linear(128, classes)
    
    def forward(self, x):
        return self.head(self.enc(x))

class HospitalAClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader, device, dp_config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.dp_config = dp_config
        self.hospital_id = "A"
        self.global_params = None
        
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
        self.global_params = [torch.tensor(p).clone().detach() for p in parameters]
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        proximal_mu = config.get("proximal_mu", 0.01)
        sr = config.get("server_round", 1)
        
        print(f"Hospital A - Round {sr} Training")
        
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.BCEWithLogitsLoss()
        
        total_loss = 0.0
        
        # Train on synthetic batches
        for data, target in self.train_loader:
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            
            if self.global_params is not None:
                proximal_loss = 0.0
                # Reconstruct dict for safe alignment (skip buffers)
                params_dict = dict(zip(self.model.state_dict().keys(), [torch.tensor(p) for p in self.global_params]))
                for name, param in self.model.named_parameters():
                    if name in params_dict:
                        global_p = params_dict[name].to(self.device)
                        proximal_loss += ((param - global_p) ** 2).sum()
                loss += (proximal_mu / 2) * proximal_loss
            
            loss.backward()
            apply_dp_to_gradients(self.model, self.dp_config, len(self.train_loader.dataset), self.device)
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(self.train_loader)
        auroc = self._evaluate_auroc()
        print(f"  Training complete. Loss: {avg_loss:.4f}, AUROC: {auroc:.4f}")
        
        return self.get_parameters(config={}), len(self.train_loader.dataset), {
            "hospital_id": "A",
            "auroc": float(auroc),
            "loss": float(avg_loss)
        }
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        criterion = nn.BCEWithLogitsLoss()
        loss = 0.0
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss += criterion(output, target).item()
        
        auroc = self._evaluate_auroc()
        return float(loss / len(self.val_loader)), len(self.val_loader.dataset), {"auroc": float(auroc)}

    def _evaluate_auroc(self):
        self.model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                out = torch.sigmoid(self.model(data))
                all_preds.append(out.cpu().numpy())
                all_targets.append(target.cpu().numpy())
        try:
             return roc_auc_score(np.vstack(all_targets), np.vstack(all_preds), average='macro')
        except:
             return 0.5

def main():
    print("Hospital A Client Starting...")
    device = torch.device("cpu")
    
    # Argument Parser
    import argparse
    parser = argparse.ArgumentParser(description='Hospital A FL Client')
    parser.add_argument('--subset', type=float, default=1.0, help='Fraction of data to use (0.0-1.0)')
    parser.add_argument('--personalize', action='store_true', help='Run in personalization mode')
    args = parser.parse_args()
    
    # Real Data Loading
    import os
    import numpy as np
    
    data_path = r"c:\Users\aishw\codered5\src\hospital_a\data"
    try:
        print(f"Loading real data from {data_path}...")
        X_train_np = np.load(os.path.join(data_path, "X_train.npy"))
        y_train_np = np.load(os.path.join(data_path, "Y_train.npy"))
        X_val_np = np.load(os.path.join(data_path, "X_val.npy"))
        y_val_np = np.load(os.path.join(data_path, "Y_val.npy"))
        
        X_train = torch.from_numpy(X_train_np).float()
        y_train = torch.from_numpy(y_train_np).float()
        X_val = torch.from_numpy(X_val_np).float()
        y_val = torch.from_numpy(y_val_np).float()
        
        # Determine classes from shape (assuming one-hot or int)
        # UnifiedModel expects 5 classes. Hospital A might have 5 or 24.
        # Check shape
        print(f"Data Shapes: X={X_train.shape}, Y={y_train.shape}")
        
    except Exception as e:
        print(f"[ERROR] Failed to load real data: {e}. Using SYNTHETIC fallback.")
        N_SAMPLES = 1000
        X_train = torch.randn(N_SAMPLES, 12, 1000)
        y_train = torch.randint(0, 2, (N_SAMPLES, 5)).float()
        X_val = torch.randn(200, 12, 1000)
        y_val = torch.randint(0, 2, (200, 5)).float()

    if args.subset < 1.0:
        N = len(X_train)
        n_sub = int(N * args.subset)
        X_train = X_train[:n_sub]
        y_train = y_train[:n_sub]
        print(f"REAL DATA SUBSET: Using {n_sub}/{N} samples ({args.subset*100:.0f}%)")
    train = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    val = DataLoader(TensorDataset(X_val, y_val), batch_size=32)

    from fl_utils.unified_model import UnifiedFLModel
    model = UnifiedFLModel().to(device)
    
    if args.personalize:
        print("🧠 PERSONALIZATION MODE: Freezing encoder...")
        from fl_utils.personalization import freeze_encoder, unfreeze_head
        freeze_encoder(model)
        unfreeze_head(model)
    
    # model = SimpleS4().to(device) # Replaced with UnifiedFLModel
    from fl_utils.unified_model import UnifiedFLModel
    model = UnifiedFLModel().to(device)
    print(f"DEBUG: Hospital A Model Architecture:\n{model}")
    dp_config = DPConfig(epsilon=5.0, delta=1e-5, max_grad_norm=1.0, num_rounds=5)
    
    client = HospitalAClient(model, train, val, device, dp_config)
    
    # Retry logic or wait logic could go here, but Flower handles retries reasonably well
    # if the server is up.
    fl.client.start_numpy_client(server_address="127.0.0.1:8081", client=client)

if __name__ == "__main__":
    main()
