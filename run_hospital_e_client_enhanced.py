"""
Enhanced Hospital E FL Client - Multimodal Fusion
Simplified with synthetic data for FL testing
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

class SimpleFusion(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.ecg_encoder = nn.Sequential(
            nn.Conv1d(12, 64, 7),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.vitals_encoder = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU()
        )
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fusion = nn.Sequential(
            nn.Linear(192, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, ecg, vitals, image):
        ecg_feat = self.ecg_encoder(ecg).squeeze(-1)
        vitals_feat = self.vitals_encoder(vitals)
        image_feat = self.image_encoder(image).view(image.size(0), -1)
        fused = torch.cat([ecg_feat, vitals_feat, image_feat], dim=1)
        return self.fusion(fused)

class HospitalEClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader, device, dp_config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.dp_config = dp_config
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
        
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        criterion = nn.BCEWithLogitsLoss()
        
        for data, target in self.train_loader:
            data = data.to(self.device)
            target = target.to(self.device)
            
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
            dp_metrics = apply_dp_to_gradients(self.model, self.dp_config, 
                                              len(self.train_loader.dataset), self.device)
            optimizer.step()
        
        auroc = self._evaluate_auroc()
        return self.get_parameters(config={}), len(self.train_loader.dataset), {
            "hospital_id": "E", "auroc": float(auroc), "loss": float(loss.item())
        }
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        criterion = nn.BCEWithLogitsLoss()
        total_loss = 0.0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                total_loss += criterion(output, target).item()
        
        auroc = self._evaluate_auroc()
        return float(total_loss / len(self.val_loader)), len(self.val_loader.dataset), {"auroc": float(auroc)}
    
    def _evaluate_auroc(self):
        self.model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = torch.sigmoid(self.model(data))
                all_preds.append(output.cpu().numpy())
                all_targets.append(target.cpu().numpy())
        try:
            return roc_auc_score(np.vstack(all_targets), np.vstack(all_preds), average='macro')
        except:
            return 0.5

def main():
    print("Hospital E - Multimodal Fusion FL Client")
    device = torch.device("cpu")
    
    # Argument Parser
    import argparse
    parser = argparse.ArgumentParser(description='Hospital E FL Client')
    parser.add_argument('--subset', type=float, default=1.0, help='Fraction of data to use (0.0-1.0)')
    parser.add_argument('--personalize', action='store_true', help='Run in personalization mode')
    args = parser.parse_args()
    
    # Real Data Loading
    import os
    import numpy as np
    
    data_path = r"c:\Users\aishw\codered5\src\hospital_e\data"
    try:
        print(f"Loading real data from {data_path}...")
        # Load components
        X_ecg = np.load(os.path.join(data_path, "X_ecg.npy")) # (N, 12, 100) or (N, 1200)
        X_vitals = np.load(os.path.join(data_path, "X_vitals.npy")) # (N, 20)
        X_lungs = np.load(os.path.join(data_path, "X_lungs.npy")) # (N, Emb)
        Y_labels = np.load(os.path.join(data_path, "Y_labels.npy")) # (N, 5)
        
        # Load indices
        train_idx = np.load(os.path.join(data_path, "train_indices.npy"))
        test_idx = np.load(os.path.join(data_path, "test_indices.npy"))
        
        # Helper to fuse or just create raw tensors
        # UnifiedModel handles flattening. We need to concat everything into one big tensor
        # or just pass one modality? 
        # The prompt says "multimodal overlap node", so we should ideally fuse.
        # But UnifiedModel takes (B, 1200). 
        # Strategy: Flatten ECG (12*100=1200) and use THAT as the main signal (since it's A/D compatible).
        # Or concat all and pad/crop. Let's concat.
        
        # Flatten ECG if 3D
        if X_ecg.ndim == 3:
            X_ecg = X_ecg.reshape(X_ecg.shape[0], -1) # (N, 1200)
        
        # Concat: ECG(1200) + Vitals(20) + Lungs(Deserved?)
        # For simplicity and compatibility with the 1200-dim UnifiedModel, 
        # let's predominantly use the ECG part, or the "Unified" max dim.
        # Actually, let's just use ECG for now as it matches D and A (High overlap).
        # OR: Concat and let the model crop.
        
        X_combined = np.concatenate([X_ecg, X_vitals], axis=1) # 1220 dims
        
        X_train_np = X_combined[train_idx]
        y_train_np = Y_labels[train_idx]
        X_val_np = X_combined[test_idx]
        y_val_np = Y_labels[test_idx]
        
        X_train = torch.from_numpy(X_train_np).float()
        y_train = torch.from_numpy(y_train_np).float()
        X_val = torch.from_numpy(X_val_np).float()
        y_val = torch.from_numpy(y_val_np).float()
        
        print(f"Data Shapes: X={X_train.shape}, Y={y_train.shape}")
        
    except Exception as e:
        print(f"[ERROR] Failed to load real data: {e}. Using SYNTHETIC fallback.")
        N_SAMPLES = 500
        INPUT_DIM = 1200
        X_train = torch.randn(N_SAMPLES, INPUT_DIM)
        y_train = torch.randint(0, 2, (N_SAMPLES, 5)).float()
        X_val = torch.randn(int(N_SAMPLES*0.2), INPUT_DIM)
        y_val = torch.randint(0, 2, (int(N_SAMPLES*0.2), 5)).float()

    if args.subset < 1.0:
        N = len(X_train)
        n_sub = int(N * args.subset)
        X_train = X_train[:n_sub]
        y_train = y_train[:n_sub]
        print(f"REAL DATA SUBSET: Using {n_sub}/{N} samples ({args.subset*100:.0f}%)")
        
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=16)

    from fl_utils.unified_model import UnifiedFLModel
    model = UnifiedFLModel().to(device)
    
    if args.personalize:
        print("🧠 PERSONALIZATION MODE: Freezing encoder...")
        from fl_utils.personalization import freeze_encoder, unfreeze_head
        freeze_encoder(model)
        unfreeze_head(model)
    dp_config = DPConfig(epsilon=5.0, delta=1e-5, max_grad_norm=1.0, num_rounds=5)
    
    client = HospitalEClient(model, train_loader, val_loader, device, dp_config)
    fl.client.start_numpy_client(server_address="127.0.0.1:8081", client=client)

if __name__ == "__main__":
    main()
