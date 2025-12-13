"""
Enhanced Hospital C FL Client - X-ray/Images
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

class SimpleResNet(nn.Module):
    def __init__(self, num_classes=14):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class HospitalCClient(fl.client.NumPyClient):
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
            dp_metrics = apply_dp_to_gradients(self.model, self.dp_config, 
                                              len(self.train_loader.dataset), self.device)
            optimizer.step()
        
        auroc = self._evaluate_auroc()
        return self.get_parameters(config={}), len(self.train_loader.dataset), {
            "hospital_id": "C", "auroc": float(auroc), "loss": float(loss.item())
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
            score = roc_auc_score(np.vstack(all_targets), np.vstack(all_preds), average='macro')
            if np.isnan(score):
                print("[WARN] AUROC is NaN. Returning 0.5 fallback.")
                return 0.5
            return score
        except Exception as e:
            print(f"[ERROR] AUROC Error: {e}")
            return 0.5

def main():
    print("Hospital C - X-ray/Images FL Client")
    device = torch.device("cpu")
    
    # Argument Parser
    import argparse
    parser = argparse.ArgumentParser(description='Hospital C FL Client')
    parser.add_argument('--subset', type=float, default=1.0, help='Fraction of data to use (0.0-1.0)')
    parser.add_argument('--personalize', action='store_true', help='Run in personalization mode')
    args = parser.parse_args()
    
    # Real Data Loading
    import os
    import pandas as pd
    from PIL import Image
    import torchvision.transforms as transforms
    
    data_dir = r"c:\Users\aishw\codered5\data\hospital_c"
    try:
        print(f"Loading real data from {data_dir}...")
        labels_df = pd.read_csv(os.path.join(data_dir, "labels.csv"))
        
        # Image Transform: Resize to match 1200 input dim (e.g., 20x20x3=1200)
        # UnifiedFLModel expects (B, 1200) vector.
        img_transform = transforms.Compose([
            transforms.Resize((20, 20)),
            transforms.ToTensor(), # (3, 20, 20)
        ])
        
        X_list = []
        y_list = []
        
        # Define 5 classes map
        classes = ['No Finding', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Atelectasis']
        
        # Limit to available images if needed, or subset
        # Only use existing files
        valid_indices = []
        image_folder = os.path.join(data_dir, "images")
        
        # To speed up, maybe don't load ALL if not needed. But simulation uses SUBSET.
        # Let's load paths first.
        
        if args.subset < 1.0:
            labels_df = labels_df.sample(frac=args.subset, random_state=42).reset_index(drop=True)
            print(f"REAL DATA SUBSET: Using {len(labels_df)} samples")
            
        for idx, row in labels_df.iterrows():
            fname = row['Image Index']
            lbl_str = row['Finding Labels']
            img_path = os.path.join(image_folder, fname)
            
            if not os.path.exists(img_path):
                # Try finding it in subdirs? No, assuming flat or fallback pattern.
                continue
                
            X_list.append(img_path)
            
            # Map label
            y_vec = torch.zeros(5)
            # Simple priority mapping
            if 'No Finding' in lbl_str:
                y_vec[0] = 1.0
            else:
                found = False
                for i, c in enumerate(classes[1:], 1):
                    if c in lbl_str:
                        y_vec[i] = 1.0
                        found = True
                if not found:
                    y_vec[4] = 1.0 # Other
            y_list.append(y_vec)
            
        if len(X_list) == 0:
            raise ValueError("No valid images found")
            
        # Custom Dataset to lazy load to save memory/startup time
        class XrayDataset(torch.utils.data.Dataset):
            def __init__(self, paths, targets, transform):
                self.paths = paths
                self.targets = targets
                self.transform = transform
            def __len__(self):
                return len(self.paths)
            def __getitem__(self, idx):
                try:
                    img = Image.open(self.paths[idx]).convert('RGB')
                    img_t = self.transform(img)
                    return img_t.view(-1), self.targets[idx] # Flatten to 1200
                except:
                    return torch.zeros(1200), self.targets[idx]

        full_dataset = XrayDataset(X_list, y_list, img_transform)
        
        # Split
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=16)
        
        print(f"Data Loaded. Train: {len(train_ds)}, Val: {len(val_ds)}")
        
    except Exception as e:
        print(f"[ERROR] Failed to load real data: {e}. Using SYNTHETIC fallback.")
        N_SAMPLES = 300
        INPUT_DIM = 1200
        X_train = torch.randn(N_SAMPLES, INPUT_DIM)
        y_train = torch.randint(0, 2, (N_SAMPLES, 5)).float()
        X_val = torch.randn(int(N_SAMPLES*0.2), INPUT_DIM)
        y_val = torch.randint(0, 2, (int(N_SAMPLES*0.2), 5)).float()
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
    
    client = HospitalCClient(model, train_loader, val_loader, device, dp_config)
    fl.client.start_numpy_client(server_address="127.0.0.1:8081", client=client)

if __name__ == "__main__":
    main()
