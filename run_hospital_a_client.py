"""
Standalone Hospital A Federated Client
Connects to FL server and participates in federated rounds.
"""
import flwr as fl
import torch
import torch.nn as nn
import numpy as np
import json
import sys
import argparse
from torch.utils.data import DataLoader, TensorDataset

sys.path.append('.')

from src.hospital_a.models.encoder import ECGClassifier

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y.float())
            total_loss += loss.item()
            all_preds.append(torch.sigmoid(logits).cpu().numpy())
            all_targets.append(y.cpu().numpy())
            
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    try:
        from sklearn.metrics import roc_auc_score
        auroc = roc_auc_score(all_targets, all_preds, average='macro')
    except:
        auroc = 0.0
        
    return total_loss / len(loader), auroc

class HospitalAClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = nn.BCEWithLogitsLoss()
        
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        
        # 1 local epoch per FL round
        train_loss = train_epoch(self.model, self.train_loader, optimizer, self.criterion, self.device)
        
        return self.get_parameters(config={}), len(self.train_loader.dataset), {"train_loss": float(train_loss)}
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        val_loss, auroc = validate(self.model, self.val_loader, self.criterion, self.device)
        return float(val_loss), len(self.val_loader.dataset), {"auroc": float(auroc)}

def main(server_address="127.0.0.1:8080"):
    print("="*60)
    print("Hospital A - Federated Client")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data (use a small subset for FL testing)
    print("Loading data...")
    X_train = np.load('src/hospital_a/data/X_train.npy')[:1000]  # First 1000 samples
    Y_train = np.load('src/hospital_a/data/Y_train.npy')[:1000]
    X_val = np.load('src/hospital_a/data/X_val.npy')[:200]
    Y_val = np.load('src/hospital_a/data/Y_val.npy')[:200]
    
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.float32)),
        batch_size=32, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(Y_val, dtype=torch.float32)),
        batch_size=32, shuffle=False
    )
    
    # Initialize model
    print("Initializing model...")
    model = ECGClassifier(
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
    
    # Create client
    client = HospitalAClient(model, train_loader, val_loader, device)
    
    print(f"Connecting to server: {server_address}")
    print("="*60)
    
    # Start client
    fl.client.start_numpy_client(server_address=server_address, client=client)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=str, default="127.0.0.1:8080")
    args = parser.parse_args()
    
    main(args.server)
