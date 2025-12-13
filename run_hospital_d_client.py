"""
Standalone Hospital D Federated Client
Connects to FL server using geriatric data.
"""
import flwr as fl
import torch
import torch.nn as nn
import numpy as np
import sys
import argparse
from torch.utils.data import DataLoader, TensorDataset

sys.path.append('.')

from src.hospital_d.models.classifier import HospitalDClassifier

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

class HospitalDClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
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
        test_loss, auroc = validate(self.model, self.test_loader, self.criterion, self.device)
        return float(test_loss), len(self.test_loader.dataset), {"auroc": float(auroc)}

def main(server_address="127.0.0.1:8080"):
    print("="*60)
    print("Hospital D - Federated Client (Geriatric)")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load geriatric data
    print("Loading geriatric data...")
    X_train = np.load('src/hospital_d/data/X_real_train_fixed.npy')
    Y_train = np.load('src/hospital_d/data/Y_real_train.npy')
    X_test = np.load('src/hospital_d/data/X_real_test_fixed.npy')
    Y_test = np.load('src/hospital_d/data/Y_real_test.npy')
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.float32)),
        batch_size=32, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(Y_test, dtype=torch.float32)),
        batch_size=32, shuffle=False
    )
    
    # Initialize model
    print("Initializing model...")
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
    
    # Create client
    client = HospitalDClient(model, train_loader, test_loader, device)
    
    print(f"Connecting to server: {server_address}")
    print("="*60)
    
    # Start client
    fl.client.start_numpy_client(server_address=server_address, client=client)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=str, default="127.0.0.1:8080")
    args = parser.parse_args()
    
    main(args.server)
