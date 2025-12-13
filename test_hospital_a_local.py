
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from sklearn.metrics import roc_auc_score

# Configuration
DATA_PATH = r"c:\Users\aishw\codered5\src\hospital_a\data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 5

def load_data():
    print(f"Loading data from {DATA_PATH}...")
    X_train = torch.from_numpy(np.load(os.path.join(DATA_PATH, "X_train.npy"))).float()
    y_train = torch.from_numpy(np.load(os.path.join(DATA_PATH, "Y_train.npy"))).float()
    X_val = torch.from_numpy(np.load(os.path.join(DATA_PATH, "X_val.npy"))).float()
    y_val = torch.from_numpy(np.load(os.path.join(DATA_PATH, "Y_val.npy"))).float()
    
    # Subset for speed
    X_train = X_train[:1000]
    y_train = y_train[:1000]
    X_val = X_val[:200]
    y_val = y_val[:200]
    
    print(f"Data Loaded. Train: {X_train.shape}, Val: {X_val.shape}")
    return X_train, y_train, X_val, y_val

class UnifiedFLModel(nn.Module):
    def __init__(self, input_dim=1200, classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, classes)
        )
        
    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        if x.size(1) != 1200:
            if x.size(1) < 1200:
                padding = torch.zeros(x.size(0), 1200 - x.size(1), device=x.device)
                x = torch.cat([x, padding], dim=1)
            else:
                x = x[:, :1200]
        return self.net(x)

def train():
    X_train, y_train, X_val, y_val = load_data()
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE)
    
    model = UnifiedFLModel().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    
    print("\nStarting Local Training...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = torch.sigmoid(model(data))
                all_preds.append(output.cpu().numpy())
                all_targets.append(target.cpu().numpy())
        
        y_true = np.vstack(all_targets)
        y_pred = np.vstack(all_preds)
        try:
            auroc = roc_auc_score(y_true, y_pred, average='macro')
        except:
            auroc = 0.5
            
        print(f"Epoch {epoch+1}: Loss = {train_loss/len(train_loader):.4f}, AUROC = {auroc:.4f}")

if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print(f"Error: {e}")
