import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# Add src to path
sys.path.append(os.getcwd())

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
        auroc = roc_auc_score(all_targets, all_preds, average='macro')
    except:
        auroc = 0.0
        
    return total_loss / len(loader), auroc

def main():
    config_path = 'src/hospital_d/train/disease_config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Data
    print("Loading synthetic data...")
    X = np.load(config['train_signals']) # [N, 8, 1000]
    Y = np.load(config['train_labels'])  # [N, 5]
    
    # Convert to Tensor
    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.float32)
    
    dataset = TensorDataset(X_tensor, Y_tensor)
    
    # 2. Split 80/20
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False)
    
    # 3. Model
    mc = config['model_config']
    model = HospitalDClassifier(
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
    ).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
    
    os.makedirs(config['output_dir'], exist_ok=True)
    best_loss = float('inf')
    
    print("Starting training...")
    for epoch in range(config['epochs']):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_auroc = validate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{config['epochs']} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUROC: {val_auroc:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            save_path = os.path.join(config['output_dir'], 'best_model_synth.pth')
            torch.save(model.state_dict(), save_path)
            print(f"  Saved best model to {save_path}")

if __name__ == "__main__":
    main()
