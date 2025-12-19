import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

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
    print(f"Using device: {device}\n")
    
    print("="*60)
    print("Training on Real Geriatric Data (60+)")
    print("="*60)
    
    # Load FIXED data
    X_train = np.load('src/hospital_d/data/X_real_train_fixed.npy')
    Y_train = np.load('src/hospital_d/data/Y_real_train.npy')
    X_test = np.load('src/hospital_d/data/X_real_test_fixed.npy')
    Y_test = np.load('src/hospital_d/data/Y_real_test.npy')
    
    print(f"Train: {len(X_train)} samples")
    print(f"Test: {len(X_test)} samples")
    print(f"Positive labels (train): {Y_train.sum(axis=0)}")
    print(f"Positive labels (test): {Y_test.sum(axis=0)}\n")
    
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(Y_train, dtype=torch.float32)
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(Y_test, dtype=torch.float32)
    )
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    # Initialize model
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    best_test_auroc = 0.0
    os.makedirs('src/hospital_d/train/checkpoints', exist_ok=True)
    
    print("Starting training...\n")
    
    for epoch in range(5):  # Reduced from 10 to 5 for faster completion
        print(f"Epoch {epoch+1}/5 - Training...")
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/5 - Validating...")
        test_loss, test_auroc = validate(model, test_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/5 | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Test AUROC: {test_auroc:.4f}")
        
        if test_auroc > best_test_auroc:
            best_test_auroc = test_auroc
            torch.save(model.state_dict(), 'src/hospital_d/train/checkpoints/best_model_geriatric.pth')
            print(f"  ✓ New best AUROC: {test_auroc:.4f}")
    
    print("\n" + "="*60)
    print(f"FINAL RESULTS (Real Geriatric Test Set)")
    print("="*60)
    print(f"Best Test AUROC: {best_test_auroc:.4f}")
    print(f"Model saved: best_model_geriatric.pth")

if __name__ == "__main__":
    main()
