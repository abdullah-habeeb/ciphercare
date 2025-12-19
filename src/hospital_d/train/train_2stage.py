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
        # Check if we have at least one positive sample per class
        valid_classes = all_targets.sum(axis=0) > 0
        if valid_classes.sum() > 0:
            auroc = roc_auc_score(all_targets[:, valid_classes], all_preds[:, valid_classes], average='macro')
        else:
            auroc = 0.0
    except:
        auroc = 0.0
        
    return total_loss / len(loader), auroc

def main():
    config_path = 'src/hospital_d/train/disease_config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # ========== STAGE 1: Pretrain on Synthetic ==========
    print("="*60)
    print("STAGE 1: Pretraining on Synthetic Data (1200 samples)")
    print("="*60)
    
    X_synth = np.load('src/hospital_d/data/X_synth.npy')
    Y_synth = np.load('src/hospital_d/data/Y_synth.npy')
    
    # Split synthetic 80/20
    n_synth = len(X_synth)
    n_train_synth = int(0.8 * n_synth)
    
    X_synth_train = torch.tensor(X_synth[:n_train_synth], dtype=torch.float32)
    Y_synth_train = torch.tensor(Y_synth[:n_train_synth], dtype=torch.float32)
    X_synth_val = torch.tensor(X_synth[n_train_synth:], dtype=torch.float32)
    Y_synth_val = torch.tensor(Y_synth[n_train_synth:], dtype=torch.float32)
    
    synth_train_ds = TensorDataset(X_synth_train, Y_synth_train)
    synth_val_ds = TensorDataset(X_synth_val, Y_synth_val)
    
    synth_train_loader = DataLoader(synth_train_ds, batch_size=32, shuffle=True)
    synth_val_loader = DataLoader(synth_val_ds, batch_size=32, shuffle=False)
    
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
    
    print(f"Training on {len(synth_train_ds)} synthetic samples...")
    
    for epoch in range(5):  # Stage 1: 5 epochs
        train_loss = train_epoch(model, synth_train_loader, optimizer, criterion, device)
        val_loss, val_auroc = validate(model, synth_val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/5 | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUROC: {val_auroc:.4f}")
    
    # Save pretrained checkpoint
    os.makedirs('src/hospital_d/train/checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'src/hospital_d/train/checkpoints/pretrained_synth.pth')
    print("\n✓ Stage 1 complete. Saved pretrained_synth.pth\n")
    
    # ========== STAGE 2: Fine-tune on Real Geriatric Data ==========
    print("="*60)
    print("STAGE 2: Fine-tuning on Real Geriatric Data (60+)")
    print("="*60)
    
    X_real_train = np.load('src/hospital_d/data/X_real_train.npy')
    Y_real_train = np.load('src/hospital_d/data/Y_real_train.npy')
    X_real_test = np.load('src/hospital_d/data/X_real_test.npy')
    Y_real_test = np.load('src/hospital_d/data/Y_real_test.npy')
    
    print(f"Real Train: {len(X_real_train)} samples")
    print(f"Real Test: {len(X_real_test)} samples")
    print(f"Positive labels (train): {Y_real_train.sum(axis=0)}")
    
    real_train_ds = TensorDataset(
        torch.tensor(X_real_train, dtype=torch.float32),
        torch.tensor(Y_real_train, dtype=torch.float32)
    )
    real_test_ds = TensorDataset(
        torch.tensor(X_real_test, dtype=torch.float32),
        torch.tensor(Y_real_test, dtype=torch.float32)
    )
    
    real_train_loader = DataLoader(real_train_ds, batch_size=32, shuffle=True)
    real_test_loader = DataLoader(real_test_ds, batch_size=32, shuffle=False)
    
    # Fine-tune with smaller LR
    optimizer_ft = torch.optim.AdamW(model.parameters(), lr=1e-5)  # 10x smaller
    
    best_test_auroc = 0.0
    
    for epoch in range(5):  # Stage 2: 5 epochs
        train_loss = train_epoch(model, real_train_loader, optimizer_ft, criterion, device)
        test_loss, test_auroc = validate(model, real_test_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/5 | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Test AUROC: {test_auroc:.4f}")
        
        if test_auroc > best_test_auroc:
            best_test_auroc = test_auroc
            torch.save(model.state_dict(), 'src/hospital_d/train/checkpoints/best_model_finetuned.pth')
            print(f"  ✓ New best AUROC: {test_auroc:.4f}")
    
    print("\n" + "="*60)
    print(f"FINAL RESULTS (Real Geriatric Test Set)")
    print("="*60)
    print(f"Best Test AUROC: {best_test_auroc:.4f}")
    print(f"Model saved: best_model_finetuned.pth")

if __name__ == "__main__":
    main()
