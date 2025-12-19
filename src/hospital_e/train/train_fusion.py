import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.append(os.getcwd())
from src.hospital_e.models.fusion_classifier import FusionClassifier

class MultimodalDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        # Load indices
        indices_file = "train_indices.npy" if split == 'train' else "test_indices.npy"
        self.indices = np.load(os.path.join(data_dir, indices_file))
        
        # Load all data (memory mapping would be better for huge data, but 3000 is small)
        self.ecg = np.load(os.path.join(data_dir, "X_ecg.npy"))[self.indices]
        self.vitals = np.load(os.path.join(data_dir, "X_vitals.npy"))[self.indices]
        self.lungs = np.load(os.path.join(data_dir, "X_lungs.npy"))[self.indices]
        self.masks = np.load(os.path.join(data_dir, "masks.npy"))[self.indices]
        self.labels = np.load(os.path.join(data_dir, "Y_labels.npy"))[self.indices]
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.ecg[idx], dtype=torch.float32),
            torch.tensor(self.vitals[idx], dtype=torch.float32),
            torch.tensor(self.lungs[idx], dtype=torch.float32),
            torch.tensor(self.masks[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32)
        )

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for ecg, vitals, lungs, masks, labels in loader:
        ecg, vitals, lungs = ecg.to(device), vitals.to(device), lungs.to(device)
        masks, labels = masks.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logits = model(ecg, vitals, lungs, masks)
        loss = criterion(logits, labels)
        
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
        for ecg, vitals, lungs, masks, labels in loader:
            ecg, vitals, lungs = ecg.to(device), vitals.to(device), lungs.to(device)
            masks, labels = masks.to(device), labels.to(device)
            
            logits = model(ecg, vitals, lungs, masks)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            all_preds.append(torch.sigmoid(logits).cpu().numpy())
            all_targets.append(labels.cpu().numpy())
            
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    try:
        auroc = roc_auc_score(all_targets, all_preds, average='macro')
    except:
        auroc = 0.0
        
    return total_loss / len(loader), auroc

def main():
    print("="*60)
    print("Hospital E: Training Multimodal Fusion Model")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data_dir = "src/hospital_e/data"
    
    # Datasets
    train_ds = MultimodalDataset(data_dir, split='train')
    val_ds = MultimodalDataset(data_dir, split='test')
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)  # Increased from 32
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    
    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples: {len(val_ds)}\n")
    
    # Model
    model = FusionClassifier(num_classes=5).to(device)
    
    # Training Config
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)  # Higher LR for faster convergence
    
    epochs = 3  # Reduced from 5
    best_auroc = 0.0
    os.makedirs('src/hospital_e/train/checkpoints', exist_ok=True)
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_auroc = validate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUROC: {val_auroc:.4f}")
        
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            torch.save(model.state_dict(), 'src/hospital_e/train/checkpoints/best_model_multimodal.pth')
            print(f"  ✓ Saved Best Model")
            
    print("\nTraining Complete!")
    print(f"Best AUROC: {best_auroc:.4f}")

if __name__ == "__main__":
    main()
