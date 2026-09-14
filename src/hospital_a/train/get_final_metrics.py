import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import numpy as np
from tqdm import tqdm

sys.path.append(os.getcwd())

from src.hospital_a.models.encoder import ECGClassifier
from src.hospital_a.utils.ecg_preprocessing import PTBXL_DiseaseDataset

def evaluate_trained_model():
    # Load config
    with open('src/hospital_a/config/disease_config.json', 'r') as f:
        config = json.load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = PTBXL_DiseaseDataset(
        signals_path=config['train_signals'],
        labels_csv_path=config['labels_csv'],
        ids_path=config.get('train_ids'),
        leads_idx=config.get('leads_idx', [0,1,6,7,8,9,10,11])
    )
    val_dataset = PTBXL_DiseaseDataset(
        signals_path=config['val_signals'],
        labels_csv_path=config['labels_csv'],
        ids_path=config.get('val_ids'),
        leads_idx=config.get('leads_idx', [0,1,6,7,8,9,10,11])
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Build model
    print("Building model...")
    mc = config['model_config']
    model = ECGClassifier(
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
    
    # Load trained weights
    checkpoint_path = 'src/hospital_a/train/checkpoints/best_model.pth'
    print(f"Loading trained model from {checkpoint_path}...\n")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    criterion = nn.BCEWithLogitsLoss()
    
    print("="*70)
    print("EVALUATING TRAINED MODEL - FINAL METRICS")
    print("="*70)
    
    # Evaluate on training set
    print("\n📊 TRAINING SET EVALUATION")
    train_loss = 0
    train_preds = []
    train_targets = []
    
    with torch.no_grad():
        for x, y in tqdm(train_loader, desc="Evaluating Train"):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            train_loss += loss.item()
            train_preds.append(torch.sigmoid(logits).cpu().numpy())
            train_targets.append(y.cpu().numpy())
    
    train_preds = np.concatenate(train_preds)
    train_targets = np.concatenate(train_targets)
    train_loss = train_loss / len(train_loader)
    
    try:
        train_auroc = roc_auc_score(train_targets, train_preds, average='macro')
    except:
        train_auroc = 0.0
    
    print(f"  Train Loss: {train_loss:.4f}")
    print(f"  Train AUROC (Macro): {train_auroc:.4f}")
    
    # Evaluate on validation set
    print("\n📊 VALIDATION SET EVALUATION")
    val_loss = 0
    val_preds = []
    val_targets = []
    
    with torch.no_grad():
        for x, y in tqdm(val_loader, desc="Evaluating Val"):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            val_loss += loss.item()
            val_preds.append(torch.sigmoid(logits).cpu().numpy())
            val_targets.append(y.cpu().numpy())
    
    val_preds = np.concatenate(val_preds)
    val_targets = np.concatenate(val_targets)
    val_loss = val_loss / len(val_loader)
    
    try:
        val_auroc = roc_auc_score(val_targets, val_preds, average='macro')
    except:
        val_auroc = 0.0
    
    print(f"  Val Loss: {val_loss:.4f}")
    print(f"  Val AUROC (Macro): {val_auroc:.4f}")
    
    # Per-class AUROC
    classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    print("\n📈 PER-CLASS AUROC (Validation Set):")
    print("-" * 40)
    for i, cls in enumerate(classes):
        try:
            auroc = roc_auc_score(val_targets[:, i], val_preds[:, i])
            print(f"  {cls:8s}: {auroc:.4f}")
        except:
            print(f"  {cls:8s}: N/A")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Final Train Loss: {train_loss:.4f}")
    print(f"Final Train AUROC: {train_auroc:.4f}")
    print(f"Final Val Loss: {val_loss:.4f}")
    print(f"Final Val AUROC: {val_auroc:.4f}")
    print("="*70)
    
    # Save metrics to file
    metrics = {
        "train_loss": float(train_loss),
        "train_auroc": float(train_auroc),
        "val_loss": float(val_loss),
        "val_auroc": float(val_auroc),
        "per_class_auroc": {
            cls: float(roc_auc_score(val_targets[:, i], val_preds[:, i]))
            for i, cls in enumerate(classes)
            if len(np.unique(val_targets[:, i])) > 1
        }
    }
    
    with open('src/hospital_a/train/final_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✓ Metrics saved to: src/hospital_a/train/final_metrics.json")

if __name__ == '__main__':
    evaluate_trained_model()
