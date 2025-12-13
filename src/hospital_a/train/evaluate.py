import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import numpy as np

sys.path.append(os.getcwd())

from src.hospital_a.models.encoder import ECGClassifier
from src.hospital_a.utils.ecg_preprocessing import PTBXL_DiseaseDataset

def evaluate_model():
    # Load config
    with open('src/hospital_a/config/disease_config.json', 'r') as f:
        config = json.load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load validation dataset
    print("Loading validation dataset...")
    val_dataset = PTBXL_DiseaseDataset(
        signals_path=config['val_signals'],
        labels_csv_path=config['labels_csv'],
        ids_path=config.get('val_ids'),
        leads_idx=config.get('leads_idx', [0,1,6,7,8,9,10,11])
    )
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
    print(f"Loading trained model from {checkpoint_path}...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    # Evaluate
    print("\nEvaluating on validation set...\n")
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    # Calculate metrics
    classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    
    print("="*60)
    print("HOSPITAL A - CARDIOLOGY MODEL EVALUATION")
    print("="*60)
    
    # Overall AUROC
    try:
        overall_auroc = roc_auc_score(all_targets, all_preds, average='macro')
        print(f"\n★ Overall AUROC (Macro): {overall_auroc:.4f}")
    except:
        overall_auroc = 0.0
        print("\n★ Overall AUROC: Could not compute")
    
    # Per-class AUROC
    print("\nPer-Class AUROC:")
    print("-" * 40)
    for i, cls in enumerate(classes):
        try:
            auroc = roc_auc_score(all_targets[:, i], all_preds[:, i])
            print(f"  {cls:8s}: {auroc:.4f}")
        except:
            print(f"  {cls:8s}: N/A")
    
    # Binary predictions (threshold = 0.5)
    binary_preds = (all_preds > 0.5).astype(int)
    
    print("\nClass Distribution in Validation Set:")
    print("-" * 40)
    for i, cls in enumerate(classes):
        pos_count = all_targets[:, i].sum()
        total = len(all_targets)
        print(f"  {cls:8s}: {int(pos_count):4d}/{total} ({pos_count/total*100:.1f}%)")
    
    print("\n" + "="*60)
    print(f"\n✓ Evaluation complete!")
    print(f"  Model: {checkpoint_path}")
    print(f"  Samples evaluated: {len(all_targets)}")
    print(f"  Overall AUROC: {overall_auroc:.4f}")
    print("="*60)

if __name__ == '__main__':
    evaluate_model()
