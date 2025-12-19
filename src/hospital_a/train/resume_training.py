"""
Resume training from a checkpoint if your laptop shut down.

Usage:
    python src/hospital_a/train/resume_training.py
"""
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

def resume_training(config_path, checkpoint_path='src/hospital_a/train/checkpoints/latest_checkpoint.pth'):
    print(f"Loading config from {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data
    print("Initializing Datasets...")
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
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Model
    print("Building Model...")
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
    
    # Freeze Encoder
    for param in model.residual_layer.parameters():
        param.requires_grad = False
    for param in model.init_conv.parameters(): 
        param.requires_grad = False
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config['lr'])
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    best_val_loss = checkpoint['best_val_loss']
    
    print(f"✓ Resuming from Epoch {start_epoch}")
    print(f"  Previous AUROC: {checkpoint['auroc']:.4f}")
    print(f"  Best Val Loss: {best_val_loss:.4f}")
    
    # Continue training
    print("Continuing Training Loop...")
    for epoch in range(start_epoch, config['epochs']):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item()
                all_preds.append(torch.sigmoid(logits).cpu().numpy())
                all_targets.append(y.cpu().numpy())
        
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        
        try:
            auroc = roc_auc_score(all_targets, all_preds, average='macro')
        except ValueError:
            auroc = 0.0
            
        print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f} | Val AUROC: {auroc:.4f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'train_loss': train_loss/len(train_loader),
            'val_loss': val_loss/len(val_loader),
            'auroc': auroc
        }
        checkpoint_path_new = os.path.join(config['output_dir'], f'checkpoint_epoch_{epoch+1}.pth')
        torch.save(checkpoint, checkpoint_path_new)
        print(f"✓ Saved checkpoint: {checkpoint_path_new}")
        
        latest_path = os.path.join(config['output_dir'], 'latest_checkpoint.pth')
        torch.save(checkpoint, latest_path)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(config['output_dir'], 'best_model.pth')
            torch.save(model.state_dict(), save_path)
            print(f"★ Saved Best Model to {save_path}")

if __name__ == '__main__':
    resume_training('src/hospital_a/config/disease_config.json')
