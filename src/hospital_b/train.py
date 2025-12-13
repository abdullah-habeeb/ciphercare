import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from src.hospital_b.dataset import ClinicalDataset
from src.hospital_b.model import get_model
import os
import json
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix

def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate binary classification metrics"""
    metrics = {}
    
    metrics['AUROC'] = float(roc_auc_score(y_true, y_prob))
    metrics['Precision'] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics['Recall'] = float(recall_score(y_true, y_pred, zero_division=0))
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['Specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    metrics['F1'] = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
    
    return metrics

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(features).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * features.size(0)
    return running_loss / len(loader.dataset)

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_probs = []
    all_preds = []
    
    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features).squeeze()
            loss = criterion(outputs, labels)
            running_loss += loss.item() * features.size(0)
            
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    avg_loss = running_loss / len(loader.dataset)
    
    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    y_pred = np.array(all_preds)
    
    metrics = calculate_metrics(y_true, y_pred, y_prob)
    metrics['loss'] = avg_loss
    
    return metrics

def local_train(csv_path, output_dir, epochs=15, batch_size=64, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    full_dataset = ClinicalDataset(csv_path)
    
    # Split: 70% train, 15% val, 15% personalization
    total_size = len(full_dataset)
    train_size = int(0.70 * total_size)
    val_size = int(0.15 * total_size)
    pers_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, pers_dataset = random_split(
        full_dataset, [train_size, val_size, pers_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    pers_loader = DataLoader(pers_dataset, batch_size=batch_size, shuffle=True)
    
    # Model
    model = get_model(input_dim=full_dataset.input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Global Training
    print("--- Starting Global Training Phase ---")
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | AUROC: {val_metrics['AUROC']:.4f}")
    
    # Save global model
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    global_model_path = os.path.join(output_dir, "hospital2_model.pth")
    torch.save(model.state_dict(), global_model_path)
    
    with open(os.path.join(output_dir, "global_results.json"), "w") as f:
        json.dump(val_metrics, f, indent=4)
    
    # Personalization Phase
    print("\n--- Starting Personalization Phase ---")
    pre_pers_metrics = evaluate(model, pers_loader, criterion, device)
    print(f"Pre-Personalization AUROC: {pre_pers_metrics['AUROC']:.4f}")
    
    # Freeze all except last layer
    for name, param in model.named_parameters():
        if "network" in name and "6" not in name:  # Freeze all except final layer
            param.requires_grad = False
    
    optimizer_pers = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    
    for epoch in range(3):
        p_loss = train_epoch(model, pers_loader, criterion, optimizer_pers, device)
        print(f"Pers Epoch {epoch+1}/3 | Loss: {p_loss:.4f}")
    
    post_pers_metrics = evaluate(model, pers_loader, criterion, device)
    print(f"Post-Personalization AUROC: {post_pers_metrics['AUROC']:.4f}")
    
    pers_results = {
        "pre_personalization": pre_pers_metrics,
        "post_personalization": post_pers_metrics,
        "improvement": post_pers_metrics['AUROC'] - pre_pers_metrics['AUROC']
    }
    
    with open(os.path.join(output_dir, "personalized_results.json"), "w") as f:
        json.dump(pers_results, f, indent=4)
    
    return global_model_path, model

if __name__ == "__main__":
    CSV_PATH = r"c:\Users\aishw\codered5\data\hospital_b\processed_vitals.csv"
    OUTPUT_DIR = r"c:\Users\aishw\codered5\ml\models"
    local_train(CSV_PATH, OUTPUT_DIR, epochs=10)
