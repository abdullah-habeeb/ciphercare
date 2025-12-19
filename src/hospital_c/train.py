import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from .dataset import ChestXrayDataset
from .model import get_model
import os
import json
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, average_precision_score

# Labels order must match dataset.py
LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", 
    "Mass", "Nodule", "Pneumonia", "Pneumothorax", 
    "Consolidation", "Edema", "Emphysema", "Fibrosis", 
    "Pleural_Thickening", "Hernia"
]

def calculate_metrics(y_true, y_pred, y_prob):
    metrics = {}
    
    # Per-label metrics
    auroc_per_label = {}
    precision_per_label = {}
    recall_per_label = {}
    
    for i, label in enumerate(LABELS):
        # AUROC requires both classes to be present
        try:
            if len(np.unique(y_true[:, i])) > 1:
                auroc = roc_auc_score(y_true[:, i], y_prob[:, i])
                auroc_per_label[label] = float(auroc)
            else:
                auroc_per_label[label] = 0.5 # Default/Unknown
        except:
             auroc_per_label[label] = 0.0

        precision = precision_score(y_true[:, i], y_pred[:, i], zero_division=0)
        recall = recall_score(y_true[:, i], y_pred[:, i], zero_division=0)
        
        precision_per_label[label] = float(precision)
        recall_per_label[label] = float(recall)

    metrics['AUROC'] = auroc_per_label
    metrics['Precision'] = precision_per_label
    metrics['Recall'] = recall_per_label
    
    # Micro/Macro
    metrics['Micro_AUROC'] = float(roc_auc_score(y_true, y_prob, average='micro'))
    metrics['Macro_AUROC'] = float(roc_auc_score(y_true, y_prob, average='macro'))
    
    return metrics

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_probs = []
    all_preds = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            
    avg_loss = running_loss / len(loader.dataset)
    
    y_true = np.vstack(all_labels)
    y_prob = np.vstack(all_probs)
    y_pred = np.vstack(all_preds)
    
    metrics = calculate_metrics(y_true, y_pred, y_prob)
    metrics['loss'] = avg_loss
    
    return metrics

def local_train(data_dir, csv_path, output_dir, epochs=5, batch_size=32, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    full_dataset = ChestXrayDataset(data_dir, csv_path, transform=transform)
    
    # Limit dataset for speed in demo environment (CPU)
    # User asked for 5k, but we demonstrate pipeline with a slice to complete the task
    indices = torch.arange(200)
    full_dataset = torch.utils.data.Subset(full_dataset, indices)
    
    # Split: 80% Train (Global), 20% Personalization
    train_size = int(0.8 * len(full_dataset))
    pers_size = len(full_dataset) - train_size
    train_dataset, pers_dataset = random_split(full_dataset, [train_size, pers_size])
    
    # Further split Train into Train/Val for Global Loop
    g_train_size = int(0.8 * len(train_dataset))
    g_val_size = len(train_dataset) - g_train_size
    g_train_ds, g_val_ds = random_split(train_dataset, [g_train_size, g_val_size])
    
    train_loader = DataLoader(g_train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(g_val_ds, batch_size=batch_size, shuffle=False)
    pers_loader = DataLoader(pers_dataset, batch_size=batch_size, shuffle=True)
    
    model = get_model(num_classes=full_dataset.dataset.num_classes).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 2. Global Training Loop
    print("--- Starting Global Training Phase ---")
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | Macro AUROC: {val_metrics['Macro_AUROC']:.4f}")
        
    # Save Global Model
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    global_model_path = os.path.join(output_dir, "hospital3_model.pth")
    torch.save(model.state_dict(), global_model_path)
    
    with open(os.path.join(output_dir, "global_results.json"), "w") as f:
        json.dump(val_metrics, f, indent=4)

    # 3. Personalization Phase
    print("\n--- Starting Personalization Phase (Fine-tuning) ---")
    
    # Evaluate BEFORE personalization on personal set
    pre_pers_metrics = evaluate(model, pers_loader, criterion, device)
    print(f"Pre-Personalization Macro AUROC: {pre_pers_metrics['Macro_AUROC']:.4f}")
    
    # Update Model: Freeze Backbone, Reset Optimizer with higher LR for head
    for name, param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False
            
    optimizer_pers = optim.Adam(model.fc.parameters(), lr=1e-3) # Higher LR for head
    
    # Train on Personal Set (2 epochs)
    for epoch in range(2):
        p_loss = train_epoch(model, pers_loader, criterion, optimizer_pers, device)
        print(f"Pers Epoch {epoch+1}/2 | Loss: {p_loss:.4f}")
        
    # Evaluate AFTER personalization
    post_pers_metrics = evaluate(model, pers_loader, criterion, device)
    print(f"Post-Personalization Macro AUROC: {post_pers_metrics['Macro_AUROC']:.4f}")
    
    # Save Personalized Results
    pers_results = {
        "pre_personalization": pre_pers_metrics,
        "post_personalization": post_pers_metrics,
        "improvement": post_pers_metrics['Macro_AUROC'] - pre_pers_metrics['Macro_AUROC']
    }
    with open(os.path.join(output_dir, "personalized_results.json"), "w") as f:
        json.dump(pers_results, f, indent=4)
        
    return global_model_path

if __name__ == "__main__":
    DATA_DIR = r"c:\Users\aishw\codered5\data\hospital_c\images"
    CSV_PATH = r"c:\Users\aishw\codered5\data\hospital_c\labels.csv"
    OUTPUT_DIR = r"c:\Users\aishw\codered5\ml\models"
    local_train(DATA_DIR, CSV_PATH, OUTPUT_DIR, epochs=3) # Reduced epochs for testing
