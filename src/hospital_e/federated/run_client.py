import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import OrderedDict
import flwr as fl
import numpy as np

sys.path.append(os.getcwd())
from src.hospital_e.models.fusion_classifier import FusionClassifier
from src.hospital_e.train.train_fusion import MultimodalDataset

# Config
SERVER_ADDRESS = "localhost:8080"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HospitalEClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = FusionClassifier(num_classes=5).to(DEVICE)
        self.train_loader = DataLoader(
            MultimodalDataset("src/hospital_e/data", split='train'),
            batch_size=32, shuffle=True
        )
        self.val_loader = DataLoader(
            MultimodalDataset("src/hospital_e/data", split='test'),
            batch_size=32, shuffle=False
        )
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        
        # Safe loading for S4
        self.model.load_state_dict(state_dict, strict=True) 

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        
        # Train 1 epoch
        epoch_loss = 0.0
        for ecg, vitals, lungs, masks, labels in self.train_loader:
            ecg, vitals, lungs = ecg.to(DEVICE), vitals.to(DEVICE), lungs.to(DEVICE)
            masks, labels = masks.to(DEVICE), labels.to(DEVICE)
            
            self.optimizer.zero_grad()
            logits = self.model(ecg, vitals, lungs, masks)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
            
        print(f"Hospital E fit complete. Loss: {epoch_loss/len(self.train_loader):.4f}")
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        loss = 0.0
        correct = 0 # Not fitting for BCE, use Loss
        
        with torch.no_grad():
            for ecg, vitals, lungs, masks, labels in self.val_loader:
                ecg, vitals, lungs = ecg.to(DEVICE), vitals.to(DEVICE), lungs.to(DEVICE)
                masks, labels = masks.to(DEVICE), labels.to(DEVICE)
                
                logits = self.model(ecg, vitals, lungs, masks)
                l = self.criterion(logits, labels)
                loss += l.item()
                
        # Return loss and size
        return float(loss / len(self.val_loader)), len(self.val_loader.dataset), {"loss": float(loss/len(self.val_loader))}

def main():
    print("Starting Hospital E FL Client...")
    
    # Try to load pretrained weights if available
    ckpt_path = "src/hospital_e/train/checkpoints/best_model_multimodal.pth"
    if os.path.exists(ckpt_path):
        print("Loading local pretrained weights...")
        client = HospitalEClient()
        try:
             client.model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
             print("✓ Loaded local weights")
        except:
             print("⚠️ Could not load local weights, starting fresh")
             
        fl.client.start_numpy_client(server_address=SERVER_ADDRESS, client=client)
    else:
        print("Starting fresh client...")
        fl.client.start_numpy_client(server_address=SERVER_ADDRESS, client=HospitalEClient())

if __name__ == "__main__":
    main()
