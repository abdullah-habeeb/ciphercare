import flwr as fl
import torch
import torch.nn as nn
import numpy as np
import json
import os
import sys

# Add src to path
sys.path.append(os.getcwd())

from src.hospital_d.models.classifier import HospitalDClassifier
from src.hospital_d.train.train_classifier import train_epoch, validate
from torch.utils.data import DataLoader, TensorDataset, random_split

class HospitalDClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion = nn.BCEWithLogitsLoss()
        
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
        
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        # Optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config['lr'])
        
        # Train locally
        epochs = self.config.get('local_epochs', 1)
        train_loss = 0.0
        for epoch in range(epochs):
            train_loss = train_epoch(self.model, self.train_loader, optimizer, self.criterion, self.device)
            
        return self.get_parameters(config={}), len(self.train_loader.dataset), {"train_loss": float(train_loss)}
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        val_loss, val_auroc = validate(self.model, self.val_loader, self.criterion, self.device)
        return float(val_loss), len(self.val_loader.dataset), {"auroc": float(val_auroc)}

def load_data_and_model(config_path='src/hospital_d/train/disease_config.json'):
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Data
    X = np.load(config['train_signals'])
    Y = np.load(config['train_labels'])
    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, Y_tensor)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False)
    
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
    
    return model, train_loader, val_loader, config

if __name__ == '__main__':
    # Standalone test
    print("Initializing Hospital D Client...")
    model, train_loader, val_loader, config = load_data_and_model()
    client = HospitalDClient(model, train_loader, val_loader, config)
    print("Hospital D Client Ready.")
    # fl.client.start_numpy_client(server_address="...", client=client)
