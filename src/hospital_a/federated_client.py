import flwr as fl
import torch
import numpy as np
import torch.nn as nn
from src.hospital_a.train.train_disease import train_epoch, validate

class HospitalAClient(fl.client.NumPyClient):
    """
    Federated Learning Client for Hospital A (Cardiology Node)
    """
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion = nn.BCEWithLogitsLoss()
        
    def get_parameters(self, config):
        """Return model parameters as a list of NumPy arrays"""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        """Set model parameters from a list of NumPy arrays"""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        """Train model locally and return updated parameters"""
        self.set_parameters(parameters)
        
        # Configure optimizer (re-initialized each round or kept persistent?) 
        # For simplicity, re-init with frozen encoder as per local training config
        # Freeze Encoder
        for param in self.model.residual_layer.parameters():
            param.requires_grad = False
        for param in self.model.init_conv.parameters(): 
            param.requires_grad = False
            
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.config['lr'])
        
        # Train
        epochs = self.config.get('local_epochs', 1)
        train_loss = 0.0
        for epoch in range(epochs):
            train_loss = train_epoch(self.model, self.train_loader, optimizer, self.criterion, self.device, epoch, epochs)
            
        return self.get_parameters(config={}), len(self.train_loader.dataset), {"train_loss": train_loss}
    
    def evaluate(self, parameters, config):
        """Evaluate model on local validation set"""
        self.set_parameters(parameters)
        val_loss, auroc = validate(self.model, self.val_loader, self.criterion, self.device)
        return float(val_loss), len(self.val_loader.dataset), {"auroc": float(auroc)}

if __name__ == "__main__":
    from src.hospital_a.utils.ecg_preprocessing import PTBXL_DiseaseDataset
    from torch.utils.data import DataLoader
    import json
    import os
    import sys
    
    # Setup for standalone test
    sys.path.append(os.getcwd())
    
    # Load Config
    config_path = 'src/hospital_a/config/disease_config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Model
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
    
    # Load Data (Small subset for testing connection)
    print("Loading Data for Client...")
    # Use validation set as mock train set for speed in testing
    val_dataset = PTBXL_DiseaseDataset(
        signals_path=config['val_signals'],
        labels_csv_path=config['labels_csv'],
        ids_path=config.get('val_ids'),
        leads_idx=config.get('leads_idx', [0,1,6,7,8,9,10,11])
    )
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Create Client
    client = HospitalAClient(model, val_loader, val_loader, config) # Using val_loader for both for testing
    
    print("Hospital A Federated Client - Initialized & Ready")
    # fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)
