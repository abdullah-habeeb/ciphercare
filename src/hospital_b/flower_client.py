import flwr as fl
import torch
from collections import OrderedDict
from src.hospital_b.model import get_model

class HospitalBClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = torch.nn.BCEWithLogitsLoss()
    
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def fit(self, parameters, config):
        # Update model with global parameters
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
        
        # Train locally
        epochs = int(config.get("local_epochs", 1))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        
        self.model.train()
        self.model.to(self.device)
        
        for _ in range(epochs):
            for features, labels in self.train_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(features).squeeze()
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}
    
    def evaluate(self, parameters, config):
        # Update model
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
        
        # Evaluate
        self.model.eval()
        loss = 0.0
        
        with torch.no_grad():
            for features, labels in self.val_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                outputs = self.model(features).squeeze()
                loss += self.criterion(outputs, labels).item()
        
        loss /= len(self.val_loader.dataset)
        
        return float(loss), len(self.val_loader.dataset), {"loss": float(loss)}

def start_client():
    print("Hospital B Client Ready to Connect...")
    # fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=HospitalBClient(...))

if __name__ == "__main__":
    start_client()
