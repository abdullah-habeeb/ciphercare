import flwr as fl
import torch
from collections import OrderedDict
from .train import local_train, get_model # Reuse logic
# Note: In a real scenario, we wouldn't call full local_train() inside fit() directly 
# without careful state management, but for this simulation, we treat 'fit' as 'run local epochs'.

# We need a Client class that wraps our training logic
class HospitalCClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        # 1. Update Local Model with Global Parameters
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
        
        # 2. Train Local (Simplified inline version of local_train logic)
        # In a real heavy app, we'd call a dedicated trainer.
        # Here we just run 1 epoch or use config['epochs']
        epochs = int(config.get("local_epochs", 1))
        
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4) # Simple optimizer
        
        self.model.train()
        self.model.to(self.device)
        
        for _ in range(epochs):
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(images), labels)
                loss.backward()
                optimizer.step()
                
        # 3. Return updated parameters
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        # 1. Update
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
        
        # 2. Evaluate
        self.model.eval()
        loss = 0.0
        correct = 0 # Not strictly applicable to multi-label exact match, but let's just log loss
        
        criterion = torch.nn.BCEWithLogitsLoss()
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss += criterion(outputs, labels).item()
                
        loss /= len(self.val_loader.dataset)
        
        return float(loss), len(self.val_loader.dataset), {"loss": float(loss)}

# Helper to start client
def start_client():
    # Load data and model (Similar setup to train.py)
    # This is a stub to demonstrate connectivity
    print("Hospital C Client Ready to Connect...")
    # fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=HospitalCClient(...))

if __name__ == "__main__":
    start_client()
