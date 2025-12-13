import torch
import torch.nn as nn

class ClinicalMLP(nn.Module):
    """Simple MLP for tabular clinical data"""
    def __init__(self, input_dim, hidden_dims=[64, 32], dropout=0.3):
        super(ClinicalMLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def get_model(input_dim=15, hidden_dims=[64, 32]):
    """Factory function to create model"""
    return ClinicalMLP(input_dim, hidden_dims)
