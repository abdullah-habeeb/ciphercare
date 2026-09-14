import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class ClinicalDataset(Dataset):
    def __init__(self, csv_path, feature_cols=None):
        self.df = pd.read_csv(csv_path)
        
        # Default feature columns (all except patient_id and target)
        if feature_cols is None:
            self.feature_cols = [col for col in self.df.columns 
                                if col not in ['patient_id', 'deterioration_risk']]
        else:
            self.feature_cols = feature_cols
        
        self.X = self.df[self.feature_cols].values.astype(np.float32)
        self.y = self.df['deterioration_risk'].values.astype(np.float32)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        features = torch.tensor(self.X[idx], dtype=torch.float32)
        label = torch.tensor(self.y[idx], dtype=torch.float32)
        
        return features, label
    
    @property
    def input_dim(self):
        return len(self.feature_cols)
