import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class ChestXrayDataset(Dataset):
    def __init__(self, data_dir, csv_file, transform=None):
        self.data_dir = data_dir
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        
        # Define all possible labels (from NIH dataset)
        # We ensure consistent ordering
        self.all_labels = [
            "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", 
            "Mass", "Nodule", "Pneumonia", "Pneumothorax", 
            "Consolidation", "Edema", "Emphysema", "Fibrosis", 
            "Pleural_Thickening", "Hernia"
        ]
        # Map label to index
        self.label_to_idx = {label: i for i, label in enumerate(self.all_labels)}
        self.num_classes = len(self.all_labels)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = self.df.iloc[idx, 0] # "Image Index" column
        labels_str = self.df.iloc[idx, 1] # "Finding Labels" column

        img_path = os.path.join(self.data_dir, img_name)
        
        # Load Image
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            # Fallback for missing/corrupt images in this rapid env
            image = Image.new("RGB", (224, 224))
        
        if self.transform:
            image = self.transform(image)
            
        # Create Multi-hot vector
        label_vec = torch.zeros(self.num_classes)
        if hasattr(labels_str, 'split'):
            for label in labels_str.split('|'):
                if label in self.label_to_idx:
                    label_vec[self.label_to_idx[label]] = 1.0
        
        return image, label_vec
