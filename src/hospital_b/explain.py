import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.hospital_b.model import get_model
from src.hospital_b.dataset import ClinicalDataset
import os

def compute_feature_importance(model_path, csv_path, output_path):
    """Compute feature importance using gradient-based method"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    dataset = ClinicalDataset(csv_path)
    
    # Load model
    model = get_model(input_dim=dataset.input_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    
    # Sample data
    num_samples = min(100, len(dataset))
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    feature_importances = np.zeros(dataset.input_dim)
    
    print("Computing feature importance via gradients...")
    for idx in indices:
        features, label = dataset[idx]
        features = features.unsqueeze(0).to(device).requires_grad_(True)
        
        output = model(features).squeeze()
        prob = torch.sigmoid(output)
        
        # Compute gradient
        prob.backward()
        
        # Accumulate absolute gradients
        feature_importances += np.abs(features.grad.cpu().numpy().squeeze())
    
    # Normalize
    feature_importances /= num_samples
    feature_importances /= feature_importances.sum()
    
    # Create visualization
    feature_names = dataset.feature_cols
    sorted_idx = np.argsort(feature_importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(feature_names)), feature_importances[sorted_idx])
    plt.yticks(range(len(feature_names)), [feature_names[i] for i in sorted_idx])
    plt.xlabel('Importance Score')
    plt.title('Hospital B - Feature Importance for Deterioration Prediction')
    plt.tight_layout()
    
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved feature importance plot to {output_path}")
    
    # Print top features
    print("\nTop 5 Important Features:")
    for i in range(min(5, len(feature_names))):
        idx = sorted_idx[i]
        print(f"  {feature_names[idx]}: {feature_importances[idx]:.4f}")
    
    return feature_importances

if __name__ == "__main__":
    MODEL_PATH = r"c:\Users\aishw\codered5\ml\models\hospital2_model.pth"
    CSV_PATH = r"c:\Users\aishw\codered5\data\hospital_b\processed_vitals.csv"
    OUTPUT_PATH = r"c:\Users\aishw\codered5\ml\shap\hospital2_shap.png"
    
    compute_feature_importance(MODEL_PATH, CSV_PATH, OUTPUT_PATH)
