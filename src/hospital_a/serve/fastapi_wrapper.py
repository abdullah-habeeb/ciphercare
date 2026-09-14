import torch
import numpy as np
import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# Ensure src is in path
sys.path.append(os.getcwd())

from src.hospital_a.models.encoder import ECGClassifier

app = FastAPI(title="Hospital A Cardiology Node")

class ECGInput(BaseModel):
    signal: List[List[float]] # Expected [8, 1000]

# Global model
model = None
device = 'cpu'

@app.on_event("startup")
async def load_model():
    global model, device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load config needed for model init (hardcoded matches training for now)
    # Ideally load from config.json
    try:
        model = ECGClassifier(
            in_channels=8,
            res_channels=256,
            skip_channels=256,
            num_classes=5,
            num_res_layers=36,
            s4_lmax=1000,
            s4_d_state=64,
            s4_dropout=0.0,
            s4_bidirectional=1,
            s4_layernorm=1
        ).to(device)
        
        checkpoint_path = "src/hospital_a/train/checkpoints/best_model.pth"
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            model.eval()
            print(f"Model loaded from {checkpoint_path}")
        else:
            print("Warning: No checkpoint found. Using random initialization.")
            model.eval()
    except Exception as e:
        print(f"Failed to load model: {e}")

@app.post("/predict")
async def predict(ecg_input: ECGInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        data = np.array(ecg_input.signal, dtype=np.float32)
        # Check shape
        if data.shape != (8, 1000):
             raise HTTPException(status_code=400, detail=f"Invalid shape {data.shape}, expected (8, 1000)")
        
        tensor = torch.tensor(data).unsqueeze(0).to(device) # [1, 8, 1000]
        
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        
        classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
        result = {cls: float(prob) for cls, prob in zip(classes, probs)}
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/explain")
async def explain(ecg_input: ECGInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        data = np.array(ecg_input.signal, dtype=np.float32)
        # Check shape [8, 1000]
        if data.shape != (8, 1000):
             raise HTTPException(status_code=400, detail=f"Invalid shape {data.shape}, expected (8, 1000)")
        
        # Saliency (Input Gradients)
        tensor = torch.tensor(data).unsqueeze(0).to(device) # [1, 8, 1000]
        tensor.requires_grad = True
        
        # Forward
        logits = model(tensor)
        # Target: Max predicted class
        target_class_idx = logits.argmax(dim=1).item()
        score = logits[0, target_class_idx]
        
        # Backward
        model.zero_grad()
        score.backward()
        
        # Get gradient
        gradients = tensor.grad.data.cpu().numpy()[0] # [8, 1000]
        
        # Process gradients (e.g., global average abs magnitude)
        saliency = np.abs(gradients)
        
        return {
            "target_class": target_class_idx,
            "saliency_map": saliency.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# To run: uvicorn src.hospital_a.serve.fastapi_wrapper:app --reload
