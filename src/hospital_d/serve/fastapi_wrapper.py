from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List
import torch
import numpy as np
import os
import sys
import json

# Add src to path
sys.path.append(os.getcwd())

from src.hospital_d.models.classifier import HospitalDClassifier
from src.hospital_d.explain.saliency import get_saliency_map, plot_saliency

app = FastAPI(title="Hospital D (Geriatric) API")

# Global model
model = None
device = None
config = None

def load_model():
    global model, device, config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Config
    with open('src/hospital_d/train/disease_config.json', 'r') as f:
        config = json.load(f)
    
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
    
    ckpt_path = os.path.join(config['output_dir'], 'best_model_synth.pth')
    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint from {ckpt_path}")
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
            # Handle DDP
            if 'module.' in list(ckpt.keys())[0]:
                ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
            
            # Safe manual load for S4 shared params
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in ckpt:
                        try:
                            param.copy_(ckpt[name])
                        except RuntimeError as e:
                            if "single memory location" in str(e):
                                # Safe fix
                                param.data = ckpt[name].clone().to(device)
                            else:
                                print(f"Error loading {name}: {e}")
            
            print(f"✓ Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print("Warning: No checkpoint found, using random weights")
    
    model.eval()

@app.on_event("startup")
async def startup_event():
    load_model()

class ECGRequest(BaseModel):
    ecg: List[List[float]] # [12, 1000] expected input (Standard 12-lead) or [8, 1000]?
    # User prompt: "Convert to NumPy → select 8 leads" implies input is 12-lead.

@app.post("/predict")
async def predict(request: ECGRequest):
    data = np.array(request.ecg) # [12, 1000]
    
    # Select leads for Hospital D: [0,2,3,4,5,6,7,11]
    leads_idx = [0,2,3,4,5,6,7,11]
    
    if data.shape[0] == 12:
        signal = data[leads_idx, :] # [8, 1000]
    elif data.shape[0] == 8:
        signal = data # Assume already selected
    else:
        # Handle transpose case or error
        if data.shape[1] == 12: 
             signal = data.T[leads_idx, :]
        else:
             return {"error": f"Invalid shape {data.shape}"}

    # Normalize? (Standard score normalization usually done in preprocessing)
    # Assuming input is raw WFDB scale? Or standard?
    # Simple z-score per lead
    epsilon = 1e-8
    mean = np.mean(signal, axis=1, keepdims=True)
    std = np.std(signal, axis=1, keepdims=True)
    signal = (signal - mean) / (std + epsilon)
    
    tensor_input = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(tensor_input)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        
    classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    
    return {
        "hospital": "D",
        "cohort": "60_plus",
        "source": "synthetic",
        "label_source": "hospital_a_pseudolabels",
        "probabilities": [float(p) for p in probs]
    }

@app.post("/explain")
async def explain(request: ECGRequest):
    data = np.array(request.ecg)
    leads_idx = [0,2,3,4,5,6,7,11]
    if data.shape[0] == 12:
        signal = data[leads_idx, :] 
    else:
        signal = data
        
    epsilon = 1e-8
    mean = np.mean(signal, axis=1, keepdims=True)
    std = np.std(signal, axis=1, keepdims=True)
    signal_norm = (signal - mean) / (std + epsilon)
    
    tensor_input = torch.tensor(signal_norm, dtype=torch.float32).unsqueeze(0).to(device)
    
    saliency_map, top_leads, top_class = get_saliency_map(model, tensor_input, device)
    
    # Generate plot (save to file for retrieval or return base64?)
    # User prompt says "Response: JSON + saliency image (base64 or path)"
    # We'll save a temp file and return path for simplicity in local demo
    save_path = "src/hospital_d/serve/latest_saliency.png"
    plot_saliency(signal_norm, saliency_map, top_leads, save_path=save_path)
    
    return {
        "hospital": "D",
        "explanation": "Saliency Top 3 Leads",
        "top_leads": [int(l) for l in top_leads],
        "image_path": os.path.abspath(save_path)
    }
