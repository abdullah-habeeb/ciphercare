import os
import sys
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

sys.path.append(os.getcwd())
try:
    from src.hospital_e.models.fusion_classifier import FusionClassifier
except ImportError:
    # Fallback for direct execution
    sys.path.append(os.path.join(os.getcwd(), 'src'))
    from hospital_e.models.fusion_classifier import FusionClassifier

app = FastAPI(title="Hospital E Multimodal Model API")

# Global variables
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MultimodalRequest(BaseModel):
    # ECG: 8 leads x 1000 timepoints
    # Expecting list of 8 lists of 1000 floats
    ecg: Optional[List[List[float]]] = None 
    
    # Vitals: 15 numeric features
    vitals: Optional[List[float]] = None
    
    # Lungs: 128 embedding dimensions (simulated input for now, 
    # real world would take image and pass through CNN)
    lungs: Optional[List[float]] = None

@app.on_event("startup")
async def load_model():
    global model
    print(f"Loading Hospital E Fusion Model on {device}...")
    model = FusionClassifier(num_classes=5).to(device)
    
    ckpt_path = "src/hospital_e/train/checkpoints/best_model_multimodal.pth"
    if os.path.exists(ckpt_path):
        try:
            # Safe loading (handling S4 memory issues just in case)
            ckpt = torch.load(ckpt_path, map_location=device)
            
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in ckpt:
                        try:
                            param.copy_(ckpt[name])
                        except RuntimeError:
                             # S4 shared params fallback
                             param.data = ckpt[name].clone().to(device)
            print(f"✓ Loaded model from {ckpt_path}")
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            print("Using random initialization.")
    else:
         print("⚠️ No checkpoint found. Using random initialization.")
    
    model.eval()

@app.on_event("shutdown")
async def shutdown():
    print("Shutting down...")

@app.post("/predict")
async def predict(request: MultimodalRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
        
    try:
        # 1. Process Inputs
        ecg_tensor = torch.zeros(1, 8, 1000).to(device)
        vitals_tensor = torch.zeros(1, 15).to(device)
        lungs_tensor = torch.zeros(1, 128).to(device)
        mask_tensor = torch.zeros(1, 3).to(device) # [ECG, Vitals, Lungs]
        
        # ECG
        if request.ecg is not None:
            ecg_arr = np.array(request.ecg, dtype=np.float32)
            if ecg_arr.shape != (8, 1000):
                # Try transpose if (1000, 8)
                if ecg_arr.shape == (1000, 8):
                    ecg_arr = ecg_arr.T
                else:
                     raise ValueError(f"Invalid ECG shape: {ecg_arr.shape}, expected (8, 1000)")
            ecg_tensor[0] = torch.tensor(ecg_arr)
            mask_tensor[0, 0] = 1
            
        # Vitals
        if request.vitals is not None:
            vitals_arr = np.array(request.vitals, dtype=np.float32)
            if len(vitals_arr) != 15:
                 raise ValueError(f"Invalid Vitals len: {len(vitals_arr)}, expected 15")
            vitals_tensor[0] = torch.tensor(vitals_arr)
            mask_tensor[0, 1] = 1
            
        # Lungs
        if request.lungs is not None:
            lungs_arr = np.array(request.lungs, dtype=np.float32)
            if len(lungs_arr) != 128:
                 raise ValueError(f"Invalid Lungs len: {len(lungs_arr)}, expected 128")
            lungs_tensor[0] = torch.tensor(lungs_arr)
            mask_tensor[0, 2] = 1
            
        # Check if at least one modality exists
        if mask_tensor.sum() == 0:
             raise HTTPException(status_code=400, detail="Must provide at least one modality (ecg, vitals, lungs)")
             
        # 2. Inference
        with torch.no_grad():
            logits = model(ecg_tensor, vitals_tensor, lungs_tensor, mask_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
            
        # 3. Format Response
        classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
        return {
            "hospital": "E",
            "type": "Multimodal Fusion Node",
            "modalities_present": {
                "ecg": bool(mask_tensor[0, 0]),
                "vitals": bool(mask_tensor[0, 1]),
                "lungs": bool(mask_tensor[0, 2])
            },
            "predictions": {cls: float(p) for cls, p in zip(classes, probs)},
            "probabilities": probs.tolist() # raw list
        }
        
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}
