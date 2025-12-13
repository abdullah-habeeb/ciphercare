from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import json
import os
from src.hospital_b.model import get_model
import numpy as np

app = FastAPI(title="Hospital B - Clinical Deterioration Prediction")

# Globals
model = None
device = None
INPUT_DIM = 15
MODEL_PATH = r"c:\Users\aishw\codered5\ml\models\hospital2_model.pth"
METRICS_PATH = r"c:\Users\aishw\codered5\ml\models\global_results.json"

FEATURE_NAMES = [
    'age', 'heart_rate', 'systolic_bp', 'diastolic_bp',
    'respiratory_rate', 'spo2', 'temperature', 'glucose',
    'creatinine', 'wbc', 'hemoglobin', 'has_hypertension',
    'has_diabetes', 'gender_encoded', 'ethnicity_encoded'
]

class VitalsRequest(BaseModel):
    age: float
    heart_rate: float
    systolic_bp: float
    diastolic_bp: float
    respiratory_rate: float
    spo2: float
    temperature: float
    glucose: float
    creatinine: float
    wbc: float
    hemoglobin: float
    has_hypertension: int
    has_diabetes: int
    gender: str  # 'M' or 'F'
    ethnicity: str  # 'White', 'Black', 'Hispanic', 'Asian', 'Other'

def load_ai_model():
    global model, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(input_dim=INPUT_DIM)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("Model loaded.")
    else:
        print("Warning: Model weights not found.")
    model.to(device)
    model.eval()

@app.on_event("startup")
async def startup_event():
    load_ai_model()

@app.post("/predict_hospital2")
async def predict(request: VitalsRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Encode categorical
        gender_encoded = 0 if request.gender == 'M' else 1
        ethnicity_map = {'White': 0, 'Black': 1, 'Hispanic': 2, 'Asian': 3, 'Other': 4}
        ethnicity_encoded = ethnicity_map.get(request.ethnicity, 4)
        
        # Create feature vector (must match training order)
        features = np.array([
            request.age, request.heart_rate, request.systolic_bp, request.diastolic_bp,
            request.respiratory_rate, request.spo2, request.temperature, request.glucose,
            request.creatinine, request.wbc, request.hemoglobin, request.has_hypertension,
            request.has_diabetes, gender_encoded, ethnicity_encoded
        ], dtype=np.float32)
        
        # Predict
        input_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor).squeeze()
            prob = torch.sigmoid(output).item()
        
        risk_level = "HIGH RISK" if prob > 0.5 else "STABLE"
        
        return {
            "deterioration_probability": float(prob),
            "risk_level": risk_level,
            "threshold": 0.5
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/metrics_hospital2")
async def get_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    return {"error": "Metrics not found"}

@app.get("/sample_vitals")
async def get_sample():
    """Return a sample vitals JSON for testing"""
    return {
        "age": 68.5,
        "heart_rate": 92.0,
        "systolic_bp": 155.0,
        "diastolic_bp": 88.0,
        "respiratory_rate": 20.0,
        "spo2": 94.0,
        "temperature": 37.2,
        "glucose": 180.0,
        "creatinine": 1.3,
        "wbc": 10.5,
        "hemoglobin": 12.0,
        "has_hypertension": 1,
        "has_diabetes": 1,
        "gender": "M",
        "ethnicity": "White"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
