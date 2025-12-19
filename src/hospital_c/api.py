from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from torchvision import transforms
from PIL import Image
import io
import base64
import json
import os
from src.hospital_c.model import get_model

app = FastAPI(title="Hospital C - Chest X-Ray AI")

# Globals
model = None
device = None
NUM_CLASSES = 14
MODEL_PATH = r"c:\Users\aishw\codered5\ml\models\hospital3_model.pth"
METRICS_PATH = r"c:\Users\aishw\codered5\ml\models\global_results.json"

LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", 
    "Mass", "Nodule", "Pneumonia", "Pneumothorax", 
    "Consolidation", "Edema", "Emphysema", "Fibrosis", 
    "Pleural_Thickening", "Hernia"
]

class PredictionRequest(BaseModel):
    image_base64: str

def load_ai_model():
    global model, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes=NUM_CLASSES)
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

@app.post("/predict_hospital3")
async def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    try:
        image_data = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            # Applicatoin of Sigmoid for multi-label logic
            probs = torch.sigmoid(outputs).squeeze().cpu().numpy()
            
        results = {label: float(prob) for label, prob in zip(LABELS, probs)}
        return {"predictions": results}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/metrics_hospital3")
async def get_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    return {"error": "Metrics not found"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
