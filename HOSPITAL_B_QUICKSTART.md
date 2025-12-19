# Hospital B - Quick Start Guide

## 🚀 Running the API

```powershell
# Start Hospital B API (port 8001)
python -m uvicorn src.hospital_b.api:app --reload --port 8001
```

Visit: **http://127.0.0.1:8001/docs**

---

## 🧪 Testing the Prediction Endpoint

### Method 1: Interactive Docs
1. Go to http://127.0.0.1:8001/docs
2. Click on `/predict_hospital2`
3. Click "Try it out"
4. Use this sample JSON:

```json
{
  "age": 72.0,
  "heart_rate": 105.0,
  "systolic_bp": 165.0,
  "diastolic_bp": 95.0,
  "respiratory_rate": 26.0,
  "spo2": 89.0,
  "temperature": 37.8,
  "glucose": 220.0,
  "creatinine": 1.8,
  "wbc": 14.0,
  "hemoglobin": 10.5,
  "has_hypertension": 1,
  "has_diabetes": 1,
  "gender": "F",
  "ethnicity": "Black"
}
```

### Method 2: PowerShell/curl

```powershell
$body = @{
    age = 72.0
    heart_rate = 105.0
    systolic_bp = 165.0
    diastolic_bp = 95.0
    respiratory_rate = 26.0
    spo2 = 89.0
    temperature = 37.8
    glucose = 220.0
    creatinine = 1.8
    wbc = 14.0
    hemoglobin = 10.5
    has_hypertension = 1
    has_diabetes = 1
    gender = "F"
    ethnicity = "Black"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:8001/predict_hospital2" -Method Post -Body $body -ContentType "application/json"
```

---

## 📊 Expected Response

```json
{
  "deterioration_probability": 0.78,
  "risk_level": "HIGH RISK",
  "threshold": 0.5
}
```

---

## 🔍 Key Features Driving Predictions

Based on gradient analysis:
1. **Systolic BP** (7.7%) - Hypertension indicator
2. **Has Hypertension** (7.5%) - Comorbidity flag
3. **Respiratory Rate** (7.2%) - Respiratory distress
4. **Temperature** (7.0%) - Infection/fever
5. **SpO2** (7.0%) - Oxygen saturation

---

## 📈 Model Performance Summary

| Metric | Value |
|--------|-------|
| **AUROC** | **0.959** |
| Precision | 0.25 |
| Recall | 1.00 |
| Specificity | 0.94 |
| F1 Score | 0.40 |

**Personalization Improvement**: +2.7% AUROC

---

## 🌸 Federated Learning

```python
from src.hospital_b.flower_client import HospitalBClient

# Client is ready to connect to FL server
# fl.client.start_numpy_client(
#     server_address="127.0.0.1:8080",
#     client=HospitalBClient(model, train_loader, val_loader)
# )
```

---

## ✅ All Tasks Completed

- [x] Data preprocessing (1000 patients)
- [x] MLP model training (10 epochs)
- [x] Personalization (+2.7% AUROC)
- [x] Feature importance analysis
- [x] Flower FL client
- [x] FastAPI deployment
- [x] Testing & documentation
