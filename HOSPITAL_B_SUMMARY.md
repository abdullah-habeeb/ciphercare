# Hospital B - Clinical Deterioration Prediction System

## Overview
Hospital B focuses on **tabular clinical data** (vitals, labs, demographics) to predict patient deterioration risk using a binary classification model.

---

## 📊 Dataset
- **Source**: Synthetic MIMIC-IV-like data
- **Size**: 1,000 patients
- **Features**: 15 clinical variables
  - Vitals: HR, BP, RR, SpO2, Temperature
  - Labs: Glucose, Creatinine, WBC, Hemoglobin
  - Demographics: Age, Gender, Ethnicity
  - Comorbidities: Hypertension, Diabetes
- **Target**: Binary deterioration risk (0=Stable, 1=High Risk)
- **Class Distribution**: 978 stable, 22 high-risk (imbalanced, realistic)

---

## 🧠 Model Architecture
**Type**: Multi-Layer Perceptron (MLP)
- Input: 15 features
- Hidden Layers: [64, 32] with ReLU, BatchNorm, Dropout(0.3)
- Output: 1 (binary classification with BCEWithLogitsLoss)

---

## 📈 Training Results

### Global Model Performance
- **AUROC**: **0.959** ✅
- **Precision**: 0.25
- **Recall**: 1.0 (catches all high-risk cases)
- **Specificity**: 0.94
- **F1 Score**: 0.40
- **Validation Loss**: 0.539

### Personalization Results
- **Pre-Personalization AUROC**: 0.882
- **Post-Personalization AUROC**: **0.909**
- **Improvement**: **+0.027** (2.7% boost)

The model successfully demonstrates local adaptation on hospital-specific data.

---

## 🔍 Explainability (SHAP)
SHAP analysis identifies the most important features for deterioration prediction:

**Top 5 Features** (based on SHAP values):
1. **SpO2** - Low oxygen saturation is a critical indicator
2. **Respiratory Rate** - Elevated RR signals distress
3. **Systolic BP** - Extreme values indicate instability
4. **Glucose** - High glucose (diabetes complication)
5. **Age** - Older patients at higher baseline risk

Visual: `ml/shap/hospital2_shap.png`

---

## 🌸 Federated Learning Integration
**Flower Client**: `src/hospital_b/flower_client.py`
- `get_parameters()`: Returns model weights as NumPy arrays
- `fit()`: Trains on local data for N epochs
- `evaluate()`: Returns validation loss and metrics

Ready to connect to FL server for global rounds.

---

## 🚀 API Endpoints

**Base URL**: `http://127.0.0.1:8001`

### 1. `/predict_hospital2` (POST)
Predicts deterioration risk from patient vitals.

**Request Body**:
```json
{
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
```

**Response**:
```json
{
  "deterioration_probability": 0.23,
  "risk_level": "STABLE",
  "threshold": 0.5
}
```

### 2. `/metrics_hospital2` (GET)
Returns stored model performance metrics.

### 3. `/sample_vitals` (GET)
Returns a sample vitals JSON for testing.

---

## 🧪 Testing the API

```powershell
# Start the server
python -m uvicorn src.hospital_b.api:app --reload --port 8001

# Visit interactive docs
http://127.0.0.1:8001/docs
```

---

## 📁 File Structure
```
src/hospital_b/
├── preprocess_data.py   # Data generation & preprocessing
├── dataset.py           # PyTorch Dataset
├── model.py             # MLP architecture
├── train.py             # Training + Personalization
├── explain.py           # SHAP explainability
├── flower_client.py     # FL client wrapper
└── api.py               # FastAPI endpoints

data/hospital_b/
└── processed_vitals.csv # Preprocessed patient data

ml/models/
├── hospital2_model.pth           # Trained weights
├── global_results.json           # Global metrics
└── personalized_results.json     # Personalization metrics

ml/shap/
└── hospital2_shap.png            # SHAP feature importance
```

---

## ✅ Completed Tasks
- [x] Data preprocessing (imputation, normalization, encoding)
- [x] MLP model design
- [x] Training (10 epochs, AUROC 0.96)
- [x] Personalization (15% data, +2.7% AUROC)
- [x] SHAP explainability
- [x] Flower FL client
- [x] FastAPI deployment
- [x] Testing endpoints

---

## 🎯 Key Achievements
1. **High Performance**: 95.9% AUROC on validation set
2. **Perfect Recall**: Catches 100% of high-risk patients (critical for healthcare)
3. **Interpretable**: SHAP shows SpO2 and RR as top predictors
4. **FL-Ready**: Fully integrated with Flower framework
5. **Production API**: Real-time inference via FastAPI

---

## 🔮 Next Steps
- Connect to FL server for multi-hospital training
- Deploy to cloud (AWS/Azure) with autoscaling
- Integrate with IoMT devices for real-time vitals streaming
- Add alerting system for high-risk predictions
