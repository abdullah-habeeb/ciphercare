# 🚀 Quick Start Guide - CipherCare ML Backend

## Step-by-Step Setup

### 1. Install Python Dependencies

```bash
pip install -r backend/requirements.txt
```

**Note**: If you get errors, try:
```bash
pip install --upgrade pip
pip install -r backend/requirements.txt
```

### 2. Generate Dataset

```bash
cd ml
python generate_dataset.py
```

**Expected output:**
```
Generating synthetic dataset for 1000 patients...
Dataset Summary:
Total patients: 1000
Stable (0): 978
High-risk (1): 22
✅ Dataset saved to data/patient_vitals.csv
```

### 3. Train Model

```bash
python train_model.py
```

**Expected output:**
```
Loading data from data/patient_vitals.csv...
Training XGBoost classifier...
📊 Model Performance:
AUROC: 0.9234
F1 Score: 0.7234
✅ Model saved to models/deterioration_model.pkl
```

### 4. Start Backend API

**Option A** (Recommended):
```bash
cd ..
python backend/run.py
```

**Option B**:
```bash
cd backend
uvicorn main:app --reload --port 8000
```

**Verify it's running:**
- Open http://localhost:8000/health
- Should see: `{"status":"healthy"}`

### 5. Start Frontend

In a **new terminal**:
```bash
npm run dev
```

Frontend will be at http://localhost:5173

### 6. Test the Integration

1. Navigate to http://localhost:5173/iomt-monitor
2. You should see:
   - Live vitals updating
   - **Risk Prediction Panel** on the right
   - Risk score gauge
   - Severity badge
   - Recommended action
   - Feature contributions

### 7. Trigger Critical Alert

Use the **Anomaly Simulator** to trigger:
- **Hypoxia** → Should show Critical risk
- **Tachycardia** → Should show Warning/Critical risk
- **Fever** → Should show Warning risk

## 🐛 Common Issues

### "Module not found" errors

**Solution**: Make sure you're in the correct directory:
```bash
# For dataset generation
cd ciphercare-insights-main/ml
python generate_dataset.py

# For training
cd ciphercare-insights-main/ml
python train_model.py

# For backend
cd ciphercare-insights-main
python backend/run.py
```

### Backend can't find model

**Solution**: Make sure you've run training first:
```bash
cd ml
python train_model.py
```

Check that these files exist:
- `models/deterioration_model.pkl`
- `models/preprocessing_pipeline.pkl`

### Frontend can't connect to API

**Solution**: 
1. Check backend is running: http://localhost:8000/health
2. Check browser console for CORS errors
3. Verify `VITE_API_URL` in `.env` (or defaults to localhost:8000)

### Low model performance

**Solution**: Regenerate dataset (random seed affects quality):
```bash
cd ml
python generate_dataset.py
python train_model.py
```

## ✅ Verification Checklist

- [ ] Python packages installed
- [ ] Dataset generated (check `data/patient_vitals.csv`)
- [ ] Model trained (check `models/deterioration_model.pkl`)
- [ ] Backend running (http://localhost:8000/health works)
- [ ] Frontend running (http://localhost:5173 works)
- [ ] IoMT page shows risk predictions
- [ ] Critical alerts work (try anomaly simulator)

## 📊 Expected Performance

After training, you should see:
- **AUROC**: 0.85-0.95
- **F1 Score**: 0.60-0.80
- **Precision**: 0.70-0.90
- **Recall**: 0.50-0.70

These metrics will vary slightly due to random data generation.

---

**Need help?** Check `ML_BACKEND_README.md` for detailed documentation.





