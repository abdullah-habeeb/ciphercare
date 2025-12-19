# Hospital A: Cardiology Disease Classification Summary

## 🎯 Project Overview
Repurposed SSSD_ECG diffusion model encoder for multi-label cardiac disease classification using PTB-XL dataset.

---

## 📊 TRAINING EFFICIENCY STATS

### Model Statistics
- **Total Parameters**: 5,631,717 (~5.6M)
- **Model Size**: 17.16 MB
- **Architecture**: S4-based Encoder (12 residual layers) + MLP Classifier Head
- **Trainable Components**: Classifier head only (encoder frozen)

### Training Configuration
- **Batch Size**: 32
- **Learning Rate**: 1e-4
- **Epochs**: 10
- **Optimizer**: AdamW
- **Loss Function**: BCEWithLogitsLoss (multi-label)
- **Training Strategy**: Frozen encoder, fine-tune classifier head

### Dataset Statistics
- **Training Samples**: 17,418
- **Validation Samples**: 2,183
- **Test Samples**: 2,198
- **Total Samples**: 21,799 (PTB-XL)
- **Input Shape**: [8 leads, 1000 timepoints]
- **Output Classes**: 5 (NORM, MI, STTC, CD, HYP)

### Optimization Details
- **Residual Channels**: 128 (reduced from 256 for speed)
- **Skip Channels**: 128
- **Num Residual Layers**: 12 (reduced from 36 for speed)
- **S4 Max Length**: 1000
- **S4 State Dimension**: 64
- **S4 Bidirectional**: Yes

### Training Efficiency
- **Training Duration**: ~3 hours (overnight run)
- **Device**: CPU
- **Estimated Time per Epoch**: ~18 minutes
- **Convergence**: ✅ Achieved (best model saved at 04:20 AM)
- **Speed Optimization**: 4x faster than initial config (12 layers vs 36)

---

## ✅ Completed Components

### 1. Data Processing
- **Source**: Raw PTB-XL `records100/` (21,799 ECG recordings)
- **Output**: Processed `.npy` arrays with correct disease labels
- **Format**: [N, 1000, 12] → [N, 8, 1000] (8 selected leads)
- **Labels**: 5 cardiac superclasses (NORM, MI, STTC, CD, HYP)

### 2. Model Architecture
- **Base**: S4-based encoder from SSSD_ECG (diffusion components removed)
- **Config**: 12 residual layers, 128 channels (optimized for CPU)
- **Classifier**: Global pooling → 256-dim MLP → 5-class output
- **Loss**: BCEWithLogitsLoss (multi-label)

### 3. Training Pipeline
- **Status**: ✅ Training completed successfully
- **Checkpointing**: Epoch-level checkpoints + best model saved
- **Resume Capability**: `resume_training.py` available

### 4. Deployment
- **API**: FastAPI on port 8000
- **Endpoints**:
  - `/predict`: Returns 5-class probability scores
  - `/explain`: Returns saliency maps for interpretability
- **Status**: ✅ Verified (200 OK responses)

### 5. Federated Learning Integration
- **Client**: `federated_client.py` (Flower wrapper)
- **Methods**: `fit()`, `evaluate()`, `get_parameters()`, `set_parameters()`
- **Status**: ✅ Ready for FL server integration

---

## 📈 Expected Performance
- **Baseline AUROC**: 0.65-0.75 (classifier head only)
- **With fine-tuning**: 0.75-0.85 (after unfreezing encoder)

---

## 🚀 Quick Start

### Run Inference API
```bash
python -m uvicorn src.hospital_a.serve.fastapi_wrapper:app --port 8000 --reload
```

### Test Predictions
```bash
python src/hospital_a/serve/test_api.py
```

### Generate Demo Visualization
```bash
python src/hospital_a/serve/demo_inference.py
```

### Resume Training (if interrupted)
```bash
python src/hospital_a/train/resume_training.py
```

---

## 📁 Key Files
- `src/hospital_a/train/checkpoints/best_model.pth` - Trained model (17.16 MB)
- `src/hospital_a/train/train_disease.py` - Training script
- `src/hospital_a/serve/fastapi_wrapper.py` - Inference API
- `src/hospital_a/models/encoder.py` - Model architecture
- `src/hospital_a/utils/process_raw.py` - Data preprocessing
- `src/hospital_a/federated_client.py` - FL client wrapper
- `src/hospital_a/serve/inference_demo.png` - Saliency map visualization
