# Hospital E: Multimodal Overlap Node Summary

## 🏥 Role
Hospital E serves as the **Multimodal Overlap Node** and **Stability Anchor** in the federated network.
- **Data**: Mix of ECG (A+D), Vitals (B), and Lungs (C).
- **Goal**: Demonstrate fusion of heterogeneous data sources and robustness to missing modalities.

## 🏗️ Architecture
### Generators
- **ECG**: Sourced from Hospital A (General) and D (Geriatric) real datasets.
- **Vitals**: Simulated 15-feature vectors correlated with cardiac labels (mimics B).
- **Lungs**: Simulated 128-dim embeddings (mimics C).

### Model: `FusionClassifier`
- **Encoders**:
    - `ECGEncoder`: S4-based (12 layers, 128 dim). Reuses Hospital A backbone.
    - `VitalsEncoder`: MLP (15 -> 128 dim).
    - `LungsEncoder`: Linear (128 -> 128 dim).
- **Fusion**:
    - Concatenation of embeddings (masked with zeros if missing).
    - MLP Head (384 -> 256 -> 5 classes).
- **Capabilities**: Handles any combination of missing data (e.g., ECG-only, Vitals-only).

## 🚀 Deployment
### API (Port 8002 recommended)
- **Endpoint**: `/predict`
- **Input**: JSON with optional `ecg`, `vitals`, `lungs` fields.
- **Output**: 5-class probabilities + metadata about present modalities.

### Federated Learning
- **Client**: `src/hospital_e/federated/run_client.py`
- **Strategy**: 
    - Currently set for **Multimodal Federation**. 
    - **Note**: Requires A/D to upgrade to Fusion wrapper to participate in the same round, or E participates in a separate multimodal round.

## 📁 Key Files
- `src/hospital_e/data/generate_multimodal.py`: Data synthesis.
- `src/hospital_e/models/fusion_classifier.py`: The fusion model.
- `src/hospital_e/train/train_fusion.py`: Training script.
- `src/hospital_e/serve/fastapi_wrapper.py`: Inference API.

## 📊 Verification Status
- [x] Data Generated (3000 samples, shapes verified).
- [x] Model Instantiated (Smoke test passed).
- [/] Training in progres.
- [ ] API Tested.
