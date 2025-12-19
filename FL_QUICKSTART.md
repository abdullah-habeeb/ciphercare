# Federated Learning Setup - Quick Start Guide

## Overview
This setup enables federated learning between:
- **Hospital A**: General cardiology (PTB-XL, all ages)
- **Hospital D**: Geriatric cardiology (PTB-XL, age ≥ 60)

## Files Created
- `fl_server.py` - Flower FL server (FedAvg strategy)
- `run_hospital_a_client.py` - Hospital A client
- `run_hospital_d_client.py` - Hospital D client

## Quick Test (Manual)

### Terminal 1: Start Server
```bash
python fl_server.py
```

### Terminal 2: Start Hospital A Client
```bash
python run_hospital_a_client.py --server 127.0.0.1:8080
```

### Terminal 3: Start Hospital D Client
```bash
python run_hospital_d_client.py --server 127.0.0.1:8080
```

## Expected Output
- Server will run 5 FL rounds
- Each round:
  1. Server sends global model to clients
  2. Clients train locally (1 epoch each)
  3. Clients send updates back to server
  4. Server aggregates using FedAvg
  5. Server evaluates on both hospitals

## Configuration
- **Strategy**: FedAvg (Federated Averaging)
- **Rounds**: 5
- **Local epochs**: 1 per round
- **Min clients**: 2
- **Server port**: 8080

## Next Steps
1. Run the test to verify FL communication
2. Monitor AUROC improvement across rounds
3. Add Hospital B (tabular MIMIC-IV data) as 3rd client
4. Experiment with:
   - More FL rounds (10-20)
   - Different strategies (FedProx, FedOpt)
   - Differential privacy (DP-FedAvg)

## Notes
- Hospital A uses 1,000 samples (subset for speed)
- Hospital D uses full 2,400 geriatric samples
- Both use same model architecture (S4-based ECGClassifier)
- Training happens on CPU (slow but functional)
