import requests
import numpy as np
import json

# The API server is already running with the trained model loaded
# Let's use it to get predictions and calculate metrics

print("="*70)
print("EXTRACTING METRICS VIA RUNNING API SERVER")
print("="*70)

# Load validation data
X_val = np.load("src/hospital_a/data/X_val.npy")  # [N, 1000, 12]
Y_val = np.load("src/hospital_a/data/Y_val.npy")  # [N, 5]

print(f"\nValidation set: {len(X_val)} samples")

# Select leads and transpose
leads_idx = [0,1,6,7,8,9,10,11]

# Test on a subset (first 100 samples to avoid timeout)
num_samples = min(100, len(X_val))
print(f"Testing on {num_samples} samples...\n")

predictions = []
url = "http://127.0.0.1:8000/predict"

for i in range(num_samples):
    signal = X_val[i][:, leads_idx].T  # [1000, 8] -> [8, 1000]
    payload = {"signal": signal.tolist()}
    
    try:
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code == 200:
            preds = resp.json()
            # Extract probabilities (should be dict with class names as keys)
            if isinstance(preds, dict):
                # Assuming order: NORM, MI, STTC, CD, HYP
                pred_values = [preds.get(cls, 0.0) for cls in ['NORM', 'MI', 'STTC', 'CD', 'HYP']]
                predictions.append(pred_values)
            else:
                predictions.append(preds)
            
            if (i+1) % 10 == 0:
                print(f"  Processed {i+1}/{num_samples} samples...")
        else:
            print(f"  Sample {i}: API error {resp.status_code}")
            break
    except Exception as e:
        print(f"  Sample {i}: Error - {e}")
        break

if len(predictions) > 0:
    predictions = np.array(predictions)
    targets = Y_val[:len(predictions)]
    
    # Calculate AUROC
    from sklearn.metrics import roc_auc_score
    
    try:
        auroc = roc_auc_score(targets, predictions, average='macro')
        print(f"\n📊 METRICS (on {len(predictions)} samples):")
        print(f"  Validation AUROC (Macro): {auroc:.4f}")
        
        # Per-class
        classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
        print(f"\n📈 PER-CLASS AUROC:")
        print("-" * 40)
        for i, cls in enumerate(classes):
            try:
                if len(np.unique(targets[:, i])) > 1:
                    cls_auroc = roc_auc_score(targets[:, i], predictions[:, i])
                    print(f"  {cls:8s}: {cls_auroc:.4f}")
            except:
                print(f"  {cls:8s}: N/A")
        
        print("\n" + "="*70)
        print(f"Estimated Validation AUROC: {auroc:.4f}")
        print("(Based on {}/{} samples)".format(len(predictions), len(X_val)))
        print("="*70)
        
    except Exception as e:
        print(f"\nError calculating AUROC: {e}")
else:
    print("\n⚠ No predictions collected")
