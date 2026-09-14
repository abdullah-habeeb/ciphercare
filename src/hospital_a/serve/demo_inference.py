import requests
import numpy as np
import matplotlib.pyplot as plt
import os
import torch

# Load a real sample from validation set
try:
    X_val = np.load("src/hospital_a/data/X_val.npy") # [N, 8, 1000]
    Y_val = np.load("src/hospital_a/data/Y_val.npy") # [N, 5]
    print(f"Loaded validation set: {X_val.shape}")
except:
    print("Could not load validation set. Generating random sample.")
    X_val = np.random.randn(10, 8, 1000)
    Y_val = np.zeros((10, 5))

# Select a sample (e.g., index 0)
idx = 0
raw_signal = X_val[idx] # [1000, 12]
label = Y_val[idx]
classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
true_labels = [classes[i] for i, val in enumerate(label) if val == 1]

# Preprocess for API: Select 8 leads and Transpose
leads_idx = [0,1,6,7,8,9,10,11]
signal = raw_signal[:, leads_idx].T # [1000, 8] -> [8, 1000]

print(f"Selected Sample {idx}")
print(f"True Labels: {true_labels}")
print(f"Input Shape: {signal.shape}")

# Send to API
url = "http://127.0.0.1:8000"
payload = {"signal": signal.tolist()}

# Predict
print("Requesting Prediction...")
resp_pred = requests.post(f"{url}/predict", json=payload)
if resp_pred.status_code == 200:
    preds = resp_pred.json()
    print("Predictions:", preds)
else:
    print("Prediction Failed:", resp_pred.text)
    exit()

# Explain
print("Requesting Explanation...")
resp_exp = requests.post(f"{url}/explain", json=payload)
if resp_exp.status_code == 200:
    explanation = resp_exp.json()
    saliency = np.array(explanation['saliency_map']) # [8, 1000]
    target_cls = classes[explanation['target_class']]
    print(f"Explanation for class: {target_cls}")
else:
    print("Explanation Failed:", resp_exp.text)
    exit()

# Visualize
print("Generating Plot...")
leads = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
# Plot first 3 leads for brevity
leads_to_plot = [1, 2, 6] # Lead II, V1, V5 (indices in 8-lead subset: 0=I, 1=II, 2=V1...)
# Wait, check dataset lead indices: [0,1,6,7,8,9,10,11] -> I, II, V1, V2, V3, V4, V5, V6
# So indices 0..7 correspond to these leads.

fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
plot_leads = [1, 2, 6] # Lead II, V1, V5
plot_names = ['Lead II', 'Lead V1', 'Lead V5']

for i, ax in enumerate(axs):
    lead_idx = plot_leads[i]
    sig = signal[lead_idx]
    sal = saliency[lead_idx]
    
    # Normalize saliency for visualization
    sal_norm = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
    
    # Plot signal
    ax.plot(sig, 'k', label='ECG Signal', alpha=0.7)
    
    # Overlay saliency as scatter or colored line
    # Scatter is easier
    # Only show high saliency
    high_sal_idx = sal_norm > 0.5
    ax.scatter(np.arange(1000)[high_sal_idx], sig[high_sal_idx], c='r', s=10, label='High Importance', alpha=0.6)
    
    ax.set_title(f"{plot_names[i]} (Target: {target_cls})")
    ax.legend(loc='upper right')

plt.suptitle(f"ECG Explanation Sample {idx}\nTrue: {true_labels} | Pred: {target_cls}", fontsize=14)
plt.tight_layout()
output_path = "src/hospital_a/serve/inference_demo.png"
plt.savefig(output_path)
print(f"Plot saved to {output_path}")
