import os
import sys
import numpy as np
import requests
import json
import matplotlib.pyplot as plt

# Test local inference logic directly without API server overhead first, 
# or via API? Prompt implies "Smoke Test" might be standalone. 
# Prompt: "test POST to /predict" = test_api.py. 
# "smoke_test_inference.py -> run 10 samples, generate demo prediction + saliency".
# I'll implement this as an INTEGRATION test calling the API.

def smoke_test():
    # Load synthetic data
    try:
        X_synth = np.load("src/hospital_d/data/X_synth.npy") # [N, 8, 1000]
        # Wait, API expects 12-lead or 8-lead? 
        # My API wrapper handles [12, 1000] or [8, 1000].
        # X_synth is [N, 8, 1000].
    except:
        print("X_synth.npy not found, using random data")
        X_synth = np.random.randn(10, 8, 1000)

    url_predict = "http://127.0.0.1:8001/predict"
    url_explain = "http://127.0.0.1:8001/explain"
    
    print(f"Running smoke test on 10 samples...")
    
    for i in range(min(10, len(X_synth))):
        sample = X_synth[i] # [8, 1000]
        payload = {"ecg": sample.tolist()}
        
        # Predict
        resp = requests.post(url_predict, json=payload)
        if resp.status_code == 200:
            res = resp.json()
            print(f"Sample {i}: {res['probabilities']}")
        else:
            print(f"Sample {i}: Predict Failed {resp.status_code}")
            print(f"Error Body: {resp.text}")
            
        # Explain (only for first one to save time)
        if i == 0:
            resp_exp = requests.post(url_explain, json=payload)
            if resp_exp.status_code == 200:
                res = resp_exp.json()
                print(f"Sample {i} Explanation: Top Leads {res['top_leads']}")
                print(f"Saliency Map saved at: {res['image_path']}")
            else:
                print(f"Sample {i}: Explain Failed {resp_exp.status_code}")

if __name__ == "__main__":
    smoke_test()
