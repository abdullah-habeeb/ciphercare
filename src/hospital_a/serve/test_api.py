import requests
import numpy as np
import time

# Wait for server
time.sleep(5)

# Generate dummy ECG (8 leads, 1000 timepoints)
dummy_signal = np.random.randn(8, 1000).tolist()

payload = {"signal": dummy_signal}

print("Testing Prediction Endpoint...")
try:
    resp = requests.post("http://127.0.0.1:8000/predict", json=payload)
    if resp.status_code == 200:
        print("Prediction Success:", resp.json())
    else:
        print("Prediction Failed:", resp.text)
except Exception as e:
    print("Connection Error (Predict):", e)

print("\nTesting Explain Endpoint...")
try:
    resp = requests.post("http://127.0.0.1:8000/explain", json=payload)
    if resp.status_code == 200:
        data = resp.json()
        print("Explain Success for Class:", data['target_class'])
        saliency = np.array(data['saliency_map'])
        print("Saliency Shape:", saliency.shape) # Should be (8, 1000)
    else:
        print("Explain Failed:", resp.text)
except Exception as e:
    print("Connection Error (Explain):", e)
