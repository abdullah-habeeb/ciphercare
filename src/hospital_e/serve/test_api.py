import requests
import numpy as np
import json

def test_predict():
    url = "http://localhost:8002/predict" # Assume Port 8002 for Hospital E
    
    # 1. Full Multimodal
    print("Testing Full Multimodal Input...")
    payload = {
        "ecg": np.random.randn(8, 1000).tolist(),
        "vitals": np.random.randn(15).tolist(),
        "lungs": np.random.randn(128).tolist()
    }
    try:
        resp = requests.post(url, json=payload)
        print(f"Status: {resp.status_code}")
        print(f"Response: {json.dumps(resp.json(), indent=2)}")
    except Exception as e:
        print(f"Failed: {e}")

    # 2. Missing Modalities (ECG Only)
    print("\nTesting ECG Only...")
    payload = {
        "ecg": np.random.randn(8, 1000).tolist()
    }
    try:
        resp = requests.post(url, json=payload)
        print(f"Status: {resp.status_code}")
        print(f"Response: {resp.json()['modalities_present']}")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    test_predict()
