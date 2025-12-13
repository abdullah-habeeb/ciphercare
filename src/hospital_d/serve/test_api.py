import requests
import json
import numpy as np

def test_predict():
    url = "http://127.0.0.1:8000/predict"
    # Create dummy 12-lead ECG [12, 1000]
    ecg_data = np.random.randn(12, 1000).tolist()
    
    payload = {"ecg": ecg_data}
    
    try:
        response = requests.post(url, json=payload, timeout=5)
        if response.status_code == 200:
            print("✓ /predict Test Passed")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"X /predict Failed: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"X /predict Error: {e}")

if __name__ == "__main__":
    test_predict()
