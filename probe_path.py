import os

path = "C:/Users/aishw/OneDrive/Dokumen/diffusion_models_ecg/result_age_cond_60+/ch256_T200_betaT0.02"

print(f"Path exists: {os.path.exists(path)}")
if os.path.exists(path):
    print("Listing:")
    try:
        print(os.listdir(path))
    except Exception as e:
        print(e)
else:
    print("Path does not exist. Parent:")
    parent = "C:/Users/aishw/OneDrive/Dokumen/diffusion_models_ecg/result_age_cond_60+"
    print(f"Parent exists: {os.path.exists(parent)}")
    if os.path.exists(parent):
        print(os.listdir(parent))
