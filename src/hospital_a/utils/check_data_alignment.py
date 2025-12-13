import pandas as pd
import numpy as np
import os

csv_path = r"C:\Users\aishw\OneDrive\Dokumen\diffusion_models_ecg\ptbxl_dataset\ptbxl_numpy\ptbxl_database.csv"
if not os.path.exists(csv_path):
    print("CSV not found")
    exit()

try:
    df = pd.read_csv(csv_path, index_col='ecg_id')
except:
    df = pd.read_csv(csv_path)

print("Total rows:", len(df))

# Check splits
train_df = df[df['strat_fold'] <= 8]
val_df = df[df['strat_fold'] == 9]
test_df = df[df['strat_fold'] == 10]

print("Train fold count (1-8):", len(train_df))
print("Val fold count (9):", len(val_df))
print("Test fold count (10):", len(test_df))

# Check actual npy sizes
y_train_path = r"C:\Users\aishw\OneDrive\Dokumen\diffusion_models_ecg\ptbxl_dataset\ptbxl_numpy\Y_train.npy"
if os.path.exists(y_train_path):
    y_train = np.load(y_train_path)
    print("Y_train.npy len:", len(y_train))
    
    # Check if df is sorted
    print("Is index sorted?", df.index.is_monotonic_increasing)
