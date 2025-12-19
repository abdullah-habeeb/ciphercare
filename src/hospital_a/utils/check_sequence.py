import pandas as pd
import numpy as np
import os

csv_path = r"C:\Users\aishw\OneDrive\Dokumen\diffusion_models_ecg\ptbxl_dataset\ptbxl_numpy\ptbxl_database.csv"
y_train_path = r"C:\Users\aishw\OneDrive\Dokumen\diffusion_models_ecg\ptbxl_dataset\ptbxl_numpy\Y_train.npy"

if not os.path.exists(csv_path) or not os.path.exists(y_train_path):
    print("Files not found")
    exit()

try:
    df = pd.read_csv(csv_path, index_col='ecg_id')
except:
    df = pd.read_csv(csv_path)

y_train = np.load(y_train_path)
print("Y_train[:10]:", y_train[:10])

# Filter standard train split
df_train = df[df['strat_fold'] <= 8]
print("DF Train Age[:10]:", df_train['age'].iloc[:10].values)

# Check exact match
if np.allclose(df_train['age'].iloc[:len(y_train)].values, y_train, equal_nan=True):
    print("MATCH FOUND: First N of standard train split!")
    exit()

# Check if it matches the *start* of the whole DF
if np.allclose(df['age'].iloc[:len(y_train)].values, y_train, equal_nan=True):
    print("MATCH FOUND: First N of WHOLE DB!")
    exit()

# Try to find the subsequence
print("Searching for subsequence...")
target = y_train[:5]
# Brute force search in df['age']
# Convert to numpy filling NaNs
ages = df['age'].fillna(-1).values
target = np.array(target)
# Simple sliding window for start
for i in range(len(ages) - 5):
    if np.allclose(ages[i:i+5], target, atol=1e-5):  # Float comparison
        # Detailed check
        # Check if length available
        if i + len(y_train) <= len(ages):
            # Check full match? 
            # Or just assume found
            print(f"Found match starting at index {i} (ecg_id {df.index[i]})")
            # Verify full length match roughly
            sub = ages[i:i+len(y_train)]
            # We need to handle NaNs in y_train if any? SSSD probably cleaned them
            # Let's check mean diff
            diff = np.abs(sub - y_train)
            if np.mean(diff) < 1e-4:
                print("Full sequence verified!")
            else:
                print("Start matched but tail diverges.")
            break
else:
    print("No subsequence match found.")
