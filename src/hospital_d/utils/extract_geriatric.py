"""
Extracts the geriatric (age >= 60) subset of PTB-XL for Hospital D.

Mirrors src/hospital_a/utils/process_raw.py's signal loading and disease
label mapping, but filters to age >= 60 and writes Hospital D's expected
train/test filenames.
"""
import argparse
import os
import sys
from pathlib import Path

import ast
import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))
from src.hospital_a.utils.process_raw import downsample_ecg

CLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']

# Same mapping used in src/hospital_a/utils/process_raw.py, kept identical so
# Hospital A and Hospital D produce label vectors over the same 5 classes.
SCP_TO_SUPERCLASS = {
    'NORM': 'NORM', 'NDT': 'STTC', 'NST_': 'STTC', 'DIG': 'STTC', 'LBBB': 'CD', 'RBBB': 'CD',
    'CLBBB': 'CD', 'CRBBB': 'CD', 'ILBBB': 'CD', 'IRBBB': 'CD', 'LMI': 'MI', 'AMI': 'MI',
    'IMI': 'MI', 'PMI': 'MI', 'ALMI': 'MI', 'ILMI': 'MI', 'IPMI': 'MI', 'ASMI': 'MI',
    'INJAS': 'MI', 'INJAL': 'MI', 'INJLA': 'MI', 'INJIL': 'MI', 'INJIN': 'MI',
    'LVH': 'HYP', 'RVH': 'HYP', 'SEHYP': 'HYP',
    'ISC_': 'STTC', 'ISCA': 'STTC', 'ISCI': 'STTC', 'ISCIL': 'STTC', 'ISCIN': 'STTC', 'ISCLA': 'STTC',
    'LAFB': 'CD', 'LPFB': 'CD', 'IVCD': 'CD', 'PAC': 'STTC', 'PVC': 'STTC', 'AFIB': 'CD', 'AFLT': 'CD',
}


def extract_geriatric(db_path, records_dir, output_dir, min_age=60, val_fraction=0.15, seed=42):
    print(f"Reading database from {db_path}")
    df = pd.read_csv(db_path, index_col='ecg_id')

    geriatric = df[df['age'] >= min_age]
    print(f"{len(geriatric)}/{len(df)} records have age >= {min_age}")

    X, Y = [], []
    for idx, row in tqdm(geriatric.iterrows(), total=len(geriatric)):
        filename = row['filename_lr']
        file_path = os.path.join(records_dir, filename)
        try:
            sig, _ = wfdb.rdsamp(file_path)
            if sig.shape != (1000, 12):
                continue

            scp_codes = ast.literal_eval(row['scp_codes'])
            y = np.zeros(len(CLASSES))
            for code in scp_codes.keys():
                cls = SCP_TO_SUPERCLASS.get(code)
                if cls:
                    y[CLASSES.index(cls)] = 1

            X.append(sig)
            Y.append(y)
        except Exception:
            continue

    X = np.array(X)  # (N, 1000, 12)
    Y = np.array(Y)  # (N, 5)
    print(f"Loaded {len(X)} valid geriatric records.")

    if len(X) == 0:
        print("No records extracted — check records_dir / db_path.")
        return

    X = downsample_ecg(X, target_len=100)  # (N, 100, 12) -> 1200-dim flattened

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    n_val = max(1, int(len(X) * val_fraction))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "X_real_train.npy"), X[train_idx])
    np.save(os.path.join(output_dir, "Y_real_train.npy"), Y[train_idx])
    np.save(os.path.join(output_dir, "X_real_test.npy"), X[val_idx])
    np.save(os.path.join(output_dir, "Y_real_test.npy"), Y[val_idx])
    print(f"Saved {len(train_idx)} train / {len(val_idx)} test geriatric records to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Hospital D's geriatric (60+) PTB-XL subset.")
    parser.add_argument("--db-path", required=True,
                        help="Path to ptbxl_database.csv from the downloaded PTB-XL dataset.")
    parser.add_argument("--records-dir", required=True,
                        help="Path to the PTB-XL root dir containing the 'records100' folder.")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "src" / "hospital_d" / "data"),
                        help="Where to write the processed .npy splits (default: src/hospital_d/data).")
    parser.add_argument("--min-age", type=int, default=60)
    args = parser.parse_args()

    extract_geriatric(args.db_path, args.records_dir, args.output_dir, min_age=args.min_age)
