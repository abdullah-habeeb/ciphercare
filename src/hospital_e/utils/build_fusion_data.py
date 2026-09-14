"""
Builds Hospital E's multimodal fusion dataset by combining real signals
already prepared for Hospitals A (ECG), B (vitals), and C (chest X-ray).

Honesty note: PTB-XL (ECG), the synthetic-MIMIC vitals, and NIH ChestX-ray14
are three independent, unlinked datasets/patient populations -- there is no
public dataset that pairs ECG + vitals + chest X-ray for the same patient
with these disease labels. So this script is upfront about what it actually
does: it takes Hospital A's *real* ECG waveforms with their *real* diagnostic
labels (NORM/MI/STTC/CD/HYP) as the ground truth, and appends Hospital B's
real vitals rows and Hospital C's real chest X-ray pixel features as
additional real-valued (but label-uninformative, since they come from
different patients/datasets) input modalities. This is a legitimate way to
exercise a genuine multimodal fusion architecture end-to-end, and it is real
data throughout, but the vitals/lungs portions are not causally linked to
the label the way the ECG portion is. That's a materially different, more
honest situation than the previous code, which fed the model pure
np.random.randn() noise for every modality.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_lungs_features(images_dir: Path, n: int, feat_dim: int = 64):
    """Small deterministic pixel-based feature vector per real chest X-ray image."""
    files = sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.jpg"))
    if not files:
        raise FileNotFoundError(f"No images found in {images_dir}")

    feats = []
    side = int(np.sqrt(feat_dim))  # 8x8 = 64
    for i in range(n):
        img = Image.open(files[i % len(files)]).convert("L").resize((side, side))
        feats.append(np.asarray(img, dtype=np.float32).flatten() / 255.0)
    return np.stack(feats)


def build(output_dir: Path, n_samples: int = 800, seed: int = 42):
    ecg_dir = PROJECT_ROOT / "src" / "hospital_a" / "data"
    vitals_csv = PROJECT_ROOT / "data" / "hospital_b" / "processed_vitals.csv"
    images_dir = PROJECT_ROOT / "data" / "hospital_c" / "images"

    X_ecg_all = np.load(ecg_dir / "X_train.npy")  # (N, 100, 12) after downsampling
    Y_all = np.load(ecg_dir / "Y_train.npy")       # (N, 5)

    import pandas as pd
    vitals_df = pd.read_csv(vitals_csv)
    vitals_cols = [c for c in vitals_df.columns if c not in ("patient_id", "deterioration_risk")]
    X_vitals_all = vitals_df[vitals_cols].values.astype(np.float32)

    n = min(n_samples, len(X_ecg_all), len(X_vitals_all))
    rng = np.random.default_rng(seed)

    ecg_idx = rng.choice(len(X_ecg_all), size=n, replace=False)
    vitals_idx = rng.choice(len(X_vitals_all), size=n, replace=False)

    X_ecg = X_ecg_all[ecg_idx].reshape(n, -1)  # flatten (100,12) -> 1200
    Y_labels = Y_all[ecg_idx]
    X_vitals = X_vitals_all[vitals_idx]
    X_vitals = np.nan_to_num(X_vitals, nan=0.0)
    X_lungs = load_lungs_features(images_dir, n)

    perm = rng.permutation(n)
    n_test = max(1, int(n * 0.15))
    test_idx, train_idx = perm[:n_test], perm[n_test:]

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "X_ecg.npy", X_ecg)
    np.save(output_dir / "X_vitals.npy", X_vitals)
    np.save(output_dir / "X_lungs.npy", X_lungs)
    np.save(output_dir / "Y_labels.npy", Y_labels)
    np.save(output_dir / "train_indices.npy", train_idx)
    np.save(output_dir / "test_indices.npy", test_idx)
    print(f"Built Hospital E fusion dataset: {n} samples "
          f"({len(train_idx)} train / {len(test_idx)} test) -> {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Hospital E's multimodal fusion dataset.")
    parser.add_argument("--n-samples", type=int, default=800)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "src" / "hospital_e" / "data"))
    args = parser.parse_args()
    build(Path(args.output_dir), n_samples=args.n_samples)
