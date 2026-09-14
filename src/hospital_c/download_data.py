import os
from pathlib import Path

from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = str(PROJECT_ROOT / "data" / "hospital_c" / "images")
CSV_PATH = str(PROJECT_ROOT / "data" / "hospital_c" / "labels.csv")

# alkzar90/NIH-Chest-X-ray-dataset uses a HF "loading script", which recent
# versions of `datasets` refuse to execute. Sohaibsoussi/NIH-Chest-X-ray-dataset-small
# is a parquet-native re-upload of the same NIH ChestX-ray14 data (same 15-class
# label schema used in src/hospital_c/api.py's LABELS list) and loads without a script.
DATASET_NAME = "Sohaibsoussi/NIH-Chest-X-ray-dataset-small"
LABEL_NAMES = [
    "No Finding", "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax", "Consolidation",
    "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
]
NUM_IMAGES = 5000


def download_data():
    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"Streaming {DATASET_NAME} (real NIH ChestX-ray14 data)... target: {NUM_IMAGES} images")
    ds = load_dataset(DATASET_NAME, split="train", streaming=True)

    records = []
    for i, item in enumerate(ds):
        if i >= NUM_IMAGES:
            break

        image = item["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")

        fname = f"img_{i:05d}.png"
        image.save(os.path.join(DATA_DIR, fname))

        label_names = "|".join(LABEL_NAMES[idx] for idx in item["labels"]) or "No Finding"
        records.append({"Image Index": fname, "Finding Labels": label_names})

        if (i + 1) % 250 == 0:
            print(f"Saved {i + 1}/{NUM_IMAGES} images...")

    import pandas as pd
    pd.DataFrame(records).to_csv(CSV_PATH, index=False)
    print(f"Saved {len(records)} real images and labels to {CSV_PATH}")


if __name__ == "__main__":
    download_data()
