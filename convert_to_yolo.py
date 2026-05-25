import json
import os
import numpy as np
import shutil
import random
from pathlib import Path

# ─── CONFIG ───────────────────────────────────────────────────────────────────
INPUT_ROOT       = "C:/landing_dataset_2"
OUTPUT_ROOT      = "C:/yolo_dataset"
IMG_W, IMG_H     = 640, 480
NUM_BATCHES      = 9
FRAMES_PER_BATCH = 1000

TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
SEED        = 42
# ──────────────────────────────────────────────────────────────────────────────

random.seed(SEED)

# Create output directories
for split in ("train", "val", "test"):
    Path(f"{OUTPUT_ROOT}/images/{split}").mkdir(parents=True, exist_ok=True)
    Path(f"{OUTPUT_ROOT}/labels/{split}").mkdir(parents=True, exist_ok=True)

# Collect all valid samples
valid_samples = []

for batch_idx in range(NUM_BATCHES):
    batch_dir = os.path.join(INPUT_ROOT, f"batch_{batch_idx:04d}")
    if not os.path.isdir(batch_dir):
        print(f"  [WARN] Missing batch: {batch_dir} — skipping")
        continue

    for frame_idx in range(FRAMES_PER_BATCH):
        fid      = f"{frame_idx:04d}"
        npy_path = os.path.join(batch_dir, f"bounding_box_2d_tight_{fid}.npy")
        lbl_path = os.path.join(batch_dir, f"bounding_box_2d_tight_labels_{fid}.json")
        img_path = os.path.join(batch_dir, f"rgb_{fid}.png")

        if not all(os.path.exists(p) for p in (npy_path, lbl_path, img_path)):
            continue

        data = np.load(npy_path, allow_pickle=True)
        if data.ndim == 0:
            data = data.item()
        data = np.atleast_1d(data)

        with open(lbl_path) as f:
            labels = json.load(f)

        label_lines = []
        for box in data:
            semantic_id = str(box["semanticId"])
            class_name  = labels.get(semantic_id, {}).get("class", "")
            if class_name != "landing_pad":
                continue

            x1, y1 = box["x_min"], box["y_min"]
            x2, y2 = box["x_max"], box["y_max"]

            if x2 <= x1 or y2 <= y1:
                continue

            x1, x2 = max(0, x1), min(IMG_W, x2)
            y1, y2 = max(0, y1), min(IMG_H, y2)

            cx = ((x1 + x2) / 2) / IMG_W
            cy = ((y1 + y2) / 2) / IMG_H
            w  = (x2 - x1) / IMG_W
            h  = (y2 - y1) / IMG_H
            label_lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        if label_lines:
            valid_samples.append((img_path, label_lines, batch_idx, frame_idx))

print(f"✓ Found {len(valid_samples)} valid frames across {NUM_BATCHES} batches")

# Shuffle and split
random.shuffle(valid_samples)
n       = len(valid_samples)
n_train = int(n * TRAIN_RATIO)
n_val   = int(n * VAL_RATIO)

splits = {
    "train": valid_samples[:n_train],
    "val":   valid_samples[n_train : n_train + n_val],
    "test":  valid_samples[n_train + n_val :],
}

print(f"  Split → train: {len(splits['train'])}  val: {len(splits['val'])}  test: {len(splits['test'])}")

# Write files
for split, samples in splits.items():
    for img_path, label_lines, b_idx, f_idx in samples:
        stem    = f"b{b_idx:04d}_f{f_idx:04d}"
        dst_img = os.path.join(OUTPUT_ROOT, "images", split, f"{stem}.png")
        dst_lbl = os.path.join(OUTPUT_ROOT, "labels", split, f"{stem}.txt")
        shutil.copy(img_path, dst_img)
        with open(dst_lbl, "w") as f:
            f.write("\n".join(label_lines))
    print(f"  ✓ {split} written ({len(samples)} samples)")

# Write dataset.yaml
yaml_content = f"""path: {OUTPUT_ROOT}
train: images/train
val:   images/val
test:  images/test

nc: 1
names: ['landing_pad']
"""

with open(os.path.join(OUTPUT_ROOT, "dataset.yaml"), "w") as f:
    f.write(yaml_content)

print(f"\n✓ dataset.yaml written to {OUTPUT_ROOT}")
print("  Next step: run train.py")