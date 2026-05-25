# Landing Pad Detection — Synthetic Data Pipeline v2

A synthetic data pipeline built in NVIDIA Isaac Sim to train a real-time landing pad detector for autonomous drone landing. The model is trained entirely on procedurally generated data and achieves perfect detection accuracy on the held-out test set.

---

## Project Overview

This is the second iteration of the landing pad detection pipeline, built as part of an internship project at ePlane Company. The goal is to train a computer vision model capable of detecting a rooftop landing pad from a downward-facing drone camera.

The full pipeline runs as follows:

Isaac Sim (Replicator) → convert_to_yolo.py → train.py → evaluate.py

---

## What's New in v2

| Feature | v1 | v2 |
|---|---|---|
| Environment | Flat ground plane | Rooftop building with industrial objects |
| Dataset size | 100 images | 9000 images (~4500 unique) |
| Train/val split | Same images for both | Proper 80/10/10 train/val/test |
| Motion blur | None | Render-level shutter + camera drift |
| Shadow variation | None | Full sunrise to sunset sun angle |
| Landing pad appearance | Single yellow | Yellow, white, worn/dirty |
| Confusion matrix | None | Per IoU threshold (0.25, 0.50, 0.75) |
| Model | YOLOv8n | YOLOv11s |

---

## Scene Setup in Isaac Sim

The scene consists of:
- A cube building with gravel rooftop texture
- A cylindrical pedestal with a flat disk landing pad on top
- Corner rails and guard rails around the rooftop perimeter
- Industrial objects: cardboard box, metal container, storage rack with crates
- SunLight (Distant Light) and DomeLight for lighting

### Randomisation Per Frame
- Camera position: X/Y +/-400 units, Z 300-600 units
- Camera look-at: +/-80 units drift to simulate drone movement
- Sun angle: -5 to -85 degree elevation, 0-360 degree compass direction
- Sun intensity: 1000-15000
- Sun colour: warm golden to pure white
- Dome intensity: 250-500
- Dome colour: cool blue-grey to neutral white
- Motion blur: enabled via OmniverseGlobalRenderSettings

---

## Data Generation

Run isaac_sim_replicator.py inside Isaac Sim's Script Editor. Change BATCH_INDEX at the top for each session (0-8).

| Batches | Landing Pad Colour |
|---|---|
| 0-3 | Yellow |
| 4-6 | White |
| 7-8 | Worn/dirty |

Each batch generates 1000 frames (~500 unique after deduplication) into:

C:/landing_dataset_2/batch_0000/
C:/landing_dataset_2/batch_0001/
...

---

## Converting to YOLO Format

After all batches are complete run:

    python convert_to_yolo.py

This reads all 9 batch folders, converts Isaac Sim pixel coordinates to YOLO normalised format, shuffles everything randomly, and writes an 80/10/10 train/val/test split to:

C:/yolo_dataset/
    images/train/   val/   test/
    labels/train/   val/   test/
    dataset.yaml

---

## Training

    python train.py

| Setting | Value |
|---|---|
| Model | YOLOv11s |
| Epochs | 100 |
| Batch size | 8 |
| Image size | 640 |
| Device | CUDA (NVIDIA GPU) |
| Training time | ~5.5 hours on RTX 3050 |

### Requirements

    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install ultralytics

---

## Evaluation

    python evaluate.py

Runs the trained model on the held-out test split (900 images) and generates a confusion matrix at three IoU thresholds.

### Results

| IoU Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.25 | 1.000 | 1.000 | 1.000 |
| 0.50 | 1.000 | 1.000 | 1.000 |
| 0.75 | 1.000 | 1.000 | 1.000 |

Best weights saved at:

    runs/detect/landing_pad_v3/weights/best.pt

---

## Repository Structure

    landing-pad-detection_v2/
        isaac_sim_replicator.py   — Isaac Sim data generation script
        convert_to_yolo.py        — converts raw output to YOLO format with 80/10/10 split
        train.py                  — YOLOv11s training script
        evaluate.py               — confusion matrix evaluation on test split
        README.md

---

## Future Work

- Real-world validation on actual drone footage
- Weather simulation: rain, fog, dust overlays
- Partial occlusion training
- Multiple building environments and rooftop layouts
- HDRI sky textures for physically accurate lighting
- Integration with flight controller for closed-loop autonomous landing

---

## Built With

- NVIDIA Isaac Sim + Omniverse Replicator
- YOLOv11s (Ultralytics)
- PyTorch 2.5.1 + CUDA 12.1
- Python 3.11
