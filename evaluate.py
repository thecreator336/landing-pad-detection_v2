import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from ultralytics import YOLO

if __name__ == '__main__':

    # ─── CONFIG ───────────────────────────────────────────────────────────────
    WEIGHTS    = r"C:\Users\epity\OneDrive\Documents\Landingrecog_2\runs\detect\landing_pad_v3\weights\best.pt"
    TEST_DIR   = "C:/yolo_dataset/images/test"
    LABELS_DIR = "C:/yolo_dataset/labels/test"
    OUTPUT_DIR = r"C:\Users\epity\OneDrive\Documents\Landingrecog_2\eval_results"
    CONF_THRESH = 0.25
    IOU_THRESHOLDS = [0.25, 0.50, 0.75]
    # ──────────────────────────────────────────────────────────────────────────

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = YOLO(WEIGHTS)

    # ── Run predictions on test set ───────────────────────────────────────────
    print("Running predictions on test set...")
    results_all = model.predict(
        source  = TEST_DIR,
        conf    = CONF_THRESH,
        verbose = False,
    )
    print(f"Predictions done — {len(results_all)} images processed")

    # ── Helper functions ──────────────────────────────────────────────────────
    def load_gt_boxes(label_path, img_w=640, img_h=480):
        boxes = []
        if not os.path.exists(label_path):
            return np.zeros((0, 4))
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                _, cx, cy, w, h = map(float, parts)
                x1 = (cx - w / 2) * img_w
                y1 = (cy - h / 2) * img_h
                x2 = (cx + w / 2) * img_w
                y2 = (cy + h / 2) * img_h
                boxes.append([x1, y1, x2, y2])
        return np.array(boxes) if boxes else np.zeros((0, 4))

    def box_iou(box_a, box_b):
        ix1 = max(box_a[0], box_b[0])
        iy1 = max(box_a[1], box_b[1])
        ix2 = min(box_a[2], box_b[2])
        iy2 = min(box_a[3], box_b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_a = (box_a[2]-box_a[0]) * (box_a[3]-box_a[1])
        area_b = (box_b[2]-box_b[0]) * (box_b[3]-box_b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def build_confusion_matrix(results_all, labels_dir, iou_thresh):
        TP = FP = FN = TN = 0
        for result in results_all:
            stem     = Path(result.path).stem
            lbl_file = os.path.join(labels_dir, f"{stem}.txt")
            gt_boxes = load_gt_boxes(lbl_file)

            if result.boxes and len(result.boxes):
                pred_boxes = result.boxes.xyxy.cpu().numpy()
            else:
                pred_boxes = np.zeros((0, 4))

            n_gt   = len(gt_boxes)
            n_pred = len(pred_boxes)

            if n_gt == 0 and n_pred == 0:
                TN += 1
                continue
            if n_gt == 0:
                FP += n_pred
                continue
            if n_pred == 0:
                FN += n_gt
                continue

            matched_gt   = set()
            matched_pred = set()

            iou_matrix = np.zeros((n_gt, n_pred))
            for gi, gb in enumerate(gt_boxes):
                for pi, pb in enumerate(pred_boxes):
                    iou_matrix[gi, pi] = box_iou(gb, pb)

            pairs = sorted(
                [(iou_matrix[gi, pi], gi, pi)
                 for gi in range(n_gt)
                 for pi in range(n_pred)],
                reverse=True,
            )

            for iou_val, gi, pi in pairs:
                if gi in matched_gt or pi in matched_pred:
                    continue
                if iou_val >= iou_thresh:
                    TP += 1
                    matched_gt.add(gi)
                    matched_pred.add(pi)

            FN += n_gt   - len(matched_gt)
            FP += n_pred - len(matched_pred)

        return TP, FP, FN, TN

    # ── Build and plot confusion matrices ─────────────────────────────────────
    print("\nBuilding confusion matrices...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Confusion Matrix — Landing Pad Detector\n(TEST split, 900 images)",
        fontsize=14, fontweight="bold"
    )

    for ax, iou_thresh in zip(axes, IOU_THRESHOLDS):
        TP, FP, FN, TN = build_confusion_matrix(results_all, LABELS_DIR, iou_thresh)

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        matrix = np.array([[TP, FN],
                            [FP, TN]])

        labels = np.array([
            [f"TP\n{TP}",  f"FN\n{FN}"],
            [f"FP\n{FP}",  f"TN\n{TN}"],
        ])

        sns.heatmap(
            matrix,
            annot=labels,
            fmt="",
            ax=ax,
            cmap="Blues",
            cbar=False,
            linewidths=1,
            linecolor="white",
            xticklabels=["Predicted Positive", "Predicted Negative"],
            yticklabels=["Actual Positive",    "Actual Negative"],
            annot_kws={"size": 13, "weight": "bold"},
        )
        ax.set_title(
            f"IoU ≥ {iou_thresh:.2f}\n"
            f"P={precision:.3f}  R={recall:.3f}  F1={f1:.3f}",
            fontsize=11,
        )

        print(f"\n  IoU ≥ {iou_thresh:.2f}")
        print(f"    TP={TP}  FP={FP}  FN={FN}  TN={TN}")
        print(f"    Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}")

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n✓ Confusion matrix saved → {save_path}")