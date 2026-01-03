import os
from pathlib import Path
import random
import numpy as np
import torch
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, classification_report
import csv

# -----------------------------
# Configuration (no CLI args)
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT    = PROJECT_ROOT / "data"     # expects train/ val/ test/ inside
MODEL_NAME   = "yolov8n-cls.pt"          # try 'yolov8s-cls.pt' if you need more capacity
IMGSZ        = 640                       # you requested 640 exactly
EPOCHS       = 25
BATCH        = 8
PATIENCE     = 5
WORKERS      = 2

# -----------------------------
# Reproducibility (light)
# -----------------------------
random.seed(42); np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

def main():
    train_dir = DATA_ROOT / "train"
    val_dir   = DATA_ROOT / "val"
    test_dir  = DATA_ROOT / "test"

    if not train_dir.exists() or not val_dir.exists():
        raise SystemExit(f"[error] Missing train/ or val/ under {DATA_ROOT}")

    print(f"[info] CUDA available: {torch.cuda.is_available()}")
    print(f"[info] Training with imgsz={IMGSZ}, epochs={EPOCHS}, batch={BATCH}")

    # -----------------------------
    # Train
    # -----------------------------
    model = YOLO(MODEL_NAME)
    results = model.train(
        data=str(train_dir),  # directory with class subfolders
        val=True,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        patience=PATIENCE,
        workers=WORKERS,
    )

    # -----------------------------
    # Validate on val/
    # -----------------------------
    print("[info] Validating on val/ ...")
    model.val(split="val", data=str(val_dir), imgsz=IMGSZ, batch=BATCH)

    # -----------------------------
    # Predict on test/ and report
    # -----------------------------
    if test_dir.exists():
        print("[info] Predicting on test/ and building metrics ...")
        preds = model.predict(source=str(test_dir), imgsz=IMGSZ, save=False, stream=False, verbose=False)

        # Map class name -> index using model.names
        # model.names is a dict like {0:'classA', 1:'classB', ...}
        idx_to_name = model.names
        name_to_idx = {v: k for k, v in idx_to_name.items()}

        # Build GT/pred arrays (label order per model.names)
        gt_labels, pred_labels, paths, pred_confs = [], [], [], []

        for r in preds:
            p = Path(r.path)
            cls_name_gt = p.parent.name  # expects .../test/<class>/<file>
            if cls_name_gt not in name_to_idx:
                # If your folder names differ in case/spelling, normalize here
                raise SystemExit(f"[error] Unknown class folder '{cls_name_gt}' in test/. "
                                 f"Expected one of: {list(name_to_idx.keys())}")
            gt_idx = name_to_idx[cls_name_gt]

            top1 = int(r.probs.top1)
            conf = float(r.probs.top1conf)

            gt_labels.append(gt_idx)
            pred_labels.append(top1)
            pred_confs.append(conf)
            paths.append(str(p))

        gt_labels   = np.array(gt_labels)
        pred_labels = np.array(pred_labels)

        # Confusion matrix & report
        # Order rows/cols by class index (0..C-1); names from idx_to_name
        num_classes = len(idx_to_name)
        label_names = [idx_to_name[i] for i in range(num_classes)]

        cm = confusion_matrix(gt_labels, pred_labels, labels=list(range(num_classes)))
        print("\nConfusion matrix (rows=GT, cols=Pred):")
        print("Labels order:", label_names)
        print(cm)

        print("\nClassification report:")
        print(classification_report(gt_labels, pred_labels, target_names=label_names, digits=4))

        # Save per-image predictions
        out_csv = PROJECT_ROOT / "test_predictions.csv"
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image_path", "gt_class", "pred_class", "pred_confidence"])
            for p, gt, pr, cf in zip(paths, gt_labels, pred_labels, pred_confs):
                w.writerow([p, idx_to_name[gt], idx_to_name[pr], f"{cf:.6f}"])
        print(f"\n[info] Wrote per-image predictions to: {out_csv}")

    else:
        print("[info] No test/ directory found; skipping test evaluation.")

    print("\n[done] Check runs/classify/* for training logs and weights.")

if __name__ == "__main__":
    main()
