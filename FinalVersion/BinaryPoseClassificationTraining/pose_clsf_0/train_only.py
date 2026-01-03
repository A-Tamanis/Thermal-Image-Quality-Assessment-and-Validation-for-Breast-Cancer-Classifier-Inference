import os
from pathlib import Path
import random
import numpy as np
import torch
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, classification_report
import csv

# -----------------------------
# Config
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT    = PROJECT_ROOT / "data_sq"    # <-- use the ALREADY padded dataset
MODEL_NAME   = "yolov8n-cls.pt"           # tiny classifier
#MODEL_NAME   = "runs/classify/train4/weights/last.pt" #pretrained classifier
IMGSZ        = 640                        # your padded images are 640x640
EPOCHS       = 40                         # change if you want
BATCH        = 2                          # CPU-friendly at 640
#PATIENCE     = 5
WORKERS      = 0 if os.name == "nt" else 2
DEVICE       = "cpu"                      # you have no GPU

# -----------------------------
# Repro
# -----------------------------
random.seed(42); np.random.seed(42); torch.manual_seed(42)

def evaluate_on_test(model: YOLO, test_dir: Path):
    preds = model.predict(
        source=str(test_dir),
        imgsz=IMGSZ,
        save=False,
        stream=False,
        verbose=False,
        device=DEVICE,
    )

    idx_to_name = model.names
    name_to_idx = {v: k for k, v in idx_to_name.items()}
    gt_labels, pred_labels, paths, pred_confs = [], [], [], []

    for r in preds:
        p = Path(r.path)
        cls_name_gt = p.parent.name  # expects .../test/<class>/<file>
        if cls_name_gt not in name_to_idx:
            raise SystemExit(
                f"[error] Unknown class folder '{cls_name_gt}' in test/. "
                f"Expected: {list(name_to_idx.keys())}"
            )
        gt_idx = name_to_idx[cls_name_gt]
        top1   = int(r.probs.top1)
        conf   = float(r.probs.top1conf)

        gt_labels.append(gt_idx)
        pred_labels.append(top1)
        pred_confs.append(conf)
        paths.append(str(p))

    gt_labels   = np.array(gt_labels)
    pred_labels = np.array(pred_labels)

    label_names = [idx_to_name[i] for i in range(len(idx_to_name))]
    cm = confusion_matrix(gt_labels, pred_labels, labels=list(range(len(idx_to_name))))

    print("\nConfusion matrix (rows=GT, cols=Pred):")
    print("Labels order:", label_names)
    print(cm)

    print("\nClassification report:")
    print(classification_report(gt_labels, pred_labels, target_names=label_names, digits=4))

    out_csv = PROJECT_ROOT / "test_predictions_from_sq.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "gt_class", "pred_class", "pred_confidence"])
        for p, gt, pr, cf in zip(paths, gt_labels, pred_labels, pred_confs):
            w.writerow([p, idx_to_name[gt], idx_to_name[pr], f"{cf:.6f}"])

    print(f"\n[info] Wrote per-image predictions to: {out_csv}")

def main():
    train_dir = DATA_ROOT / "train"
    val_dir   = DATA_ROOT / "val"
    test_dir  = DATA_ROOT / "test"

    if not train_dir.exists() or not val_dir.exists():
        raise SystemExit(f"[error] Missing train/ or val/ under {DATA_ROOT}")

    print(f"[info] Using existing padded dataset at: {DATA_ROOT}")
    print(f"[info] CUDA available: {torch.cuda.is_available()} (ignored, using CPU)")
    print(f"[info] Training on {train_dir} with imgsz={IMGSZ}, epochs={EPOCHS}, batch={BATCH}")

    # 1) Train
    model = YOLO(MODEL_NAME)
    model.train(
        data=str(train_dir),
        val=True,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        #patience=PATIENCE,
        workers=WORKERS,
        device=DEVICE,
    )

    # 2) Validate
    print("[info] Validating on val/ ...")
    model.val(
        split="val",
        data=str(val_dir),
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
    )

    # 3) Test
    if test_dir.exists():
        print("[info] Predicting on test/ ...")
        evaluate_on_test(model, test_dir)
    else:
        print("[info] No test/ directory found; skipping test evaluation.")

    print("\n[done] Training/evaluation complete. Check runs/classify/* for logs and weights.")

if __name__ == "__main__":
    main()
