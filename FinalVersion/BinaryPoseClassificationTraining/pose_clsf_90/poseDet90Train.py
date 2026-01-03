import os
from pathlib import Path
import cv2
import numpy as np
import random
import torch
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, classification_report
import csv
import shutil

# -----------------------------
# Config (no CLI args)
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DATA     = PROJECT_ROOT / "dataset_90"        # expects train/ val/ test/
DST_DATA     = PROJECT_ROOT / "dataset_90_sq"     # will be (re)created
SQUARE_SIDE  = 640                          # final padded size
MODEL_NAME   = "yolov8n-cls.pt"
IMGSZ        = 640                          # train size (square)
EPOCHS       = 1
BATCH        = 4                            # CPU-friendly at 640
PATIENCE     = 5
WORKERS      = 0 if os.name == "nt" else 2  # 0 on Windows; 2 on Linux/macOS
DEVICE       = "cpu"                        # force CPU

# -----------------------------
# Repro
# -----------------------------
random.seed(42); np.random.seed(42); torch.manual_seed(42)

def ensure_binary(gray):
    # force strictly 0/255
    _, out = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return out

def pad_to_square(gray, target):
    h, w = gray.shape[:2]
    # keep aspect: fit the longer side to target
    scale = min(target / w, target / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((target, target), dtype=resized.dtype)  # black
    top  = (target - new_h) // 2
    left = (target - new_w) // 2
    canvas[top:top+new_h, left:left+new_w] = resized
    return ensure_binary(canvas)

def to_gray(img):
    if img is None:
        return None
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def build_squared_copy(src_root: Path, dst_root: Path, target: int):
    if not (src_root / "train").exists():
        raise SystemExit(f"[error] {src_root} missing train/")

    # clean & rebuild
    if dst_root.exists():
        shutil.rmtree(dst_root)
    print(f"[info] Creating square dataset at: {dst_root}")
    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    for split in ["train", "val", "test"]:
        split_dir = src_root / split
        if not split_dir.exists():
            continue
        for cls_name in sorted([d.name for d in split_dir.iterdir() if d.is_dir()]):
            in_dir  = split_dir / cls_name
            out_dir = dst_root / split / cls_name
            out_dir.mkdir(parents=True, exist_ok=True)
            for fn in os.listdir(in_dir):
                if not fn.lower().endswith(exts): 
                    continue
                p_in = in_dir / fn
                img = cv2.imread(str(p_in), cv2.IMREAD_UNCHANGED)
                if img is None:
                    print(f"[warn] failed to read {p_in}")
                    continue
                gray = to_gray(img)
                gray = ensure_binary(gray)
                sq   = pad_to_square(gray, target)
                cv2.imwrite(str(out_dir / (Path(fn).stem + ".png")), sq)
    print("[info] Square copy complete.")

def evaluate_on_test(model: YOLO, test_dir: Path):
    preds = model.predict(source=str(test_dir), imgsz=IMGSZ, save=False, stream=False, verbose=False, device=DEVICE)
    idx_to_name = model.names
    name_to_idx = {v: k for k, v in idx_to_name.items()}
    gt_labels, pred_labels, paths, pred_confs = [], [], [], []
    for r in preds:
        p = Path(r.path)
        cls_name_gt = p.parent.name  # expects .../test/<class>/<file>
        if cls_name_gt not in name_to_idx:
            raise SystemExit(f"[error] Unknown class folder '{cls_name_gt}' in test/. Expected: {list(name_to_idx.keys())}")
        gt_idx = name_to_idx[cls_name_gt]
        top1   = int(r.probs.top1)
        conf   = float(r.probs.top1conf)
        gt_labels.append(gt_idx); pred_labels.append(top1); pred_confs.append(conf); paths.append(str(p))
    gt_labels = np.array(gt_labels); pred_labels = np.array(pred_labels)
    label_names = [idx_to_name[i] for i in range(len(idx_to_name))]
    cm = confusion_matrix(gt_labels, pred_labels, labels=list(range(len(idx_to_name))))
    print("\nConfusion matrix (rows=GT, cols=Pred):")
    print("Labels order:", label_names)
    print(cm)
    print("\nClassification report:")
    print(classification_report(gt_labels, pred_labels, target_names=label_names, digits=4))
    out_csv = PROJECT_ROOT / "test_predictions.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "gt_class", "pred_class", "pred_confidence"])
        for p, gt, pr, cf in zip(paths, gt_labels, pred_labels, pred_confs):
            w.writerow([p, idx_to_name[gt], idx_to_name[pr], f"{cf:.6f}"])
    print(f"\n[info] Wrote per-image predictions to: {out_csv}")

def main():
    if not SRC_DATA.exists():
        raise SystemExit(f"[error] data/ not found at {SRC_DATA}")

    # 1) Build padded square copy
    build_squared_copy(SRC_DATA, DST_DATA, SQUARE_SIDE)

    train_dir = DST_DATA / "train"
    val_dir   = DST_DATA / "val"
    test_dir  = DST_DATA / "test"
    if not train_dir.exists() or not val_dir.exists():
        raise SystemExit(f"[error] Missing train/ or val/ under {DST_DATA}")

    print(f"[info] CUDA available: {torch.cuda.is_available()} (ignored, using CPU)")
    print(f"[info] Training on {train_dir} with imgsz={IMGSZ}")

    # 2) Train on CPU
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

    # 3) Validate
    print("[info] Validating on val/ ...")
    model.val(split="val", data=str(val_dir), imgsz=IMGSZ, batch=BATCH, device=DEVICE)

    # 4) Test (optional)
    if test_dir.exists():
        print("[info] Predicting on test/ ...")
        evaluate_on_test(model, test_dir)
    else:
        print("[info] No test/ directory found; skipping test evaluation.")

    print("\n[done] Check runs/classify/* for logs and weights.")

if __name__ == "__main__":
    main()
