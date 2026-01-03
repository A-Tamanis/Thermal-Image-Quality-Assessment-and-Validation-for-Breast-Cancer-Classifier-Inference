import os
import glob
import numpy as np

from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, recall_score, f1_score, roc_auc_score

# -----------------------------
# CONFIG (change these)
# -----------------------------
MODEL_PATH = r"runs/classify/train6/weights/best.pt"

# Dataset must be: DATA_DIR/class_name/*.jpg (or png, etc.)
DATA_DIR = r"data_sq/test"   # e.g. .../binary_pose/test

# Choose which folder/class is the "positive" class for recall / ROC-AUC
POSITIVE_CLASS_NAME = "valid"          # e.g. "good" or "valid"

# Image extensions to include
EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
# -----------------------------


def collect_images(data_dir, exts):
    """Returns list of (image_path, class_name) from folder-structured dataset."""
    items = []
    for class_name in sorted(os.listdir(data_dir)):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for ext in exts:
            for p in glob.glob(os.path.join(class_dir, ext)):
                items.append((p, class_name))
    return items


def main():
    model = YOLO(MODEL_PATH)

    # Ultralytics class index -> class name mapping
    # e.g. {0: 'bad', 1: 'good'}
    names = model.names
    inv_names = {v: k for k, v in names.items()}

    if POSITIVE_CLASS_NAME not in inv_names:
        raise ValueError(
            f"POSITIVE_CLASS_NAME='{POSITIVE_CLASS_NAME}' not found in model.names={names}"
        )
    pos_idx = inv_names[POSITIVE_CLASS_NAME]

    samples = collect_images(DATA_DIR, EXTS)
    if not samples:
        raise RuntimeError(f"No images found under {DATA_DIR} with extensions {EXTS}")

    y_true = []
    y_pred = []
    y_score = []  # probability score for positive class (needed for ROC-AUC)

    for img_path, true_class_name in samples:
        if true_class_name not in inv_names:
            # If your folder names differ from model class names, fix them or add a mapping here.
            raise ValueError(
                f"Folder class '{true_class_name}' not in model.names={names}"
            )

        true_idx = inv_names[true_class_name]

        r = model(img_path, verbose=False)[0]  # one result
        probs = r.probs.data.cpu().numpy()     # shape: (num_classes,)

        pred_idx = int(np.argmax(probs))
        score_pos = float(probs[pos_idx])

        y_true.append(true_idx)
        y_pred.append(pred_idx)
        y_score.append(score_pos)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_score = np.array(y_score)

    # Confusion matrix in the order [0..C-1]
    cm = confusion_matrix(y_true, y_pred, labels=list(names.keys()))

    # Binary metrics (treat POSITIVE_CLASS_NAME as "positive")
    # Convert multiclass indices -> binary (positive vs not positive)
    y_true_bin = (y_true == pos_idx).astype(int)
    y_pred_bin = (y_pred == pos_idx).astype(int)

    recall = recall_score(y_true_bin, y_pred_bin)  # Sensitivity
    f1 = f1_score(y_true_bin, y_pred_bin)

    # Specificity = TN / (TN + FP)
    # In binary:
    # TN = count(true=0 and pred=0), FP = count(true=0 and pred=1)
    tn = int(np.sum((y_true_bin == 0) & (y_pred_bin == 0)))
    fp = int(np.sum((y_true_bin == 0) & (y_pred_bin == 1)))
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    # ROC-AUC needs probability scores for the positive class
    # Only valid if both classes exist in y_true_bin
    if len(np.unique(y_true_bin)) < 2:
        roc_auc = float("nan")
        roc_note = "ROC-AUC undefined (only one class present in ground truth)."
    else:
        roc_auc = roc_auc_score(y_true_bin, y_score)
        roc_note = ""

    print("Model classes:", names)
    print(f"Positive class: '{POSITIVE_CLASS_NAME}' (index={pos_idx})")
    print(f"Samples: {len(samples)}\n")

    print("Confusion Matrix (rows=true, cols=pred, order=class indices):")
    print(cm, "\n")

    print(f"Recall/Sensitivity (pos='{POSITIVE_CLASS_NAME}'): {recall:.4f}")
    print(f"Specificity (neg = not '{POSITIVE_CLASS_NAME}'): {specificity:.4f}")
    print(f"F1-score (pos='{POSITIVE_CLASS_NAME}'): {f1:.4f}")
    print(f"ROC-AUC (pos='{POSITIVE_CLASS_NAME}'): {roc_auc:.4f}")
    if roc_note:
        print(roc_note)


if __name__ == "__main__":
    main()
