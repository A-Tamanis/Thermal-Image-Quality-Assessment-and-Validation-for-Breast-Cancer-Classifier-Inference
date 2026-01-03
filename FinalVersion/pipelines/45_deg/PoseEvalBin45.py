import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from ultralytics import YOLO


# Model configurations

MODEL_PATH_45 = "models/best_binary_45.pt"   
CORRECT_CLASS_NAME_45 = "valid"            
IMGSZ_45 = 640
DEVICE_45 = "cpu"

# cache model globally
_model_45: YOLO | None = None



# Preprocessing

# transforms the grayscale image to a binary image
def ensure_binary(gray: np.ndarray) -> np.ndarray:
    # force strictly 0/255
    _, out = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return out

# pads the image from a 640x480 to a 640x640 square image
def pad_to_square(gray: np.ndarray, target: int) -> np.ndarray:
    h, w = gray.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("Empty image passed to pad_to_square")

    # keep aspect: fit the longer side to target
    scale = min(target / w, target / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))

    # nearest neighbor (same as training)
    resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # black canvas
    canvas = np.zeros((target, target), dtype=resized.dtype)
    top  = (target - new_h) // 2
    left = (target - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized

    return ensure_binary(canvas)

# transforms the image to grayscale
def to_gray(img: np.ndarray) -> np.ndarray:
    if img is None:
        return None
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# root function that prepares the image for the Yolov8cls model to run
def preprocess_binary_pose_image(image_path: str, imgsz: int = 640) -> np.ndarray:
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to read image: {img_path}")

    gray = to_gray(img)
    gray = ensure_binary(gray)
    sq   = pad_to_square(gray, imgsz)

    # YOLO expects 3 channels; stack the binary image
    sq_3ch = np.stack([sq, sq, sq], axis=-1)
    return sq_3ch



# YOLO helper
def _load_model_45() -> YOLO:
    model = YOLO(MODEL_PATH_45)
    return model


def run_check(image_path: str) -> Dict[str, Any]:
    """
    45 degree binary pose classifier check.
    """
    global _model_45

    name = "binary_pose_45deg"
    reasons: List[str] = []
    extra: Dict[str, Any] = {}

    # load model
    if _model_45 is None:
        try:
            _model_45 = _load_model_45()
        except Exception as e:
            return {
                "ok": False,
                "name": name,
                "reasons": [f"FAIL: could not load YOLO classifier '{MODEL_PATH_45}': {e}"],
                "extra": {},
            }

    # preprocess image
    try:
        img_3ch = preprocess_binary_pose_image(image_path, imgsz=IMGSZ_45)
    except (FileNotFoundError, ValueError) as e:
        return {
            "ok": False,
            "name": name,
            "reasons": [f"FAIL: {e}"],
            "extra": {},
        }

    # run the model
    try:
        results = _model_45(img_3ch, imgsz=IMGSZ_45, device=DEVICE_45, verbose=True)[0]
    except Exception as e:
        return {
            "ok": False,
            "name": name,
            "reasons": [f"FAIL: YOLO classification error: {e}"],
            "extra": {},
        }

    probs = results.probs
    top1_id   = int(probs.top1)
    top1_conf = float(probs.top1conf)
    class_name = _model_45.names[top1_id]

    extra["class_name"] = class_name
    extra["class_id"] = top1_id
    extra["confidence"] = top1_conf

    # failure
    if class_name != CORRECT_CLASS_NAME_45:
        reasons.append(
            f"FAIL: predicted class '{class_name}' != expected '{CORRECT_CLASS_NAME_45}' "
            f"(conf={top1_conf:.3f})."
        )
        ok = False
    # success
    else:
        reasons.append(
            f"PASS: predicted class '{class_name}' with confidence {top1_conf:.3f}."
        )
        ok = True

    return {
        "ok": ok,
        "name": name,
        "reasons": reasons,
        "extra": extra,
    }


# fro standalone use
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="45° binary pose classifier check")
    parser.add_argument("image", help="Path to input image")
    args = parser.parse_args()

    res = run_check(args.image)
    print("OK:", res["ok"])
    for r in res["reasons"]:
        print("-", r)
    print("Extra:", res["extra"])
