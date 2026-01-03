import cv2
import numpy as np
from typing import Dict, Any, List
from ultralytics import YOLO


#for YOLOv8 classification, the input image must be square
SQUARE_SIDE = 640
MODEL_PATH_0 = "models/best_binary0.pt"

# The class name that represents a correct 0-degree pose, essentialy what the model returns
CORRECT_CLASS_NAME_0 = "valid"     

# Load YOLO model once
_model_0 = YOLO(MODEL_PATH_0)


# transforms the grayscale image to a binary image
def ensure_binary(gray):
    _, out = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return out

# transforms the image to grayscale
def to_gray(img):
    if img is None:
        return None
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# pads the image from a 640x480 to a 640x640 square image
def pad_to_square(gray, target):
    h, w = gray.shape[:2]
    scale = min(target / w, target / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))

    resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    canvas = np.zeros((target, target), dtype=resized.dtype)
    top  = (target - new_h) // 2
    left = (target - new_w) // 2
    canvas[top:top+new_h, left:left+new_w] = resized

    return ensure_binary(canvas)

# root function that prepares the image for the Yolov8cls model to run
def preprocess_single_image_for_classifier(img_path, target=SQUARE_SIDE):
    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to read image: {img_path}")

    gray = to_gray(img)
    gray = ensure_binary(gray)
    sq   = pad_to_square(gray, target)

    sq_bgr = cv2.cvtColor(sq, cv2.COLOR_GRAY2BGR)
    return sq_bgr


# function called from pipeline script
#Runs the 0-degree binary pose classifier and returns a pipeline result dict
def run_check(image_path: str) -> Dict[str, Any]:
   
    name = "binary_pose_0deg"
    reasons: List[str] = []
    extra: Dict[str, Any] = {}

    # Load and preprocess
    try:
        img = preprocess_single_image_for_classifier(image_path, SQUARE_SIDE)
    except Exception as e:
        return {
            "ok": False,
            "name": name,
            "reasons": [f"FAIL: Could not preprocess image: {e}"],
            "extra": {}
        }

    # Run YOLOv8 classifier
    try:
        res = _model_0(img, imgsz=SQUARE_SIDE, device="cpu")[0]
    except Exception as e:
        return {
            "ok": False,
            "name": name,
            "reasons": [f"FAIL: YOLO inference error: {e}"],
            "extra": {}
        }

    pred_idx   = int(res.probs.top1)
    pred_name  = res.names[pred_idx]
    pred_conf  = float(res.probs.top1conf)

    extra["pred_class"] = pred_name
    extra["confidence"] = pred_conf

    # failure
    if pred_name != CORRECT_CLASS_NAME_0:
        reasons.append(
            f"FAIL: predicted class '{pred_name}' != expected '{CORRECT_CLASS_NAME_0}'."
        )
        reasons.append(f"Confidence: {pred_conf:.3f}")
        return {
            "ok": False,
            "name": name,
            "reasons": reasons,
            "extra": extra,
        }

    # Success
    reasons.append(f"PASS: predicted class '{pred_name}' with confidence {pred_conf:.3f}.")
    return {
        "ok": True,
        "name": name,
        "reasons": reasons,
        "extra": extra,
    }


# for standalone use
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="0 degree Binary Pose Check")
    parser.add_argument("image", help="Path to image")
    args = parser.parse_args()

    result = run_check(args.image)

    print("OK:", result["ok"])
    for r in result["reasons"]:
        print("-", r)
    print("Extra:", result["extra"])
