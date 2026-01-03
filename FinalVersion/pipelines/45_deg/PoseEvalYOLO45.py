import cv2
from ultralytics import YOLO
from PoseEval45 import evaluate_pose   


# model configuration
DETECTION_MODEL_45 = "models/best_box_45.pt" 
DEVICE_45 = "cpu"      

_model_45 = None   



# function used by the pipeline
def run_check(image_path: str):
    """
    45° detection-based pose evaluation.
    Returns:
      {
        "ok": bool,
        "name": "pose_eval_45deg",
        "reasons": [...],
        "extra": {
            "detections": [...],
            "pose_info": {...}
        }
      }
    """
    global _model_45
    name = "pose_eval_45deg"
    reasons = []
    extra = {}


    # Load model once
    if _model_45 is None:
        try:
            _model_45 = YOLO(DETECTION_MODEL_45)
        except Exception as e:
            return {
                "ok": False,
                "name": name,
                "reasons": [f"FAIL: Could not load YOLO model '{DETECTION_MODEL_45}': {e}"],
                "extra": {}
            }


    # load image
    img = cv2.imread(image_path)
    if img is None:
        return {
            "ok": False,
            "name": name,
            "reasons": [f"FAIL: Could not read image '{image_path}'"],
            "extra": {}
        }


    # run YOLO detection
    try:
        results = _model_45(img, device=DEVICE_45)[0]
    except Exception as e:
        return {
            "ok": False,
            "name": name,
            "reasons": [f"FAIL: YOLO inference error: {e}"],
            "extra": {}
        }

    img_h, img_w = results.orig_shape
    detections = results.boxes.data.cpu().numpy().tolist()
    extra["detections"] = detections

    if len(detections) == 0:
        return {
            "ok": False,
            "name": name,
            "reasons": ["FAIL: No detections found in image."],
            "extra": extra
        }


    # evaluate pose
    pose_info = evaluate_pose(detections, img_w, img_h)
    extra["pose_info"] = pose_info

    ok = pose_info.get("ok", False)

    # reasoning
    if not ok:
        for r in pose_info.get("reasons", []):
            #reasons.append(f"FAIL: {r}")
            reasons.append(r)
    else:
        reasons.append("PASS: pose acceptable.")

    return {
        "ok": ok,
        "name": name,
        "reasons": reasons,
        "extra": extra,
    }



# for standalone use
if __name__ == "__main__":
    test_img = "pz.jpg"
    out = run_check(test_img)

    print("OK:", out["ok"])
    print("Reasons:")
    for r in out["reasons"]:
        print(" -", r)
    print("Extra:", list(out["extra"].keys()))
    print("Pose Info:")
    for p in out["extra"]["pose_info"]["reasons"]:
        print("-", p)
    print("Detections:")
    for b in out["extra"]["pose_info"]["used_boxes"]:
        print("-", b)


