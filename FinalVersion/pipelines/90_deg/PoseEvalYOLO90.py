import cv2
from ultralytics import YOLO
from PoseEval90 import evaluate_pose_left



# function called from the pipeline

def run_check(image_path: str):
    """
    90° pose evaluation wrapper.
    Returns:
        {
            "ok": bool,
            "name": "pose_eval_90deg",
            "reasons": [...],
            "extra": {
                "detections": [...],
                "raw_pose": {...}
            }
        }
    """

    name = "pose_eval_90deg"
    reasons = []
    extra = {}

    # load the YOLO model
    try:
        model = YOLO("models/best_box_90.pt")
    except Exception as e:
        return {
            "ok": False,
            "name": name,
            "reasons": [f"FAIL: Could not load YOLO model: {e}"],
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
        results = model(img)[0]
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

    # No detections at all, fail
    if len(detections) == 0:
        return {
            "ok": False,
            "name": name,
            "reasons": ["FAIL: No objects detected in image."],
            "extra": extra,
        }


    # evaluate pose
    pose_info = evaluate_pose_left(detections, img_w, img_h)
    extra["raw_pose"] = pose_info

    ok = pose_info.get("ok", False)

    # reasoning
    if not ok:
        for r in pose_info.get("reasons", []):
            reasons.append(r)
    else:
        reasons.append("PASS: pose is acceptable.")

    return {
        "ok": ok,
        "name": name,
        "reasons": reasons,
        "extra": extra
    }



# for standalone use
if __name__ == "__main__":
    test_img = "p058.jpg"

    out = run_check(test_img)

    print("Pose OK:", out["ok"])
    print("Reasons:")
    for r in out["reasons"]:
        print("-", r)

    print("Extra keys: ")
    for e in out["extra"]:
        print(f" {e}")
