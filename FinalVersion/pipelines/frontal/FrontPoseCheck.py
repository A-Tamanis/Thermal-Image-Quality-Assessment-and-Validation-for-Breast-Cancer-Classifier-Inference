
from ultralytics import YOLO
from FrontPoseAnalysis import analyze_pose


class FrontalPoseChecker:

    def __init__(self, model_path: str):
        """
        Loads the YOLO model once
        """
        self.model = YOLO(model_path)

    def check(self, image_path: str):
        """
        Runs YOLO and analyze_pose on a single image
        Returns:
            valid: bool
            verdict: str
            metrics: dict
        """
        # Run YOLO
        results = self.model(image_path)[0]

        # Extract image size
        img_h, img_w = results.orig_img.shape[:2]

        # Parse detections into correct format
        detections = []
        for b in results.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            cls_id = int(b.cls[0].item())
            conf   = float(b.conf[0].item())

            detections.append({
                "cls": cls_id,         # 0=breast, 1=armpit, 2=nipple
                "conf": conf,
                "xyxy": (x1, y1, x2, y2),
            })

        # Run full analysis
        valid, verdict, metrics = analyze_pose(
            detections=detections,
            img_w=img_w,
            img_h=img_h
        )

        return valid, verdict, metrics


MODEL_PATH = "models/best_pose0.pt"   

# Creates a single global checker so it loads YOLO once
_checker = FrontalPoseChecker(MODEL_PATH)

def run_check(image_path: str):
    """
    Pipeline-compatible wrapper around FrontalPoseChecker.check().
    Returns a dict of the form the pipeline uses.
    """

    valid, verdict, metrics = _checker.check(image_path)

    if valid:
        return {
            "ok": True,
            "name": "frontal_pose_check",
            "reasons": [],
            "extra": metrics
        }
    else:
        return {
            "ok": False,
            "name": "frontal_pose_check",
            "reasons": [verdict],   # verdict explains the first fail reason
            "extra": metrics
        }

# for standalone use
if __name__ == "__main__":
    test_img = "p12.jpg"
    out = run_check(test_img)

    print("OK:", out["ok"])
    print("Reasons:")
    for r in out["reasons"]:
        print(" -", r)
    print("Extra:", list(out["extra"].keys()))
    # print("Pose Info:")
    # for p in out["extra"]["metrics"]["reasons"]:
    #     print("-", p)
    # print("Detections:")
    # for b in out["extra"]["pose_info"]["used_boxes"]:
    #     print("-", b)
