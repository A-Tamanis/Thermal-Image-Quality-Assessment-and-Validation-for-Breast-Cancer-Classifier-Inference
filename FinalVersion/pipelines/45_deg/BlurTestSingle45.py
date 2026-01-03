import cv2
import argparse
from typing import Dict, Any, List

#The threshold an image needs to pass to be considered sharp
SHARPNESS_MIN_VARIANCE = 5.0


def run_check(image_path: str) -> Dict[str, Any]:
    """
    Sharpness check for 45 degree images using Laplacian variance.

    Returns a dict:
      {
        "ok": bool,
        "name": "sharpness_0deg",
        "reasons": [str, ...],
        "extra": {
            "variance": float,
            "threshold": float,
        }
      }
    """
    name = "sharpness_0deg"
    reasons: List[str] = []
    extra: Dict[str, Any] = {}

    # Read the image in grayscale
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if gray is None:
        reasons.append(f"FAIL: failed to read image '{image_path}'.")
        return {
            "ok": False,
            "name": name,
            "reasons": reasons,
            "extra": extra,
        }

    # Apply Laplacian and calculate variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = float(laplacian.var())

    extra["variance"] = variance
    extra["threshold"] = SHARPNESS_MIN_VARIANCE

    if variance < SHARPNESS_MIN_VARIANCE:
        reasons.append(
            f"FAIL: Laplacian variance {variance:.2f} < threshold "
            f"{SHARPNESS_MIN_VARIANCE:.2f} (image too blurry)."
        )
        ok = False
    else:
        reasons.append(
            f"PASS: Laplacian variance {variance:.2f} ≥ threshold "
            f"{SHARPNESS_MIN_VARIANCE:.2f} (sharpness acceptable)."
        )
        ok = True

    return {
        "ok": ok,
        "name": name,
        "reasons": reasons,
        "extra": extra,
    }

