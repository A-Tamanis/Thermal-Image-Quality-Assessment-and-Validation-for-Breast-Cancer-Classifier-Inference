"""
90° QUALITY PIPELINE

Runs, in order:
  1. sharpness_90deg.run_check
  2. binary_pose_90deg.run_check
  3. background_90deg.run_check
  4. pose_eval_90deg.run_check
  5. metrics_90deg.run_check

Stops at the first failure.
Each step must return a dict:
  {
    "ok": bool,
    "name": str,
    "reasons": [str, ...],
    "extra": {...}
  }
"""

from typing import List, Dict, Any, Optional

# --- import your step modules (adjust names if needed) ---
from BlurTestSingle90 import run_check as sharpness_run_90
from PoseEvalBin90 import run_check as binary_pose_run_90
#from BackgroundTest90 import run_check as background_run_90
from PoseEvalYOLO90 import run_check as pose_eval_run_90
from BreastQualityMask90 import run_check as metrics_run_90


CHECKS_90 = [
    sharpness_run_90,
    binary_pose_run_90,
    #background_run_90,
    pose_eval_run_90,
    metrics_run_90,
]


def run_pipeline_90(image_path: str) -> Dict[str, Any]:
    """
    Run the full 90° pipeline on a single image.

    Returns:
      {
        "ok": bool,
        "image_path": str,
        "failed_step": Optional[str],
        "steps": [ step_result0, step_result1, ... ]
      }

    Each step_result is exactly what the step's run_check() returned.
    """
    step_results: List[Dict[str, Any]] = []
    failed_step: Optional[str] = None

    for check_fn in CHECKS_90:
        result = check_fn(image_path)

        ok = bool(result.get("ok", False))
        name = result.get("name", check_fn.__name__)

        # normalise name field if missing
        if "name" not in result:
            result["name"] = name

        step_results.append(result)

        if not ok:
            failed_step = name
            break

    overall_ok = failed_step is None

    return {
        "ok": overall_ok,
        "image_path": image_path,
        "failed_step": failed_step,
        "steps": step_results,
    }



# For Standalone use
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="90° quality pipeline")
    parser.add_argument("image", help="Path to 90° thermal image")
    args = parser.parse_args()

    out = run_pipeline_90(args.image)

    print(f"IMAGE: {out['image_path']}")
    print(f"OVERALL OK: {out['ok']}")
    if out["failed_step"] is not None:
        print(f"FAILED STEP: {out['failed_step']}")

    print("\n--- STEP DETAILS ---")
    for i, step in enumerate(out["steps"]):
        print(f"\n[{i+1}] {step.get('name', '<unnamed>')}")
        print("  OK:", step.get("ok", False))
        for r in step.get("reasons", []):
            print("   -", r)
