"""
45° QUALITY PIPELINE

Runs, in order:
  1. sharpness_45deg.run_check
  2. binary_pose_45deg.run_check
  3. background_45deg.run_check
  4. pose_eval_45deg.run_check
  5. metrics_45deg.run_check

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
from BlurTestSingle45 import run_check as sharpness_run_45
from PoseEvalBin45 import run_check as binary_pose_run_45
#from BackgroundTest45 import run_check as background_run_45
from PoseEvalYOLO45 import run_check as pose_eval_run_45
from BreastQualityMask45 import run_check as metrics_run_45


CHECKS_45 = [
    sharpness_run_45,
    binary_pose_run_45,
    #background_run_45,
    pose_eval_run_45,
    metrics_run_45,
]


def run_pipeline_45(image_path: str) -> Dict[str, Any]:
    """
    Run the full 45° pipeline on a single image.

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

    for check_fn in CHECKS_45:
        result = check_fn(image_path)
        # Be defensive: ensure required keys exist
        ok = bool(result.get("ok", False))
        name = result.get("name", check_fn.__name__)

        # Normalise name field if missing
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


# -----------------------------
# Simple CLI for testing
# -----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="45° quality pipeline")
    parser.add_argument("image", help="Path to 45° thermal image")
    args = parser.parse_args()

    out = run_pipeline_45(args.image)

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
