"""
0° (FRONTAL) QUALITY PIPELINE

Runs in order:
  1. FrontBlurTestSingle.run_check      sharpness / blur
  2. FrontPoseDetBin.run_check          binary pose / silhouette check
  3. FrontBackgroundTest.run_check      background quality
  4. FrontPoseCheck.run_check           YOLO-based frontal pose analysis
  5. FrontBreastQualityMask.run_check   breast-level quality metrics

Stops at the first failure.
Each step returns a dict:
  {
    "ok": bool,                   if the image passed this check
    "name": str,                  the name of the check
    "reasons": [str, ...],        reasons why the check failed or passed, explanation for each check
    "extra": {...}                metrics reguarding each check (for example, the Laplacian variance for image sharpness)
  }
"""

from typing import List, Dict, Any, Optional

from FrontBlurTestSingle import run_check as sharpness_run_front
from FrontPoseDetBin import run_check as binary_pose_run_front
#from FrontBackgroundTest import run_check as background_run_front
from FrontPoseCheck import run_check as pose_eval_run_front
from FrontBreastQualityMask import run_check as metrics_run_front


CHECKS_FRONT = [
    sharpness_run_front,
    binary_pose_run_front,
    #background_run_front,
    pose_eval_run_front,
    metrics_run_front,
]


def run_pipeline_0deg(image_path: str) -> Dict[str, Any]:
    """
    Run the full 0 degree (frontal) pipeline on a single image

    Returns:
      {
        "ok": bool,
        "image_path": str,
        "failed_step": Optional[str],
        "steps": [ step_result0, step_result1, ... ]
      }

    Each step_result is exactly what the step's run_check() returned
    """
    step_results: List[Dict[str, Any]] = []
    failed_step: Optional[str] = None

    for check_fn in CHECKS_FRONT:
        result = check_fn(image_path)

        # ensuring required keys exist
        ok = bool(result.get("ok", False))
        name = result.get("name", check_fn.__name__)

        # Normalising name field if missing
        if "name" not in result:
            result["name"] = name

        step_results.append(result)

        if not ok:
            failed_step = name
            #break

    overall_ok = failed_step is None

    return {
        "ok": overall_ok,
        "image_path": image_path,
        "failed_step": failed_step,
        "steps": step_results,
    }


# for running through command line
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="0 degree (frontal) quality pipeline")
    parser.add_argument("image", help="Path to frontal (0 degrees) thermal image")
    args = parser.parse_args()

    out = run_pipeline_0deg(args.image)

    print(f"IMAGE: {out['image_path']}")
    print(f"OVERALL OK: {out['ok']}")
    if out["failed_step"] is not None:
        print(f"FAILED STEP: {out['failed_step']}")

    print("\n--- STEP DETAILS---")
    for i, step in enumerate(out["steps"]):
        print(f"\n[{i+1}] {step.get('name', '<unnamed>')}")
        print("  OK:", step.get("ok", False))
        for r in step.get("reasons", []):
            print("   -", r)
