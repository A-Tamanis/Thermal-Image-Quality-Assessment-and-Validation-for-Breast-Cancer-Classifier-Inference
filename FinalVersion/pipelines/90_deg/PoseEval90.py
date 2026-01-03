from dataclasses import dataclass
from typing import List, Dict, Any, Optional

# Model configuration and thresholds

# class ids for this model
BREAST_CLS  = 0
ARMPIT_CLS  = 1

# confidence thresholds
MIN_CONF_BREAST = 0.6
MIN_CONF_ARMPIT = 0.4

# box area sanity (fraction of image area)
BREAST_AREA_MIN = 0.02   # too small, invalid
BREAST_AREA_MAX = 0.50   # too big, invalid
ARMPIT_AREA_MIN = 0.005
ARMPIT_AREA_MAX = 0.25

# expected breasr center (normalized [0,1])
BREAST_CX_MIN = 0.25   # can go a bit left
BREAST_CX_MAX = 0.60   # not too far right
BREAST_CY_MIN = 0.35   # not too high
BREAST_CY_MAX = 0.80   # not too low

# expected armpit center region (top-right)
ARMPIT_CX_MIN = 0.50    # not too centered
ARMPIT_CX_MAX = 0.90    # not too much to the right
ARMPIT_CY_MIN = 0.05    # not close to top border
ARMPIT_CY_MAX = 0.45    # not too low

# relative constraints (armpit vs breast)
ARMPIT_ABOVE_MARGIN   = 0.15  # armpit must be at least this much higher
ARMPIT_RIGHT_MARGIN   = 0.15  # armpit must be this much more to the right

# border margins (normalized)
TOP_MARGIN    = 0.05
BOTTOM_MARGIN = 0.05
LEFT_MARGIN   = 0.05
RIGHT_MARGIN  = 0.05


@dataclass
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls: int
    img_w: int
    img_h: int

    # image width
    @property
    def w(self) -> float:
        return max(0.0, self.x2 - self.x1)

    # image height
    @property
    def h(self) -> float:
        return max(0.0, self.y2 - self.y1)

    # image area
    @property
    def area(self) -> float:
        return self.w * self.h

    # center of x-axis
    @property
    def cx(self) -> float:
        return self.x1 + self.w / 2.0

    # center of y-axis
    @property
    def cy(self) -> float:
        return self.y1 + self.h / 2.0

    # normalizing (0-1)
    @property
    def cx_n(self) -> float:
        return self.cx / self.img_w

    @property
    def cy_n(self) -> float:
        return self.cy / self.img_h

    @property
    def x1_n(self) -> float:
        return self.x1 / self.img_w

    @property
    def y1_n(self) -> float:
        return self.y1 / self.img_h

    @property
    def x2_n(self) -> float:
        return self.x2 / self.img_w

    @property
    def y2_n(self) -> float:
        return self.y2 / self.img_h

    @property
    def area_frac(self) -> float:
        return self.area / (self.img_w * self.img_h + 1e-6)

# keep best (highest-conf) box for each class, if any
def _select_best_box(
    detections: List[List[float]],
    cls_id: int,
    min_conf: float,
    img_w: int,
    img_h: int,
) -> Optional[Box]:
    
    best_box: Optional[Box] = None
    for det in detections:
        if len(det) < 6:
            continue
        x1, y1, x2, y2, conf, c = det
        c = int(c)
        if c != cls_id:
            continue
        if conf < min_conf:
            continue
        box = Box(x1, y1, x2, y2, conf, c, img_w, img_h)
        if best_box is None or box.conf > best_box.conf:
            best_box = box
    return best_box


def evaluate_pose_left(
    detections: List[List[float]],
    img_w: int,
    img_h: int,
) -> Dict[str, Any]:
    """
    Evaluate 'left side to camera' pose based on detections from the model:
        class 0: breast
        class 1: armpit

    detections: list of [x1, y1, x2, y2, conf, cls] in pixel coords.
    Returns:
        {
          "ok": bool,          # pass / fail
          "reasons": [str...], # explanation of checks
          "boxes": { "breast": {...}, "armpit": {...} }  # used boxes
        }
    """
    reasons: List[str] = []
    boxes_out: Dict[str, Dict[str, float]] = {}

    # pick best box per class
    breast = _select_best_box(detections, BREAST_CLS, MIN_CONF_BREAST, img_w, img_h)
    armpit = _select_best_box(detections, ARMPIT_CLS, MIN_CONF_ARMPIT, img_w, img_h)

    if breast is None:
        reasons.append("FAIL: breast not detected with sufficient confidence.")
        return {"ok": False, "reasons": reasons, "boxes": boxes_out}

    if armpit is None:
        reasons.append("FAIL: armpit not detected with sufficient confidence.")
        boxes_out["breast"] = breast.__dict__
        return {"ok": False, "reasons": reasons, "boxes": boxes_out}

    boxes_out["breast"] = breast.__dict__
    boxes_out["armpit"] = armpit.__dict__

    # breast area in allowed margin check
    if not (BREAST_AREA_MIN <= breast.area_frac <= BREAST_AREA_MAX):
        reasons.append(
            f"FAIL: breast area fraction {breast.area_frac:.3f} "
            f"outside [{BREAST_AREA_MIN}, {BREAST_AREA_MAX}]."
        )
        return {"ok": False, "reasons": reasons, "boxes": boxes_out}

    # armpit area in allowed margin check
    if not (ARMPIT_AREA_MIN <= armpit.area_frac <= ARMPIT_AREA_MAX):
        reasons.append(
            f"FAIL: armpit area fraction {armpit.area_frac:.3f} "
            f"outside [{ARMPIT_AREA_MIN}, {ARMPIT_AREA_MAX}]."
        )
        return {"ok": False, "reasons": reasons, "boxes": boxes_out}

    # breast center position in allowed window check
    if not (BREAST_CX_MIN <= breast.cx_n <= BREAST_CX_MAX):
        reasons.append(
            f"FAIL: breast center X {breast.cx_n:.3f} not in [{BREAST_CX_MIN}, {BREAST_CX_MAX}]."
        )
        return {"ok": False, "reasons": reasons, "boxes": boxes_out}

    if not (BREAST_CY_MIN <= breast.cy_n <= BREAST_CY_MAX):
        reasons.append(
            f"FAIL: breast center Y {breast.cy_n:.3f} not in [{BREAST_CY_MIN}, {BREAST_CY_MAX}]."
        )
        return {"ok": False, "reasons": reasons, "boxes": boxes_out}

    # breast border checks
    if breast.y1_n < TOP_MARGIN:
        reasons.append(f"FAIL: breast too close to top border (y1_n={breast.y1_n:.3f}).")
        return {"ok": False, "reasons": reasons, "boxes": boxes_out}

    if (1.0 - breast.y2_n) < BOTTOM_MARGIN:
        reasons.append(f"FAIL: breast too close to bottom border (y2_n={breast.y2_n:.3f}).")
        return {"ok": False, "reasons": reasons, "boxes": boxes_out}

    if breast.x1_n < LEFT_MARGIN:
        reasons.append(f"FAIL: breast too close to left border (x1_n={breast.x1_n:.3f}).")
        return {"ok": False, "reasons": reasons, "boxes": boxes_out}

    if (1.0 - breast.x2_n) < RIGHT_MARGIN:
        reasons.append(f"FAIL: breast too close to right border (x2_n={breast.x2_n:.3f}).")
        return {"ok": False, "reasons": reasons, "boxes": boxes_out}

    # armpit location, top-right region
    if not (ARMPIT_CX_MIN <= armpit.cx_n <= ARMPIT_CX_MAX):
        reasons.append(
            f"FAIL: armpit center X {armpit.cx_n:.3f} not in [{ARMPIT_CX_MIN}, {ARMPIT_CX_MAX}]."
        )
        return {"ok": False, "reasons": reasons, "boxes": boxes_out}

    if not (ARMPIT_CY_MIN <= armpit.cy_n <= ARMPIT_CY_MAX):
        reasons.append(
            f"FAIL: armpit center Y {armpit.cy_n:.3f} not in [{ARMPIT_CY_MIN}, {ARMPIT_CY_MAX}]."
        )
        return {"ok": False, "reasons": reasons, "boxes": boxes_out}

    # armpit above and to the right of breast check
    if not (armpit.cy_n + ARMPIT_ABOVE_MARGIN < breast.cy_n):
        reasons.append(
            "FAIL: armpit is not clearly above breast "
            f"(armpit.cy_n={armpit.cy_n:.3f}, breast.cy_n={breast.cy_n:.3f})."
        )
        return {"ok": False, "reasons": reasons, "boxes": boxes_out}

    if not (armpit.cx_n > breast.cx_n + ARMPIT_RIGHT_MARGIN):
        reasons.append(
            "FAIL: armpit is not clearly to the right of breast "
            f"(armpit.cx_n={armpit.cx_n:.3f}, breast.cx_n={breast.cx_n:.3f})."
        )
        return {"ok": False, "reasons": reasons, "boxes": boxes_out}

    # no checks were violated, success
    reasons.append("PASS: all left-pose checks satisfied.")
    return {"ok": True, "reasons": reasons, "boxes": boxes_out}
