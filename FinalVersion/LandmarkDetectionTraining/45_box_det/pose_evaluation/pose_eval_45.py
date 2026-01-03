from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import math


# ==========================
# Config / thresholds
# ==========================

CLASS_MAP = {
    0: "right_breast",
    1: "left_breast",
    2: "armpit",
    3: "right_nip",
    4: "left_nip",
}

MIN_CONF_STRONG = 0.6   # right_breast, armpit
MIN_CONF_WEAK   = 0.3   # left_breast, nipples

# Right breast expected relative size (fraction of image area)
RB_AREA_MIN = 0.05
RB_AREA_MAX = 0.45

# Right breast expected center (normalized 0–1)
RB_X_MIN = 0.40   # not too far left
RB_X_MAX = 0.75   # not hugging right border
RB_Y_MIN = 0.35
RB_Y_MAX = 0.75

# Armpit must be clearly above RB (margin in normalized units)
ARMPIT_ABOVE_MARGIN = 0.05

# Border margins (normalized)
TOP_MARGIN    = 0.05
BOTTOM_MARGIN = 0.05
LEFT_MARGIN   = 0.03
RIGHT_MARGIN  = 0.03

# Aspect ratio constraints for breast boxes
MIN_ASPECT = 0.6   # h/w
MAX_ASPECT = 1.8

# Min fraction of image area to consider a box non-tiny
MIN_BOX_AREA_FRAC = 0.001


@dataclass
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls: int
    name: str
    img_w: int
    img_h: int

    @property
    def w(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def h(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.w * self.h

    @property
    def cx(self) -> float:
        return self.x1 + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y1 + self.h / 2.0

    # normalized (0–1)
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

    @property
    def aspect(self) -> float:
        return (self.h + 1e-6) / (self.w + 1e-6)  # h / w

    def contains_point(self, x: float, y: float) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


def _select_best_boxes(
    detections: List[List[float]],
    img_w: int,
    img_h: int,
) -> Dict[str, Box]:
    """
    Keep best (highest-conf) box for each class, with basic conf thresholds.
    """
    best: Dict[str, Box] = {}
    for det in detections:
        if len(det) < 6:
            continue
        x1, y1, x2, y2, conf, cls_id = det
        cls_id = int(cls_id)
        name = CLASS_MAP.get(cls_id)
        if name is None:
            continue

        # confidence gating
        min_conf = MIN_CONF_STRONG if name in ("right_breast", "armpit") else MIN_CONF_WEAK
        if conf < min_conf:
            continue

        box = Box(x1, y1, x2, y2, conf, cls_id, name, img_w, img_h)

        # ignore absurdly tiny boxes
        if box.area_frac < MIN_BOX_AREA_FRAC:
            continue

        prev = best.get(name)
        if prev is None or box.conf > prev.conf:
            best[name] = box

    return best


def _virtual_nipple(breast_box: Box, real_nip: Optional[Box]) -> (float, float, bool):
    """
    Returns (x, y, is_real). If real nipple box is provided, use its center.
    Otherwise, create a virtual nipple slightly left of the center of the breast box.
    """
    if real_nip is not None:
        return real_nip.cx, real_nip.cy, True
    # virtual nipple: center x, a bit above center y
    x = breast_box.cx - 0.25 * breast_box.w
    y = breast_box.cy
    return x, y, False


def evaluate_pose(
    detections: List[List[float]],
    img_w: int,
    img_h: int,
) -> Dict[str, Any]:
    """
    Evaluate whether the patient is in a good ~45° pose based on YOLO detections.

    detections: list of [x1, y1, x2, y2, conf, class_id] in pixel coords.
    Returns dict with:
        ok: bool
        score: float (0–1)
        reasons: list of human-readable strings
        used_boxes: {name: {x1, y1, x2, y2, conf, ...}}
    """
    reasons: List[str] = []
    used_boxes: Dict[str, Dict[str, float]] = {}

    boxes = _select_best_boxes(detections, img_w, img_h)

    rb = boxes.get("right_breast")
    lb = boxes.get("left_breast")
    ap = boxes.get("armpit")
    rn = boxes.get("right_nip")
    ln = boxes.get("left_nip")

    # score bookkeeping
    score = 0.0
    max_score = 0.0

    # Helper to add a scoring component
    def add_component(value: float, weight: float, desc: str):
        nonlocal score, max_score, reasons
        max_score += weight
        score += max(0.0, min(value, 1.0)) * weight
        reasons.append(f"{desc}: {value:.2f} (weight {weight})")

    # =======================
    # Hard checks
    # =======================

    hard_fail = False

    if rb is None:
        reasons.append("HARD FAIL: right_breast not detected with sufficient confidence.")
        hard_fail = True
    else:
        used_boxes["right_breast"] = rb.__dict__

    if ap is None:
        reasons.append("HARD FAIL: armpit not detected with sufficient confidence.")
        hard_fail = True
    else:
        used_boxes["armpit"] = ap.__dict__

    # If we failed both essential landmarks, no point in going further
    if hard_fail:
        return {
            "ok": False,
            "score": 0.0,
            "reasons": reasons,
            "used_boxes": used_boxes,
        }

    # =======================
    # Right breast size & position (hard + soft)
    # =======================

    # 1) right breast area fraction
    if rb.area_frac < RB_AREA_MIN or rb.area_frac > RB_AREA_MAX:
        reasons.append(
            f"HARD FAIL: right_breast area_frac={rb.area_frac:.3f} "
            f"outside [{RB_AREA_MIN}, {RB_AREA_MAX}]."
        )
        return {
            "ok": False,
            "score": 0.0,
            "reasons": reasons,
            "used_boxes": used_boxes,
        }

    # 2) center position of RB (x, y)
    if not (RB_X_MIN <= rb.cx_n <= RB_X_MAX and RB_Y_MIN <= rb.cy_n <= RB_Y_MAX):
        reasons.append(
            "HARD FAIL: right_breast center not in expected central region "
            f"(cx_n={rb.cx_n:.3f}, cy_n={rb.cy_n:.3f})."
        )
        return {
            "ok": False,
            "score": 0.0,
            "reasons": reasons,
            "used_boxes": used_boxes,
        }

    # 3) aspect ratio sanity (soft)
    max_score += 1.0
    if MIN_ASPECT <= rb.aspect <= MAX_ASPECT:
        score += 1.0
        reasons.append(f"OK: right_breast aspect ratio {rb.aspect:.2f} within expected range.")
    else:
        reasons.append(
            f"SOFT: right_breast aspect ratio {rb.aspect:.2f} outside "
            f"[{MIN_ASPECT}, {MAX_ASPECT}]."
        )

    # =======================
    # Armpit relation (hard + soft)
    # =======================

    # hard: armpit clearly above right breast
    if (ap.cy_n + ARMPIT_ABOVE_MARGIN) >= rb.cy_n:
        reasons.append(
            "HARD FAIL: armpit not sufficiently above right_breast "
            f"(ap.cy_n={ap.cy_n:.3f}, rb.cy_n={rb.cy_n:.3f})."
        )
        return {
            "ok": False,
            "score": 0.0,
            "reasons": reasons,
            "used_boxes": used_boxes,
        }

    # soft: armpit horizontal position – should be same or more lateral (to the right)
    max_score += 1.0
    if ap.cx_n >= rb.cx_n:
        score += 1.0
        reasons.append(
            "OK: armpit horizontally at or to the right of right_breast (consistent with 45°)."
        )
    else:
        reasons.append(
            "SOFT: armpit horizontally left of right_breast center; may indicate off-angle pose."
        )

    # =======================
    # Border / cropping checks (soft)
    # =======================

    def border_score(box: Box, name: str) -> float:
        s = 1.0
        # top
        if box.y1_n < TOP_MARGIN:
            s -= 0.25
        # bottom
        if (1.0 - box.y2_n) < BOTTOM_MARGIN:
            s -= 0.25
        # left
        if box.x1_n < LEFT_MARGIN:
            s -= 0.25
        # right
        if (1.0 - box.x2_n) < RIGHT_MARGIN:
            s -= 0.25
        s = max(0.0, s)
        return s

    rb_border_s = border_score(rb, "right_breast")
    add_component(rb_border_s, weight=1.0, desc="right_breast border margin score")

    ap_border_s = border_score(ap, "armpit")
    add_component(ap_border_s, weight=0.5, desc="armpit border margin score")

    # =======================
    # Left breast soft checks
    # =======================

    if lb is not None:
        used_boxes["left_breast"] = lb.__dict__

        # left breast should be left of right breast
        max_score += 0.5
        if lb.cx_n < rb.cx_n:
            score += 0.5
            reasons.append("OK: left_breast is left of right_breast (expected).")
        else:
            reasons.append("SOFT: left_breast is not left of right_breast; suspect misalignment.")

        # smaller or equal area
        max_score += 0.5
        if lb.area_frac <= rb.area_frac:
            score += 0.5
            reasons.append("OK: left_breast area <= right_breast area (expected oblique view).")
        else:
            reasons.append("SOFT: left_breast appears larger than right_breast.")

    # =======================
    # Nipples / virtual nipples geometry (soft)
    # =======================

    # Right side
    rn_x, rn_y, rn_is_real = _virtual_nipple(rb, rn)
    if rn is not None:
        used_boxes["right_nip"] = rn.__dict__

    # If we have a left breast, we can define left nipple (real or virtual)
    if lb is not None:
        ln_x, ln_y, ln_is_real = _virtual_nipple(lb, ln)
        if ln is not None:
            used_boxes["left_nip"] = ln.__dict__
    else:
        ln_x = ln_y = None
        ln_is_real = False

    # (A) nipple inside own breast (if real)
    if rn is not None:
        max_score += 0.5
        v = 1.0 if rb.contains_point(rn.cx, rn.cy) else 0.0
        score += v * 0.5
        if v == 1.0:
            reasons.append("OK: right_nip lies inside right_breast box.")
        else:
            reasons.append("SOFT: right_nip outside right_breast box.")

    if ln is not None and lb is not None:
        max_score += 0.5
        v = 1.0 if lb.contains_point(ln.cx, ln.cy) else 0.0
        score += v * 0.5
        if v == 1.0:
            reasons.append("OK: left_nip lies inside left_breast box.")
        else:
            reasons.append("SOFT: left_nip outside left_breast box.")

    # (B) nip-to-nip line geometry (only if we have both breasts)
    if lb is not None:
        dx = (rn_x - ln_x)
        dy = (rn_y - ln_y)

        if dx != 0:
            angle = math.degrees(math.atan2(dy, dx))  # mainly for debugging / logging
        else:
            angle = 90.0 if dy > 0 else -90.0

        # We mostly care that |dy| is not huge relative to |dx|
        dist = math.hypot(dx, dy) + 1e-6
        horiz_frac = abs(dx) / dist  # 1.0 = purely horizontal, 0.0 = purely vertical

        # Expect nip line to be more horizontal than vertical → horiz_frac near 1
        horiz_score = max(0.0, min(1.0, (horiz_frac - 0.5) / 0.5))  # map [0.5,1] -> [0,1]
        add_component(horiz_score, weight=1.0,
                      desc=f"nipple-line horizontality (angle={angle:.1f}°)")

    # =======================
    # Normalize score and final decision
    # =======================

    if max_score == 0:
        final_score = 0.0
    else:
        final_score = score / max_score

    # Decision thresholds – you can tune these
    ok = final_score >= 0.8

    reasons.append(f"FINAL SCORE: {final_score:.3f} (ok={ok})")

    return {
        "ok": ok,
        "score": float(final_score),
        "reasons": reasons,
        "used_boxes": used_boxes,
    }


