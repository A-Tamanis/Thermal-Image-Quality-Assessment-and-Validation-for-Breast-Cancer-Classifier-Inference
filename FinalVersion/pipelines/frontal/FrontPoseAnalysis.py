"""
FrontPoseAnalysis.py

Logic for analyzing frontal stance using YOLO detections of:
- breast
- armpit
- nipple

Main entry:
    valid, verdict, metrics = analyze_pose(detections, img_w, img_h)

Expected detections format:
    detections: list of dicts, each:
        {
            "cls": int or str,            # 0/1/2 or "breast"/"armpit"/"nipple"
            "conf": float,                # confidence score
            "xyxy": (x1, y1, x2, y2),     # pixel coordinates
        }

"""

import math
from typing import List, Dict, Tuple, Any, Optional


# THRESHOLDS. Below are the threshold the metrics need to be within

CLASS_ID_TO_NAME = {
    0: "breast",
    1: "armpit",
    2: "nipple",
}

MIN_CONF_BREAST = 0.8
MIN_CONF_ARMPIT = 0.4
MIN_CONF_NIPPLE = 0.5

# torso width relative to image width (between breasts)
MIN_BREAST_WIDTH_RATIO = 0.18
MAX_BREAST_WIDTH_RATIO = 0.80

# armpit width vs breast width
MIN_BREAST_TO_ARMPIT_WIDTH_RATIO = 0.65
MAX_BREAST_TO_ARMPIT_WIDTH_RATIO = 2.0

# tilt thresholds in degrees
MAX_NIPPLE_ANGLE_DEG = 12.0
MAX_ARMPIT_ANGLE_DEG = 15.0    # armpit detections are not as percise as nipples, so we give some more degrees of freedomg before failing the image 

# cropping thresholds (normalized distance to border)
MIN_BORDER_DIST = 0.08  # fail detections that are less than 8% of the image width or height of distance from the border

# symmetry height thresholds
MAX_NIPPLE_Y_DIFF = 0.15    
MAX_ARMPIT_Y_DIFF = 0.10
MAX_NIPPLE_ARMPIT_MID_DIFF = 0.2

# torso centering
MAX_TORSO_CENTER_OFFSET = 0.20


# virtual nipple placement within breast box (relative to breast height)
VIRTUAL_NIPPLE_VERTICAL_OFFSET = 0.10  # 0.1 * breast_height below breast center


# helpers

def _cls_to_name(cls_value: Any) -> str:
    # Map cls (int or str) to canonical name: 'breast', 'armpit', 'nipple'
    if isinstance(cls_value, str):
        name = cls_value.lower()
        if name not in ("breast", "armpit", "nipple"):
            raise ValueError(f"Unknown class name: {cls_value}")
        return name
    
    if cls_value not in CLASS_ID_TO_NAME:
        raise ValueError(f"Unknown class id: {cls_value}")
    return CLASS_ID_TO_NAME[cls_value]


def _normalize_box(xyxy, img_w: int, img_h: int) -> Dict[str, float]:
    # Convert (x1, y1, x2, y2) in pixels to normalized dict with center & size
    x1, y1, x2, y2 = xyxy
    w = max(x2 - x1, 1e-6)
    h = max(y2 - y1, 1e-6)
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    return {
        "x1n": x1 / img_w,
        "y1n": y1 / img_h,
        "x2n": x2 / img_w,
        "y2n": y2 / img_h,
        "cxn": cx / img_w,
        "cyn": cy / img_h,
        "wn": w / img_w,
        "hn": h / img_h,
    }


def _point_inside_box(cx: float, cy: float, box: Dict[str, float]) -> bool:
    return (box["x1n"] <= cx <= box["x2n"]) and (box["y1n"] <= cy <= box["y2n"])

# Angle in degrees of the line from left->right, absolute value
def _angle_deg(p_left: Tuple[float, float], p_right: Tuple[float, float]) -> float:
    
    (xL, yL), (xR, yR) = p_left, p_right
    dx = xR - xL
    dy = yR - yL
    if abs(dx) < 1e-6:
        return 90.0
    theta = math.atan2(dy, dx)
    return abs(theta * 180.0 / math.pi)

# Minimum normalized distance of any point to any image border
def _min_border_dist(points: List[Tuple[float, float]]) -> float:
    
    if not points:
        return 1.0
    dists = []
    for x, y in points:
        dists.append(x)          # left
        dists.append(1.0 - x)    # right
        dists.append(y)          # top
        dists.append(1.0 - y)    # bottom
    return min(dists)




# Core selection steps

def _select_breasts_and_armpits(
    detections: List[Dict[str, Any]],
    img_w: int,
    img_h: int,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], List[str]]:
    """
    Select one left/right breast and one left/right armpit, using midline.
    Returns:
        LB, RB, LA, RA, fail_reasons
        each of LB/RB/LA/RA is dict with box info or None if missing.
    """
    breasts = []
    armpits = []

    for det in detections:
        cls_name = _cls_to_name(det["cls"])
        conf = float(det["conf"])
        box_norm = _normalize_box(det["xyxy"], img_w, img_h)
        if cls_name == "breast" and conf >= MIN_CONF_BREAST:
            breasts.append({"conf": conf, "box": box_norm})
        elif cls_name == "armpit" and conf >= MIN_CONF_ARMPIT:
            armpits.append({"conf": conf, "box": box_norm})

    fail_reasons = []

    if len(breasts) < 2:
        fail_reasons.append("not_enough_breasts_detected")
    if len(armpits) < 2:
        fail_reasons.append("not_enough_armpits_detected")

    if len(breasts) == 0:
        return None, None, None, None, fail_reasons

    # Use breasts to estimate midline.
    if len(breasts) >= 2:
        # pick top 2 by confidence as initial torso breasts
        breasts_sorted = sorted(breasts, key=lambda d: d["conf"], reverse=True)[:2]
        xs = [b["box"]["cxn"] for b in breasts_sorted]
        x_mid_body = sum(xs) / len(xs)
    else:
        x_mid_body = 0.5

    # Helper to pick best left/right by side
    def pick_lr(candidates: List[Dict[str, Any]]) -> Tuple[Optional[Dict], Optional[Dict]]:
        left_cands = [c for c in candidates if c["box"]["cxn"] < x_mid_body]
        right_cands = [c for c in candidates if c["box"]["cxn"] >= x_mid_body]

        left = max(left_cands, key=lambda c: c["conf"]) if left_cands else None
        right = max(right_cands, key=lambda c: c["conf"]) if right_cands else None
        return left, right

    LB, RB = pick_lr(breasts)
    LA, RA = pick_lr(armpits)

    if LB is None:
        fail_reasons.append("missing_left_breast")
    if RB is None:
        fail_reasons.append("missing_right_breast")
    if LA is None:
        fail_reasons.append("missing_left_armpit")
    if RA is None:
        fail_reasons.append("missing_right_armpit")

    return LB, RB, LA, RA, fail_reasons


def _build_nipples(
    detections: List[Dict[str, Any]],
    img_w: int,
    img_h: int,
    LB: Dict[str, Any],
    RB: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], int]:
    """
    For each side, pick real nipple inside breast if available, otherwise create virtual.

    Logic:
      - If both sides have a real nipple inside their breast boxes -> use both real.
      - If exactly one side has a real nipple:
          * That side uses the real nipple.
          * The other side gets a virtual nipple whose position is:
              - horizontally mirrored relative to its own breast box center
              - vertically in the same relative position inside its own breast box
            using the normalized offset of the real nipple in its breast.
      - If no real nipples for either side -> geometric virtuals for both
        (center x, slightly below center y).

    Returns:
        LN, RN, real_nipples_count
        LN/RN each: {
            "cxn", "cyn",
            "is_virtual": bool,
            "source_conf": float or 0.0
        }
    """

    # choose detected nipples
    nipples = []
    for det in detections:
        cls_name = _cls_to_name(det["cls"])
        if cls_name != "nipple":
            continue
        conf = float(det["conf"])
        if conf < MIN_CONF_NIPPLE:
            continue
        box = _normalize_box(det["xyxy"], img_w, img_h)
        nipples.append({"conf": conf, "box": box})

    #Return best real nipple inside this breast or None
    def real_nipple_for_breast(breast_box: Dict[str, float]) -> Optional[Dict[str, Any]]:
        cands = []
        for n in nipples:
            cx = n["box"]["cxn"]
            cy = n["box"]["cyn"]
            if _point_inside_box(cx, cy, breast_box):
                cands.append(n)
        if not cands:
            return None
        return max(cands, key=lambda n: n["conf"])

    # Fallback virtual nipple: center x, slightly below center y
    def geometric_virtual_from_breast(breast_box: Dict[str, float]) -> Dict[str, Any]:
        
        cx = (breast_box["x1n"] + breast_box["x2n"]) / 2.0
        cy_center = (breast_box["y1n"] + breast_box["y2n"]) / 2.0
        h = breast_box["hn"]
        cy = cy_center + VIRTUAL_NIPPLE_VERTICAL_OFFSET * h
        return {
            "cxn": cx,
            "cyn": cy,
            "is_virtual": True,
            "source_conf": 0.0,
        }


    # Build virtual nipple in target breast, using the normalized position of a real nipple
    # inside real_breast_box, mirrored horizontally and mapped into target_breast_box.
    def mirrored_virtual_from_other(
        real_box: Dict[str, float],
        real_breast_box: Dict[str, float],
        target_breast_box: Dict[str, float],
    ) -> Dict[str, Any]:
        
        # Real nipple center (normalized)
        cx_real = real_box["cxn"]
        cy_real = real_box["cyn"]

        # Real breast geometry
        cx_b_real = (real_breast_box["x1n"] + real_breast_box["x2n"]) / 2.0
        cy_b_real = (real_breast_box["y1n"] + real_breast_box["y2n"]) / 2.0
        w_real = real_breast_box["wn"]
        h_real = real_breast_box["hn"]

        # Target breast geometry
        cx_b_t = (target_breast_box["x1n"] + target_breast_box["x2n"]) / 2.0
        cy_b_t = (target_breast_box["y1n"] + target_breast_box["y2n"]) / 2.0
        w_t = target_breast_box["wn"]
        h_t = target_breast_box["hn"]

        # Normalized offsets within the real breast box
        if w_real <= 1e-6 or h_real <= 1e-6:
            # Degenerate box, fallback to geometric
            return geometric_virtual_from_breast(target_breast_box)

        dx_norm = (cx_real - cx_b_real) / w_real  # horizontal offset
        dy_norm = (cy_real - cy_b_real) / h_real  # vertical offset

        # Mirror horizontally for the other side, keep vertical pattern
        cx_virtual = cx_b_t - dx_norm * w_t
        cy_virtual = cy_b_t + dy_norm * h_t

        # Clamp to inside the target breast box
        cx_virtual = max(target_breast_box["x1n"], min(target_breast_box["x2n"], cx_virtual))
        cy_virtual = max(target_breast_box["y1n"], min(target_breast_box["y2n"], cy_virtual))

        return {
            "cxn": cx_virtual,
            "cyn": cy_virtual,
            "is_virtual": True,
            "source_conf": 0.0,
        }

    # Try to find real nipples per side
    real_L = real_nipple_for_breast(LB["box"])
    real_R = real_nipple_for_breast(RB["box"])

    # Case 1: both real, use directly
    if real_L is not None and real_R is not None:
        LN = {
            "cxn": real_L["box"]["cxn"],
            "cyn": real_L["box"]["cyn"],
            "is_virtual": False,
            "source_conf": real_L["conf"],
        }
        RN = {
            "cxn": real_R["box"]["cxn"],
            "cyn": real_R["box"]["cyn"],
            "is_virtual": False,
            "source_conf": real_R["conf"],
        }
        real_count = 2
        return LN, RN, real_count

    # Case 2: only left real, right is mirrored virtual from left
    if real_L is not None and real_R is None:
        LN = {
            "cxn": real_L["box"]["cxn"],
            "cyn": real_L["box"]["cyn"],
            "is_virtual": False,
            "source_conf": real_L["conf"],
        }
        RN = mirrored_virtual_from_other(
            real_box=real_L["box"],
            real_breast_box=LB["box"],
            target_breast_box=RB["box"],
        )
        real_count = 1
        return LN, RN, real_count

    # Case 3: only right real, left is mirrored virtual from right
    if real_L is None and real_R is not None:
        RN = {
            "cxn": real_R["box"]["cxn"],
            "cyn": real_R["box"]["cyn"],
            "is_virtual": False,
            "source_conf": real_R["conf"],
        }
        LN = mirrored_virtual_from_other(
            real_box=real_R["box"],
            real_breast_box=RB["box"],
            target_breast_box=LB["box"],
        )
        real_count = 1
        return LN, RN, real_count

    # Case 4: no real nipples, geometric virtuals both sides
    LN = geometric_virtual_from_breast(LB["box"])
    RN = geometric_virtual_from_breast(RB["box"])
    real_count = 0
    return LN, RN, real_count



# Metrics + decision
def analyze_pose(
    detections: List[Dict[str, Any]],
    img_w: int,
    img_h: int,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Main entry point

    Args:
        detections: list of {"cls", "conf", "xyxy"} for a single image.
        img_w, img_h: image dimensions in pixels.

    Returns:
        valid: bool
        verdict: short string ("valid" or first failing reason)
        metrics: dict of all computed metrics + auxiliary info
    """
    metrics: Dict[str, Any] = {}
    fail_reasons: List[str] = []

    # Select breasts and armpits
    LB, RB, LA, RA, selection_failures = _select_breasts_and_armpits(detections, img_w, img_h)
    fail_reasons.extend(selection_failures)

    has_left_breast = LB is not None
    has_right_breast = RB is not None
    has_left_armpit = LA is not None
    has_right_armpit = RA is not None

    metrics["has_left_breast"] = has_left_breast
    metrics["has_right_breast"] = has_right_breast
    metrics["has_left_armpit"] = has_left_armpit
    metrics["has_right_armpit"] = has_right_armpit

    # Hard fail if any breast or armpit is missing
    if not (has_left_breast and has_right_breast and has_left_armpit and has_right_armpit):
        
        if not has_left_breast:
            fail_reasons.append("hard_fail_missing_left_breast")
        if not has_right_breast:
            fail_reasons.append("hard_fail_missing_right_breast")
        if not has_left_armpit:
            fail_reasons.append("hard_fail_missing_left_armpit")
        if not has_right_armpit:
            fail_reasons.append("hard_fail_missing_right_armpit")

        metrics["fail_reasons"] = fail_reasons
        return False, (fail_reasons[0] if fail_reasons else "invalid_missing_detections"), metrics

    # Extract centers
    LBc = (LB["box"]["cxn"], LB["box"]["cyn"])
    RBc = (RB["box"]["cxn"], RB["box"]["cyn"])
    LAc = (LA["box"]["cxn"], LA["box"]["cyn"])
    RAc = (RA["box"]["cxn"], RA["box"]["cyn"])

    metrics["LB_cx"], metrics["LB_cy"] = LBc
    metrics["RB_cx"], metrics["RB_cy"] = RBc
    metrics["LA_cx"], metrics["LA_cy"] = LAc
    metrics["RA_cx"], metrics["RA_cy"] = RAc

    # Build nipples (real or virtual)
    LN, RN, real_nipples_count = _build_nipples(detections, img_w, img_h, LB, RB)
    LNc = (LN["cxn"], LN["cyn"])
    RNc = (RN["cxn"], RN["cyn"])

    metrics["LN_cx"], metrics["LN_cy"] = LNc
    metrics["RN_cx"], metrics["RN_cy"] = RNc
    metrics["left_nipple_is_virtual"] = LN["is_virtual"]
    metrics["right_nipple_is_virtual"] = RN["is_virtual"]
    metrics["real_nipples_count"] = real_nipples_count
    metrics["has_any_real_nipple"] = real_nipples_count > 0

    # Width metrics
    breast_width_ratio = abs(RBc[0] - LBc[0])
    armpit_width_ratio = abs(RAc[0] - LAc[0])
    breast_to_armpit_width_ratio = (
        armpit_width_ratio / breast_width_ratio if breast_width_ratio > 1e-6 else 0.0
    )

    metrics["breast_width_ratio"] = breast_width_ratio
    metrics["armpit_width_ratio"] = armpit_width_ratio
    metrics["breast_to_armpit_width_ratio"] = breast_to_armpit_width_ratio

    # Symmetry / centering
    nipple_y_diff_norm = abs(RNc[1] - LNc[1])
    armpit_y_diff_norm = abs(RAc[1] - LAc[1])
    torso_center_x = 0.5 * (LBc[0] + RBc[0])
    torso_center_offset = abs(torso_center_x - 0.5)
    nipple_center = 0.5 * (LNc[0] + RNc[0])
    armpit_center = 0.5 * (LAc[0] + RAc[0])
    nipple_armpit_diff = abs(nipple_center - armpit_center)

    metrics["nipple_y_diff_norm"] = nipple_y_diff_norm
    metrics["armpit_y_diff_norm"] = armpit_y_diff_norm
    metrics["torso_center_x"] = torso_center_x
    metrics["torso_center_offset"] = torso_center_offset
    metrics["nipple_armpit_diff"] = nipple_armpit_diff

    # Tilt
    nipple_angle_deg = _angle_deg(LNc, RNc)
    armpit_angle_deg = _angle_deg(LAc, RAc)
    metrics["nipple_angle_deg"] = nipple_angle_deg
    metrics["armpit_angle_deg"] = armpit_angle_deg
    metrics["nip_vs_armpit_angle_diff_deg"] = abs(nipple_angle_deg - armpit_angle_deg)

    # Vertical ordering
    mean_y_armpit = 0.5 * (LAc[1] + RAc[1])
    mean_y_nipple = 0.5 * (LNc[1] + RNc[1])
    mean_y_breast = 0.5 * (LBc[1] + RBc[1])

    metrics["mean_y_armpit"] = mean_y_armpit
    metrics["mean_y_nipple"] = mean_y_nipple
    metrics["mean_y_breast"] = mean_y_breast
    metrics["armpits_above_nipples"] = mean_y_armpit < mean_y_nipple

    # Border / cropping
    breast_points = [LBc, RBc]
    armpit_points = [LAc, RAc]
    nipple_points = [LNc, RNc]

    min_border_dist_breasts = _min_border_dist(breast_points)
    min_border_dist_armpits = _min_border_dist(armpit_points)
    min_border_dist_nipples = _min_border_dist(nipple_points)

    metrics["min_border_dist_breasts"] = min_border_dist_breasts
    metrics["min_border_dist_armpits"] = min_border_dist_armpits
    metrics["min_border_dist_nipples"] = min_border_dist_nipples



    # Hard rule checks (fail_reasons)

    # torso width plausible
    if breast_width_ratio < MIN_BREAST_WIDTH_RATIO:
        fail_reasons.append("breast_width_too_small")
    if breast_width_ratio > MAX_BREAST_WIDTH_RATIO:
        fail_reasons.append("breast_width_too_large")

    if not (MIN_BREAST_TO_ARMPIT_WIDTH_RATIO <= breast_to_armpit_width_ratio <= MAX_BREAST_TO_ARMPIT_WIDTH_RATIO):
        fail_reasons.append("breast_to_armpit_width_ratio_out_of_range")

    # tilt too large
    if nipple_angle_deg > MAX_NIPPLE_ANGLE_DEG:
        fail_reasons.append("nipple_tilt_too_large")
    if armpit_angle_deg > MAX_ARMPIT_ANGLE_DEG:
        fail_reasons.append("armpit_tilt_too_large")

    # vertical ordering
    if not metrics["armpits_above_nipples"]:
        fail_reasons.append("armpits_not_above_nipples")

    # cropping
    if min_border_dist_breasts < MIN_BORDER_DIST:
        fail_reasons.append("breasts_too_close_to_border")
    if min_border_dist_armpits < MIN_BORDER_DIST:
        fail_reasons.append("armpits_too_close_to_border")

    # torso centering
    if torso_center_offset > MAX_TORSO_CENTER_OFFSET:
        fail_reasons.append("torso_too_off_center")

    # symmetry
    if nipple_y_diff_norm > MAX_NIPPLE_Y_DIFF:
        fail_reasons.append("nipple_height_asymmetry_too_large")
    if armpit_y_diff_norm > MAX_ARMPIT_Y_DIFF:
        fail_reasons.append("armpit_height_asymmetry_too_large")
    
    if nipple_armpit_diff > MAX_NIPPLE_ARMPIT_MID_DIFF:
        fail_reasons.append("nipple_to_armpit_mid_diff_too_large")


    # Final decision
    valid = len(fail_reasons) == 0
    verdict = "valid" if valid else fail_reasons[0]

    metrics["fail_reasons"] = fail_reasons

    return valid, verdict, metrics
