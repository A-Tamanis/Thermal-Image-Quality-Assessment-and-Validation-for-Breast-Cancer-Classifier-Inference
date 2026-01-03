import cv2
import numpy as np


# Colors in BGR
BOX_COLORS = {
    "right_breast": (255, 0, 0),    # blue
    "left_breast":  (255, 255, 0),  # cyan
    "armpit":       (0, 0, 255),    # red
    "right_nip":    (0, 255, 0),    # green
    "left_nip":     (255, 0, 255),  # magenta
}

def _center_from_box(b):
    x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))


def draw_pose_overlay(
    img_bgr: np.ndarray,
    pose_info: dict,
    show_reasons: bool = False,
    max_reasons: int = 4,
) -> np.ndarray:
    """
    Draw bounding boxes, (real or virtual) nipples, and pose score on a copy of img_bgr.

    img_bgr   : original image (OpenCV BGR)
    pose_info : output from evaluate_pose(...)
    returns   : annotated image (BGR)
    """
    out = img_bgr.copy()
    h, w = out.shape[:2]

    used = pose_info.get("used_boxes", {})
    ok   = pose_info.get("ok", False)
    score = pose_info.get("score", 0.0)
    reasons = pose_info.get("reasons", [])

    # 1) Draw boxes
    for name, box_dict in used.items():
        x1, y1 = int(box_dict["x1"]), int(box_dict["y1"])
        x2, y2 = int(box_dict["x2"]), int(box_dict["y2"])
        conf   = box_dict["conf"]

        color = BOX_COLORS.get(name, (200, 200, 200))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        label = f"{name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
        cv2.putText(out, label, (x1 + 1, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # 2) Nipples (real + virtual) – right side
    rb = used.get("right_breast")
    rn = used.get("right_nip")
    lb = used.get("left_breast")
    ln = used.get("left_nip")

    # Right virtual/real nipple
    if rb is not None:
        rb_cx, rb_cy = _center_from_box(rb)
        rb_w = int(rb["x2"] - rb["x1"])
        if rn is not None:
            # real
            rn_cx, rn_cy = _center_from_box(rn)
            cv2.circle(out, (rn_cx, rn_cy), 5, BOX_COLORS.get("right_nip", (0, 255, 0)), -1)
        else:
            # virtual
            vn_cx = int(rb_cx - 0.25 * rb_w)
            vn_cy = int(rb_cy)
            cv2.circle(out, (vn_cx, vn_cy), 5, (255, 255, 255), -1)  # white
            cv2.putText(out, "virt RN", (vn_cx + 4, vn_cy - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    # Left virtual/real nipple (only if left breast exists)
    if lb is not None:
        lb_cx, lb_cy = _center_from_box(lb)
        lb_w = int(lb["x2"] - lb["x1"])
        if ln is not None:
            ln_cx, ln_cy = _center_from_box(ln)
            cv2.circle(out, (ln_cx, ln_cy), 5, BOX_COLORS.get("left_nip", (255, 0, 255)), -1)
        else:
            vn_cx = int(lb_cx - 0.25 * lb_w)
            vn_cy = int(lb_cy)
            cv2.circle(out, (vn_cx, vn_cy), 5, (200, 200, 200), -1)  # grey
            cv2.putText(out, "virt LN", (vn_cx + 4, vn_cy - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

    # 3) Pose summary banner at top
    banner_color = (0, 180, 0) if ok else (0, 0, 200)
    text = f"Pose OK: {ok}  |  score={score:.3f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(out, (0, 0), (max(w, tw) + 10, th + 10), banner_color, -1)
    cv2.putText(out, text, (5, th + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # 4) Optional: draw a few reasons at bottom-left
    if show_reasons and len(reasons) > 0:
        y0 = h - 10 - 15 * min(max_reasons, len(reasons))
        for i, r in enumerate(reasons[:max_reasons]):
            y = y0 + i * 15
            cv2.putText(out, r[:80], (5, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    return out
