import cv2
import numpy as np
from typing import Dict, Any, List



# Make a breast focused mask from the body mask

def make_breast_mask_from_body(
    body_mask,
    top_crop_frac=0.30,    # remove 20% of torso height from the top
    bottom_crop_frac=0.10,  # remove 20% of torso height from the bottom
    erode_size=40
):
    """
    Create a breast/body band mask by:
    - Taking the body bounding box (from body_mask)
    - Cropping some percentage from top and bottom
    - Using full torso width
    All operations are done in mask space.

    Returns:
        breast_mask : 0/1 mask (same shape as body_mask)
        bbox        : (y_min, y_max, x_min, x_max) of the band (for debug/visualization)
    """
    cv2.imshow("body mask", (body_mask*255).astype(np.uint8))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_size, erode_size))
    body_mask = cv2.erode(body_mask, kernel, iterations=1)

    cv2.imshow("body mask eroded", (body_mask*255).astype(np.uint8))

    ys, xs = np.where(body_mask > 0)
    if ys.size == 0:
        return np.zeros_like(body_mask, dtype=np.uint8), None

    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    H = y_max - y_min + 1
    W = x_max - x_min + 1

    # Crop some percentage of image height from top and bottom of torso in mask space
    y1 = int(round(y_min + top_crop_frac * H))
    y2 = int(round(y_max - bottom_crop_frac * H))

    # Use full torso width
    x1 = x_min
    x2 = x_max

    H_img, W_img = body_mask.shape
    y1 = max(0, min(H_img - 1, y1))
    y2 = max(0, min(H_img - 1, y2))
    x1 = max(0, min(W_img - 1, x1))
    x2 = max(0, min(W_img - 1, x2))

    breast_mask = np.zeros_like(body_mask, dtype=np.uint8)
    if y2 <= y1 or x2 <= x1:
        return breast_mask, None

    # copy body in that vertical band to breast/body region
    band = body_mask[y1:y2+1, x1:x2+1]
    breast_mask[y1:y2+1, x1:x2+1] = band
    
    cv2.imshow("breast mask", (breast_mask*255).astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return breast_mask, (y1, y2, x1, x2)




# Quality checks on the body/breast mask

def evaluate_breast_quality_on_mask(
    img,
    body_mask,
    # vertical crop of the body to define breast band
    top_crop_frac=0.20,
    bottom_crop_frac=0.20,
    # thresholds
    cutoff_border_frac=0.05,    # frac of body pixels touching top/bottom of the band
    min_mean_intensity = 150.0, # minimal intensity mean
    max_mean_intensity = 195.0, # max intensity mean
    cold_k_sigma=0.8,           # cold pixels = mu - k*sigma
    max_cold_frac=0.15,         # max allowed cold-blob fraction of breast band area
    min_var_intensity=200.0,    # minimal variance of intensities
    max_sat_frac=0.30,          # max fraction of pixels at extremes
    low_sat_val=5,              # this is considered "low extreme"
    high_sat_val=250            # this is considered "high extreme"
):
    """
    All metrics are computed on the body/breast mask
    No background pixels are used

    Args:
        img       : grayscale image (H x W)
        body_mask : 0/1 (H x W), 1 = body (from your background pipeline)
    Returns:
        quality_ok : bool
        metrics    : dict
    """
    metrics = {}

    # derive breast band mask from body mask
    breast_mask, band_bbox = make_breast_mask_from_body(
        body_mask,
        top_crop_frac=top_crop_frac,
        bottom_crop_frac=bottom_crop_frac
    )
    #metrics["band_bbox"] = band_bbox

    if band_bbox is None or breast_mask.sum() == 0:
        metrics["reason"] = "no_breast_band"
        metrics["quality_ok"] = False
        return False, metrics

    y1, y2, x1, x2 = band_bbox
    band_mask = breast_mask[y1:y2+1, x1:x2+1]
    band_img = img[y1:y2+1, x1:x2+1]

    H_band, W_band = band_mask.shape
    total_band_pixels = H_band * W_band

    # coverage of band by body
    body_pixels_band = int(band_mask.sum())
    coverage = body_pixels_band / float(total_band_pixels)
    metrics["coverage"] = coverage

    # cutoff on top/bottom in band mask
    if body_pixels_band > 0:
        top_touch = band_mask[0, :].sum()
        bottom_touch = band_mask[-1, :].sum()
        frac_top = top_touch / float(body_pixels_band)
        frac_bottom = bottom_touch / float(body_pixels_band)
    else:
        frac_top = frac_bottom = 0.0

    cutoff_top_bottom = (frac_top > cutoff_border_frac) or (frac_bottom > cutoff_border_frac)
    metrics["frac_top_border"] = frac_top
    metrics["frac_bottom_border"] = frac_bottom
    metrics["cutoff_top_bottom"] = cutoff_top_bottom

    # thermal stats on body pixels in the band
    vals = band_img[band_mask == 1].astype(np.float32)
    mean_intensity = float(vals.mean())
    var_intensity = float(vals.var())
    min_intensity = float(vals.min())
    max_intensity = float(vals.max())
    dyn_range = max_intensity - min_intensity

    metrics["mean_intensity"] = mean_intensity
    metrics["var_intensity"] = var_intensity
    metrics["min_intensity"] = min_intensity
    metrics["max_intensity"] = max_intensity
    metrics["dynamic_range"] = dyn_range

    # saturation only on body pixels in band
    pct_low = float((vals <= low_sat_val).sum()) / vals.size
    pct_high = float((vals >= high_sat_val).sum()) / vals.size
    metrics["pct_low"] = pct_low
    metrics["pct_high"] = pct_high

    # occlusion via cold blobs
    sigma = float(vals.std())
    if sigma == 0:
        cold_frac = 0.0
    else:
        cold_threshold = mean_intensity - cold_k_sigma * sigma
        cold_mask = (band_mask == 1) & (band_img.astype(np.float32) < cold_threshold)
        cold_mask_u8 = cold_mask.astype(np.uint8) * 255

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            cold_mask_u8, connectivity=8
        )
        max_frac = 0.0
        for label in range(1, num_labels):  # skip background
            area = stats[label, cv2.CC_STAT_AREA]
            frac = area / float(body_pixels_band)
            if frac > max_frac:
                max_frac = frac
        cold_frac = max_frac

    metrics["cold_frac"] = cold_frac
    occlusion = cold_frac > max_cold_frac
    metrics["occlusion"] = occlusion

    # final rules
    rules = {
        "no_cutoff": not cutoff_top_bottom,
        "no_occlusion": not occlusion,
        "min_mean_ok": mean_intensity >= min_mean_intensity,
        "max_mean_ok": mean_intensity <= max_mean_intensity,
        "var_ok": var_intensity >= min_var_intensity,
        "sat_low_ok": pct_low <= max_sat_frac,
        "sat_high_ok": pct_high <= max_sat_frac
    }
    metrics["rules"] = rules

    quality_ok = all(rules.values())
    metrics["quality_ok"] = quality_ok

    if not quality_ok and "reason" not in metrics:
        metrics["reason"] = "rules_failed"

    return quality_ok, metrics



# Background quality function (also used to get body_mask)

def check_background_quality(
    image_path,
    bandage_close_size=15,   # used to remove the bandage in dataset images
    noise_open_size=5,       # removes small hot noise in background for a reliable body mask
    body_margin_size=40,     # margin around body excluded from background
    threshold=20.0           # Laplacian variance threshold
):
    """
    Returns:
        lap_var   : Laplacian variance on background
        is_clean  : True if background is acceptable
        bg_mask   : uint8 mask (1=background, 0=body+margin)
        body_mask : uint8 mask (1=body after cleaning)
    """

    # Load grayscale image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    # Threshold to get initial binary (body - white, background - black)
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bin01 = (binary > 0).astype(np.uint8)  # 0/1

    # Morphological closing to fill small dark gaps inside body (like the square bandage)
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (bandage_close_size, bandage_close_size)
    )
    bin_closed = cv2.morphologyEx(bin01, cv2.MORPH_CLOSE, close_kernel)

    # Morphological opening to remove small bright noise in the background
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (noise_open_size, noise_open_size)
    )
    bin_clean = cv2.morphologyEx(bin_closed, cv2.MORPH_OPEN, open_kernel)

    # Largest connected component on the cleaned binary is the body mask
    bin_clean_u8 = (bin_clean * 255).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        bin_clean_u8, connectivity=8
    )

    if num_labels <= 1:
        print("[warn] No body detected.")
        H, W = img.shape
        return 0.0, False, np.ones((H, W), dtype=np.uint8), np.zeros((H, W), dtype=np.uint8)

    areas = stats[1:, cv2.CC_STAT_AREA]  # skip background row
    largest_label = 1 + np.argmax(areas)
    body_mask = (labels == largest_label).astype(np.uint8)  # binary 0/1

    # Smooth and dilate the body mask to exclude boundary and add a margin
    body_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (body_margin_size, body_margin_size)
    )
    body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_CLOSE, body_kernel)

    # dialate mask to get a margin between body and background edge
    body_mask_with_margin = cv2.dilate(body_mask, body_kernel)

    # Raw background mask: everything outside body+margin
    bg_mask = (1 - body_mask_with_margin).astype(np.uint8)  # 0/1

    # Keep only background connected to the image border
    # (this removes inner "islands" like bandages and noise)
    H, W = bg_mask.shape
    bg_u8 = (bg_mask * 255).astype(np.uint8)
    num_labels_bg, labels_bg, stats_bg, _ = cv2.connectedComponentsWithStats(
        bg_u8, connectivity=8
    )

    bg_mask_filtered = np.zeros_like(bg_mask, dtype=np.uint8)

    for label in range(1, num_labels_bg):
        x, y, w, h, area = stats_bg[label]

        touches_border = (
            x == 0 or y == 0 or (x + w) == W or (y + h) == H
        )

        if touches_border:
            bg_mask_filtered[labels_bg == label] = 1 # keep component if in touch with the border

    bg_mask = bg_mask_filtered  # final background mask

    # Laplacian on original image
    lap = cv2.Laplacian(img, cv2.CV_64F, ksize=3)
    lap = cv2.convertScaleAbs(lap)

    cv2.imshow("",lap)
    cv2.waitKey(0)

    # Take only background pixels
    bg_pixels = lap[bg_mask.astype(bool)]
    if bg_pixels.size == 0:
        print("[warn] Background mask has zero pixels.")
        lap_var = 0.0
    else:
        lap_var = bg_pixels.var()

        # ---- Background block statistics check (NEW) ----
    block_size = 64  # adjust as needed
    mean_thresh = 40.0
    var_thresh  = 90.0

    H, W = img.shape
    bg_fail = False

    for y in range(0, H, block_size):
        for x in range(0, W, block_size):
            y2 = min(y + block_size, H)
            x2 = min(x + block_size, W)

            block_mask = bg_mask[y:y2, x:x2]
            if block_mask.sum() < 10:
                continue

            block_vals = img[y:y2, x:x2][block_mask.astype(bool)]

            m = float(block_vals.mean())
            v = float(block_vals.var())

            if m > mean_thresh or v > var_thresh:
                bg_fail = True
                print(m)
                print(v)
                break
        if bg_fail:
            break

    #is_clean = lap_var < threshold
    is_clean = (lap_var < threshold) and (not bg_fail)
    return lap_var, is_clean, bg_mask, body_mask



# Pipeline function

def run_check(image_path: str) -> Dict[str, Any]:
    """
    Pipeline wrapper for 45 degree background check
    and breast/band quality metrics.

    Returns:
      {
        "ok": bool,
        "name": "metrics_45deg",
        "reasons": [...],
        "extra": {
            "background_lap_var": float,
            "background_clean": bool,
            "metrics": {...}  # all breast-band metrics
        }
      }
    """
    name = "metrics_0deg"
    reasons: List[str] = []
    extra: Dict[str, Any] = {}

    
    bandage_close_size = 5
    noise_open_size    = 5
    body_margin_size   = 15
    bg_threshold       = 25.0

    # get body mask
    try:
        lap_var_bg, is_clean_bg, bg_mask, body_mask = check_background_quality(
            image_path,
            bandage_close_size=bandage_close_size,
            noise_open_size=noise_open_size,
            body_margin_size=body_margin_size,
            threshold=bg_threshold,
        )
    except FileNotFoundError as e:
        return {
            "ok": False,
            "name": name,
            "reasons": [f"FAIL: {e}"],
            "extra": {},
        }

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {
            "ok": False,
            "name": name,
            "reasons": [f"FAIL: could not load image '{image_path}'."],
            "extra": {},
        }

    # breast and body quality on mask
    breast_ok, metrics = evaluate_breast_quality_on_mask(img, body_mask)

    extra["background_lap_var"] = float(lap_var_bg)
    extra["background_clean"]   = bool(is_clean_bg)
    extra["metrics"]            = metrics

    #ok = bool(breast_ok)
    ok = True

    if not is_clean_bg:
        reasons.append(
            # f"FAIL: background Laplacian variance {lap_var_bg:.3f} "
            # f">= threshold {bg_threshold:.3f} (noisy background)."
            f"FAIL: NOISY background"
        )
        ok = False
    else:
        reasons.append(
            # f"PASS: background Laplacian variance {lap_var_bg:.3f} "
            # f"< threshold {bg_threshold:.3f} (background clean)."
            f"PASS: background clean"
        )
        

    if not breast_ok:
        reason = metrics.get("reason", "rules_failed")
        reasons.append(f"FAIL: breast/band quality check failed ({reason}).")

        rules = metrics.get("rules", {})
        if rules:
            failed_rules = [k for k, v in rules.items() if not v]
            if failed_rules:
                reasons.append("Failed rules: " + ", ".join(failed_rules))
        
        ok = False
    else:
        reasons.append("PASS: breast/band quality metrics are within acceptable limits.")

    return {
        "ok": ok,
        "name": name,
        "reasons": reasons,
        "extra": extra,
    }



# Standalone use

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="45 metrics / breast-band quality check")
    parser.add_argument("image", help="Path to thermal image")
    args = parser.parse_args()

    result = run_check(args.image)

    print("OK:", result["ok"])
    for r in result["reasons"]:
        print("-", r)
    print("Extra keys:", list(result["extra"].keys()))
    print("Breast metrics:", result["extra"].get("metrics", {}))

