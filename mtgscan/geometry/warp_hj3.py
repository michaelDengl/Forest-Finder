# mtgscan/geometry/warp_hj3.py
"""
Alternative card warping based on hj3yoo/mtg_card_detector.

Main ideas we "steal":
  - find_card(): contour + hierarchy search to find 4-point card quads
  - four_point_transform(): PyImageSearch 4-point warp
  - optional glare reduction

Usage patterns:

1) Simple "just warp the card from this image" (recommended for card_crop jpgs):

   python -m mtgscan.geometry.warp_hj3 in.jpg out.jpg

2) Warp using a YOLO box on a full frame (less important, but supported):

   python -m mtgscan.geometry.warp_hj3 in.jpg x1 y1 x2 y2 out.jpg
"""

from __future__ import annotations
import cv2
import numpy as np
from typing import List, Optional, Tuple
import math

# How far *inside* the crop we start looking for card edges (fraction of width/height)
EDGE_INNER_MARGIN_X = 0.005   # was effectively ~0.08 before → move search closer to border
EDGE_INNER_MARGIN_Y = 0.04   # leave this as-is for now
EDGE_MAX_INNER_FRAC = 0.08   # search up to 45% of width from each side

HOUGH_MIN_AREA_RATIO = 0.70   # reject if Hough area < 90% of 1D bbox area
HOUGH_MAX_AREA_RATIO = 1.25   # reject if Hough area > 125% of 1D bbox area

HOUGH_MAX_SIDE_X_OFFSET_FRAC = 0.25  # how far Hough lines may deviate from 1D bbox (25% of size)
HOUGH_MAX_SIDE_Y_OFFSET_FRAC = 0.06


# ---------------------------------------------------------------------------
# 1) Point ordering + four-point perspective transform (from hj3yoo)
# ---------------------------------------------------------------------------

def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order 4 points as: top-left, top-right, bottom-right, bottom-left.
    """
    pts = np.asarray(pts, dtype="float32").reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    # smallest sum → top-left, largest sum → bottom-right
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # smallest diff → top-right, largest diff → bottom-left
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Transform a quadrilateral section of an image into a rectangular area.

    This is essentially the PyImageSearch 4-point perspective transform.
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # width
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    # height
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    maxWidth = max(1, maxWidth)
    maxHeight = max(1, maxHeight)

    dst = np.array(
        [
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1],
        ],
        dtype="float32",
    )

    mat = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, mat, (maxWidth, maxHeight))

    # If image is landscape, rotate to portrait (like in mtg_card_detector)
    if warped.shape[1] > warped.shape[0]:
        center = (maxHeight / 2.0, maxHeight / 2.0)
        mat_rot = cv2.getRotationMatrix2D(center, 270, 1.0)
        warped = cv2.warpAffine(warped, mat_rot, (maxHeight, maxWidth))

    return warped


# ---------------------------------------------------------------------------
# 2) Glare reduction (straight from the repo, slightly cleaned)
# ---------------------------------------------------------------------------

def remove_glare(img: np.ndarray) -> np.ndarray:
    """
    Reduce bright specular highlights (sleeve glare etc.) in a BGR image.
    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(img_hsv)

    # low saturation → possible reflection region
    non_sat = (s < 32) * 255
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    non_sat = cv2.erode(non_sat.astype(np.uint8), disk)

    # kill brightness where it's not saturated
    v[non_sat == 0] = 0
    # high brightness → glare candidates
    glare = (v > 200) * 255
    glare = cv2.dilate(glare.astype(np.uint8), disk)

    glare_reduced = np.ones((img.shape[0], img.shape[1], 3), dtype=np.uint8) * 200
    glare_bgr = cv2.cvtColor(glare, cv2.COLOR_GRAY2BGR)

    corrected = np.where(glare_bgr > 0, glare_reduced, img)
    return corrected


# ---------------------------------------------------------------------------
# 3) Core: contour-based card detection (faithful to hj3yoo)
# ---------------------------------------------------------------------------
def _rects_from_mask(
    mask: np.ndarray,
    size_thresh: int,
) -> List[np.ndarray]:
    """
    Given a binary mask (0/255), find large blobs and return minAreaRect boxes.
    """
    # Make sure mask is uint8 with only 0 / 255
    mask = (mask > 0).astype("uint8") * 255

    # Slight closing to connect gaps along the card border
    kernel = np.ones((5, 5), np.uint8)
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    cnts_info = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts_info) == 3:
        _, cnts, _ = cnts_info
    else:
        cnts, _ = cnts_info

    rects: List[np.ndarray] = []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < size_thresh:
            continue

        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        rects.append(np.array(box, dtype=np.float32))

    return rects


def find_card(
    img: np.ndarray,
    thresh_c: int = 1,                     # not used anymore, kept for signature
    kernel_size: Tuple[int, int] = (5, 5), # not used anymore, kept for signature
    size_thresh: int = 1000,
) -> List[np.ndarray]:
    """
    Find card-like rectangles using brightness segmentation instead of edge soup.

    Strategy:
      1) convert to grayscale and blur heavily
      2) apply Otsu threshold (bright bg vs dark objects)
      3) apply inverted Otsu threshold
      4) from each mask, take big blobs via _rects_from_mask
      5) return union of all candidate rectangles
    """
    h, w = img.shape[:2]
    img_area = float(h * w)
    size_thresh = max(size_thresh, int(img_area * 0.05))  # at least 5% of image

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # strong blur to suppress artwork/text texture
    blur = cv2.GaussianBlur(gray, (21, 21), 0)

    # Otsu thresholds: normal and inverted
    _, mask1 = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    _, mask2 = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # DEBUG: save masks
    try:
        cv2.imwrite("/home/lubuharg/Documents/MTG/debug_mask1.jpg", mask1)
        cv2.imwrite("/home/lubuharg/Documents/MTG/debug_mask2.jpg", mask2)
    except Exception as e:
        print("[DEBUG] Could not write debug_mask images:", e)

    rects1 = _rects_from_mask(mask1, size_thresh)
    rects2 = _rects_from_mask(mask2, size_thresh)

    rects = rects1 + rects2

    print(
        f"[DEBUG] find_card (brightness): "
        f"size_thresh={size_thresh}, rects1={len(rects1)}, rects2={len(rects2)}, total={len(rects)}"
    )

    return rects


# ---------------------------------------------------------------------------
# 4) High-level helpers
# ---------------------------------------------------------------------------
def detect_card_edges_1d(
    img: np.ndarray,
    min_margin: int = 8,
) -> Tuple[int, int, int, int]:
    """
    Detect card borders by scanning 1D brightness profiles from each side.

    We explicitly restrict the search to bands near the outer borders using
    EDGE_INNER_MARGIN_X / EDGE_INNER_MARGIN_Y and EDGE_MAX_INNER_FRAC, so we
    don't accidentally lock onto inner frames (textbox, artwork, etc.).

    Returns:
        (x_left, x_right, y_top, y_bottom)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)

    h, w = blur.shape

    # --- DEBUG IMAGES (unchanged) ---
    try:
        cv2.imwrite("/home/lubuharg/Documents/MTG/debug_gray.jpg", gray)
        cv2.imwrite("/home/lubuharg/Documents/MTG/debug_blur.jpg", blur)
    except Exception as e:
        print("[DEBUG] Could not write gray/blur debug:", e)

    try:
        _, mask_light = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        _, mask_dark = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        cv2.imwrite("/home/lubuharg/Documents/MTG/debug_mask1.jpg", mask_light)
        cv2.imwrite("/home/lubuharg/Documents/MTG/debug_mask2.jpg", mask_dark)
    except Exception as e:
        print("[DEBUG] Could not write mask debug:", e)

    try:
        edges = cv2.Laplacian(blur, cv2.CV_8U, ksize=3)
        _, edges_bin = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        edges_closed = cv2.morphologyEx(edges_bin, cv2.MORPH_CLOSE, kernel, iterations=1)
        cv2.imwrite("/home/lubuharg/Documents/MTG/debug_thresh.jpg", edges_bin)
        cv2.imwrite("/home/lubuharg/Documents/MTG/debug_erode.jpg", edges_closed)
    except Exception as e:
        print("[DEBUG] Could not write edge debug:", e)

    # 1-D profiles
    col_mean = blur.mean(axis=0)
    row_mean = blur.mean(axis=1)

    grad_x = np.diff(col_mean)
    grad_y = np.diff(row_mean)

    def clamp(v, lo, hi):
        return max(lo, min(hi, v))

    # ---- Horizontal (left / right) using EDGE_* ----
    # Left search window: between inner_margin_x and max_inner_frac of width
    left_lo = int(max(min_margin, EDGE_INNER_MARGIN_X * w))
    left_hi = int(min(w - 2 - min_margin, EDGE_MAX_INNER_FRAC * w))
    if left_hi <= left_lo:
        left_hi = left_lo + 1

    left_band = np.abs(grad_x[left_lo:left_hi])
    x_left = left_lo + int(left_band.argmax()) if left_band.size > 0 else min_margin

    # Right search window: symmetric near the right border
    right_hi = w - 1 - min_margin
    right_lo = int(max(0, w - EDGE_MAX_INNER_FRAC * w))
    # also respect inner margin
    right_lo = max(right_lo, int(EDGE_INNER_MARGIN_X * w))
    if right_hi <= right_lo:
        right_lo = right_hi - 1

    right_band = np.abs(grad_x[right_lo:right_hi])
    band_max = right_band.max() if right_band.size > 0 else 0.0
    thresh = band_max * 0.5  # “strong enough” threshold

    x_right = None
    if band_max > 0 and right_band.size > 0:
        # iterate from border inward (highest index → lowest),
        # but only inside the outer band
        for idx in range(right_hi - 1, right_lo - 1, -1):
            if abs(grad_x[idx]) >= thresh:
                x_right = idx + 1  # +1 because gradient is between columns
                break

    # fallback: argmax in the band
    if x_right is None:
        x_right = right_lo + int(right_band.argmax()) + 1 if right_band.size > 0 else w - 1 - min_margin

    # ---- Vertical (top / bottom) using EDGE_* ----
    top_lo = int(max(min_margin, EDGE_INNER_MARGIN_Y * h))
    top_hi = int(min(h - 2 - min_margin, EDGE_MAX_INNER_FRAC * h))
    if top_hi <= top_lo:
        top_hi = top_lo + 1

    top_band = np.abs(grad_y[top_lo:top_hi])
    y_top = top_lo + int(top_band.argmax()) if top_band.size > 0 else min_margin

    bottom_hi = h - 1 - min_margin
    bottom_lo = int(max(0, h - EDGE_MAX_INNER_FRAC * h))
    bottom_lo = max(bottom_lo, int(EDGE_INNER_MARGIN_Y * h))
    if bottom_hi <= bottom_lo:
        bottom_lo = bottom_hi - 1

    bottom_band = np.abs(grad_y[bottom_lo:bottom_hi])
    y_bottom = bottom_lo + int(bottom_band.argmax()) + 1 if bottom_band.size > 0 else h - 1 - min_margin

    # final clamp
    x_left = clamp(x_left, 0, w - 2)
    x_right = clamp(x_right, x_left + 1, w - 1)
    y_top = clamp(y_top, 0, h - 2)
    y_bottom = clamp(y_bottom, y_top + 1, h - 1)

    print(
        f"[WARP_1D] edges: x_left={x_left}, x_right={x_right}, "
        f"y_top={y_top}, y_bottom={y_bottom}, w={w}, h={h}, "
        f"right_band_max={band_max:.2f}, right_thresh={thresh:.2f}"
    )

    return x_left, x_right, y_top, y_bottom


import math

def _line_from_points(p1, p2):
    """Return line in ax + by + c = 0 from two points."""
    x1, y1 = p1
    x2, y2 = p2
    a = y1 - y2
    b = x2 - x1
    c = x1*y2 - x2*y1
    return a, b, c

def _intersect_lines(l1, l2):
    """Intersect two lines (a,b,c). Return (x,y) or None if parallel."""
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    det = a1*b2 - a2*b1
    if abs(det) < 1e-6:
        return None
    x = (b1*c2 - b2*c1) / det
    y = (c1*a2 - c2*a1) / det
    return x, y

def refine_corners_hough(
    img: np.ndarray,
    bbox: Tuple[int, int, int, int],
    pad: int = 20,
) -> Optional[np.ndarray]:
    """
    Use Canny + Hough inside a padded bbox to find 4 card edges.

    bbox: (x_left, x_right, y_top, y_bottom) in full image coords.
    Returns 4x2 float32 (tl, tr, br, bl) in full image coords, or None.
    """
    x_left, x_right, y_top, y_bottom = bbox
    h, w = img.shape[:2]

    # Crop ROI around bbox (a bit larger with pad)
    x0 = max(0, x_left - pad)
    x1 = min(w - 1, x_right + pad)
    y0 = max(0, y_top - pad)
    y1 = min(h - 1, y_bottom + pad)

    roi = img[y0:y1, x0:x1]
    if roi.size == 0:
        print("[Hough] empty ROI")
        return None

    rh, rw = roi.shape[:2]

    # Bbox in ROI coordinates (for distance checks)
    xL_roi = x_left   - x0
    xR_roi = x_right  - x0
    yT_roi = y_top    - y0
    yB_roi = y_bottom - y0
    
    # midpoints of the bbox in ROI coords (for left/right & top/bottom splitting)
    mid_x = 0.5 * (xL_roi + xR_roi)
    mid_y = 0.5 * (yT_roi + yB_roi)

    print(
        f"[Hough] ROI size: {rw}x{rh}, "
        f"bbox_roi=(xL={xL_roi}, xR={xR_roi}, yT={yT_roi}, yB={yB_roi})"
    )

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Make the *card* (darker than background) white, background black
    _, th = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Close gaps along the outer border
    kernel = np.ones((5, 5), np.uint8)
    th_closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Now take edges of that mask – this is very similar to debug_mask1
    edges = cv2.Canny(th_closed, 50, 150)

    # Extra debug: see what Hough actually sees
    try:
        cv2.imwrite("/home/lubuharg/Documents/MTG/debug_hough_edges_roi.jpg", edges)
    except Exception as e:
        print("[Hough] could not save debug_hough_edges_roi:", e)


    # Save Canny for extra visibility
    try:
        cv2.imwrite("/home/lubuharg/Documents/MTG/debug_hough_edges_roi.jpg", edges)
    except Exception as e:
        print("[Hough] could not save debug_hough_edges_roi:", e)

    # Hough lines
    lines = cv2.HoughLinesP(
    edges,
    rho=1,
    theta=np.pi / 180,
    threshold=40,                       # a bit easier to trigger
    minLineLength=max(rh, rw) // 8,     # shorter lines are allowed
    maxLineGap=int(0.10 * max(rh, rw)), # allow bigger gaps to be merged
)

    if lines is None:
        print("[Hough] no lines found")
        return None

    verticals: List[np.ndarray] = []
    horizontals: List[np.ndarray] = []

    for l in lines[:, 0]:
        x1p, y1p, x2p, y2p = l
        dx = x2p - x1p
        dy = y2p - y1p
        if dx == 0:
            angle = 90.0
        else:
            angle = abs(math.degrees(math.atan2(dy, dx)))
        # treat near-vertical as vertical, the rest as horizontal-ish
        if angle > 45:
            verticals.append(l)
        else:
            horizontals.append(l)

    print(f"[Hough] found {len(verticals)} verticals, {len(horizontals)} horizontals")

    # --- DEBUG 1: all lines + bbox in ROI (ALWAYS saved if we got any lines) ---
    dbg_all = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Draw all verticals in green, horizontals in red
    for v in verticals:
        cv2.line(dbg_all, (v[0], v[1]), (v[2], v[3]), (0, 255, 0), 1)
    for hline in horizontals:
        cv2.line(dbg_all, (hline[0], hline[1]), (hline[2], hline[3]), (0, 0, 255), 1)

    # 1-D bbox in ROI coords (yellow)
    bbox_roi_poly = np.array(
        [[xL_roi, yT_roi], [xR_roi, yT_roi], [xR_roi, yB_roi], [xL_roi, yB_roi]],
        dtype=np.int32,
    )
    cv2.polylines(dbg_all, [bbox_roi_poly], True, (0, 255, 255), 2)

    try:
        cv2.imwrite("/home/lubuharg/Documents/MTG/debug_hough_all_roi.jpg", dbg_all)
        print("[Hough] Saved debug_hough_all_roi.jpg")
    except Exception as e:
        print("[Hough] could not save debug_hough_all_roi:", e)

    # If we really have almost no vertical/horizontal lines, stop here
    if len(verticals) < 1 or len(horizontals) < 1:
        print("[Hough] not enough vertical/horizontal lines")
        return None

    # --- helper: choose closest line to a given side of the bbox in ROI coords ---

    def avg_x(line: np.ndarray) -> float:
        return (float(line[0]) + float(line[2])) * 0.5

    def avg_y(line: np.ndarray) -> float:
        return (float(line[1]) + float(line[3])) * 0.5

    # maximum allowed distances of the chosen line from the bbox side
    max_dx = max(20.0, HOUGH_MAX_SIDE_X_OFFSET_FRAC * rw)
    max_dy = max(20.0, HOUGH_MAX_SIDE_Y_OFFSET_FRAC * rh)

    # choose verticals for left/right (closest center x to each side)
    best_left = None
    best_left_dist = 1e9
    best_right = None
    best_right_dist = 1e9

    for v in verticals:
        cx = avg_x(v)
        dL = abs(cx - xL_roi)
        dR = abs(cx - xR_roi)
        if dL < best_left_dist:
            best_left_dist = dL
            best_left = v
        if dR < best_right_dist:
            best_right_dist = dR
            best_right = v

    print(
        f"[Hough] vertical candidate dL={best_left_dist:.1f}, "
        f"dR={best_right_dist:.1f}, max_dx={max_dx:.1f}"
    )

    if best_left is None or best_right is None:
        print("[Hough] could not choose both verticals")
        return None
    if best_left_dist > max_dx or best_right_dist > max_dx:
        print("[Hough] verticals too far from bbox → reject")
        return None

    v_left = best_left
    v_right = best_right

    # choose horizontals for top/bottom (closest center y to each side)
    best_top = None
    best_top_dist = 1e9
    best_bottom = None
    best_bottom_dist = 1e9

    for hh in horizontals:
        cy = avg_y(hh)
        dT = abs(cy - yT_roi)
        dB = abs(cy - yB_roi)
        if dT < best_top_dist:
            best_top_dist = dT
            best_top = hh
        if dB < best_bottom_dist:
            best_bottom_dist = dB
            best_bottom = hh

    print(
        f"[Hough] horizontal candidate dT={best_top_dist:.1f}, "
        f"dB={best_bottom_dist:.1f}, max_dy={max_dy:.1f}"
    )

    if best_top is None or best_bottom is None:
        print("[Hough] could not choose both horizontals")
        return None
    if best_top_dist > max_dy or best_bottom_dist > max_dy:
        print("[Hough] horizontals too far from bbox → reject")
        return None


    h_top = best_top
    h_bottom = best_bottom

    print(
        "[Hough] chosen lines (ROI coords):\n"
        f"  left   dL={best_left_dist:.1f}px\n"
        f"  right  dR={best_right_dist:.1f}px\n"
        f"  top    dT={best_top_dist:.1f}px\n"
        f"  bottom dB={best_bottom_dist:.1f}px"
    )

    # --- Convert chosen segments to infinite lines (ax + by + c = 0) in ROI coords ---

    v_left_line   = _line_from_points((v_left[0],  v_left[1]),  (v_left[2],  v_left[3]))
    v_right_line  = _line_from_points((v_right[0], v_right[1]), (v_right[2], v_right[3]))
    h_top_line    = _line_from_points((h_top[0],   h_top[1]),   (h_top[2],   h_top[3]))
    h_bottom_line = _line_from_points((h_bottom[0],h_bottom[1]),(h_bottom[2],h_bottom[3]))

    tl = _intersect_lines(v_left_line,  h_top_line)
    tr = _intersect_lines(v_right_line, h_top_line)
    br = _intersect_lines(v_right_line, h_bottom_line)
    bl = _intersect_lines(v_left_line,  h_bottom_line)

    if any(p is None for p in (tl, tr, br, bl)):
        print("[Hough] intersection failed")
        return None

    corners_roi = np.array(
        [
            [tl[0], tl[1]],
            [tr[0], tr[1]],
            [br[0], br[1]],
            [bl[0], bl[1]],
        ],
        dtype=np.float32,
    )

    # Map ROI corners back to full-image coordinates
    corners = corners_roi.copy()
    corners[:, 0] += x0
    corners[:, 1] += y0

    # --- Sanity check vs 1-D bbox: area ratio ---
    bbox_rect = np.array(
        [[x_left, y_top], [x_right, y_top], [x_right, y_bottom], [x_left, y_bottom]],
        dtype=np.float32,
    )

    def poly_area(pts: np.ndarray) -> float:
        x = pts[:, 0]
        y = pts[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    area_bbox = poly_area(bbox_rect)
    area_hough = poly_area(corners)
    ratio = area_hough / area_bbox if area_bbox > 0 else 0.0

    print(
        f"[Hough] area check: bbox={area_bbox:.1f}, hough={area_hough:.1f}, "
        f"ratio={ratio:.3f}, allowed=({HOUGH_MIN_AREA_RATIO}, {HOUGH_MAX_AREA_RATIO})"
    )

    if ratio < HOUGH_MIN_AREA_RATIO or ratio > HOUGH_MAX_AREA_RATIO:
        print("[Hough] area mismatch vs 1D bbox → reject")
        return None

    # --- DEBUG 2: chosen lines in ROI and full image ---

    # ROI with chosen lines highlighted
    dbg_chosen = dbg_all.copy()
    cv2.line(dbg_chosen, (v_left[0],  v_left[1]),  (v_left[2],  v_left[3]),  (0, 255, 0), 3)
    cv2.line(dbg_chosen, (v_right[0], v_right[1]), (v_right[2], v_right[3]), (0, 255, 0), 3)
    cv2.line(dbg_chosen, (h_top[0],   h_top[1]),   (h_top[2],   h_top[3]),   (0, 0, 255), 3)
    cv2.line(dbg_chosen, (h_bottom[0],h_bottom[1]),(h_bottom[2],h_bottom[3]),(0, 0, 255), 3)

    try:
        cv2.imwrite("/home/lubuharg/Documents/MTG/debug_hough_chosen_roi.jpg", dbg_chosen)
        print("[Hough] Saved debug_hough_chosen_roi.jpg")
    except Exception as e:
        print("[Hough] could not save debug_hough_chosen_roi:", e)

    # full-image overlay of bbox (red) + Hough quad (green)
    dbg_full = img.copy()
    cv2.polylines(dbg_full, [bbox_rect.astype(np.int32)], True, (0, 0, 255), 2)
    cv2.polylines(dbg_full, [corners.astype(np.int32)], True, (0, 255, 0), 3)
    try:
        cv2.imwrite("/home/lubuharg/Documents/MTG/debug_rects_hough.jpg", dbg_full)
        print("[Hough] Saved debug_rects_hough.jpg")
    except Exception as e:
        print("[Hough] could not save debug_rects_hough:", e)

    return corners


def _pick_largest_quad(cnts: List[np.ndarray]) -> Optional[np.ndarray]:
    if not cnts:
        return None
    return max(cnts, key=lambda c: cv2.contourArea(c))

def _debug_draw_rects(
    img: np.ndarray,
    rects: List[np.ndarray],
    out_path: str = "/home/lubuharg/Documents/MTG/debug_rects.jpg",
) -> None:
    """
    Draw rectangles on a copy of the image for debugging.
    If rects is empty, still save a copy of the original image.
    """
    dbg = img.copy()

    for r in rects:
        pts = r.astype(np.int32)
        cv2.polylines(dbg, [pts], isClosed=True, color=(0, 0, 255), thickness=4)

    try:
        cv2.imwrite(out_path, dbg)
        print(f"[DEBUG] Saved rect overlay to: {out_path}")
    except Exception as e:
        print("[DEBUG] Could not write debug_rects.jpg:", e)



def warp_card_from_image_1d(
    img: np.ndarray,
    use_glare_reduction: bool = False,
) -> Tuple[np.ndarray, bool]:
    """
    1D border detection (primary) + optional Hough refinement.
    Returns (warped_img, used_hough).
    """
    try:
        x_left, x_right, y_top, y_bottom = detect_card_edges_1d(img)
    except Exception as e:
        print("[WARP_1D] edge detection failed:", e)
        return img.copy(), False

    bbox = (x_left, x_right, y_top, y_bottom)
    bbox_rect = np.array(
        [[x_left, y_top], [x_right, y_top], [x_right, y_bottom], [x_left, y_bottom]],
        dtype=np.float32,
    )

    # Try to refine with Hough
    corners = refine_corners_hough(img, bbox, pad=20)

    if corners is None:
        print("[WARP_1D] Using 1D bbox only (no valid Hough corners).")
        dbg = img.copy()
        cv2.polylines(dbg, [bbox_rect.astype(np.int32)], True, (0, 0, 255), 3)
        try:
            cv2.imwrite("/home/lubuharg/Documents/MTG/debug_rects_1d.jpg", dbg)
        except Exception as e:
            print("[WARP_1D] Could not write debug_rects_1d.jpg:", e)

        warped = four_point_transform(img, bbox_rect)
        if use_glare_reduction:
            warped = remove_glare(warped)
        return warped, False

    # Use refined corners (green), but also show the 1D bbox (red)
    dbg = img.copy()
    cv2.polylines(dbg, [bbox_rect.astype(np.int32)], True, (0, 0, 255), 2)
    cv2.polylines(dbg, [corners.astype(np.int32)], True, (0, 255, 0), 3)
    try:
        cv2.imwrite("/home/lubuharg/Documents/MTG/debug_rects_1d.jpg", dbg)
        print("[WARP_1D] Saved debug_rects_1d.jpg with 1D (red) + Hough (green).")
    except Exception as e:
        print("[WARP_1D] Could not write debug_rects_1d.jpg:", e)

    warped = four_point_transform(img, corners)
    if use_glare_reduction:
        warped = remove_glare(warped)

    return warped, True


def warp_card_from_yolo(
    full_img: np.ndarray,
    box: Tuple[float, float, float, float],
    size_thresh_rel: float = 0.02,
    use_glare_reduction: bool = False,
) -> Tuple[np.ndarray, bool]:
    """
    Alternative path: use YOLO box as a rough crop, then run contour logic inside it.

        full_img: full BGR frame
        box: (x1, y1, x2, y2)

    Returns (warped_image, success_flag).
    """
    x1, y1, x2, y2 = map(int, box)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(full_img.shape[1], x2)
    y2 = min(full_img.shape[0], y2)

    if x2 <= x1 or y2 <= y1:
        return full_img.copy(), False

    roi = full_img[y1:y2, x1:x2]
    warped, ok = warp_card_from_image(roi, size_thresh_rel=size_thresh_rel,
                                      use_glare_reduction=use_glare_reduction)
    if not ok:
        return roi, False

    return warped, True


# ---------------------------------------------------------------------------
# 5) CLI demo
# ---------------------------------------------------------------------------

def _demo_cli():
    """
    CLI:

    1) Simple mode (recommended for now):
        python -m mtgscan.geometry.warp_hj3 in.jpg out.jpg

       → runs full contour pipeline on `in.jpg` and saves warped card.

    2) YOLO-box mode:
        python -m mtgscan.geometry.warp_hj3 in.jpg x1 y1 x2 y2 out.jpg

       → crops using (x1,y1,x2,y2) from full frame, then warps contour inside.
    """
    import sys

    if len(sys.argv) not in (3, 7):
        print(
            "Usage:\n"
            "  python -m mtgscan.geometry.warp_hj3 in.jpg out.jpg\n"
            "  python -m mtgscan.geometry.warp_hj3 in.jpg x1 y1 x2 y2 out.jpg"
        )
        sys.exit(1)

    if len(sys.argv) == 3:
        in_path = sys.argv[1]
        out_path = sys.argv[2]
        img = cv2.imread(in_path)
        if img is None:
            print(f"Could not read image: {in_path}")
            sys.exit(1)
        warped, ok = warp_card_from_image_1d(img)
        cv2.imwrite(out_path, warped)
        print(f"Saved warped card to: {out_path} (success={ok})")
        return


    # len == 7: YOLO-box mode
    in_path = sys.argv[1]
    x1 = float(sys.argv[2])
    y1 = float(sys.argv[3])
    x2 = float(sys.argv[4])
    y2 = float(sys.argv[5])
    out_path = sys.argv[6]

    img = cv2.imread(in_path)
    if img is None:
        print(f"Could not read image: {in_path}")
        sys.exit(1)

    warped, ok = warp_card_from_yolo(img, (x1, y1, x2, y2))
    cv2.imwrite(out_path, warped)
    print(f"Saved warped card to: {out_path} (success={ok})")


if __name__ == "__main__":
    _demo_cli()
