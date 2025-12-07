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
EDGE_INNER_MARGIN_X = 0.01   # was effectively ~0.08 before → move search closer to border
EDGE_INNER_MARGIN_Y = 0.04   # leave this as-is for now
EDGE_MAX_INNER_FRAC = 0.20   # search up to 45% of width from each side

HOUGH_MIN_AREA_RATIO = 0.80   # reject if Hough area < 90% of 1D bbox area
HOUGH_MAX_AREA_RATIO = 1.25   # reject if Hough area > 125% of 1D bbox area

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
    min_margin: int = 4,
) -> Tuple[int, int, int, int]:
    """
    Detect card borders by scanning 1D brightness profiles from each side.

    Returns:
        (x_left, x_right, y_top, y_bottom)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)

    h, w = blur.shape

    # --- DEBUG IMAGES ---
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

    # Use the constants at the top of the file
    sx = max(int(w * EDGE_MAX_INNER_FRAC), 24)
    sy = max(int(h * EDGE_MAX_INNER_FRAC), 24)

    def clamp(v, lo, hi):
        return max(lo, min(hi, v))

    # ---------------- LEFT EDGE (from outside inward) ----------------
    left_lo = max(int(w * EDGE_INNER_MARGIN_X), min_margin)
    left_hi = clamp(left_lo + sx, left_lo + 1, w - 2)
    left_band = np.abs(grad_x[left_lo:left_hi])
    left_band_max = left_band.max() if left_band.size > 0 else 0.0
    left_thresh = left_band_max * 0.5

    x_left = None
    if left_band_max > 0 and left_band.size > 0:
        # scan from border inward → first strong jump
        for idx in range(left_lo, left_hi):
            if abs(grad_x[idx]) >= left_thresh:
                x_left = idx
                break

    # fallback: if nothing exceeded threshold, use argmax in the band
    if x_left is None:
        x_left = left_lo + int(left_band.argmax()) if left_band.size > 0 else min_margin

    # ---------------- RIGHT EDGE (outside inward, existing logic) ----
    right_hi = w - 1 - min_margin
    right_lo = clamp(right_hi - sx, 0, right_hi - 1)
    right_band = np.abs(grad_x[right_lo:right_hi])
    right_band_max = right_band.max() if right_band.size > 0 else 0.0
    right_thresh = right_band_max * 0.5

    x_right = None
    if right_band_max > 0 and right_band.size > 0:
        for idx in range(right_hi - 1, right_lo - 1, -1):
            if abs(grad_x[idx]) >= right_thresh:
                x_right = idx + 1
                break

    if x_right is None:
        x_right = right_lo + int(right_band.argmax()) + 1 if right_band.size > 0 else w - 1 - min_margin

    # ---------------- TOP EDGE ---------------------------------------
    top_lo = max(int(h * EDGE_INNER_MARGIN_Y), min_margin)
    top_hi = clamp(top_lo + sy, top_lo + 1, h - 2)
    top_band = np.abs(grad_y[top_lo:top_hi])
    y_top = top_lo + int(top_band.argmax()) if top_band.size > 0 else min_margin

    # ---------------- BOTTOM EDGE ------------------------------------
    bottom_hi = h - 1 - min_margin
    bottom_lo = clamp(bottom_hi - sy, 0, bottom_hi - 1)
    bottom_band = np.abs(grad_y[bottom_lo:bottom_hi])
    y_bottom = bottom_lo + int(bottom_band.argmax()) + 1 if bottom_band.size > 0 else h - 1 - min_margin

    # clamp everything
    x_left = clamp(x_left, 0, w - 2)
    x_right = clamp(x_right, x_left + 1, w - 1)
    y_top = clamp(y_top, 0, h - 2)
    y_bottom = clamp(y_bottom, y_top + 1, h - 1)

    print(
        f"[WARP_1D] edges: x_left={x_left}, x_right={x_right}, "
        f"y_top={y_top}, y_bottom={y_bottom}, w={w}, h={h}, "
        f"left_band_max={left_band_max:.2f}, left_thresh={left_thresh:.2f}, "
        f"right_band_max={right_band_max:.2f}, right_thresh={right_thresh:.2f}"
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

    x0 = max(0, x_left - pad)
    x1 = min(w - 1, x_right + pad)
    y0 = max(0, y_top - pad)
    y1 = min(h - 1, y_bottom + pad)

    roi = img[y0:y1, x0:x1]
    if roi.size == 0:
        return None

    rh, rw = roi.shape[:2]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=max(rh, rw) // 3,
        maxLineGap=20,
    )
    if lines is None:
        print("[Hough] no lines found")
        return None

    verticals = []
    horizontals = []

    for l in lines[:, 0]:
        x1p, y1p, x2p, y2p = l
        dx = x2p - x1p
        dy = y2p - y1p
        if dx == 0:
            angle = 90.0
        else:
            angle = abs(math.degrees(math.atan2(dy, dx)))
        if angle > 45:
            verticals.append(l)
        else:
            horizontals.append(l)

    if len(verticals) < 2 or len(horizontals) < 2:
        print("[Hough] not enough vertical/horizontal lines")
        return None

    # Only consider verticals close to left/right ROI border
    def avg_x(line):
        return (line[0] + line[2]) / 2.0

    v_candidates_left = [v for v in verticals if avg_x(v) < rw * 0.35]
    v_candidates_right = [v for v in verticals if avg_x(v) > rw * 0.65]

    if not v_candidates_left or not v_candidates_right:
        print("[Hough] verticals not near borders")
        return None

    v_left = min(v_candidates_left, key=avg_x)
    v_right = max(v_candidates_right, key=avg_x)

    # Only consider horizontals close to top/bottom ROI border
    def avg_y(line):
        return (line[1] + line[3]) / 2.0

    h_candidates_top = [h for h in horizontals if avg_y(h) < rh * 0.35]
    h_candidates_bottom = [h for h in horizontals if avg_y(h) > rh * 0.65]

    if not h_candidates_top or not h_candidates_bottom:
        print("[Hough] horizontals not near borders")
        return None

    h_top = min(h_candidates_top, key=avg_y)
    h_bottom = max(h_candidates_bottom, key=avg_y)

    # Convert to line equations in ROI coords
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

    # Map back to full image
    corners = corners_roi.copy()
    corners[:, 0] += x0
    corners[:, 1] += y0

    # Sanity check vs 1-D bbox: area and overlap must be reasonable
    bbox_rect = np.array(
        [[x_left, y_top], [x_right, y_top], [x_right, y_bottom], [x_left, y_bottom]],
        dtype=np.float32,
    )

    # area ratio
    def poly_area(pts):
        x = pts[:, 0]
        y = pts[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    area_bbox = poly_area(bbox_rect)
    area_hough = poly_area(corners)

    ratio = area_hough / area_bbox if area_bbox > 0 else 0.0

    if ratio < HOUGH_MIN_AREA_RATIO or ratio > HOUGH_MAX_AREA_RATIO:
        print(
            f"[Hough] area mismatch vs 1D bbox, reject "
            f"(bbox={area_bbox:.1f}, hough={area_hough:.1f}, ratio={ratio:.3f})"
        )
        return None


    # Debug overlay: green = Hough quad, red = 1D bbox
    dbg = img.copy()
    cv2.polylines(dbg, [bbox_rect.astype(np.int32)], True, (0, 0, 255), 2)
    cv2.polylines(dbg, [corners.astype(np.int32)], True, (0, 255, 0), 3)
    try:
        cv2.imwrite("/home/lubuharg/Documents/MTG/debug_rects_hough.jpg", dbg)
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
