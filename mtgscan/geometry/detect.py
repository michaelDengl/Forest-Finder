# mtgscan/geometry/detect.py
from __future__ import annotations
from typing import Dict, Optional, Tuple, List, Set
import math
import cv2
import numpy as np
from mtgscan.core.contracts import Corners

# Defaults tuned for Raspberry Pi + OpenCV 4.6 and our synthetic tests
_DEFAULT_CFG: Dict = {
    # ↓ relaxed to allow smaller/further cards in real photos
    "min_area_ratio": 0.005,           # relative to frame area
    # Absolute floor on contour area; high enough to ignore inner artwork-only quads
    "min_abs_area_px": 2800000.0,
    "max_area_ratio": 0.95,            # already used by _choose_best_quad
    "aspect_range": (0.40, 3.00),
    "require_portrait": True,    
    # Expected card size relative to full frame (optional, can be tuned)
    # Example: 0.25 = card ≈ 25% of the image area
    "target_area_ratio": 0.36,        # tuned to typical framing; avoid overshoot
    "area_ratio_tol": 0.35,           # not used as a hard gate, only for ranking  
    "max_quad_epsilon": 0.08,
    "min_solidity": 0.75,
    "canny": {"low": 40, "high": 160, "clahe": True},
    "blur": {"ksize": 5},
    "feature": {"nfeatures": 2000, "ratio": 0.85, "ransac_thresh": 5.0, "min_inliers": 8},
    "prefer": "contours",
    "debug": False,
    "dark_border": {"percentile": 20, "blur": 3, "morph": 7, "min_area_ratio": 0.18},
    "saturation": {"low": 40, "blur": 3, "morph": 7, "min_area_ratio": 0.22},

    "auto_relax": True,
    "relax": {
        "min_area_ratio": 0.003,      # NEW: relax more than first pass
        "min_abs_area_px": 1800000.0,      # NEW: relaxed absolute area
        "aspect_range": (0.40, 3.00),
        "max_quad_epsilon": 0.08,
        "min_solidity": 0.75
    },

    # Fallback (largest portrait quad)
    "fallback_min_abs_area_px": 2000000.0,
    "fallback_min_area_ratio": 0.25,

    # MTG-likeness helpers
    "card_aspect": 1.395,
    "card_aspect_tol": 0.18,
    "card_border_min": 0.22,
    "quad_expand_max_scale": 1.20,    # modest growth headroom
    "quad_expand_aspect_weight": 0.32,# gentle aspect push
    "quad_refine_enabled": True,
    "quad_refine_scale_max": 1.20,    # keep refine close to detected size
    "local_refine": {"enabled": True, "margin_px": 32},  # widen ROI to catch top/bottom edges
    "upright_snap": False,            # keep natural tilt; card may not be level in frame
    "perspective_taper": 0.04,        # gently widen bottom vs top to counter camera tilt
    "save_last_quad_path": "tests/output/last_quad.json",

    # contour housekeeping
    "border_margin_px": 2,
}


# ----------------------------------------------------------------------------- #
# Config / utilities                                                            #
# ----------------------------------------------------------------------------- #

def card_likeness_score(quad: np.ndarray, frame: np.ndarray, cfg: Dict) -> float:
    """Return a 0..1 score for how much this quad looks like an MTG card."""
    H, W = frame.shape[:2]
    if quad is None or quad.shape != (4, 2):
        return 0.0

    # Aspect check
    aspect = (np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3])) / \
             (np.linalg.norm(quad[3] - quad[0]) + np.linalg.norm(quad[2] - quad[1]) + 1e-6)
    aspect_target = cfg.get("card_aspect", 1.395)
    tol = cfg.get("card_aspect_tol", 0.18)
    aspect_ok = (1 - tol) * aspect_target <= aspect <= (1 + tol) * aspect_target

    # Very simple scoring right now (aspect only)
    return 1.0 if aspect_ok else 0.0

def _merge_cfg(cfg: Optional[Dict]) -> Dict:
    if not cfg:
        return dict(_DEFAULT_CFG)
    merged = dict(_DEFAULT_CFG)
    for k, v in cfg.items():
        if isinstance(v, dict) and k in merged and isinstance(merged[k], dict):
            merged[k] = {**merged[k], **v}
        else:
            merged[k] = v
    return merged

def order_corners_clockwise(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, np.float32)
    if pts.shape != (4, 2):
        pts = pts.reshape(4, 2)
    sorted_y = pts[np.argsort(pts[:, 1])]
    top2 = sorted_y[:2]
    bottom2 = sorted_y[2:]
    tl, tr = top2[np.argsort(top2[:, 0])]
    bl, br = bottom2[np.argsort(bottom2[:, 0])]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def is_plausible_quad(pts: np.ndarray, frame_shape: Tuple[int, int, int], cfg: Optional[Dict] = None) -> bool:
    cfg = _merge_cfg(cfg)
    H, W = frame_shape[:2]
    frame_area = float(H * W)

    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    if not np.isfinite(pts).all():
        if cfg.get("debug"): print("[plaus] non-finite pts")
        return False

    area = abs(cv2.contourArea(pts))
    area_ratio = area / frame_area if frame_area > 0 else 0.0
    min_ratio = float(cfg["min_area_ratio"])
    min_abs   = float(cfg.get("min_abs_area_px", 0.0))
    eps = 5e-6

    # Require BOTH relative and absolute size → kills tiny inner blobs
    if not ((area_ratio + eps >= min_ratio) and (area + eps >= min_abs)):
        if cfg.get("debug"):
            print(f"[plaus] area fail: area={area:.1f}, ratio={area_ratio:.4f}, "
                  f"need ratio>={min_ratio} AND area>={min_abs}")
        return False

    hull = cv2.convexHull(pts.reshape(-1, 1, 2))
    if len(hull) < 3:
        if cfg.get("debug"): print("[plaus] hull too small:", len(hull))
        return False

    def dist(a, b) -> float:
        return math.hypot(float(a[0] - b[0]), float(a[1] - b[1]))
    tl, tr, br, bl = pts
    width  = (dist(tl, tr) + dist(bl, br)) / 2.0
    height = (dist(tl, bl) + dist(tr, br)) / 2.0
    if width <= 1e-3 or height <= 1e-3:
        if cfg.get("debug"):
            print("[plaus] degenerate width/height:", width, height)
        return False

    # Enforce portrait orientation (card taller than wide) to avoid matching
    # the horizontal artwork window etc.
    if cfg.get("require_portrait", False) and height <= width:
        if cfg.get("debug"):
            print(f"[plaus] reject landscape-ish quad: h={height:.1f}, w={width:.1f}")
        return False


    aspect = height / width
    a_min, a_max = cfg["aspect_range"]
    ok_aspect = (a_min <= aspect <= a_max)
    if cfg.get("debug"):
        print(f"[plaus] area%={(area_ratio):.4f}, aspect={aspect:.3f}, range=({a_min},{a_max}) -> {ok_aspect}")
    return ok_aspect



def _clip_quad(pts: np.ndarray, frame_shape) -> np.ndarray:
    H, W = frame_shape[:2]
    q = np.asarray(pts, np.float32).reshape(4, 2).copy()
    q[:, 0] = np.clip(q[:, 0], 0, W - 1)
    q[:, 1] = np.clip(q[:, 1], 0, H - 1)
    return q

def _inliers_from_mask(dst_pts: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    dst_xy_all = dst_pts.reshape(-1, 2)
    if mask is None:
        return dst_xy_all
    m = mask.ravel().astype(bool)
    return dst_xy_all[m] if m.any() else dst_xy_all

def _points_inside_fraction(quad: np.ndarray, pts_xy: np.ndarray) -> float:
    if pts_xy.size == 0:
        return 0.0
    quad_cnt = quad.reshape(-1, 1, 2).astype(np.float32)
    inside = 0
    for p in pts_xy:
        if cv2.pointPolygonTest(quad_cnt, (float(p[0]), float(p[1])), False) >= 0:
            inside += 1
    return inside / float(len(pts_xy))

def _score_quad(quad: np.ndarray, inliers_xy: np.ndarray, frame_shape, *, aspect_target: Optional[float] = None, hull_area: Optional[float] = None, hull_overlap: Optional[float] = None) -> float:
    H, W = frame_shape[:2]
    frame_area = float(H * W)
    q = quad.astype(np.float32).reshape(4, 2)

    frac = _points_inside_fraction(q, inliers_xy) if inliers_xy.size > 0 else 0.0

    area = abs(cv2.contourArea(q))
    area_ratio = min(1.0, area / frame_area if frame_area > 0 else 0.0)
    score = frac + 0.06 * area_ratio

    if aspect_target:
        def dist(a, b): return math.hypot(float(a[0] - b[0]), float(a[1] - b[1]))
        tl, tr, br, bl = q
        width  = (dist(tl, tr) + dist(bl, br)) / 2.0
        height = (dist(tl, bl) + dist(tr, br)) / 2.0
        if width > 1e-3 and height > 1e-3:
            aspect = height / width
            delta = abs(math.log(max(1e-6, aspect / float(aspect_target))))
            score -= 0.30 * min(1.5, delta)

    if hull_area is not None and hull_area > 1.0 and area > 1.0:
        ratio = float(area / hull_area)
        score -= 0.18 * min(2.0, abs(math.log(ratio)))

    if hull_overlap is not None:
        score += 0.30 * min(1.0, hull_overlap)

    return score

def _pca_stats(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
    pts = np.asarray(pts, np.float32)
    if pts.shape[0] < 2:
        return np.zeros(2, np.float32), np.eye(2, dtype=np.float32), 0.0, 0.0
    mean, vecs = cv2.PCACompute(pts, mean=None)
    mean = mean.ravel().astype(np.float32)
    proj = (pts - mean) @ vecs.T
    s1 = float(np.std(proj[:, 0])) + 1e-6
    s2 = float(np.std(proj[:, 1])) + 1e-6
    return mean, vecs, max(s1, s2), min(s1, s2)

def _quad_centroid(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, np.float32).reshape(4, 2)
    return q.mean(axis=0)

def _scale_quad_to_min_area(pts: np.ndarray, frame_shape, min_area_ratio: float) -> np.ndarray:
    H, W = frame_shape[:2]
    req_area = float(H * W) * float(min_area_ratio)
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    base_area = abs(cv2.contourArea(pts))
    if base_area < 1e-6:
        return pts
    if base_area < req_area:
        s = math.sqrt((req_area * 1.02) / (base_area + 1e-6))
        c = pts.mean(axis=0)
        pts = (pts - c) * s + c
    pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)
    return pts.astype(np.float32)

# ----------------------------------------------------------------------------- #
# Rect fitting helpers                                                           #
# ----------------------------------------------------------------------------- #

def _rect_from_pca(dst_xy: np.ndarray, aspect: float, margin: float = 1.12) -> Optional[np.ndarray]:
    dst_xy = np.asarray(dst_xy, np.float32)
    if dst_xy.shape[0] < 4:
        return None
    mean, vecs = cv2.PCACompute(dst_xy, mean=None)
    mean = mean.ravel()
    proj = (dst_xy - mean) @ vecs.T
    l1 = float(np.quantile(np.abs(proj[:, 0]), 0.95))
    l2 = float(np.quantile(np.abs(proj[:, 1]), 0.95))
    if l1 < 1.0 or l2 < 1.0:
        return None
    A = float(aspect) if aspect > 1e-6 else 1.4
    if (l2 / l1) < A: l2 = A * l1
    else:             l1 = l2 / A
    l1 *= margin; l2 *= margin
    rect_local = np.array([[ l1,  l2],[ l1, -l2],[-l1, -l2],[-l1,  l2]], np.float32)
    pts = rect_local @ vecs + mean
    return order_corners_clockwise(pts.astype(np.float32))

def _fixed_aspect_rect_from_points(dst_xy: np.ndarray, aspect: float, cover_q: float = 0.90, margin: float = 1.02) -> Optional[np.ndarray]:
    pts = np.asarray(dst_xy, np.float32)
    if pts.shape[0] < 4:
        return None
    mean, vecs = cv2.PCACompute(pts, mean=None)
    mean = mean.ravel()
    proj = (pts - mean) @ vecs.T
    u = np.abs(proj[:, 0]); v = np.abs(proj[:, 1])
    u_q = float(np.quantile(u, cover_q)); v_q = float(np.quantile(v, cover_q))
    if u_q < 1.0 and v_q < 1.0:
        return None
    A = float(aspect) if aspect > 1e-6 else 1.4
    l1 = max(u_q, v_q / A); l2 = A * l1
    l1 *= margin; l2 *= margin
    rect_local = np.array([[ l1,  l2],[ l1, -l2],[-l1, -l2],[-l1,  l2]], np.float32)
    quad = rect_local @ vecs + mean
    return order_corners_clockwise(quad.astype(np.float32))

# ----------------------------------------------------------------------------- #
# MTG-card likeness helpers (orientation invariant aspect + border contrast)     #
# ----------------------------------------------------------------------------- #

def _quad_hw(quad: np.ndarray) -> Tuple[float, float]:
    q = np.asarray(quad, np.float32).reshape(4, 2)
    def d(a, b) -> float: return math.hypot(float(a[0]-b[0]), float(a[1]-b[1]))
    tl, tr, br, bl = q
    w = 0.5 * (d(tl, tr) + d(bl, br))
    h = 0.5 * (d(tl, bl) + d(tr, br))
    return h, w

def _orientation_invariant_aspect(h: float, w: float) -> float:
    if h <= 1e-3 or w <= 1e-3:
        return 0.0
    a = h / w
    return max(a, 1.0 / a)

def _warp_quad(frame: np.ndarray, quad: np.ndarray, out_hw=(1050, 750)) -> np.ndarray:
    Ht, Wt = out_hw
    dst = np.float32([[0,0],[Wt-1,0],[Wt-1,Ht-1],[0,Ht-1]])
    M = cv2.getPerspectiveTransform(np.asarray(quad, np.float32), dst)
    return cv2.warpPerspective(frame, M, (Wt, Ht))

def _expand_quad(frame_shape, quad: np.ndarray, cfg: Dict) -> np.ndarray:
    """Scale quad about its centroid toward target area and card aspect, clipped to frame."""
    H, W = frame_shape[:2]
    q = np.asarray(quad, np.float32).reshape(4, 2)
    center = q.mean(axis=0)
    area = abs(cv2.contourArea(q)) + 1e-6
    frame_area = float(H * W)
    target = cfg.get("target_area_ratio", None)
    card_aspect = float(cfg.get("card_aspect", 1.395))
    max_scale = float(cfg.get("quad_expand_max_scale", 1.40))
    aspect_w = float(cfg.get("quad_expand_aspect_weight", 0.35))

    # Base isotropic scale from target area
    scale_iso = 1.00
    if target:
        desired_area = float(target) * frame_area
        if desired_area > area:
            scale_iso = max(scale_iso, math.sqrt(desired_area / area))

    # Anisotropic tweak toward target aspect
    h, w = _quad_hw(q)
    cur_aspect = (h / (w + 1e-6)) if w > 1e-6 else card_aspect
    scale_h = scale_iso
    scale_w = scale_iso
    if cur_aspect < card_aspect:  # too wide → grow height more
        factor = min(max_scale, 1.0 + aspect_w * (card_aspect / (cur_aspect + 1e-6) - 1.0))
        scale_h *= factor
    elif cur_aspect > card_aspect:  # too tall → grow width more
        factor = min(max_scale, 1.0 + aspect_w * (cur_aspect / card_aspect - 1.0))
        scale_w *= factor

    scale_h = min(max_scale, scale_h)
    scale_w = min(max_scale, scale_w)

    q[:, 0] = (q[:, 0] - center[0]) * scale_w + center[0]
    q[:, 1] = (q[:, 1] - center[1]) * scale_h + center[1]
    q[:, 0] = np.clip(q[:, 0], 0, W - 1)
    q[:, 1] = np.clip(q[:, 1], 0, H - 1)
    return q.astype(np.float32)


def _refine_quad(frame_shape, quad: np.ndarray, cfg: Dict) -> np.ndarray:
    """Square up quad using minAreaRect orientation, target aspect and area."""
    H, W = frame_shape[:2]
    q = np.asarray(quad, np.float32).reshape(4, 2)
    area = abs(cv2.contourArea(q)) + 1e-6
    frame_area = float(H * W)
    target_area = cfg.get("target_area_ratio", None)
    card_aspect = float(cfg.get("card_aspect", 1.395))
    max_scale = float(cfg.get("quad_refine_scale_max", 1.65))

    rect = cv2.minAreaRect(q)
    center = rect[0]
    w0, h0 = rect[1]
    angle = rect[2]
    if h0 < w0:
        w0, h0 = h0, w0
        angle += 90.0

    desired_area = area * 1.02
    if target_area:
        desired_area = max(desired_area, float(target_area) * frame_area)

    h_new = math.sqrt(desired_area * card_aspect)
    w_new = h_new / card_aspect
    # slight oversize to include full border (keep tiny to avoid blow-up)
    h_new *= 1.01
    w_new *= 1.01

    # Cap scaling
    scale_cap = max_scale * math.sqrt(area / desired_area)
    h_new = min(h_new, h0 * max_scale)
    w_new = min(w_new, w0 * max_scale)

    box = cv2.boxPoints((center, (w_new, h_new), angle))
    box = order_corners_clockwise(box.astype(np.float32))
    box[:, 0] = np.clip(box[:, 0], 0, W - 1)
    box[:, 1] = np.clip(box[:, 1], 0, H - 1)
    return box.astype(np.float32)


def _local_edge_refine(frame: np.ndarray, quad: np.ndarray, cfg: Dict) -> np.ndarray:
    """Align quad to strongest edge contour inside a small ROI around it."""
    if frame is None or quad is None:
        return quad
    H, W = frame.shape[:2]
    q = np.asarray(quad, np.float32).reshape(4, 2)
    margin = int(cfg.get("local_refine", {}).get("margin_px", 24))
    x0 = max(0, int(np.floor(q[:, 0].min() - margin)))
    x1 = min(W - 1, int(np.ceil(q[:, 0].max() + margin)))
    y0 = max(0, int(np.floor(q[:, 1].min() - margin)))
    y1 = min(H - 1, int(np.ceil(q[:, 1].max() + margin)))
    if x1 <= x0 or y1 <= y0:
        return quad

    roi = frame[y0:y1+1, x0:x1+1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    edges = cv2.Canny(clahe, 40, 160)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), 1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return quad

    # Local pass should not reuse the global min_abs_area_px gate (cards are
    # typically < 2.8M px in our 12MP frame), otherwise we never snap to the
    # real border and stay on the inner art. Use a relaxed copy.
    local_cfg = dict(cfg)
    local_cfg["min_abs_area_px"] = min(float(cfg.get("min_abs_area_px", 0.0)), 450_000.0)
    local_cfg["min_area_ratio"] = min(float(cfg.get("min_area_ratio", 0.0)), 0.001)

    # pick largest plausible contour in ROI
    frame_area = float(H * W)
    orig_area = float(abs(cv2.contourArea(q))) + 1e-6
    card_aspect = float(cfg.get("card_aspect", 1.395))
    best = None
    best_area = 0.0
    for c in cnts:
        if len(c) < 4:
            continue
        c_area = float(cv2.contourArea(c))
        if c_area < best_area:
            continue
        # reject contours that are wildly smaller/larger than the current quad
        ratio = c_area / orig_area
        if ratio < 0.65 or ratio > 1.35:
            continue
        r = cv2.minAreaRect(c)
        box = cv2.boxPoints(r).astype(np.float32)
        box[:, 0] += x0
        box[:, 1] += y0
        if not is_plausible_quad(box, frame.shape, local_cfg):
            continue
        # additional aspect sanity: stay within a reasonable card-like band
        h_new, w_new = _quad_hw(box)
        aspect = h_new / (w_new + 1e-6)
        if aspect < card_aspect * 0.72 or aspect > card_aspect * 1.32:
            continue
        best = order_corners_clockwise(box)
        best_area = c_area

    return best.astype(np.float32) if best is not None else quad


def _upright_snap(frame_shape, quad: np.ndarray, cfg: Dict) -> np.ndarray:
    """Force quad to axis-aligned card aspect box centered at current centroid."""
    if quad is None:
        return quad
    H, W = frame_shape[:2]
    q = np.asarray(quad, np.float32).reshape(4, 2)
    center = q.mean(axis=0)
    area = abs(cv2.contourArea(q)) + 1e-6
    card_aspect = float(cfg.get("card_aspect", 1.395))
    frame_area = float(H * W)
    target = cfg.get("target_area_ratio", None)

    desired_area = area
    if target:
        desired_area = max(desired_area, float(target) * frame_area)
    h_new = math.sqrt(desired_area * card_aspect)
    w_new = h_new / card_aspect

    # build axis-aligned box
    half_w = 0.5 * w_new
    half_h = 0.5 * h_new
    box = np.array([
        [center[0] - half_w, center[1] - half_h],
        [center[0] + half_w, center[1] - half_h],
        [center[0] + half_w, center[1] + half_h],
        [center[0] - half_w, center[1] + half_h],
    ], dtype=np.float32)
    box[:, 0] = np.clip(box[:, 0], 0, W - 1)
    box[:, 1] = np.clip(box[:, 1], 0, H - 1)
    return box


def _apply_perspective_taper(frame_shape, quad: np.ndarray, cfg: Dict) -> np.ndarray:
    """Slightly taper quad horizontally: widen bottom vs top to compensate camera tilt."""
    t = float(cfg.get("perspective_taper", 0.0))
    if abs(t) < 1e-4:
        return quad
    H, W = frame_shape[:2]
    q = np.asarray(quad, np.float32).reshape(4, 2)
    cy = q[:, 1].mean()
    # NumPy 2.0 removed ndarray.ptp; use np.ptp for compatibility
    y_span = max(1.0, float(np.ptp(q[:, 1])))
    tapered = q.copy()
    for i, (x, y) in enumerate(q):
        # normalize y around center: -0.5 at top, +0.5 at bottom
        yn = (y - cy) / y_span
        scale = 1.0 + t * yn
        tapered[i, 0] = (x - q[:, 0].mean()) * scale + q[:, 0].mean()
    tapered[:, 0] = np.clip(tapered[:, 0], 0, W - 1)
    tapered[:, 1] = np.clip(tapered[:, 1], 0, H - 1)
    return tapered.astype(np.float32)

def _filled_mask_from_edges(edges: np.ndarray, close_ksize: int = 7) -> np.ndarray:
    """Close edge gaps, flood-fill background, and return filled foreground mask."""
    ksize = max(3, close_ksize | 1)
    kernel = np.ones((ksize, ksize), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    closed = cv2.dilate(closed, np.ones((3, 3), np.uint8), 1)
    h, w = closed.shape
    filled = closed.copy()
    cv2.floodFill(filled, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 255)
    filled_inv = cv2.bitwise_not(filled)
    return cv2.morphologyEx(filled_inv, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)


def _save_quad_if_needed(quad: np.ndarray, cfg: Dict):
    path = cfg.get("save_last_quad_path", None)
    if not path:
        return
    try:
        import json, os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(quad.reshape(4, 2).tolist(), f)
    except Exception:
        pass

def _border_contrast_score(warp: np.ndarray) -> float:
    g = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
    H, W = g.shape
    m   = max(2, int(0.02 * min(H, W)))      # outer ring thickness
    i1  = max(6, int(0.06 * min(H, W)))      # inner ring start
    i2  = max(i1 + 1, int(0.12 * min(H, W))) # inner ring end

    outer = np.zeros_like(g, np.uint8)
    inner = np.zeros_like(g, np.uint8)

    cv2.rectangle(outer, (0,0), (W-1, H-1), 255, thickness=m)
    cv2.rectangle(inner, (i1,i1), (W-1-i1, H-1-i1), 255, thickness=i2 - i1)

    mo = float(g[outer > 0].mean()) if (outer > 0).any() else 255.0
    mi = float(g[inner > 0].mean()) if (inner > 0).any() else 0.0

    diff = max(0.0, (mi - mo))  # positive if border darker
    rng  = max(1.0, float(g.max() - g.min()))
    return float(np.clip(diff / rng * 2.5, 0.0, 1.0))

def _choose_best_quad(cnts, frame: np.ndarray, cfg: Dict) -> Optional[Corners]:
    """Pick the most plausible card quad from a list of contours.

    Strategy:
    - ignore contours touching the image border (likely the page / tray)
    - approximate each contour to 4 points (or minAreaRect fallback)
    - run is_plausible_quad() (area + aspect + portrait) as the main gate
    - extra MTG check via is_mtg_card_like()
    - rank remaining quads by:
        1) area closeness to target_area_ratio (if provided)
        2) larger area preferred
    """
    if not cnts:
        return None

    H, W = frame.shape[:2]
    frame_area = float(H * W)
    target_area = cfg.get("target_area_ratio")
    if target_area is not None:
        target_area = float(target_area)
    eps = 1e-6

    eps_scale = float(cfg.get("max_quad_epsilon", 0.08))
    max_area_ratio = float(cfg.get("max_area_ratio", 0.95))
    border_margin = int(cfg.get("border_margin_px", 6))

    # biggest contours first
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    def touches_border(rect):
        x, y, w, h = rect
        if x <= border_margin or y <= border_margin:
            return True
        if x + w >= W - border_margin or y + h >= H - border_margin:
            return True
        return False

    candidates: List[Tuple[np.ndarray, float, float]] = []
    big_rejected_once = False

    for c in cnts:
        if len(c) < 4:
            continue

        rect = cv2.boundingRect(c)

        # Only skip border-touching contours if they are GIANT
        if touches_border(rect):
            area_c = float(cv2.contourArea(c))
            area_ratio_c = area_c / frame_area if frame_area > 0 else 0.0

            # If it covers more than ~40% of the frame, it's tray/background
            if area_ratio_c > 0.40:
                if cfg.get("debug"):
                    print(f"[contours] skip border-touching giant: area_ratio={area_ratio_c:.3f}")
                continue
            # otherwise keep it (card touching image edge, etc.)

        peri = cv2.arcLength(c, True)

        # Try direct 4-point approx; otherwise force to 4 points; then minAreaRect
        pts = None
        approx = cv2.approxPolyDP(c, eps_scale * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype(np.float32)
        else:
            for factor in (1.5, 2.0, 3.0):
                approx2 = cv2.approxPolyDP(c, eps_scale * factor * peri, True)
                if len(approx2) == 4:
                    pts = approx2.reshape(4, 2).astype(np.float32)
                    break
            if pts is None:
                box = cv2.boxPoints(cv2.minAreaRect(c))
                pts = box.astype(np.float32)

        pts = order_corners_clockwise(pts)

        # Drop anything that is clearly the whole page / huge region
        area = float(abs(cv2.contourArea(pts)))
        area_ratio = area / frame_area if frame_area > 0 else 0.0
        if area_ratio > max_area_ratio:
            if not big_rejected_once and cfg.get("debug"):
                print(f"[contours] skip giant: area_ratio={area_ratio:.3f} > {max_area_ratio}")
            big_rejected_once = True
            continue

        # Main plausibility gate (area + aspect + portrait etc.)
        if not is_plausible_quad(pts, frame.shape, cfg):
            continue

        # Extra MTG-card gate: orientation-invariant aspect + dark outer border
        mtg_ok, bscore = is_mtg_card_like(frame, pts, cfg)
        if cfg.get("debug"):
            print(f"[plaus] mtg_like={mtg_ok}, border_score={bscore:.3f}")
        if not mtg_ok:
            continue

        candidates.append((pts, area, area_ratio))

    # --- pick best candidate -------------------------------------------------
    if not candidates:
        return None

    best_quad = None
    best_score = -1.0

    for quad, area, area_ratio in candidates:
        if target_area is not None:
            # score by closeness to target area + prefer larger
            score = -abs(area_ratio - target_area) + 0.05 * area_ratio
        else:
            # simple: prefer larger quads
            score = area_ratio

        if score > best_score + eps:
            best_score = score
            best_quad = quad

    if best_quad is None:
        return None

    return Corners(pts=best_quad.astype(np.float32))


def _largest_portrait_quad(cnts, frame: np.ndarray, cfg: Dict) -> Optional[Corners]:
    """Fallback: pick the largest portrait-ish quad ignoring MTG border score.
    Useful when the card outline is clear in edges/binary but MTG contrast is weak."""
    if not cnts:
        return None
    H, W = frame.shape[:2]
    frame_area = float(H * W)
    min_abs = float(cfg.get("fallback_min_abs_area_px", 1_200_000.0))
    min_ratio = float(cfg.get("fallback_min_area_ratio", 0.12))
    target = cfg.get("target_area_ratio", 0.30)
    border_margin = int(cfg.get("border_margin_px", 6))

    def touches_border(rect):
        x, y, w, h = rect
        return (x <= border_margin or y <= border_margin or
                x + w >= W - border_margin or y + h >= H - border_margin)

    best = None
    best_score = -1.0
    for c in cnts:
        if len(c) < 4:
            continue
        if touches_border(cv2.boundingRect(c)):
            continue
        area = float(abs(cv2.contourArea(c)))
        if area < min_abs:
            continue
        area_ratio = area / (frame_area + 1e-6)
        if area_ratio < min_ratio:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, float(cfg.get("max_quad_epsilon", 0.08)) * peri, True)
        if len(approx) != 4:
            approx = cv2.boxPoints(cv2.minAreaRect(c)).astype(np.float32).reshape(-1, 1, 2)
        pts = order_corners_clockwise(approx.reshape(4, 2).astype(np.float32))
        h, w = _quad_hw(pts)
        if h <= w:  # portrait only
            continue
        score = area_ratio
        if target is not None:
            score -= 0.8 * abs(area_ratio - float(target))
        # prefer aspect near expected card aspect
        card_aspect = float(cfg.get("card_aspect", 1.4))
        score -= 0.5 * abs((h / (w + 1e-6)) - card_aspect)
        if score > best_score:
            best_score = score
            best = pts
    return Corners(pts=best.astype(np.float32)) if best is not None else None

def is_mtg_card_like(frame: np.ndarray, quad: np.ndarray, cfg: Optional[Dict] = None) -> Tuple[bool, float]:
    """
    Extra orientation-invariant gate for MTG card shape + dark outer border.
    Returns (ok, border_score[0..1]).
    """
    cfg = _merge_cfg(cfg)
    h, w = _quad_hw(quad)
    A = _orientation_invariant_aspect(h, w)
    A0  = float(cfg.get("card_aspect", _DEFAULT_CFG["card_aspect"]))
    tol = float(cfg.get("card_aspect_tol", _DEFAULT_CFG["card_aspect_tol"]))
    a_min, a_max = (A0 * (1.0 - tol), A0 * (1.0 + tol))
    if not (a_min <= A <= a_max):
        return False, 0.0

    warp   = _warp_quad(frame, quad, out_hw=(1050, 750))
    bscore = _border_contrast_score(warp)
    return (bscore >= float(cfg.get("card_border_min", _DEFAULT_CFG["card_border_min"]))), bscore

# ----------------------------------------------------------------------------- #
# Contour path                                                                   #
# ----------------------------------------------------------------------------- #

def detect_by_contours(frame: np.ndarray, cfg: Optional[Dict] = None) -> Optional[Corners]:
    cfg = _merge_cfg(cfg)
    H, W = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --- Saturation mask: grab colorful/dark card vs white tray -----------------
    def _saturation_mask() -> Optional[np.ndarray]:
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            s = hsv[:, :, 1]
        except Exception:
            return None
        low = int(cfg["saturation"].get("low", 40))
        mask = cv2.inRange(s, low, 255)
        b = int(cfg["saturation"].get("blur", 3))
        if b > 1:
            if b % 2 == 0: b += 1
            mask = cv2.GaussianBlur(mask, (b, b), 0)
        m = int(cfg["saturation"].get("morph", 7))
        if m > 1:
            k = np.ones((m, m), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
            mask = cv2.dilate(mask, k, iterations=1)
        return mask

    sat_mask = _saturation_mask()
    if sat_mask is not None:
        cnts, _ = cv2.findContours(sat_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        min_ratio_sat = float(cfg["saturation"].get("min_area_ratio", cfg["min_area_ratio"]))
        frame_area = float(H * W)
        cnts = [c for c in cnts if cv2.contourArea(c) / (frame_area + 1e-6) >= min_ratio_sat]
        corners = _choose_best_quad(cnts, frame, cfg)
        if corners is not None:
            return corners
        fallback = _largest_portrait_quad(cnts, frame, cfg)
        if fallback is not None:
            return fallback

    # --- Try to isolate the dark outer border before generic binarization ---
    def _dark_border_mask() -> Optional[np.ndarray]:
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            v = hsv[:, :, 2]
        except Exception:
            v = gray
        eq = cv2.equalizeHist(v)
        p = float(cfg["dark_border"].get("percentile", 30))
        p = np.clip(p, 5.0, 60.0)
        thr = int(np.percentile(eq, p))
        thr = int(np.clip(thr + 2, 8, 220))
        mask = cv2.inRange(eq, 0, thr)
        b = int(cfg["dark_border"].get("blur", 3))
        if b > 1:
            if b % 2 == 0:
                b += 1
            mask = cv2.GaussianBlur(mask, (b, b), 0)
        m = int(cfg["dark_border"].get("morph", 5))
        if m > 1:
            kernel = np.ones((m, m), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=1)
        return mask

    dark_mask = _dark_border_mask()
    if dark_mask is not None:
        cnts, _ = cv2.findContours(dark_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        min_ratio_dark = float(cfg["dark_border"].get("min_area_ratio", cfg["min_area_ratio"]))
        frame_area = float(H * W)
        border_margin = int(cfg.get("border_margin_px", 6))
        def _touches_border(rect):
            x, y, w, h = rect
            return (x <= border_margin or y <= border_margin or
                    x + w >= W - border_margin or y + h >= H - border_margin)

        filtered = []
        for c in cnts:
            area_ratio = cv2.contourArea(c) / (frame_area + 1e-6)
            if area_ratio < min_ratio_dark:
                continue
            if _touches_border(cv2.boundingRect(c)):
                continue
            filtered.append(c)
        cnts = filtered
        corners = _choose_best_quad(cnts, frame, cfg)
        if corners is not None:
            return corners

    # light blur
    k = int(cfg["blur"]["ksize"])
    if k > 1:
        if k % 2 == 0:
            k += 1
        gray_blur = cv2.GaussianBlur(gray, (k, k), 0)
    else:
        gray_blur = gray

    def _binary_candidates(g):
        _, b1 = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, b2 = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        b3 = cv2.adaptiveThreshold(
            g, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31, 5
        )
        return [b1, b2, b3]

    # Try binary masks first
    for b in _binary_candidates(gray_blur):
        b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), 1)
        cnts, _ = cv2.findContours(b, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        corners = _choose_best_quad(cnts, frame, cfg)
        if corners is not None:
            return corners
        fallback = _largest_portrait_quad(cnts, frame, cfg)
        if fallback is not None:
            return fallback

    # If no result → try edges (CLAHE helps separate dark border from bright tray)
    edge_src = gray_blur
    if cfg["canny"].get("clahe", False):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        edge_src = clahe.apply(gray_blur)
    edges = cv2.Canny(edge_src, cfg["canny"]["low"], cfg["canny"]["high"])
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), 1)
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    c = _choose_best_quad(cnts, frame, cfg)
    if c is not None:
        return c
    fallback = _largest_portrait_quad(cnts, frame, cfg)
    if fallback is not None:
        return fallback

    # Edge-fill fallback: close gaps, fill foreground blobs, then contour again
    filled = _filled_mask_from_edges(edges, close_ksize=9)
    cnts, _ = cv2.findContours(filled, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    c = _choose_best_quad(cnts, frame, cfg)
    if c is not None:
        return c
    return _largest_portrait_quad(cnts, frame, cfg)

# ----------------------------------------------------------------------------- #
# Match clustering (k-means)                                                     #
# ----------------------------------------------------------------------------- #

def _best_kmeans_cluster(dst_xy_all: np.ndarray, debug: bool = False) -> Optional[np.ndarray]:
    pts = np.asarray(dst_xy_all, np.float32)
    N = pts.shape[0]
    if N < 8: return None
    best_pts = None; best_score = float("inf")
    for K in (3, 2):
        if N < K: continue
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 1.0)
        compactness, labels, centers = cv2.kmeans(pts, K, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        labels = labels.ravel()
        for j in range(K):
            cl = pts[labels == j]; m = cl.shape[0]
            if m < max(20, int(0.02 * N)): continue
            mean, vecs = cv2.PCACompute(cl, mean=None)
            proj = (cl - mean.ravel()) @ vecs.T
            l1 = float(np.var(proj[:, 0])) + 1e-6
            l2 = float(np.var(proj[:, 1])) + 1e-6
            spread = l1 * l2
            score = spread / (m ** 1.2)
            if score < best_score: best_score = score; best_pts = cl.copy()
    if debug and best_pts is None: print("[features] kmeans: no suitable cluster")
    return best_pts

# ----------------------------------------------------------------------------- #
# Rasterized hull-overlap                                                        #
# ----------------------------------------------------------------------------- #

def _hull_overlap_score(frame_shape: Tuple[int, int, int], hull: np.ndarray, quad: np.ndarray) -> float:
    H, W = frame_shape[:2]
    if hull is None or len(hull) < 3: return 0.0
    hull_mask = np.zeros((H, W), np.uint8)
    quad_mask = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(hull_mask, hull.reshape(-1, 2).astype(np.int32), 255)
    cv2.fillConvexPoly(quad_mask,  quad.reshape(-1, 2).astype(np.int32), 255)
    inter = cv2.countNonZero(cv2.bitwise_and(hull_mask, quad_mask))
    hull_area = cv2.countNonZero(hull_mask)
    if hull_area <= 0: return 0.0
    return float(inter) / float(hull_area)

# ----------------------------------------------------------------------------- #
# Feature path                                                                   #
# ----------------------------------------------------------------------------- #

def detect_by_features(frame: np.ndarray, template: Optional[np.ndarray], cfg: Optional[Dict] = None) -> Optional[Corners]:
    if template is None: return None
    cfg = _merge_cfg(cfg)

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    templ_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    Hf, Wf = frame.shape[:2]
    Ht, Wt = template.shape[:2]
    aspect = (float(Ht) / float(Wt)) if Wt > 0 else 1.4

    orb = cv2.ORB_create(nfeatures=int(cfg["feature"]["nfeatures"]))
    kpf, desf = orb.detectAndCompute(frame_gray, None)
    kpt, dest = orb.detectAndCompute(templ_gray, None)
    if desf is None or dest is None or len(kpf) == 0 or len(kpt) == 0: return None

    ratio  = float(cfg["feature"]["ratio"])
    ransac = float(cfg["feature"]["ransac_thresh"])
    min_inl = int(cfg["feature"]["min_inliers"])

    # --- Matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(dest, desf, k=2)
    good: List[cv2.DMatch] = []
    for pair in knn:
        if len(pair) < 2: continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)

    # Cross-check (strict) + mutual-good intersection
    bf_x = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    mm_x = bf_x.match(dest, desf)  # strict symmetric matches
    set_good: Set[Tuple[int,int]] = {(m.queryIdx, m.trainIdx) for m in good}
    mutual = [m for m in mm_x if (m.queryIdx, m.trainIdx) in set_good]

    if cfg.get("debug"):
        print("[features] kpt_template:", len(kpt), "kpt_frame:", len(kpf),
              f"good:{len(good)} cross:{len(mm_x)} mutual:{len(mutual)}")

    if len(good) < 4 and len(mutual) < 4:
        return None

    src_pts = np.float32([kpt[m.queryIdx].pt for m in good]).reshape(-1, 1, 2) if len(good) >= 4 else None
    dst_pts = np.float32([kpf[m.trainIdx].pt for m in good]).reshape(-1, 1, 2) if len(good) >= 4 else None
    src_pts_mut = np.float32([kpt[m.queryIdx].pt for m in mutual]).reshape(-1, 1, 2) if len(mutual) >= 4 else None
    dst_pts_mut = np.float32([kpf[m.trainIdx].pt for m in mutual]).reshape(-1, 1, 2) if len(mutual) >= 4 else None

    # --- Prefer a homography from MUTUAL-GOOD matches when it looks solid
    if src_pts_mut is not None and dst_pts_mut is not None:
        Hm, maskm = cv2.findHomography(src_pts_mut, dst_pts_mut, cv2.RANSAC, ransacReprojThreshold=max(3.0, 0.7 * ransac))
        if Hm is not None and maskm is not None and int(maskm.sum()) >= max(min_inl, 8):
            templ_corners = np.float32([[0,0],[Wt-1,0],[Wt-1,Ht-1],[0,Ht-1]]).reshape(-1,1,2)
            mapped_m = cv2.perspectiveTransform(templ_corners, Hm).reshape(4,2).astype(np.float32)
            mapped_m = _clip_quad(order_corners_clockwise(mapped_m), frame.shape)

            # Build support hull from mutual inliers only
            inl_xy_m = _inliers_from_mask(dst_pts_mut, maskm)
            hull_m = None
            if inl_xy_m.shape[0] >= 3:
                hull_m = cv2.convexHull(inl_xy_m.astype(np.float32).reshape(-1,1,2))
            ov_m = _hull_overlap_score(frame.shape, hull_m, mapped_m) if hull_m is not None else 0.0

            if cfg.get("debug"):
                print(f"[mutual] inliers={int(maskm.sum())} hullOv={ov_m:.2f}")

            if ov_m >= 0.65:
                return Corners(pts=mapped_m.astype(np.float32))

    # --- If mutual was weak, proceed with the original (loose) pipeline
    dst_xy_all = dst_pts.reshape(-1, 2) if dst_pts is not None else np.empty((0,2), np.float32)
    candidates: List[np.ndarray] = []

    Hmat, mask = (cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac) if (src_pts is not None and dst_pts is not None) else (None, None))
    inliers_xy = _inliers_from_mask(dst_pts, mask) if (dst_pts is not None) else np.empty((0,2), np.float32)

    if Hmat is not None and mask is not None and int(mask.sum()) >= min_inl:
        templ_corners = np.float32([[0, 0], [Wt - 1, 0], [Wt - 1, Ht - 1], [0, Ht - 1]]).reshape(-1, 1, 2)
        mapped = cv2.perspectiveTransform(templ_corners, Hmat).reshape(4, 2)
        candA = _clip_quad(order_corners_clockwise(mapped.astype(np.float32)), frame.shape)
        candidates.append(candA)

        area_after_clip = abs(cv2.contourArea(candA))
        if area_after_clip < 64.0 and cfg.get("debug"):
            print("[features] homography quad degenerate; rebuilding via perimeter→PCA→minAreaRect")

        if area_after_clip < 64.0:
            n = 60
            top    = np.stack([np.linspace(0, Wt - 1, n), np.zeros(n)], axis=1)
            right  = np.stack([np.full(n, Wt - 1), np.linspace(0, Ht - 1, n)], axis=1)
            bottom = np.stack([np.linspace(Wt - 1, 0, n), np.full(n, Ht - 1)], axis=1)
            left   = np.stack([np.zeros(n), np.linspace(Ht - 1, 0, n)], axis=1)
            perim  = np.vstack([top, right, bottom, left]).astype(np.float32).reshape(-1, 1, 2)
            mapped_perim = cv2.perspectiveTransform(perim, Hmat).reshape(-1, 2)
            vis = mapped_perim[
                (mapped_perim[:, 0] >= 0) & (mapped_perim[:, 0] < Wf) &
                (mapped_perim[:, 1] >= 0) & (mapped_perim[:, 1] < Hf)
            ]
            if vis.shape[0] >= 20:
                hull = cv2.convexHull(vis.reshape(-1, 1, 2).astype(np.float32))
                peri_len = cv2.arcLength(hull, True)
                quad = None
                for eps in (0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12):
                    approx = cv2.approxPolyDP(hull, eps * peri_len, True)
                    if len(approx) == 4:
                        quad = approx.reshape(4, 2).astype(np.float32)
                        break
                if quad is not None:
                    candidates.append(_clip_quad(order_corners_clockwise(quad), frame.shape))

            pca_fixed = _fixed_aspect_rect_from_points(inliers_xy, aspect, cover_q=0.90, margin=1.02)
            if pca_fixed is not None:
                candidates.append(_clip_quad(pca_fixed, frame.shape))
            pca_quad = _rect_from_pca(inliers_xy, aspect, margin=1.10)
            if pca_quad is not None:
                candidates.append(_clip_quad(pca_quad, frame.shape))
            if inliers_xy.shape[0] >= 4:
                box = cv2.boxPoints(cv2.minAreaRect(inliers_xy.astype(np.float32)))
                candidates.append(_clip_quad(order_corners_clockwise(box.astype(np.float32)), frame.shape))
    else:
        inliers_xy = dst_xy_all
        pca_fixed = _fixed_aspect_rect_from_points(inliers_xy, aspect, cover_q=0.90, margin=1.02)
        if pca_fixed is not None:
            candidates.append(_clip_quad(pca_fixed, frame.shape))
        else:
            pca_quad = _rect_from_pca(inliers_xy, aspect, margin=1.10)
            if pca_quad is not None:
                candidates.append(_clip_quad(pca_quad, frame.shape))
        if inliers_xy.shape[0] >= 4:
            box = cv2.boxPoints(cv2.minAreaRect(inliers_xy.astype(np.float32)))
            candidates.append(_clip_quad(order_corners_clockwise(box.astype(np.float32)), frame.shape))

    # KMEANS dense cluster
    cluster_xy = _best_kmeans_cluster(dst_xy_all, debug=cfg.get("debug", False))
    if cluster_xy is not None:
        kq = _fixed_aspect_rect_from_points(cluster_xy, aspect, cover_q=0.90, margin=1.02)
        if kq is None:
            kq = _rect_from_pca(cluster_xy, aspect, margin=1.08)
        if kq is None and cluster_xy.shape[0] >= 4:
            kq = cv2.boxPoints(cv2.minAreaRect(cluster_xy.astype(np.float32))).astype(np.float32)
        if kq is not None:
            candidates.append(_clip_quad(order_corners_clockwise(kq), frame.shape))

    if len(candidates) == 0: return None

    # Support set
    scoring_xy = cluster_xy if cluster_xy is not None else inliers_xy
    center, vecs, s_major, s_minor = _pca_stats(scoring_xy)
    rad = max(8.0, 1.5 * s_major)

    # Build inlier/cluster hull
    inlier_hull = None
    if scoring_xy.shape[0] >= 3:
        inlier_hull = cv2.convexHull(scoring_xy.astype(np.float32).reshape(-1, 1, 2))
    hull_area = None
    if inlier_hull is not None and len(inlier_hull) >= 3:
        hull_area = abs(cv2.contourArea(inlier_hull))

    # Hull-overlap early pick (kept)
    hull_overlaps: List[Tuple[np.ndarray, float]] = []
    if inlier_hull is not None and len(inlier_hull) >= 3:
        for q in candidates:
            if not is_plausible_quad(q, frame.shape, cfg): continue
            ov = _hull_overlap_score(frame.shape, inlier_hull, q)
            hull_overlaps.append((q, ov))
            if cfg.get("debug"):
                cov = _points_inside_fraction(q, scoring_xy) if scoring_xy.size > 0 else 0.0
                print(f"[select] cand cov={cov:.2f} hullOv={ov:.2f}")
        if len(hull_overlaps) > 0:
            hull_overlaps.sort(key=lambda x: x[1], reverse=True)
            best_q, best_ov = hull_overlaps[0]
            second_ov = hull_overlaps[1][1] if len(hull_overlaps) > 1 else 0.0
            if (best_ov - second_ov) >= 0.10 or best_ov >= 0.60:
                if cfg.get("debug"):
                    print(f"[select] early pick by hull overlap: best={best_ov:.2f}, second={second_ov:.2f}")
                return Corners(pts=best_q.astype(np.float32))

    # Gating
    n_support = scoring_xy.shape[0]
    cover_thresh = (0.72 if cluster_xy is not None else 0.68) if n_support < 200 else (0.75 if cluster_xy is not None else 0.70)
    filtered: List[np.ndarray] = []
    for q in candidates:
        if not is_plausible_quad(q, frame.shape, cfg): continue
        cover = _points_inside_fraction(q, scoring_xy) if scoring_xy.size > 0 else 0.0
        if cover < cover_thresh:
            if cfg.get("debug"): print(f"[features] drop cand: low coverage {cover:.2f}")
            continue
        if hull_area is not None and hull_area > 1.0:
            qa = abs(cv2.contourArea(q.astype(np.float32)))
            ratio = qa / hull_area
            upper = 1.75 if cluster_xy is not None else 1.60
            lower = 0.60 if cluster_xy is not None else 0.65
            if ratio > upper or ratio < lower:
                if cfg.get("debug"):
                    print(f"[features] drop cand: area ratio {ratio:.2f} (qa={qa:.1f}, hull={hull_area:.1f})")
                continue
        qc = _quad_centroid(q)
        d = float(np.linalg.norm(qc - center))
        if d > 2.5 * rad:
            if cfg.get("debug"): print(f"[features] drop cand: centroid far (d={d:.1f}, rad={rad:.1f})")
            continue
        filtered.append(q)

    if len(filtered) == 0:
        if len(hull_overlaps) > 0:
            if cfg.get("debug"): print("[select] fallback to best hull-overlap")
            return Corners(pts=hull_overlaps[0][0].astype(np.float32))
        return None

    # Final scoring
    best_quad = None; best_score = -1.0
    for q in filtered:
        hull_ov = _hull_overlap_score(frame.shape, inlier_hull, q) if (inlier_hull is not None and len(inlier_hull) >= 3) else None
        s = _score_quad(q, scoring_xy, frame.shape, aspect_target=aspect, hull_area=hull_area, hull_overlap=hull_ov)
        qc = _quad_centroid(q)
        d_norm = float(np.linalg.norm(qc - center)) / (rad + 1e-6)
        s -= 0.45 * min(2.0, d_norm)
        if s > best_score: best_score = s; best_quad = q

    # --- Final scoring: add hull-overlap bonus & stronger center bias (duplicate kept)
    best_quad = None
    best_score = -1.0
    scoring_for_score = scoring_xy
    for q in filtered:
        hull_ov = None
        if inlier_hull is not None and len(inlier_hull) >= 3:
            hull_ov = _hull_overlap_score(frame.shape, inlier_hull, q)
        s = _score_quad(q, scoring_for_score, frame.shape, aspect_target=aspect, hull_area=hull_area, hull_overlap=hull_ov)

        qc = _quad_centroid(q)
        d_norm = float(np.linalg.norm(qc - center)) / (rad + 1e-6)
        s -= 0.45 * min(2.0, d_norm)

        if s > best_score:
            best_score = s
            best_quad = q

    if best_quad is None:
        return None

    return Corners(pts=best_quad.astype(np.float32))

# ----------------------------------------------------------------------------- #
# Public entrypoint                                                              #
# ----------------------------------------------------------------------------- #

def _relaxed_cfg(cfg: Dict) -> Dict:
    r = dict(cfg)
    relax = cfg.get("relax", {})

    r["min_area_ratio"] = min(float(cfg.get("min_area_ratio", 0.005)),
                              float(relax.get("min_area_ratio", 0.003)))

    r["min_abs_area_px"] = min(float(cfg.get("min_abs_area_px", 2000.0)),
                               float(relax.get("min_abs_area_px", 2000.0)))

    lo, hi = cfg.get("aspect_range", (0.40, 3.00))
    r_lo, r_hi = relax.get("aspect_range", (0.40, 3.00))
    r["aspect_range"] = (min(lo, r_lo), max(hi, r_hi))

    r["max_quad_epsilon"] = max(float(cfg.get("max_quad_epsilon", 0.08)),
                                float(relax.get("max_quad_epsilon", 0.08)))

    r["min_solidity"] = min(float(cfg.get("min_solidity", 0.75)),
                            float(relax.get("min_solidity", 0.75)))

    return r


def detect(frame: np.ndarray, cfg: Optional[Dict] = None, template: Optional[np.ndarray] = None) -> Optional[Corners]:
    cfg = _merge_cfg(cfg)

    if cfg.get("debug"):
        print("[detect] merged cfg:", {
            "prefer": cfg.get("prefer"),
            "min_area_ratio": cfg.get("min_area_ratio"),
            "aspect_range": cfg.get("aspect_range"),
            "max_quad_epsilon": cfg.get("max_quad_epsilon"),
            "min_solidity": cfg.get("min_solidity"),
            "auto_relax": cfg.get("auto_relax", True)
        })

    def _run_once(effective_cfg: Dict) -> Optional[Corners]:
        if effective_cfg.get("prefer", "contours") == "contours":
            c = detect_by_contours(frame, effective_cfg)
            if c is not None:
                return c
            return detect_by_features(frame, template, effective_cfg) if template is not None else None
        else:
            c = detect_by_features(frame, template, effective_cfg) if template is not None else None
            if c is not None:
                return c
            return detect_by_contours(frame, effective_cfg)

    # First pass
    result = _run_once(cfg)
    if result is not None:
        pts = _expand_quad(frame.shape, result.pts, cfg)
        if cfg.get("quad_refine_enabled", True):
            pts = _refine_quad(frame.shape, pts, cfg)
        if cfg.get("local_refine", {}).get("enabled", True):
            pts = _local_edge_refine(frame, pts, cfg)
        pts = _apply_perspective_taper(frame.shape, pts, cfg)
        if cfg.get("upright_snap", False):
            pts = _upright_snap(frame.shape, pts, cfg)
        pts = order_corners_clockwise(pts)
        _save_quad_if_needed(pts, cfg)
        return Corners(pts=pts)

    # Auto-relax
    if cfg.get("auto_relax", True):
        rcfg = _relaxed_cfg(cfg)
        if cfg.get("debug"):
            print("[detect] no result → relaxing")
        res = _run_once(rcfg)
        if res is not None:
            pts = _expand_quad(frame.shape, res.pts, rcfg)
            if rcfg.get("quad_refine_enabled", True):
                pts = _refine_quad(frame.shape, pts, rcfg)
            if rcfg.get("local_refine", {}).get("enabled", True):
                pts = _local_edge_refine(frame, pts, rcfg)
            pts = _apply_perspective_taper(frame.shape, pts, rcfg)
            if rcfg.get("upright_snap", False):
                pts = _upright_snap(frame.shape, pts, rcfg)
            pts = order_corners_clockwise(pts)
            _save_quad_if_needed(pts, rcfg)
            return Corners(pts=pts)

    return None
