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
    "min_area_ratio": 0.001,           # was 0.005
    "aspect_range": (0.40, 3.00),      # was (0.5, 2.2)
    "max_quad_epsilon": 0.08,          # was 0.05
    "min_solidity": 0.75,              # was 0.80
    "canny": {"low": 75, "high": 200},
    "blur": {"ksize": 5},
    "feature": {
        "nfeatures": 2000,
        "ratio": 0.85,
        "ransac_thresh": 5.0,
        "min_inliers": 8
    },
    "prefer": "contours",
    "debug": False,

    # --- automatic fallback pass if nothing is detected on the first pass
    "auto_relax": True,
    "relax": {
        "min_area_ratio": 0.001,
        "aspect_range": (0.40, 3.00),
        "max_quad_epsilon": 0.08,
        "min_solidity": 0.75
    }
}


# ----------------------------------------------------------------------------- #
# Config / utilities                                                            #
# ----------------------------------------------------------------------------- #

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
    eps = 5e-6
    if area <= eps or area_ratio + eps < min_ratio:
        if cfg.get("debug"):
            print(f"[plaus] area fail: area={area:.1f}, ratio={area_ratio:.4f}, min={min_ratio}")
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
        if cfg.get("debug"): print("[plaus] degenerate width/height:", width, height)
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
# Contour path                                                                   #
# ----------------------------------------------------------------------------- #

def detect_by_contours(frame: np.ndarray, cfg: Optional[Dict] = None) -> Optional[Corners]:
    cfg = _merge_cfg(cfg)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    k = int(cfg["blur"]["ksize"])
    if k > 1:
        if k % 2 == 0: k += 1
        gray_blur = cv2.GaussianBlur(gray, (k, k), 0)
    else:
        gray_blur = gray
    def _binary_candidates(g):
        _, b1 = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, b2 = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        b3 = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5)
        return [b1, b2, b3]
    for b in _binary_candidates(gray_blur):
        b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), 1)
        cnts, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        corners = _choose_best_quad(cnts, frame, cfg)
        if corners is not None: return corners
    edges = cv2.Canny(gray_blur, cfg["canny"]["low"], cfg["canny"]["high"])
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), 1)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return _choose_best_quad(cnts, frame, cfg)

def _choose_best_quad(cnts, frame: np.ndarray, cfg: Dict) -> Optional[Corners]:
    if not cnts: return None
    H, W = frame.shape[:2]
    frame_area = float(H * W)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    eps_scale = float(cfg["max_quad_epsilon"])
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps_scale * peri, True)
        pts = None
        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype(np.float32)
        else:
            for factor in (1.5, 2.0, 3.0):
                approx2 = cv2.approxPolyDP(c, eps_scale * factor * peri, True)
                if len(approx2) == 4:
                    pts = approx2.reshape(4, 2).astype(np.float32); break
            if pts is None:
                box = cv2.boxPoints(cv2.minAreaRect(c))
                pts = box.astype(np.float32)
        pts = order_corners_clockwise(pts)
        if not is_plausible_quad(pts, frame.shape, cfg): continue
        area = cv2.contourArea(pts)
        hull = cv2.convexHull(pts)
        hull_area = cv2.contourArea(hull)
        solidity = (area / hull_area) if hull_area > 0 else 0.0
        if solidity < cfg["min_solidity"]: continue
        if (area / frame_area) < cfg["min_area_ratio"]: continue
        return Corners(pts=pts.astype(np.float32))
    return None

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
    """Build a second-pass config that widens the gates but preserves user overrides where they’re already looser."""
    r = dict(cfg)
    relax = cfg.get("relax", {})
    # Only loosen — never tighten user-supplied values
    r["min_area_ratio"]   = min(float(cfg.get("min_area_ratio", 0.005)), float(relax.get("min_area_ratio", 0.001)))
    lo, hi = cfg.get("aspect_range", (0.5, 2.2))
    r_lo, r_hi = relax.get("aspect_range", (0.40, 3.00))
    r["aspect_range"] = (min(lo, r_lo), max(hi, r_hi))
    r["max_quad_epsilon"] = max(float(cfg.get("max_quad_epsilon", 0.05)), float(relax.get("max_quad_epsilon", 0.08)))
    r["min_solidity"]     = min(float(cfg.get("min_solidity", 0.80)), float(relax.get("min_solidity", 0.75)))
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
            return c if c is not None else detect_by_contours(frame, effective_cfg)

    # Pass 1: user/CLI config
    result = _run_once(cfg)
    if result is not None:
        return result

    # Optional Pass 2: auto-relax if nothing found
    if cfg.get("auto_relax", True):
        rcfg = _relaxed_cfg(cfg)
        if cfg.get("debug"):
            print("[detect] no result → relaxing gates:", {
                "min_area_ratio": rcfg.get("min_area_ratio"),
                "aspect_range": rcfg.get("aspect_range"),
                "max_quad_epsilon": rcfg.get("max_quad_epsilon"),
                "min_solidity": rcfg.get("min_solidity"),
            })
        return _run_once(rcfg)

    return None
