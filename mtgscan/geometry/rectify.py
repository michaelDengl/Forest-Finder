# mtgscan/geometry/rectify.py
from __future__ import annotations
from typing import Tuple, Optional
import cv2
import numpy as np

# MTG card aspect (H/W). 63×88 mm ≈ 1.3968; we’ll default to that.
_MTG_ASPECT = 1.395

def _order_corners_clockwise(pts: np.ndarray) -> np.ndarray:
    """Return TL, TR, BR, BL (clockwise) given 4 unordered points."""
    p = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    # sort by y, then split to top/bottom and sort by x within each
    idx = np.argsort(p[:, 1])
    top = p[idx[:2]][np.argsort(p[idx[:2], 0])]
    bot = p[idx[2:]][np.argsort(p[idx[2:], 0])]
    tl, tr = top
    bl, br = bot
    return np.array([tl, tr, br, bl], dtype=np.float32)

def compute_target_size(
    *,
    width: Optional[int] = None,
    height: Optional[int] = None,
    aspect: float = _MTG_ASPECT,
) -> Tuple[int, int]:
    """
    Pick (W, H) (width, height) that matches the requested aspect (H/W).
    You can provide either width or height; the other side is derived.
    If neither is provided, use a sensible default of 744×1039 (~1.397).
    """
    if width is None and height is None:
        return 744, 1039  # crisp, fast to compute; keeps aspect ≈ MTG
    if width is not None and height is not None:
        return int(width), int(height)
    if width is not None:
        h = int(round(aspect * float(width)))
        return int(width), max(1, h)
    # else height provided
    w = int(round(float(height) / max(1e-6, aspect)))
    return max(1, w), int(height)

def warp_card(
    image: np.ndarray,
    quad_xy: np.ndarray,
    *,
    width: Optional[int] = None,
    height: Optional[int] = None,
    aspect: float = _MTG_ASPECT,
) -> np.ndarray:
    """
    Perspective-warp the detected quad into a rectified MTG card image.

    Args:
        image: BGR image.
        quad_xy: 4×2 float32 array (TL,TR,BR,BL order preferred; any order ok).
        width/height: optional target size; if one is missing, we derive it
                      from MTG aspect; if both missing, 744×1039 is used.
        aspect: target H/W (defaults to MTG).

    Returns:
        Rectified BGR image of shape (H, W, 3).
    """
    H, W = image.shape[:2]
    q = _order_corners_clockwise(quad_xy)
    # ensure inside frame
    q[:, 0] = np.clip(q[:, 0], 0, W - 1)
    q[:, 1] = np.clip(q[:, 1], 0, H - 1)

    dst_w, dst_h = compute_target_size(width=width, height=height, aspect=aspect)

    src = q.astype(np.float32)
    dst = np.array([[0, 0],
                    [dst_w - 1, 0],
                    [dst_w - 1, dst_h - 1],
                    [0, dst_h - 1]], dtype=np.float32)
    Hmat = cv2.getPerspectiveTransform(src, dst)
    rectified = cv2.warpPerspective(image, Hmat, (dst_w, dst_h), flags=cv2.INTER_LINEAR)
    return rectified
