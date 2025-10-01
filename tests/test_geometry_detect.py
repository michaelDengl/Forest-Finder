"""
Pytest for Story 1: Robust Card Detection with feature fallback.
These tests generate synthetic images on the fly, so no test assets are required.
"""
from __future__ import annotations
import os
import uuid

import numpy as np
import cv2
import pytest

from mtgscan.geometry.detect import (
    detect,
    detect_by_contours,
    detect_by_features,
    order_corners_clockwise,
    is_plausible_quad,
)
from mtgscan.core.contracts import Corners

# ---------- Utilities to build synthetic scenes ---------- #

def _make_textured_template_card(w: int = 400, h: int = 560) -> np.ndarray:
    """Create a synthetic 'card' template with internal texture so ORB has features."""
    card = np.full((h, w, 3), 235, np.uint8)
    cv2.rectangle(card, (6, 6), (w - 7, h - 7), (20, 20, 20), 3)
    for x in range(30, w, 40):
        cv2.line(card, (x, 20), (x, h - 20), (60, 60, 60), 1)
    for y in range(40, h, 40):
        cv2.line(card, (20, y), (w - 20, y), (60, 60, 60), 1)
    cv2.putText(card, "MAGIC", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(card, "THE GATHERING", (40, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.circle(card, (int(w * 0.85), int(h * 0.1)), 18, (0, 0, 0), 3)
    cv2.circle(card, (int(w * 0.15), int(h * 0.85)), 18, (0, 0, 0), 3)
    return card

def _place_card_in_frame(template: np.ndarray, frame_w: int = 1000, frame_h: int = 750,
                         jitter: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """Warp the template by a random homography and paste into a larger frame."""
    Ht, Wt = template.shape[:2]
    frame = np.full((frame_h, frame_w, 3), 30, np.uint8)
    rng = np.random.default_rng(42)
    margin = 80
    pts_dst = np.array([
        [margin + rng.integers(0, jitter), margin + rng.integers(0, jitter)],
        [frame_w - margin - rng.integers(0, jitter), margin + rng.integers(0, jitter)],
        [frame_w - margin - rng.integers(0, jitter), frame_h - margin - rng.integers(0, jitter)],
        [margin + rng.integers(0, jitter), frame_h - margin - rng.integers(0, jitter)],
    ], dtype=np.float32)
    pts_src = np.array([[0, 0], [Wt - 1, 0], [Wt - 1, Ht - 1], [0, Ht - 1]], dtype=np.float32)
    Hmat = cv2.getPerspectiveTransform(pts_src, pts_dst)
    warped = cv2.warpPerspective(template, Hmat, (frame_w, frame_h))
    mask = np.zeros((frame_h, frame_w), np.uint8)
    cv2.fillConvexPoly(mask, pts_dst.astype(np.int32), 255)
    inv = cv2.bitwise_not(mask)
    bg = cv2.bitwise_and(frame, frame, mask=inv)
    fg = cv2.bitwise_and(warped, warped, mask=mask)
    out = cv2.add(bg, fg)
    return out, pts_dst

def _add_shadow(frame: np.ndarray) -> np.ndarray:
    """Add a soft shadow to challenge contour detection a bit."""
    H, W = frame.shape[:2]
    overlay = frame.copy()
    shadow = np.zeros_like(frame)
    poly = np.array([
        [int(W * 0.2), int(H * 0.1)],
        [int(W * 0.9), int(H * 0.2)],
        [int(W * 0.8), int(H * 0.9)],
        [int(W * 0.15), int(H * 0.7)],
    ], dtype=np.int32)
    cv2.fillConvexPoly(shadow, poly, (0, 0, 0))
    shadow = cv2.GaussianBlur(shadow, (51, 51), 0)
    return cv2.addWeighted(overlay, 1.0, shadow, 0.25, 0)

# ---------- Debug helper (NEW) ---------- #

def _debug_visualize_on_low_iou(frame: np.ndarray,
                                true_corners: np.ndarray,
                                got_corners: np.ndarray,
                                iou: float,
                                tag: str = "feature_fallback") -> None:
    """
    If IoU is low, draw GT (green) and predicted (red) quads and save a PNG to tests/output/.
    """
    try:
        dbg = frame.copy()
        cv2.polylines(dbg, [true_corners.astype(np.int32)], True, (0, 255, 0), 2)  # green = GT
        cv2.polylines(dbg, [got_corners.astype(np.int32)], True, (0, 0, 255), 2)   # red = predicted
        os.makedirs("tests/output", exist_ok=True)
        fname = f"tests/output/{tag}_{str(uuid.uuid4())[:8]}_iou_{iou:.3f}.png"
        cv2.imwrite(fname, dbg)
        print(f"[debug] Saved low-IoU visualization → {fname}")
    except Exception as e:
        print(f"[debug] Failed to save visualization: {e}")

# ---------- Tests ---------- #

def test_order_corners_clockwise_basic():
    pts = np.array([[100, 50], [400, 60], [420, 500], [90, 480]], dtype=np.float32)
    np.random.default_rng(0).shuffle(pts)
    ordered = order_corners_clockwise(pts)
    s = ordered.sum(axis=1)
    assert np.argmin(s) == 0  # TL
    assert np.argmax(s) == 2  # BR

def test_detect_by_contours_happy_path():
    template = _make_textured_template_card()
    frame, _ = _place_card_in_frame(template)
    frame = _add_shadow(frame)
    cfg = {
        "prefer": "features",
        "debug": True,
        "feature": {
            "nfeatures": 6000,
            "ratio": 0.90,        # was 0.98 — much stricter
            "ransac_thresh": 3.0, # was 8.0 — tighter model
            "min_inliers": 12,    # was 3  — avoid flimsy H
        },
        "min_area_ratio": 0.002,
        "aspect_range": (0.4, 2.6),
    }

    # Override with lenient, synthetic-friendly settings + debug
    cfg.update({
        "prefer": "features",  # force feature path
        "debug": True,         # show debug prints from detect_by_features
        "feature": {"nfeatures": 6000, "ratio": 0.98, "ransac_thresh": 8.0, "min_inliers": 3},
        "min_area_ratio": 0.002,   # accept small minAreaRect fallback
        "aspect_range": (0.4, 2.6),
    })

    got = detect_by_contours(frame, cfg)
    assert isinstance(got, Corners)
    pts = got.pts
    assert is_plausible_quad(pts, frame.shape, cfg)
    assert pts.shape == (4, 2)
    assert pts.dtype == np.float32

def test_feature_fallback_works():
    template = _make_textured_template_card()
    frame, true_corners = _place_card_in_frame(template)

    # Lenient, synthetic-friendly config + debug
    cfg = {
        "prefer": "features",   # force feature path
        "debug": True,          # will print match counts/inliers
        "feature": {
            "nfeatures": 6000,
            "ratio": 0.98,
            "ransac_thresh": 8.0,
            "min_inliers": 3,
        },
        "min_area_ratio": 0.002,   # accept small minAreaRect fallback
        "aspect_range": (0.4, 2.6),
    }

    # Call feature path directly to isolate it
    got = detect_by_features(frame, template, cfg)
    assert isinstance(got, Corners), "feature path returned None"
    pts = got.pts
    assert is_plausible_quad(pts, frame.shape, cfg)

    # IoU vs ground truth
    def quad_to_mask(quad: np.ndarray, shape):
        m = np.zeros(shape[:2], np.uint8)
        cv2.fillConvexPoly(m, quad.astype(np.int32), 255)
        return m

    m_true = quad_to_mask(true_corners, frame.shape)
    m_got = quad_to_mask(pts, frame.shape)
    inter = np.logical_and(m_true > 0, m_got > 0).sum()
    union = np.logical_or(m_true > 0, m_got > 0).sum()
    iou = inter / max(1, union)

    # NEW: save a visualization when IoU is low to speed up debugging
    if iou < 0.7:
        _debug_visualize_on_low_iou(frame, true_corners, pts, iou, tag="feature_fallback")

    assert iou >= 0.7

def test_low_iou_visualization_writes_file():
    # Make a simple black frame
    frame = np.zeros((200, 300, 3), np.uint8)

    # Ground truth square (green) and a shifted predicted square (red) to ensure low IoU
    true_corners = np.array([[10, 10], [110, 10], [110, 110], [10, 110]], dtype=np.float32)
    got_corners  = true_corners + 80  # shift so they barely overlap
    fake_iou = 0.2  # definitely < 0.7

    # Capture directory contents before
    outdir = "tests/output"
    before = set(os.listdir(outdir)) if os.path.isdir(outdir) else set()

    # Call the hook directly
    _debug_visualize_on_low_iou(frame, true_corners, got_corners, fake_iou, tag="viz_test")

    # Check that at least one new PNG appeared with our tag
    after = set(os.listdir(outdir))
    new_files = sorted(list(after - before))
    assert any(f.startswith("viz_test_") and f.endswith(".png") for f in new_files), \
        "Expected a visualization PNG to be created in tests/output/"
