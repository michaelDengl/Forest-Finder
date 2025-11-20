#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os
import cv2
import numpy as np

from mtgscan.geometry.detect import detect, is_plausible_quad, card_likeness_score
from mtgscan.geometry.rectify import warp_card, compute_target_size
from mtgscan.core.contracts import Corners
import sys
from datetime import datetime

# --- Simple log-to-file wrapper ---
logfile = f"detect_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log = open(logfile, "w")

class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

sys.stdout = Tee(sys.stdout, log)
sys.stderr = Tee(sys.stderr, log)

print(f"[logging] Writing debug output to: {logfile}")


def draw_quad(img, quad, color, thickness=2):
    q = quad.astype(int).reshape(4, 2)
    cv2.polylines(img, [q], True, color, thickness, lineType=cv2.LINE_AA)

def load_quad(path):
    # expects JSON: [[x,y],[x,y],[x,y],[x,y]] in TL,TR,BR,BL order
    with open(path, "r") as f:
        arr = np.array(json.load(f), dtype=np.float32)
    return arr.reshape(4, 2)

def mask_from_quad(quad, shape):
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillConvexPoly(m, quad.astype(np.int32), 255)
    return m

def iou_quads(q1, q2, shape):
    m1 = mask_from_quad(q1, shape)
    m2 = mask_from_quad(q2, shape)
    inter = np.logical_and(m1 > 0, m2 > 0).sum()
    union = np.logical_or(m1 > 0, m2 > 0).sum()
    return inter / max(1, union)

def main():
    ap = argparse.ArgumentParser(description="Run mtgscan detect() on a real image, visualize, and rectify.")
    ap.add_argument("image", help="Path to input image (BGR/RGB supported).")
    ap.add_argument("--template", help="Path to template image (optional, enables feature path fallback).")
    ap.add_argument("--prefer", choices=["contours","features"], default="contours",
                    help='Pipeline preference; "features" requires --template.')
    ap.add_argument("--out_dir", default="tests/output", help="Directory for outputs.")
    ap.add_argument("--out", default=None, help="Output viz PNG path. Default: <out_dir>/<image_basename>_viz.png")
    ap.add_argument("--rect", default=None, help="Rectified PNG path. Default: <out_dir>/<image_basename>_rect.png")
    ap.add_argument("--rect_w", type=int, default=None, help="Rectified width (H will follow MTG aspect if not given).")
    ap.add_argument("--rect_h", type=int, default=None, help="Rectified height (W will follow MTG aspect if not given).")
    ap.add_argument("--gt", help="Path to ground-truth quad JSON [[x,y],...]. Optional.")
    ap.add_argument("--debug", action="store_true", help="Enable debug prints in detector.")
    # sensible defaults for real photos
    ap.add_argument("--min_area_ratio", type=float, default=None)
    ap.add_argument("--aspect_min", type=float, default=None)
    ap.add_argument("--aspect_max", type=float, default=None)

    args = ap.parse_args()

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Could not read image: {args.image}")
    
    # Rotate camera image 90° clockwise because camera is mounted 90° left
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    # After rotation
    cv2.imwrite("tests/output/DEBUG_01_rotated_input.png", img)
    print("[dbg] saved DEBUG_01_rotated_input.png", img.shape)
    # --- Save debug preprocessing images ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("tests/output/DEBUG_02_gray.png", gray)

    blur = cv2.GaussianBlur(gray, (5,5), 0)
    cv2.imwrite("tests/output/DEBUG_03_blur.png", blur)

    canny = cv2.Canny(blur, 75, 200)
    cv2.imwrite("tests/output/DEBUG_04_canny.png", canny)

    # Binary thresholding tests
    _, b1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite("tests/output/DEBUG_05_b1.png", b1)

    _, b2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imwrite("tests/output/DEBUG_06_b2.png", b2)

    b3 = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 5
    )
    cv2.imwrite("tests/output/DEBUG_07_b3.png", b3)

    print("[dbg] saved DEBUG_02..07 preprocessing images")


    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.image))[0]
    out_viz = args.out or os.path.join(args.out_dir, f"{base}_viz.png")
    out_rect = args.rect or os.path.join(args.out_dir, f"{base}_rect.png")

    template = None
    if args.template:
        template = cv2.imread(args.template, cv2.IMREAD_COLOR)
        if template is None:
            raise SystemExit(f"Could not read template: {args.template}")

    cfg = {
    "prefer": args.prefer,
    "debug": args.debug,
    # TEMP: relax filters so we can see what the detector *would* pick
    "require_portrait": False,      # turn off portrait gate for now
    "min_area_ratio": 0.0015,       # allow smaller candidates
    "min_abs_area_px": 8000.0,      # slightly relaxed absolute area
    }  

    if args.min_area_ratio is not None:
        cfg["min_area_ratio"] = args.min_area_ratio
    if args.aspect_min is not None or args.aspect_max is not None:
        lo = args.aspect_min if args.aspect_min is not None else _DEFAULT_LO  # 0.40
        hi = args.aspect_max if args.aspect_max is not None else _DEFAULT_HI  # 3.00
        cfg["aspect_range"] = (lo, hi)
    # keep feature params if you want, or let detect() defaults handle them


    res = detect(img, cfg=cfg, template=template)
    vis = img.copy()

    if isinstance(res, Corners):
        quad = res.pts  # get the points first

        # Extra debug info
        score = card_likeness_score(quad, img, cfg)  # ← pass img, not img.shape
        h, w = img.shape[:2]
        area = cv2.contourArea(quad.astype(np.float32))
        area_ratio = area / float(w * h)
        print(f"[dbg] likeness_score={score:.3f}, area={area:.1f}, area%={area_ratio:.4f}, h={h}, w={w}")

        # visualize detection
        draw_quad(vis, quad, (0, 255, 0), 3)
        ok = is_plausible_quad(quad, img.shape, cfg)
        print(f"Detection: OK={ok}, quad=\n{quad}")

        # --- Perspective correction
        dst_w, dst_h = compute_target_size(width=args.rect_w, height=args.rect_h)
        rectified = warp_card(img, quad, width=dst_w, height=dst_h)
        cv2.imwrite(out_rect, rectified)
        print(f"Saved rectified → {out_rect}")



    else:
        print("No Corners detected. Card-like check skipped.")
        cv2.putText(vis, "NO DETECTION", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2, cv2.LINE_AA)

    cv2.imwrite(out_viz, vis)
    print(f"Saved visualization → {out_viz}")

if __name__ == "__main__":
    main()
