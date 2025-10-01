#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os
import cv2
import numpy as np

from mtgscan.geometry.detect import detect, is_plausible_quad, card_likeness_score
from mtgscan.geometry.rectify import warp_card, compute_target_size
from mtgscan.core.contracts import Corners

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
        quad = res.pts
        # visualize detection
        draw_quad(vis, quad, (0, 255, 0), 3)  # green
        ok = is_plausible_quad(quad, img.shape, cfg)
        print(f"Detection: OK={ok}, quad=\n{quad}")


        # optional IoU to GT
        if args.gt:
            gt_quad = load_quad(args.gt)
            draw_quad(vis, gt_quad, (0, 255, 0), 2)  # green
            iou = iou_quads(gt_quad, quad, img.shape)
            print(f"IoU vs GT: {iou:.3f}")
            cv2.putText(vis, f"IoU: {iou:.3f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)

        # --- Perspective correction
        dst_w, dst_h = compute_target_size(width=args.rect_w, height=args.rect_h)  # uses MTG aspect when one/both missing
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
