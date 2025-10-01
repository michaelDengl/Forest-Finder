#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os
import cv2
import numpy as np

from mtgscan.geometry.detect import detect, is_plausible_quad
from mtgscan.core.contracts import Corners

def draw_quad(img, quad, color, thickness=2):
    q = quad.astype(int).reshape(4, 2)
    cv2.polylines(img, [q], True, color, thickness, lineType=cv2.LINE_AA)

def load_quad(path):
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
    ap = argparse.ArgumentParser(description="Run mtgscan detect() on a real image and visualize.")
    ap.add_argument("image", help="Path to input image (BGR/RGB supported).")
    ap.add_argument("--template", help="Path to template image (optional, enables feature path fallback).")
    ap.add_argument("--prefer", choices=["contours","features"], default=None,
                    help='Pipeline preference; "features" requires --template.')
    ap.add_argument("--out", default=None, help="Output PNG path. Defaults to <image_basename>_viz.png")
    ap.add_argument("--gt", help="Path to ground-truth quad JSON [[x,y],...]. Optional.")
    ap.add_argument("--debug", action="store_true", help="Enable debug prints in detector.")

    # --- make all these optional (no override if not passed)
    ap.add_argument("--min_area_ratio", type=float, default=None, help="Min area ratio for plausible quads.")
    ap.add_argument("--aspect_min", type=float, default=None, help="Min card aspect (H/W).")
    ap.add_argument("--aspect_max", type=float, default=None, help="Max card aspect (H/W).")

    args = ap.parse_args()

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Could not read image: {args.image}")

    template = None
    if args.template:
        template = cv2.imread(args.template, cv2.IMREAD_COLOR)
        if template is None:
            raise SystemExit(f"Could not read template: {args.template}")

    # build config only from provided flags
    cfg = {}
    if args.prefer is not None:
        cfg["prefer"] = args.prefer
    if args.debug:
        cfg["debug"] = True
    if args.min_area_ratio is not None:
        cfg["min_area_ratio"] = args.min_area_ratio
    if args.aspect_min is not None and args.aspect_max is not None:
        cfg["aspect_range"] = (args.aspect_min, args.aspect_max)

    # add stronger feature settings if template is used
    if args.template:
        cfg.setdefault("feature", {})
        cfg["feature"].update({"nfeatures": 3000, "ratio": 0.90,
                               "ransac_thresh": 3.0, "min_inliers": 12})

    res = detect(img, cfg=cfg, template=template)
    vis = img.copy()

    if isinstance(res, Corners):
        quad = res.pts
        draw_quad(vis, quad, (0, 0, 255), 3)  # red
        ok = is_plausible_quad(quad, img.shape, cfg)
        print(f"Detection: OK={ok}, quad=\n{quad}")

        if args.gt:
            gt_quad = load_quad(args.gt)
            draw_quad(vis, gt_quad, (0, 255, 0), 2)
            iou = iou_quads(gt_quad, quad, img.shape)
            print(f"IoU vs GT: {iou:.3f}")
            cv2.putText(vis, f"IoU: {iou:.3f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)
    else:
        print("No Corners detected.")
        cv2.putText(vis, "NO DETECTION", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2, cv2.LINE_AA)

    out_dir = "tests/output"
    os.makedirs(out_dir, exist_ok=True)
    out = args.out or os.path.join(
        out_dir, os.path.splitext(os.path.basename(args.image))[0] + "_viz.png")
    cv2.imwrite(out, vis)
    print(f"Saved visualization → {out}")

if __name__ == "__main__":
    main()
