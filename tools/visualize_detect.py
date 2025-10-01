#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os
import cv2
import numpy as np

from mtgscan.geometry.detect import detect, is_plausible_quad, is_mtg_card_like
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
    ap = argparse.ArgumentParser(description="Run mtgscan detect() on a real image and visualize.")
    ap.add_argument("image", help="Path to input image (BGR/RGB supported).")
    ap.add_argument("--template", help="Path to template image (optional, enables feature path fallback).")
    ap.add_argument("--prefer", choices=["contours","features"], default="contours",
                    help='Pipeline preference; "features" requires --template.')
    ap.add_argument("--out", default=None, help="Output PNG filename (ignored if --out_dir is set).")
    ap.add_argument("--out_dir", default="tests/output", help="Directory to place the visualization PNG.")
    ap.add_argument("--gt", help="Path to ground-truth quad JSON [[x,y],...]. Optional.")
    ap.add_argument("--debug", action="store_true", help="Enable debug prints in detector.")
    # some sensible defaults for real photos
    ap.add_argument("--min_area_ratio", type=float, default=0.001, help="Min area ratio for plausible quads.")
    ap.add_argument("--aspect_min", type=float, default=0.40, help="Min card aspect (H/W) for geometry gate.")
    ap.add_argument("--aspect_max", type=float, default=3.00, help="Max card aspect (H/W) for geometry gate.")
    # card-likeness knobs
    ap.add_argument("--card_aspect", type=float, default=1.395, help="MTG oriented aspect target.")
    ap.add_argument("--card_aspect_tol", type=float, default=0.18, help="± tolerance around MTG oriented aspect.")
    ap.add_argument("--card_border_min", type=float, default=0.22, help="Min border contrast score to accept.")

    args = ap.parse_args()

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Could not read image: {args.image}")

    template = None
    if args.template:
        template = cv2.imread(args.template, cv2.IMREAD_COLOR)
        if template is None:
            raise SystemExit(f"Could not read template: {args.template}")

    cfg = {
        "prefer": args.prefer,
        "debug": args.debug,
        "min_area_ratio": args.min_area_ratio,
        "aspect_range": (args.aspect_min, args.aspect_max),
        # feature params that work well on real scenes
        "feature": {"nfeatures": 3000, "ratio": 0.90, "ransac_thresh": 3.0, "min_inliers": 12},
        # card-likeness
        "card_aspect": args.card_aspect,
        "card_aspect_tol": args.card_aspect_tol,
        "card_border_min": args.card_border_min,
    }

    res = detect(img, cfg=cfg, template=template)
    vis = img.copy()

    # prepare output path
    os.makedirs(args.out_dir, exist_ok=True)
    if args.out:
        out_path = os.path.join(args.out_dir, args.out)
    else:
        base = os.path.splitext(os.path.basename(args.image))[0] + "_viz.png"
        out_path = os.path.join(args.out_dir, base)

    if isinstance(res, Corners):
        quad = res.pts

        # 1) geometric plausibility (area/aspect gate that detect.py already uses)
        ok_geom = is_plausible_quad(quad, img.shape, cfg)

        # 2) our extra “Magic-card-like” check (orientation-invariant aspect + dark outer border)
        ok_card, score = is_mtg_card_like(img, quad, cfg)

        color = (0, 255, 0) if (ok_geom and ok_card) else (0, 0, 255)
        draw_quad(vis, quad, color, 3)

        print(f"Detection: OK={ok_geom and ok_card}, quad=\n{quad}")
        print(f"Card-like check: OK={ok_card}, score={score:.2f}")

        cv2.putText(vis, f"cardness:{score:.2f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

        # optional IoU to GT
        if args.gt:
            gt_quad = load_quad(args.gt)
            draw_quad(vis, gt_quad, (0, 255, 0), 2)  # green for GT
            iou = iou_quads(gt_quad, quad, img.shape)
            print(f"IoU vs GT: {iou:.3f}")
            cv2.putText(vis, f"IoU: {iou:.3f}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)
    else:
        print("No Corners detected.")
        cv2.putText(vis, "NO DETECTION", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2, cv2.LINE_AA)

    cv2.imwrite(out_path, vis)
    print(f"Saved visualization → {out_path}")

if __name__ == "__main__":
    main()
