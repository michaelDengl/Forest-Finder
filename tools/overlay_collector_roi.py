#!/usr/bin/env python3
import sys, argparse, cv2
from pathlib import Path
from mtgscan.roi.collector import extract_collector_roi, _load_cfg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--out", default=None)
    ap.add_argument("--cfg", default="config/roi.yaml")
    ap.add_argument("--resize", action="store_true")
    args = ap.parse_args()

    inp = Path(args.input)
    img = cv2.imread(str(inp))
    if img is None:
        print(f"[ERR] Cannot read {inp}")
        sys.exit(2)

    cfg = _load_cfg(args.cfg)
    Hc, Wc = cfg["canonical"]["height"], cfg["canonical"]["width"]
    if args.resize and (img.shape[0] != Hc or img.shape[1] != Wc):
        img = cv2.resize(img, (Wc, Hc), interpolation=cv2.INTER_AREA)

    r = extract_collector_roi(img, cfg_path=args.cfg)
    vis = img.copy()
    x0,y0,x1,y1 = r.xyxy_px
    cv2.rectangle(vis, (x0,y0), (x1,y1), (0,255,0), 2)
    cv2.putText(vis, f"tpl={r.template}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
    out = Path(args.out) if args.out else inp.with_name(inp.stem + "_collector_overlay.png")
    cv2.imwrite(str(out), vis)
    print(f"[OK] template={r.template}, overlay saved -> {out}")

if __name__ == "__main__":
    main()
