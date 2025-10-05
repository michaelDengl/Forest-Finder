#!/usr/bin/env python3
import sys, cv2
from pathlib import Path
from mtgscan.roi.collector import extract_collector_roi

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 -m tools.dump_collector_roi <rectified.png> [out.png]")
        sys.exit(2)
    inp = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv)>2 else inp.with_name(inp.stem + "_collector.png")
    img = cv2.imread(str(inp))
    r = extract_collector_roi(img)
    if r.crop is None:
        print("[ERR] Empty crop (check rectification size or ROI config).")
        sys.exit(1)
    cv2.imwrite(str(out), r.crop)
    print(f"[OK] template={r.template}, crop saved to: {out}")
if __name__ == "__main__":
    main()
