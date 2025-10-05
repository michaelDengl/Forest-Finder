# tools/read_collector_line.py
#!/usr/bin/env python3
import sys, argparse, cv2
from pathlib import Path
from mtgscan.roi.collector import extract_collector_roi, _load_cfg
from mtgscan.ocr.collector_line import read_collector_line, _split_blocks, _prep

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rectified")
    ap.add_argument("--resize", action="store_true")
    ap.add_argument("--dump", action="store_true")
    args = ap.parse_args()

    p = Path(args.rectified)
    img = cv2.imread(str(p))
    if img is None:
        print(f"[ERR] cannot read {p}"); sys.exit(2)

    cfg = _load_cfg("config/roi.yaml")
    Hc, Wc = cfg["canonical"]["height"], cfg["canonical"]["width"]
    if args.resize and (img.shape[0] != Hc or img.shape[1] != Wc):
        img = cv2.resize(img, (Wc, Hc), interpolation=cv2.INTER_AREA)

    r = extract_collector_roi(img)
    out = read_collector_line(r.crop)

    print(f"template={r.template} holo={r.meta.get('holo_detected')}")
    print(f"raw='{out['raw']}'  conf={out['conf']:.2f}")
    print(f"set_code={out['set_code']}  collector_number={out['collector_number']}  language={out.get('language')}")
    #                                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    if args.dump:
        num_bgr, set_bgr = _split_blocks(r.crop)
        num_bin = _prep(num_bgr, scale=4)
        set_bin = _prep(set_bgr, scale=6)
        cv2.imwrite(str(p.with_name(p.stem+"_num_bin.png")), num_bin)
        cv2.imwrite(str(p.with_name(p.stem+"_set_bin.png")), set_bin)
        print("[DBG] wrote", p.with_name(p.stem+"_num_bin.png"), "and", p.with_name(p.stem+"_set_bin.png"))

if __name__ == "__main__":
    main()
